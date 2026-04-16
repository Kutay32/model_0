"""
CBIS-DDSM (Kaggle awsaf49 bundle) path helpers and manifest builders.

Expected layout under dataset root:
  csv/  (mass_case_description_* , calc_case_description_* , dicom_info.csv)
  jpeg/ (tree referenced by CSV paths)
"""
from __future__ import annotations

import os
import random
import re
from typing import Iterable

import pandas as pd


def _norm_key(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {_norm_key(c): c for c in df.columns}


def pick_column(df: pd.DataFrame, *candidates: str) -> str | None:
    cmap = column_lookup(df)
    for c in candidates:
        k = _norm_key(c)
        if k in cmap:
            return cmap[k]
    return None


_PATIENT_TOKEN = re.compile(r"(P_\d+)", re.IGNORECASE)


def _clean_path_cell(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip().replace("\\", "/")
    s = s.split("\n")[0].strip().strip('"').strip()
    return s


def resolve_cbis_path(raw: object, dataset_root: str) -> str:
    """Map a path from CBIS CSV to a local filesystem path (Kaggle uses jpeg/*.jpg)."""
    s = _clean_path_cell(raw)
    if not s:
        return ""
    root = os.path.abspath(dataset_root)
    if "CBIS-DDSM" in s:
        idx = s.find("CBIS-DDSM")
        tail = s[idx + len("CBIS-DDSM") :].lstrip("/")
        candidate = os.path.normpath(os.path.join(root, tail))
    elif s.lower().startswith("jpeg/"):
        candidate = os.path.normpath(os.path.join(root, s))
    else:
        candidate = os.path.normpath(os.path.join(root, "jpeg", s.lstrip("/")))

    if os.path.isfile(candidate):
        return candidate
    base, ext = os.path.splitext(candidate)
    if ext.lower() == ".dcm":
        for e in (".jpg", ".jpeg", ".png", ".dcm"):
            alt = base + e
            if os.path.isfile(alt):
                return alt
        return base + ".jpg"
    return candidate


def _read_case_csvs(
    dataset_root: str, which: Iterable[str]
) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    csv_dir = os.path.join(dataset_root, "csv")
    for name in which:
        p = os.path.join(csv_dir, name)
        if not os.path.isfile(p):
            continue
        out.append((name, pd.read_csv(p)))
    return out


_JPEG_UID_RE = re.compile(r"jpeg/([0-9.]+)/", re.IGNORECASE)


def _uid_jpeg_maps_from_dicom(dicom: pd.DataFrame, dataset_root: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Map SeriesInstanceUID-style folder (numeric.dots under jpeg/) to one crop JPEG and one ROI mask JPEG.
    CBIS case CSV paths do not match on-disk layout; dicom_info.csv is authoritative.
    """
    c_img = pick_column(dicom, "image_path", "ImagePath")
    c_series = pick_column(dicom, "SeriesDescription", "Series Description")
    crop_by_uid: dict[str, str] = {}
    mask_by_uid: dict[str, str] = {}
    if not c_img or not c_series:
        return crop_by_uid, mask_by_uid
    for _, row in dicom.iterrows():
        if pd.isna(row.get(c_img)) or pd.isna(row.get(c_series)):
            continue
        raw_ip = str(row[c_img])
        m = _JPEG_UID_RE.search(raw_ip.replace("\\", "/"))
        if not m:
            continue
        uid = m.group(1)
        full = resolve_cbis_path(raw_ip, dataset_root)
        if not full or not os.path.isfile(full):
            continue
        sd = str(row[c_series]).lower()
        if "cropped images" in sd:
            crop_by_uid.setdefault(uid, full)
        elif "roi mask" in sd:
            mask_by_uid.setdefault(uid, full)
    return crop_by_uid, mask_by_uid


def collect_abnormal_patient_tokens(dataset_root: str) -> set[str]:
    """Normalized P_XXXXX tokens for patients with annotated masses or calcifications."""
    ids: set[str] = set()
    case_names = [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    ]
    for _, df in _read_case_csvs(dataset_root, case_names):
        for _, row in df.iterrows():
            for c in df.columns:
                if df[c].dtype != object:
                    continue
                val = row.get(c)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                m = _PATIENT_TOKEN.search(str(val))
                if m:
                    ids.add(m.group(1).upper())
    return ids


def build_segmentation_manifest(dataset_root: str) -> pd.DataFrame:
    """
    Rows: cropped lesion patch + ROI mask. Official train CSV -> split train;
    official test CSV -> split val.
    """
    dicom_path = os.path.join(dataset_root, "csv", "dicom_info.csv")
    if not os.path.isfile(dicom_path):
        return pd.DataFrame()
    dicom = pd.read_csv(dicom_path)
    crop_by_uid, mask_by_uid = _uid_jpeg_maps_from_dicom(dicom, dataset_root)

    rows: list[dict] = []
    specs = [
        ("mass_case_description_train_set.csv", "train"),
        ("calc_case_description_train_set.csv", "train"),
        ("mass_case_description_test_set.csv", "val"),
        ("calc_case_description_test_set.csv", "val"),
    ]
    for fname, split in specs:
        p = os.path.join(dataset_root, "csv", fname)
        if not os.path.isfile(p):
            continue
        df = pd.read_csv(p)
        c_crop = pick_column(df, "cropped image file path", "cropped_image_file_path")
        c_path = pick_column(df, "pathology", "Pathology")
        c_pid = pick_column(df, "patient_id", "Patient ID")
        if not c_crop:
            continue
        for _, row in df.iterrows():
            patho = str(row[c_path]).strip().upper() if c_path and pd.notna(row.get(c_path)) else ""
            if patho not in {"BENIGN", "MALIGNANT"}:
                continue
            rel = _clean_path_cell(row[c_crop])
            parts = rel.split("/")
            if len(parts) < 2:
                continue
            folder_uid = parts[-2]
            img_p = crop_by_uid.get(folder_uid)
            mask_p = mask_by_uid.get(folder_uid)
            if not img_p or not mask_p:
                continue
            pid = ""
            if c_pid and pd.notna(row.get(c_pid)):
                pid = str(row[c_pid]).strip()
            rows.append(
                {
                    "split": split,
                    "image_path": img_p,
                    "mask_path": mask_p,
                    "pathology": patho.lower(),
                    "source_csv": fname,
                    "patient_id": pid,
                }
            )
    return pd.DataFrame(rows)


def build_classification_manifest(
    dataset_root: str,
    max_normal: int = 4000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Benign / malignant: cropped patches from case CSVs (same as segmentation table).
    Normal: full-field mammogram JPEGs from dicom_info.csv, excluding abnormal patients.
    """
    seg_df = build_segmentation_manifest(dataset_root)
    clip_rows: list[dict] = []
    for _, r in seg_df.iterrows():
        label = "malignant" if r["pathology"] == "malignant" else "benign"
        clip_rows.append(
            {
                "split": r["split"],
                "image_path": r["image_path"],
                "label": label,
                "kind": "lesion_crop",
            }
        )

    dicom_path = os.path.join(dataset_root, "csv", "dicom_info.csv")
    if os.path.isfile(dicom_path):
        dicom = pd.read_csv(dicom_path)
        c_img = pick_column(dicom, "image_path", "ImagePath", "File Location")
        c_series = pick_column(dicom, "SeriesDescription", "Series Description")
        c_patient = pick_column(dicom, "Patient ID", "patient_id")
        abnormal = collect_abnormal_patient_tokens(dataset_root)
        rng = random.Random(random_seed)
        normals: list[dict] = []
        c_name = pick_column(dicom, "PatientName", "patient_name", "Patient Name")
        for _, row in dicom.iterrows():
            if not c_img or not c_series or pd.isna(row.get(c_series)) or pd.isna(row.get(c_img)):
                continue
            desc = str(row[c_series]).lower()
            if "full mammogram" not in desc:
                continue
            if "cropped" in desc or "mask" in desc or "roi" in desc:
                continue
            blob = ""
            if c_patient and pd.notna(row.get(c_patient)):
                blob += str(row[c_patient])
            if c_name and pd.notna(row.get(c_name)):
                blob += str(row[c_name])
            m = _PATIENT_TOKEN.search(blob)
            if m and m.group(1).upper() in abnormal:
                continue
            img_p = resolve_cbis_path(str(row[c_img]), dataset_root)
            if not img_p or not os.path.isfile(img_p):
                continue
            low = img_p.lower()
            if "test" in low or "-test-" in low or "_test_" in low:
                sp = "val"
            elif "training" in low or "-train-" in low or "_train_" in low:
                sp = "train"
            else:
                sp = "train"
            normals.append({"split": sp, "image_path": img_p, "label": "normal", "kind": "full_field"})
        rng.shuffle(normals)
        clip_rows.extend(normals[:max_normal])

    return pd.DataFrame(clip_rows)


def write_manifests(dataset_root: str, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    seg = build_segmentation_manifest(dataset_root)
    seg_path = os.path.join(out_dir, "manifest_segmentation.csv")
    seg.to_csv(seg_path, index=False)
    cls = build_classification_manifest(dataset_root)
    cls_path = os.path.join(out_dir, "manifest_classification.csv")
    cls.to_csv(cls_path, index=False)
    return seg_path, cls_path
