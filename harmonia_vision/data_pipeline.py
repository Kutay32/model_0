"""
CBIS-DDSM mass segmentation preprocessing for Harmonia Vision.

Loads only mass_*.csv manifests (ignores calcification CSVs), converts DICOM with
windowing, extracts breast region (Otsu + largest connected component), merges
multiple ROI masks with OR logic, resizes to 256x256, and writes per-client .npy folders.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


def _norm_key(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {_norm_key(c): c for c in df.columns}


def pick_column(df: pd.DataFrame, *candidates: str) -> str:
    cmap = column_lookup(df)
    for c in candidates:
        k = _norm_key(c)
        if k in cmap:
            return cmap[k]
    return ""


def resolve_cbis_path(raw: object, dataset_root: str) -> str:
    """Resolve a relative CBIS path string to a local filesystem path."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip().replace("\\", "/")
    s = s.split("\n")[0].strip().strip('"').strip()
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


def _window_dicom_pixels(ds: pydicom.dataset.FileDataset, arr: np.ndarray) -> np.ndarray:
    """Apply DICOM VOILUT / window center-width and return float32 in [0, 1]."""
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    x = arr.astype(np.float64) * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax <= xmin:
            return np.zeros_like(x, dtype=np.float32)
        y = (x - xmin) / (xmax - xmin)
        if photometric == "MONOCHROME1":
            y = 1.0 - y
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    def _first(v: Any) -> float:
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return float(list(v)[0])
        return float(v)

    center = _first(wc)
    width = _first(ww)
    low = center - width / 2.0
    high = center + width / 2.0
    y = (x - low) / max(high - low, 1e-8)
    if photometric == "MONOCHROME1":
        y = 1.0 - y
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def load_image_grayscale(path: str) -> np.ndarray:
    """Load mammogram as 2D float32 in [0, 1]. Supports DICOM and common lossless formats."""
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
        return _window_dicom_pixels(ds, arr)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = img.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def extract_breast_region(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Otsu thresholding + largest connected component, then crop to bounding box.
    `gray` is single-channel float32 in [0, 1]; `mask` matches spatial size.
    """
    gray = np.clip(gray.astype(np.float32), 0.0, 1.0)
    gray_u8 = (gray * 255.0).astype(np.uint8)

    _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return gray, mask

    largest_label = 1
    largest_area = 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = i

    component_mask = (labels == largest_label).astype(np.uint8)
    clean_u8 = cv2.bitwise_and(gray_u8, gray_u8, mask=component_mask)
    clean_f = clean_u8.astype(np.float32) / 255.0

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    cropped_img = clean_f[y : y + h, x : x + w]
    cropped_mask = mask[y : y + h, x : x + w]
    return cropped_img, cropped_mask


def iter_mass_csv_paths(dataset_root: str) -> list[str]:
    """Only `mass_*.csv` files (excludes calcification CSVs)."""
    root = Path(dataset_root)
    paths = sorted({str(p) for p in root.glob("**/mass_*.csv")})
    if not paths:
        csv_dir = root / "csv"
        paths = sorted({str(p) for p in csv_dir.glob("mass_*.csv")}) if csv_dir.is_dir() else []
    return paths


def build_dicom_map(dicom_csv: str, dataset_root: str) -> dict[str, dict[str, str]]:
    dicom_df = pd.read_csv(dicom_csv, engine="python", on_bad_lines="skip")
    dicom_map: dict[str, dict[str, str]] = {}
    for _, r in dicom_df.iterrows():
        pid = str(r.get("PatientID", "")).strip()
        desc = str(r.get("SeriesDescription", "")).strip().lower()
        ipath = str(r.get("image_path", "")).strip()
        if "CBIS-DDSM" in ipath:
            idx = ipath.find("CBIS-DDSM")
            tail = ipath[idx + len("CBIS-DDSM") :].lstrip("/")
            local_path = os.path.normpath(os.path.join(dataset_root, tail))
        else:
            local_path = os.path.normpath(os.path.join(dataset_root, ipath))
        dicom_map.setdefault(pid, {})[desc] = local_path
    return dicom_map


def gather_rows_from_mass_csvs(
    csv_paths: list[str], dicom_map: dict[str, dict[str, str]], dataset_root: str
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        c_patient = pick_column(df, "patient_id", "Patient ID")
        c_breast = pick_column(df, "left or right breast")
        c_view = pick_column(df, "image view")
        c_pathology = pick_column(df, "pathology")
        c_abnormal_id = pick_column(df, "abnormality id")
        if not all([c_patient, c_breast, c_view, c_pathology, c_abnormal_id]):
            continue

        for _, row in df.iterrows():
            patient = str(row.get(c_patient, "")).strip()
            breast = str(row.get(c_breast, "")).strip()
            view = str(row.get(c_view, "")).strip()
            abn_id = str(row.get(c_abnormal_id, "")).strip()

            pathology_orig = str(row.get(c_pathology, "")).strip().upper()
            if "BENIGN" in pathology_orig:
                pathology = "BENIGN"
            elif "MALIGNANT" in pathology_orig:
                pathology = "MALIGNANT"
            else:
                continue

            prefixes = [
                f"Mass-Training_{patient}_{breast}_{view}",
                f"Mass-Test_{patient}_{breast}_{view}",
            ]
            base_pid = next((p for p in prefixes if p in dicom_map), None)
            if not base_pid:
                continue

            full_img = dicom_map[base_pid].get("full mammogram images", "")
            mask_key = f"{base_pid}_{abn_id}"
            if mask_key in dicom_map:
                mask_path = dicom_map[mask_key].get("roi mask images", "")
            else:
                mask_path = dicom_map[base_pid].get("roi mask images", "")

            if not full_img or not mask_path:
                continue
            if not os.path.isfile(full_img) or not os.path.isfile(mask_path):
                continue

            rows_out.append(
                {
                    "patient_id": patient,
                    "breast": breast,
                    "view": view,
                    "pathology": pathology,
                    "image_path": full_img,
                    "mask_path": mask_path,
                }
            )
    return rows_out


def split_patients_iid(patient_ids: list[str], seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ids = list(patient_ids)
    rng.shuffle(ids)
    mid = len(ids) // 2
    return set(ids[:mid]), set(ids[mid:])


def split_patients_non_iid(patient_ids: list[str], labels_by_patient: dict[str, str], seed: int) -> tuple[set[str], set[str]]:
    """
    Non-IID: Client A gets mostly BENIGN, Client B mostly MALIGNANT.
    Majority pathology per patient assigns the patient to A (benign) or B (malignant);
    ties broken deterministically by hash.
    """
    rng = np.random.default_rng(seed)
    set_ids = list(dict.fromkeys(patient_ids))
    pool_a: list[str] = []
    pool_b: list[str] = []
    mixed: list[str] = []

    for pid in set_ids:
        lab = labels_by_patient.get(pid, "")
        if lab == "BENIGN":
            pool_a.append(pid)
        elif lab == "MALIGNANT":
            pool_b.append(pid)
        else:
            mixed.append(pid)

    rng.shuffle(mixed)
    # Push tie/unknown patients to the smaller pool to balance counts
    for pid in mixed:
        if len(pool_a) <= len(pool_b):
            pool_a.append(pid)
        else:
            pool_b.append(pid)

    return set(pool_a), set(pool_b)


def majority_pathology_per_patient(rows: list[dict[str, Any]]) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        pid = r["patient_id"]
        p = r["pathology"]
        counts.setdefault(pid, {"BENIGN": 0, "MALIGNANT": 0})
        if p in counts[pid]:
            counts[pid][p] += 1
    out: dict[str, str] = {}
    for pid, c in counts.items():
        b, m = c["BENIGN"], c["MALIGNANT"]
        if b > m:
            out[pid] = "BENIGN"
        elif m > b:
            out[pid] = "MALIGNANT"
        else:
            out[pid] = "TIE"
    return out


def grouper_key(r: dict[str, Any]) -> tuple[str, str, str]:
    return r["patient_id"], r["breast"], r["view"]


def process_pipeline(
    dataset_root: str,
    out_root: str | None,
    split_mode: str,
    target_size: int,
    seed: int,
    *,
    dry_run: bool = False,
    max_groups: int | None = None,
) -> int:
    """Returns number of samples written (0 on dry-run)."""
    dataset_root = os.path.abspath(dataset_root)
    out_root_abs = os.path.abspath(out_root) if out_root else ""

    csv_dir = os.path.join(dataset_root, "csv")
    dicom_csv = os.path.join(csv_dir, "dicom_info.csv")
    if not os.path.isfile(dicom_csv):
        raise FileNotFoundError(f"Expected dicom_info.csv at {dicom_csv}")

    mass_csvs = iter_mass_csv_paths(dataset_root)
    if not mass_csvs:
        raise FileNotFoundError(f"No mass_*.csv under {dataset_root}")

    print(f"Using mass CSV files: {mass_csvs}")
    dicom_map = build_dicom_map(dicom_csv, dataset_root)
    raw_rows = gather_rows_from_mass_csvs(mass_csvs, dicom_map, dataset_root)
    if not raw_rows:
        raise RuntimeError("No valid mass rows after path validation.")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in raw_rows:
        grouped[grouper_key(r)].append(r)

    patient_ids = list({r["patient_id"] for r in raw_rows})
    labels_by_patient = majority_pathology_per_patient(raw_rows)

    if split_mode.lower() == "iid":
        a_set, b_set = split_patients_iid(patient_ids, seed)
        clients = {"client_a": a_set, "client_b": b_set}
    elif split_mode.lower() == "non_iid":
        a_set, b_set = split_patients_non_iid(patient_ids, labels_by_patient, seed)
        clients = {"client_a": a_set, "client_b": b_set}
    else:
        raise ValueError("split_mode must be 'iid' or 'non_iid'")

    n_groups = len(grouped)
    print(
        f"Stats: raw_rows={len(raw_rows)}, lesion_groups={n_groups}, "
        f"patients={len(patient_ids)}, split={split_mode}"
    )

    if dry_run:
        missing_img = 0
        sample_try = 0
        for key, group_rows in sorted(grouped.items(), key=lambda x: x[0])[: min(3, n_groups)]:
            mp = group_rows[0]["image_path"]
            if not os.path.isfile(mp):
                missing_img += 1
            else:
                sample_try += 1
        print(
            f"Dry-run: would write to {out_root_abs or '(no out-root)'} - "
            f"first-3 groups with existing image file: {sample_try}/3 (heuristic)."
        )
        return 0

    if not out_root_abs:
        raise ValueError("out_root is required unless dry_run=True")

    for name, pset in clients.items():
        d = os.path.join(out_root_abs, name, "images")
        m = os.path.join(out_root_abs, name, "masks")
        os.makedirs(d, exist_ok=True)
        os.makedirs(m, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    items = sorted(grouped.items(), key=lambda x: x[0])
    if max_groups is not None:
        items = items[: max(0, max_groups)]

    for key, group_rows in tqdm(items, desc="Groups"):
        patient, breast, view = key
        owner = "client_a" if patient in clients["client_a"] else "client_b"

        master_path = group_rows[0]["image_path"]
        pathology = group_rows[0]["pathology"]

        try:
            img = load_image_grayscale(master_path)
        except (OSError, ValueError, pydicom.errors.InvalidDicomError):
            continue

        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            imin, imax = float(img.min()), float(img.max())
            img = (img - imin) / (imax - imin + 1e-8)

        h, w = img.shape
        merged = np.zeros((h, w), dtype=np.uint8)
        valid = True
        for gr in group_rows:
            mp = gr["mask_path"]
            if not os.path.isfile(mp):
                valid = False
                break
            mimg = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mimg is None:
                valid = False
                break
            if mimg.shape[:2] != (h, w):
                mimg = cv2.resize(mimg, (w, h), interpolation=cv2.INTER_NEAREST)
            merged = cv2.bitwise_or(merged, mimg)

        if not valid:
            continue
        if np.sum(merged) == 0:
            continue

        cropped_img, cropped_m = extract_breast_region(img, merged)
        gray = cropped_img.astype(np.float32)

        mbin = (cropped_m > 127).astype(np.float32)
        if mbin.sum() == 0:
            continue

        gimg = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        gmask = cv2.resize(mbin, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        base = f"{patient}_{breast}_{view}"
        out_img = os.path.join(out_root_abs, owner, "images", f"{base}.npy")
        out_mask = os.path.join(out_root_abs, owner, "masks", f"{base}.npy")
        np.save(out_img, gimg.astype(np.float32))
        np.save(out_mask, gmask.astype(np.float32))

        manifest_rows.append(
            {
                "patient_id": patient,
                "client": owner,
                "pathology": pathology,
                "image": out_img,
                "mask": out_mask,
            }
        )

    manifest_path = os.path.join(out_root_abs, f"manifest_{split_mode}.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest_rows)} samples. Manifest: {manifest_path}")
    return len(manifest_rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Harmonia Vision CBIS-DDSM preprocessing")
    p.add_argument("--dataset-root", required=True, help="Root of extracted CBIS-DDSM (contains csv/dicom_info.csv)")
    p.add_argument(
        "--out-root",
        default="",
        help="Output root for client_a / client_b npy folders (optional with --dry-run)",
    )
    p.add_argument("--split-mode", choices=("iid", "non_iid"), required=True)
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load CSVs, print stats, and skip all image I/O and disk writes.",
    )
    p.add_argument("--max-groups", type=int, default=None, help="Process at most N lesion groups (for smoke tests).")
    args = p.parse_args()
    if not args.dry_run and not args.out_root:
        p.error("--out-root is required unless --dry-run is set")
    process_pipeline(
        args.dataset_root,
        args.out_root or None,
        args.split_mode,
        args.target_size,
        args.seed,
        dry_run=args.dry_run,
        max_groups=args.max_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
