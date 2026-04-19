"""
Preprocess CBIS-DDSM dataset according to the requirements:
- Filter mass cases only
- Merge train and test CSVs
- Read full mammograms and actual ROI masks
- Patient-based 70/15/15 split
- Breast extraction via Otsu thresholding + largest connected component
- Export to .npy format
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def _norm_key(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def column_lookup(df: pd.DataFrame) -> dict:
    return {_norm_key(c): c for c in df.columns}


def pick_column(df: pd.DataFrame, *candidates: str) -> str:
    cmap = column_lookup(df)
    for c in candidates:
        k = _norm_key(c)
        if k in cmap:
            return cmap[k]
    return ""


def resolve_cbis_path(raw: object, dataset_root: str) -> str:
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


def extract_breast_region(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Otsu thresholding + largest connected component to isolate breast and remove artifacts.
    Then bounding box crop.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Find the largest component skipping background (label 0)
    if num_labels <= 1:
        # Failsafe if image is completely black or something went wrong
        return image, mask

    largest_label = 1
    largest_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > largest_area:
            largest_area = stats[i, cv2.CC_STAT_AREA]
            largest_label = i

    # 3. Apply mask to remove L/R labels, marks, pectoral muscle (partially)
    component_mask = (labels == largest_label).astype(np.uint8)
    clean_image = cv2.bitwise_and(image, image, mask=component_mask)

    # 4. Bounding box crop
    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    cropped_image = clean_image[y : y + h, x : x + w]
    cropped_mask = mask[y : y + h, x : x + w]

    return cropped_image, cropped_mask


def find_actual_mask_in_dir(folder_path: str) -> str:
    """Finds the actual binary mask among files in the ROI mask folder."""
    if not os.path.isdir(folder_path):
        return ""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return ""
    if len(files) == 1:
        return files[0]
    
    # If multiple files, actual mask typically has only exactly two unique values (0 and 255) and is usually smaller in file size.
    best_file = files[0]
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        uniques = np.unique(img)
        if len(uniques) <= 3 and 0 in uniques and 255 in uniques:
            # almost certainly the mask
            return f
    
    # Fallback: return the smallest file size (masks compress really well)
    files.sort(key=lambda x: os.path.getsize(x))
    return files[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to CBIS-DDSM extracted dataset.")
    parser.add_argument("--out-dir", default="mammography/cache/processed", help="Output directory for .npy arrays")
    parser.add_argument("--target-size", type=int, default=256, help="Target resize dimension")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    csv_dir = os.path.join(data_root, "csv")
    train_csv = os.path.join(csv_dir, "mass_case_description_train_set.csv")
    test_csv = os.path.join(csv_dir, "mass_case_description_test_set.csv")
    dicom_csv = os.path.join(csv_dir, "dicom_info.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv) or not os.path.exists(dicom_csv):
        print(f"Dataset CSVs not found in {csv_dir}")
        return 1

    print("Loading dicom_info.csv mapping...")
    dicom_df = pd.read_csv(dicom_csv)
    
    # Pre-build path mapping from DICOM Info.
    # Format: PatientID -> SeriesDescription -> image_path
    dicom_map = {}
    for _, r in dicom_df.iterrows():
        pid = str(r["PatientID"]).strip()
        desc = str(r["SeriesDescription"]).strip().lower()
        ipath = str(r["image_path"]).strip()
        # Convert CBIS-DDSM/jpeg/... to local path format
        if "CBIS-DDSM" in ipath:
            idx = ipath.find("CBIS-DDSM")
            tail = ipath[idx + len("CBIS-DDSM"):].lstrip("/")
            local_path = os.path.normpath(os.path.join(data_root, tail))
        else:
            local_path = os.path.normpath(os.path.join(data_root, ipath))
            
        if pid not in dicom_map:
            dicom_map[pid] = {}
        dicom_map[pid][desc] = local_path


    print("Merging dataset CSVs...")
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    # Katman 1: Merge only mass cases.
    df = pd.concat([df_train, df_test], ignore_index=True)

    c_patient = pick_column(df, "patient_id", "Patient ID")
    c_breast = pick_column(df, "left or right breast")
    c_view = pick_column(df, "image view")
    c_pathology = pick_column(df, "pathology")
    c_abnormal_id = pick_column(df, "abnormality id")

    grouped_data = []

    print("Resolving paths from dicom_info...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        patient = str(row.get(c_patient)).strip()
        breast = str(row.get(c_breast)).strip()
        view = str(row.get(c_view)).strip()
        abn_id = str(row.get(c_abnormal_id)).strip()
        
        # Katman 2: Etiket temizliği
        pathology_orig = str(row.get(c_pathology)).strip().upper()
        if "BENIGN" in pathology_orig:
            pathology = "BENIGN"
        elif "MALIGNANT" in pathology_orig:
            pathology = "MALIGNANT"
        else:
            continue

        # Look up in dicom_map
        # Full mammogram key matches e.g. "Mass-Training_P_01265_RIGHT_MLO"
        # Mask key matches e.g. "Mass-Training_P_01265_RIGHT_MLO_1"
        # Note: test set uses "Mass-Test_..."
        
        # Find prefix
        candidate_prefixes = [f"Mass-Training_{patient}_{breast}_{view}", f"Mass-Test_{patient}_{breast}_{view}"]
        base_pid = None
        for p in candidate_prefixes:
            if p in dicom_map:
                base_pid = p
                break
                
        if not base_pid:
            continue
            
        full_img_candidate = dicom_map[base_pid].get("full mammogram images", "")
        
        mask_pid = f"{base_pid}_{abn_id}"
        if mask_pid in dicom_map:
            actual_mask_path = dicom_map[mask_pid].get("roi mask images", "")
        else:
            # Maybe the abnormality id is omitted if there's only 1?
            actual_mask_path = dicom_map[base_pid].get("roi mask images", "")

        if not os.path.isfile(full_img_candidate) or not os.path.isfile(actual_mask_path):
            continue

        grouped_data.append({
            "patient_id": patient,
            "breast": breast,
            "view": view,
            "pathology": pathology,
            "pathology_orig": pathology_orig,
            "image_path": full_img_candidate,
            "mask_path": actual_mask_path
        })
    
    df_clean = pd.DataFrame(grouped_data)

    print(f"Grouped Data Entries before merging duplicates: {len(df_clean)}")

    # Patient Split (70/15/15) Katman 6
    unique_patients = list(df_clean["patient_id"].unique())
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    n = len(unique_patients)
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)
    
    train_patients = set(unique_patients[:train_n])
    val_patients = set(unique_patients[train_n:train_n+val_n])
    test_patients = set(unique_patients[train_n+val_n:])

    def get_split(pid):
        if pid in train_patients: return "train"
        if pid in val_patients: return "val"
        return "test"

    manifest_rows = []

    groups = df_clean.groupby(["patient_id", "breast", "view"])
    print(f"Processing {len(groups)} unique lesion groups...")

    for (patient, breast, view), group in tqdm(groups):
        master_img_path = group.iloc[0]["image_path"]
        pathology = group.iloc[0]["pathology"]
        
        img = cv2.imread(master_img_path)
        if img is None:
            continue
            
        merged_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        valid_group = True
        for _, row in group.iterrows():
            m_path = row["mask_path"]
            m_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m_img is None:
                valid_group = False
                break
            
            if m_img.shape != merged_mask.shape:
                m_img = cv2.resize(m_img, (merged_mask.shape[1], merged_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            merged_mask = cv2.bitwise_or(merged_mask, m_img)
            
        if not valid_group:
            continue

        # Katman 4: Mask Validation
        if np.sum(merged_mask) == 0:
            continue

        # Katman 6: Image Cleaning & Bounding Box Crop
        cropped_img, cropped_mask = extract_breast_region(img, merged_mask)

        # Resize to Target Size (e.g., 256x256)
        final_img = cv2.resize(cropped_img, (args.target_size, args.target_size), interpolation=cv2.INTER_LINEAR)
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        final_mask = cv2.resize(cropped_mask, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
        final_mask = (final_mask > 127).astype(np.float32) # Store as float directly or uint8

        base_name = f"{patient}_{breast}_{view}.npy"
        out_img_path = os.path.join(out_dir, "img_" + base_name)
        out_msk_path = os.path.join(out_dir, "mask_" + base_name)

        np.save(out_img_path, final_img)
        np.save(out_msk_path, final_mask)

        manifest_rows.append({
            "patient_id": patient,
            "split": get_split(patient),
            "image_path": out_img_path,
            "mask_path": out_msk_path,
            "pathology": pathology.lower(),
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_out = os.path.join(out_dir, "manifest_dataset.csv")
    manifest_df.to_csv(manifest_out, index=False)
    print(f"Total processed samples: {len(manifest_rows)}")
    print(f"Manifest written to: {manifest_out}")

    return 0

if __name__ == '__main__':
    main()
