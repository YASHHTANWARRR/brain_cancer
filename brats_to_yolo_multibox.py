import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

ROOT = "/home/hornet/dataset_folders/archive/BraTS2021_Training_Data"

OUT_ROOT = "/home/hornet/dataset_folders/dataset_multibox"

OUT_IMG = os.path.join(OUT_ROOT, "images")
OUT_LBL = os.path.join(OUT_ROOT, "labels")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

def apply_clahe(img):
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(img)

image_count = 0
box_count = 0

patients = sorted(os.listdir(ROOT))

for patient in tqdm(patients, desc="Processing Patients"):

    patient_dir = os.path.join(ROOT, patient)

    t1ce_path = os.path.join(
        patient_dir,
        f"{patient}_t1ce.nii.gz"
    )

    seg_path = os.path.join(
        patient_dir,
        f"{patient}_seg.nii.gz"
    )

    if not os.path.exists(t1ce_path):
        continue

    if not os.path.exists(seg_path):
        continue

    try:
        img_vol = nib.load(t1ce_path).get_fdata()
        seg_vol = nib.load(seg_path).get_fdata()

    except Exception as e:
        print(f"Failed loading {patient}: {e}")
        continue

    depth = img_vol.shape[2]

    for z in range(depth):

        img_slice = img_vol[:, :, z]
        mask_slice = seg_vol[:, :, z]

        if np.sum(mask_slice) == 0:
            continue

        img_slice = apply_clahe(img_slice)

        tumor_mask = (mask_slice > 0).astype(np.uint8)

        h, w = tumor_mask.shape

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(
                tumor_mask,
                connectivity=8
            )

        label_lines = []

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < 20:
                continue

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h

            bw_norm = bw / w
            bh_norm = bh / h

            label_lines.append(
                f"0 "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{bw_norm:.6f} "
                f"{bh_norm:.6f}"
            )

            box_count += 1

        if len(label_lines) == 0:
            continue

        filename = f"{patient}_{z:03d}"

        img_path = os.path.join(
            OUT_IMG,
            f"{filename}.png"
        )

        lbl_path = os.path.join(
            OUT_LBL,
            f"{filename}.txt"
        )

        cv2.imwrite(img_path, img_slice)

        with open(lbl_path, "w") as f:
            f.write("\n".join(label_lines))

        image_count += 1

print("\n")
print("=" * 50)
print("MULTIBOX DATASET CREATED")
print("=" * 50)
print(f"Images Created : {image_count}")
print(f"Boxes Created  : {box_count}")
print("=" * 50)
print(OUT_ROOT)
print("=" * 50)