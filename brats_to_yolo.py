import os
import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

ROOT = "/home/hornet/dataset_folders/archive/BraTS2021_Training_Data"

OUT_IMG = "dataset/images"
OUT_LBL = "dataset/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

def apply_clahe(img):
    img = cv2.normalize(img, None, 0, 255,
                        cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    return clahe.apply(img)
count = 0
patients = sorted(os.listdir(ROOT))

for patient in tqdm(patients):
    pdir = os.path.join(ROOT, patient)
    t1ce_path = os.path.join(
        pdir,
        f"{patient}_t1ce.nii.gz"
    )
    seg_path = os.path.join(
        pdir,
        f"{patient}_seg.nii.gz"
    )
    if not os.path.exists(t1ce_path):
        continue
    
    img_vol = nib.load(t1ce_path).get_fdata()
    seg_vol = nib.load(seg_path).get_fdata()
    depth = img_vol.shape[2]
    
    for z in range(depth):
        img = img_vol[:, :, z]
        mask = seg_vol[:, :, z]
        
        if np.sum(mask) == 0:
            continue
        
        img = apply_clahe(img)
        tumor = (mask > 0).astype(np.uint8)
        ys, xs = np.where(tumor > 0)
        
        if len(xs) == 0:
            continue
        
        xmin = xs.min()
        xmax = xs.max()
        ymin = ys.min()
        ymax = ys.max()
        
        h, w = img.shape
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        name = f"{patient}_{z:03d}"
        
        cv2.imwrite(f"{OUT_IMG}/{name}.png",img)
        with open(f"{OUT_LBL}/{name}.txt","w") as f:
            f.write(
                f"0 {x_center} {y_center} {bw} {bh}"
            )
        count += 1

print("Created:", count)