import os
import shutil
import random
from collections import defaultdict

DATASET_ROOT = "/home/hornet/dataset_folders/dataset"

IMG_DIR = os.path.join(DATASET_ROOT, "images")
LBL_DIR = os.path.join(DATASET_ROOT, "labels")

random.seed(42)

patients = defaultdict(list)

for file in os.listdir(IMG_DIR):
    if not file.endswith(".png"):
        continue

    patient = "_".join(file.split("_")[:2])
    patients[patient].append(file)

patient_ids = list(patients.keys())
random.shuffle(patient_ids)
