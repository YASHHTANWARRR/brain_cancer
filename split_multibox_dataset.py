import os
import shutil
import random
from collections import defaultdict

DATASET_ROOT = "/home/hornet/dataset_folders/dataset_multibox_v1"

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

n = len(patient_ids)

train_patients = patient_ids[:int(0.7 * n)]
val_patients   = patient_ids[int(0.7 * n):int(0.9 * n)]
test_patients  = patient_ids[int(0.9 * n):]

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(IMG_DIR, split),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(LBL_DIR, split),
        exist_ok=True
    )

def move_files(patient_list, split):

    for patient in patient_list:

        for img_file in patients[patient]:

            lbl_file = img_file.replace(
                ".png",
                ".txt"
            )

            shutil.move(
                os.path.join(
                    IMG_DIR,
                    img_file
                ),
                os.path.join(
                    IMG_DIR,
                    split,
                    img_file
                )
            )

            shutil.move(
                os.path.join(
                    LBL_DIR,
                    lbl_file
                ),
                os.path.join(
                    LBL_DIR,
                    split,
                    lbl_file
                )
            )

move_files(train_patients, "train")
move_files(val_patients, "val")
move_files(test_patients, "test")

print("Done")