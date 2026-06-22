# Brain Tumor Detection using Attention-Enhanced YOLOv8 on BraTS 2021

## Overview

This project develops a lightweight and accurate brain tumor detection system using MRI scans from the BraTS 2021 dataset. The work focuses on converting volumetric MRI segmentation data into an object detection dataset and evaluating multiple attention-enhanced YOLOv8 architectures for tumor localization.

The primary objective is to investigate whether modern attention mechanisms such as Convolutional Block Attention Module (CBAM) and Coordinate Attention (CA) can improve brain tumor detection performance while maintaining real-time inference on consumer-grade GPUs.

---

## Dataset

Dataset Used:

* BraTS 2021 Training Dataset
* 1,252 patient MRI volumes
* Multi-modal MRI scans:

  * T1
  * T1CE
  * T2
  * FLAIR
* Expert-annotated tumor segmentation masks

Source:

https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

### Preprocessing Pipeline

The original BraTS dataset contains 3D MRI volumes and segmentation masks.

The following preprocessing pipeline is applied:

BraTS MRI Volume
→ T1CE Modality Selection
→ Slice Extraction
→ CLAHE Contrast Enhancement
→ Tumor Mask Extraction
→ Bounding Box Generation
→ YOLO Format Conversion

Only slices containing tumor regions are retained.

---

## Dataset Conversion

The segmentation masks are converted into YOLO object detection labels.

Each tumor slice is transformed into:

Image:

* PNG format

Label:

* YOLO bounding box format

Resulting dataset:

* 81,437 MRI slices
* 81,437 YOLO labels
* Single tumor class

Class Mapping:

0 → Brain Tumor

---

## Proposed Architectures

### Experiment 1 — Baseline YOLOv8n

MRI
→ CLAHE
→ YOLOv8n
→ Tumor Detection

### Experiment 2 — YOLOv8n + CBAM

MRI
→ CLAHE
→ YOLOv8n
→ CBAM
→ Tumor Detection

### Experiment 3 — YOLOv8n + Coordinate Attention

MRI
→ CLAHE
→ YOLOv8n
→ Coordinate Attention
→ Tumor Detection

### Experiment 4 — YOLOv8n + CBAM + Coordinate Attention

MRI
→ CLAHE
→ YOLOv8n
→ CBAM
→ Coordinate Attention
→ Tumor Detection

---

## Evaluation Metrics

The models are evaluated using:

* Precision
* Recall
* F1 Score
* mAP@50
* mAP@50–95
* Inference Time
* Frames Per Second (FPS)
* Parameter Count

---

## Hardware

Training Platform:

* NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB VRAM)
* CUDA Enabled
* Ubuntu Linux
* Python 3.10
* PyTorch 2.12
* Ultralytics YOLOv8

---

## Installation

Create environment:

```bash
conda create -n brain_tumor python=3.10
conda activate brain_tumor
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install ultralytics
pip install nibabel opencv-python scikit-image tqdm matplotlib pandas
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Project Structure

```text
brain_tumor/
│
├── dataset/
│   ├── images/
│   └── labels/
│
├── scripts/
│   ├── brats_to_yolo.py
│   ├── split_dataset.py
│   └── train.py
│
├── models/
│   ├── yolov8n_baseline/
│   ├── yolov8n_cbam/
│   ├── yolov8n_ca/
│   └── yolov8n_cbam_ca/
│
├── results/
│   ├── figures/
│   ├── confusion_matrix/
│   └── metrics/
│
├── data.yaml
├── README.md
└── requirements.txt
```

---

## Training

Baseline Training:

```bash
python train.py
```

Typical Settings:

* Image Size: 512 × 512
* Batch Size: 4
* Epochs: 100
* Mixed Precision Training Enabled

---

## Expected Contributions

* Conversion of BraTS 2021 segmentation data into an object detection dataset
* Lightweight brain tumor localization using YOLOv8
* Integration of CBAM attention mechanism
* Integration of Coordinate Attention mechanism
* Comparative ablation study on attention modules
* Deployment on low-memory consumer GPUs

---

## Future Work

* Tumor segmentation using U-Net and nnU-Net
* Multi-modal MRI fusion
* YOLOv11-based architectures
* Vision Transformer integration
* Clinical validation on external datasets

---

## License

This project is intended for academic and research purposes.
BraTS dataset usage must comply with the dataset's original license and citation requirements.

```
```
