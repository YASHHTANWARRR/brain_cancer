from ultralytics import YOLO
import torch

print("=" * 50)
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA not available!")

print("=" * 50)

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=512,
    batch=4,
    device=0,
    workers=4,
    amp=True,
    cache=False,
    pretrained=True,
    project="runs",
    name="yolov8n_clahe_baseline",

    # save settings
    save=True,
    save_period=10,

    # validation
    val=True,

    # reproducibility
    seed=42,

    # optimizer
    optimizer="auto",

    # augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    mosaic=1.0,

    verbose=True
)

print("Training Complete")