from ultralytics import YOLO
import torch
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_CFG = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8_cbam.yaml"
DATA_CFG = ROOT / "data.yaml"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("=" * 50)

model = YOLO(str(MODEL_CFG))

model.train(
    data=str(DATA_CFG),
    pretrained="yolov8n.pt",
    epochs=50,
    imgsz=512,
    batch=4,
    device=DEVICE,
    workers=4,
    amp=True,
    cache=False,
    project="runs",
    name="yolov8n_cbam",
)
