from ultralytics import YOLO
import torch

print("=" * 50)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)

model = YOLO("ultralytics/cfg/models/v8/yolov8_cbam.yaml")

model.train(
    data="/home/hornet/Desktop/brain_cancer/data.yaml",
    pretrained="yolov8n.pt",
    epochs=50,
    imgsz=512,
    batch=4,
    device=0,
    workers=4,
    amp=True,
    cache=False,
    project="runs",
    name="yolov8n_cbam"
)