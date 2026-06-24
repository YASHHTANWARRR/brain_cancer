from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/yolov8n_clahe_baseline/weights/best.pt"
)

metrics = model.val(
    data="data.yaml",
    split="test"
)

print(metrics)