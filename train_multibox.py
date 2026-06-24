from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data_multibox.yaml",
    epochs=50,
    imgsz=512,
    batch=4,
    device=0,
    workers=4,
    amp=True,
    project="runs",
    name="yolov8n_multibox"
)