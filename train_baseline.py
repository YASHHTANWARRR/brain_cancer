from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=10,          # test run first
    imgsz=512,
    batch=4,
    device=0,
    workers=4,
    amp=True,
    cache=False,
    pretrained=True,
    project="runs",
    name="yolov8n_baseline"
)