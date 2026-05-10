from ultralytics import YOLO

model = YOLO("best.pt")

rknn_path = model.export(
    format="rknn",
    name="rk3588",
    imgsz=736,
)

print(rknn_path)