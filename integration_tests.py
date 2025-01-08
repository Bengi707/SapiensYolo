import torch
from ultralytics.ultralytics.nn.tasks import DetectionModel

model = DetectionModel("yolo11n.yaml")  # build model
im = torch.randn(1, 3, 640, 640)  # requires min imgsz=64
_ = model.predict(im, profile=True)
print()