import torch
from ultralytics.ultralytics import YOLO
# Build a new model from your custom YAML
model = YOLO('yolo11n.yaml')

model.train(
    data='wider_person.yaml',
    batch=1,          # Specify batch size (e.g., 16)
    imgsz=1024,          # Specify image size (e.g., 640x640)
    save_period = 4,
    name = "/home/bengi/PycharmProjects/SapiensYOLO_trainable/saved_models",
    pretrained=False,
    amp= False,
    epochs = 50,
    single_cls=True
)