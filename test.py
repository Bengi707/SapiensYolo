import cv2
import torch
from ultralytics.ultralytics import YOLO
# Build a new model from your custom YAML
model = YOLO("yolo11n")

results = model.val(data="wider_person.yaml", imgsz=1024,batch=2, save_json=True)
print(results.box.map)  # Print mAP50-95