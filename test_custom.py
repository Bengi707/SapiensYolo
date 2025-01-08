import glob
import os.path

import cv2
import torch
from ultralytics.ultralytics import YOLO
# Build a new model from your custom YAML
# model = YOLO(r"C:\Users\Bengi\Downloads\best.pt")
model = YOLO("ultralytics/yolo11n.pt")

# results = model.val(data="wider_person.yaml", imgsz=1024,batch=2, save_json=True)
# print(results.box.map)  # Print mAP50-95

folder = r"C:\Users\Bengi\PycharmProjects\Datasets\CrowdHuman\CrowdHuman_val\images"
images = glob.glob(os.path.join(folder,"*.jpg"))

for img_path in images[10:]:

    results = model.predict(source=img_path,conf=0.25)

    # Load the image
    original_img = cv2.imread(img_path)
    original_img_ = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)


    for r in results:
        boxes = r.boxes.data.cpu().numpy()  # Bounding boxes
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box  # Unpack box data
            class_id = int(cls)
            class_name = model.names[class_id]  # Class name from model
            color = (0, 255, 0)  # Green for bounding box
            cv2.rectangle(original_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(original_img, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    height, width = original_img_.shape[:2]

    # Set the target width or height (you can choose one and calculate the other)
    target_height = 650

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Compute the new dimensions based on the target height
    new_height = target_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(original_img, (new_width, new_height))

    # Display or save the image
    cv2.imshow("Prediction",resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
