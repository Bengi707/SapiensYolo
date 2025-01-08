import cv2
import os
from tqdm import tqdm


filename =r'C:\\Users\\Bengi\\PycharmProjects\\Datasets\\CrowdHuman\\CrowdHuman_val\\images_va\\1066405,100a000985ef070.jpg'
labelname =r'C:\\Users\\Bengi\\PycharmProjects\\Datasets\\CrowdHuman\\CrowdHuman_test\\\\1066405,100a000985ef070.jpg'

img_path = filename
label_path = os.path.join(label_dir, filename.replace("jpg","txt"))

# Read the image
img = cv2.imread(img_path)
h, w, _ = img.shape

# Check if label file exists
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Draw each bounding box
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        # Convert normalized coordinates to pixel values
        x_center = int(x_center * w)
        y_center = int(y_center * h)
        box_width = int(box_width * w)
        box_height = int(box_height * h)

        # Calculate top-left and bottom-right corners of the bounding box
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(img, f'Class {class_id}', (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    height, width = img.shape[:2]

    # Set the target width or height (you can choose one and calculate the other)
    target_height = 650

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Compute the new dimensions based on the target height
    new_height = target_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height))

    # Display the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", resized_image)
    cv2.waitKey(0)  # Press any key to move to the next image