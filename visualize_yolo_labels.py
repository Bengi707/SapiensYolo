import cv2
import os
from tqdm import tqdm

# Define paths
image_dir = r"C:\Users\Bengi\PycharmProjects\Datasets\CrowdHuman\CrowdHuman_val\images"   # Directory containing images
label_dir = r"C:\Users\Bengi\PycharmProjects\Datasets\CrowdHuman\CrowdHuman_val\labels"           # Directory containing YOLO format label files
# output_dir = "path/to/output_images"   # Optional: save images with bboxes overlaid
# os.makedirs(output_dir, exist_ok=True)

# Loop through images and corresponding labels
for filename in tqdm(os.listdir(image_dir)):

        # img_path = os.path.join(image_dir, filename)
        img_path = os.path.join(image_dir, filename)
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

            # Display the image with bounding boxes
            cv2.imshow("Image with Bounding Boxes", img)
            cv2.waitKey(0)  # Press any key to move to the next image


cv2.destroyAllWindows()
