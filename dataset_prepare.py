import os
import json
from tqdm import tqdm

def convert_to_yolo(json_file, output_dir, images_dir, use_box="vbox"):
    """
    Converts CrowdHuman annotations to YOLO format.
    Args:
        json_file: Path to CrowdHuman annotation file (.odgt format).
        output_dir: Directory to save YOLO annotation files.
        images_dir: Path to the directory containing images.
        use_box: Choose which box to use ('fbox', 'vbox', or 'hbox').
        fbox: full body box
        vbox: visible body box
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON Lines (one JSON object per line)
    with open(json_file, 'r') as f:
        annotations = [json.loads(line.strip()) for line in f]

    for annotation in tqdm(annotations, desc="Converting"):
        image_id = annotation["ID"]
        image_path = os.path.join(images_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found!")
            continue

        img_width, img_height = get_image_dimensions(image_path)  # Replace with your image dimension function
        yolo_annotations = []

        for box in annotation["gtboxes"]:
            if box["tag"] != "person":
                continue  # Skip non-person objects

            # Select the desired box (e.g., 'fbox')
            x1, y1, w, h = box[use_box]

            # Convert to YOLO format
            x_center = (x1 + w / 2) / img_width
            y_center = (y1 + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            # Ensure values are within [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            # Class ID for 'person' is 0
            yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")

        # Write to .txt file
        output_file = os.path.join(output_dir, f"{image_id}.txt")
        with open(output_file, 'w') as f:
            f.write("\n".join(yolo_annotations))

def get_image_dimensions(image_path):
    """Retrieve dimensions of an image."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.width, img.height

# Paths
json_file = "../Datasets/CrowdHuman/annotation_val.odgt"  # CrowdHuman annotations
output_dir = r"C:\Users\Bengi\PycharmProjects\Datasets\CrowdHuman\CrowdHuman_val\labels"
images_dir = "../Datasets/CrowdHuman/CrowdHuman_val/Images"

# Convert
convert_to_yolo(json_file, output_dir, images_dir, use_box="fbox")