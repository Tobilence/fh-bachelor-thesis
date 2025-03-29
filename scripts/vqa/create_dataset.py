## Create dataset for VQA
from PIL import Image
from typing import List


def map_class_to_name(class_id: int) -> str:
    names = {
        0: "quartzite",
        1: "live_knot",
        2: "marrow", 
        3: "resin",
        4: "dead_knot",
        5: "knot_with_crack",
        6: "missing_knot",
        7: "crack"
    }
    return names[class_id]


def get_image_size(dataset_split: str, image_name: str) -> tuple[int, int]:
    image_path = Path(f"data/wood-defects-parsed/images/{dataset_split}/{image_name}.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Image {image_path} not found")
        
    img = Image.open(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image {image_path}")
        
    width, height = img.size
    return width, height


def map_annotations_to_json(dataset_split: str, file_name_str: str, annotations: List[str]):
    annotations = [annotation.replace("\n", "") for annotation in annotations]
    json_annotations = []
    width, height = get_image_size(dataset_split, file_name_str)
    
    for annotation in annotations:
        if not annotation.strip():
            continue
        # YOLO format: class x_center y_center width height
        parts = annotation.split()
        if len(parts) == 5:
            class_id, x_center, y_center, w, h = map(float, parts)
            
            # Convert normalized YOLO coordinates to absolute x1,y1,x2,y2 format
            x1 = int((x_center - (w / 2)) * width)
            y1 = int((y_center - (h / 2)) * height)
            x2 = int((x_center + (w / 2)) * width)
            y2 = int((y_center + (h / 2)) * height)
            
            # Ensure coordinates are within image bounds and normalize between 0 and 1
            x1 = max(0.0, min(1.0, x1 / width))
            y1 = max(0.0, min(1.0, y1 / height))
            x2 = max(0.0, min(1.0, x2 / width))
            y2 = max(0.0, min(1.0, y2 / height))
            
            json_annotation = {
                "class_id": map_class_to_name(int(class_id)),
                "bounding_box": {
                    "x1": x1,
                    "y1": y1, 
                    "x2": x2,
                    "y2": y2
                }
            }
            json_annotations.append(json_annotation)
    
    return json_annotations


## Expected Format (JSON):
"""
{
  "id": "example_detection_empty",
  "image": "path/to/empty_image.jpg",
  "conversations": [
    {
      "role": "user",
      "content": "Detect objects in this image and return them in JSON format."
    },
    {
      "role": "assistant", 
      "content": "```json\n{\n  \"objects\": []\n}\n```"
    }
  ]
}
"""

import os
import json
import random
from pathlib import Path

def create_vqa_dataset(split_type: str):
    # Define paths
    base_path = Path("data/wood-defects-parsed")
    labels_path = base_path / "labels"
    
    # Get all json files from test folder (could be train/val as well)
    # Get all text files containing YOLO format annotations
    txt_files = list(Path(labels_path / split_type).glob("*.txt"))
    
    dataset = []
    
    for json_file in txt_files:
        # Get corresponding image path
        image_name = json_file.stem + ".jpg"
        image_path = f"data/wood-defects-parsed/images/{split_type}/{image_name}"
        
        # Read the annotations
        with open(json_file) as f:
            annotations = f.readlines()
            
        # Create conversation format
        conversation = {
            "id": json_file.stem,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can detect wood defects in images."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}"
                        },
                        {
                            "type": "text",
                            "text": "What type of defects do you see in this wood? Return the defects and their relative locations in JSON format."
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": map_annotations_to_json(split_type, json_file.stem, annotations)
                }
            ]
        }
        
        dataset.append(conversation)
    return dataset

if __name__ == "__main__":
    for split_type in ["val", "train", "test"]:
        dataset = create_vqa_dataset("val")
        os.makedirs("data/wood-defects-parsed/qwen", exist_ok=True)
        with open(f"data/wood-defects-parsed/qwen/{split_type}.json", "w+") as f:
            json.dump(dataset, f, indent=2)
