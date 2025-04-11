import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_boxes(predictions: list, ground_truths: list, img_id: str, img_dir: str = "data/wood-defects/images"):
    """
    Visualize predicted and ground truth bounding boxes on an image.
    
    Args:
        predictions: List of dictionaries containing predicted bounding boxes
        ground_truths: List of dictionaries containing ground truth bounding boxes
        img_id: Image ID to load
        img_dir: Directory containing the images
    """
    # Load image
    img_path = Path(img_dir) / f"{img_id}.jpg"
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    # Colors for predictions and ground truth
    pred_color = 'red'
    gt_color = 'green'
    
    # Draw predictions
    for pred in predictions:
        box = pred['bounding_box']
        x1, y1 = box['x1'] * width, box['y1'] * height
        x2, y2 = box['x2'] * width, box['y2'] * height
        width_box = x2 - x1
        height_box = y2 - y1
        
        rect = Rectangle((x1, y1), width_box, height_box, 
                        linewidth=2, edgecolor=pred_color, facecolor='none')
        ax.add_patch(rect)
        
        # Add class label and confidence if available
        label = f"{pred['class_id']}"
        if 'confidence' in pred:
            label += f" ({pred['confidence']:.2f})"
        ax.text(x1, y1-5, label, color=pred_color, fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw ground truth
    for gt in ground_truths:
        box = gt['bounding_box']
        x1, y1 = box['x1'] * width, box['y1'] * height
        x2, y2 = box['x2'] * width, box['y2'] * height
        width_box = x2 - x1
        height_box = y2 - y1
        
        rect = Rectangle((x1, y1), width_box, height_box, 
                        linewidth=2, edgecolor=gt_color, facecolor='none')
        ax.add_patch(rect)
        
        # Add class label
        ax.text(x1, y1-5, gt['class_id'], color=gt_color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=pred_color, label='Predictions'),
        Line2D([0], [0], color=gt_color, label='Ground Truth')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"Image {img_id}")
    plt.axis('off')
    return fig

import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path

if __name__ == "__main__":
    
    test_image_path = Path(__file__) / "../data" / "wood-defects-parsed/images/test/99100008.jpg"
    test_annotations_path = Path(__file__) / "../data" / "wood-defects-parsed/vqa/test.json"
    # Load test image and annotations
    
    # Load image
    img = Image.open(test_image_path)
    
    # Load annotations
    with open(test_annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Example predictions and ground truths
    predictions = [
        {
            'class_id': 'defect1',
            'bounding_box': {'x1': 0.2, 'y1': 0.3, 'x2': 0.4, 'y2': 0.5},
            'confidence': 0.95
        }
    ]
    
    ground_truths = [
        {
            'class_id': 'defect1',
            'bounding_box': {'x1': 0.25, 'y1': 0.35, 'x2': 0.45, 'y2': 0.55}
        }
    ]
    
    # Create visualization
    fig = visualize_boxes(predictions, ground_truths, "test_image")
    
    # Display the plot
    plt.show()
