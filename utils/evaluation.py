import numpy as np
from typing import List, Dict, Tuple

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    # Calculate intersection coordinates
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_ap(recall: List[float], precision: List[float]) -> float:
    """Calculate Average Precision using 11-point interpolation."""
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11.
    return ap

def calculate_map(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of dictionaries containing predicted bounding boxes and classes.
                    Each dict should have format:
                    {
                        'class_id': str,
                        'bounding_box': {
                            'x1': float,  # normalized coordinates (0-1)
                            'y1': float,
                            'x2': float,
                            'y2': float
                        },
                        'confidence': float  # optional
                    }
        ground_truths: List of dictionaries containing ground truth bounding boxes and classes.
        iou_threshold: IoU threshold for considering a detection as correct (default: 0.5)
    
    Returns:
        float: Mean Average Precision across all classes
    """
    # Get unique classes
    classes = set()
    for gt in ground_truths:
        classes.add(gt['class_id'])
    
    aps = []
    
    for class_id in classes:
        # Get predictions and ground truths for current class
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_gts = [g for g in ground_truths if g['class_id'] == class_id]
        
        if not class_gts:  # Skip if no ground truth for this class
            continue
            
        # Sort predictions by confidence if available, otherwise use order
        if 'confidence' in class_preds[0]:
            class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize arrays for precision and recall
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        gt_matched = np.zeros(len(class_gts))
        
        # For each prediction
        for pred_idx, pred in enumerate(class_preds):
            best_iou = -np.inf
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(class_gts):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = calculate_iou(pred['bounding_box'], gt['bounding_box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If best match exceeds threshold, mark as true positive
            if best_iou >= iou_threshold:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Calculate cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(class_gts)
        
        # Calculate AP for this class
        ap = calculate_ap(recall, precision)
        aps.append(ap)
    
    # Return mean of APs
    return np.mean(aps) if aps else 0.0

