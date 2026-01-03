"""
Evaluation metrics for object detection
"""
import torch
import numpy as np
from collections import defaultdict


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) using 11-point interpolation
    """
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap


class DetectionMetrics:
    """
    Calculate mAP and other detection metrics
    """
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = defaultdict(list)  # {class_id: [(confidence, is_correct)]}
        self.num_gt = defaultdict(int)  # {class_id: count}
    
    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        """
        Update metrics with predictions and ground truth
        
        Args:
            pred_boxes: (N, 4) tensor
            pred_labels: (N,) tensor
            pred_scores: (N,) tensor
            gt_boxes: (M, 4) tensor
            gt_labels: (M,) tensor
        """
        # Count ground truth objects per class
        for label in gt_labels:
            self.num_gt[label.item()] += 1
        
        # Match predictions to ground truth
        if len(pred_boxes) == 0:
            return
        
        matched_gt = set()
        
        # Sort predictions by confidence (descending)
        sorted_indices = torch.argsort(pred_scores, descending=True)
        
        for idx in sorted_indices:
            pred_box = pred_boxes[idx].cpu().numpy()
            pred_label = pred_labels[idx].item()
            pred_score = pred_scores[idx].item()
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_label.item() != pred_label or gt_idx in matched_gt:
                    continue
                
                iou = compute_iou(pred_box, gt_box.cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if prediction is correct
            is_correct = best_iou >= self.iou_threshold and best_gt_idx >= 0
            if is_correct:
                matched_gt.add(best_gt_idx)
            
            self.predictions[pred_label].append((pred_score, is_correct))
    
    def compute(self):
        """
        Compute mAP and per-class AP
        """
        aps = []
        
        for class_id in range(self.num_classes):
            if class_id not in self.predictions or self.num_gt[class_id] == 0:
                continue
            
            # Sort by confidence
            preds = sorted(self.predictions[class_id], key=lambda x: x[0], reverse=True)
            
            tp = np.array([1 if p[1] else 0 for p in preds])
            fp = 1 - tp
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / self.num_gt[class_id]
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            ap = compute_ap(recall, precision)
            aps.append(ap)
        
        mAP = np.mean(aps) if len(aps) > 0 else 0.0
        
        return {
            'mAP': mAP,
            'AP_per_class': aps
        }


def psnr(img1, img2, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))
