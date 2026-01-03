"""
YOLO-style detection losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLODetectionLoss(nn.Module):
    """
    YOLO-style detection loss
    Combines:
    - Localization loss (bbox regression)
    - Objectness loss (object confidence)
    - Classification loss (class probabilities)
    """
    def __init__(self, num_classes, anchors, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLODetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
    
    def compute_iou(self, box1, box2):
        """
        Compute IoU between two sets of boxes
        
        Args:
            box1: (N, 4) [x1, y1, x2, y2]
            box2: (N, 4) [x1, y1, x2, y2]
        """
        # Intersection
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = box1_area + box2_area - intersection
        
        iou = intersection / (union + 1e-6)
        return iou
    
    def build_targets(self, predictions, targets, anchors, stride):
        """
        Build target tensors for YOLO loss computation
        
        Args:
            predictions: (B, H, W, num_anchors, 5 + num_classes)
            targets: List of target dictionaries for each image
            anchors: (num_anchors, 2)
            stride: Feature map stride
        
        Returns:
            Target masks and values for loss computation
        """
        batch_size, height, width, num_anchors, _ = predictions.shape
        device = predictions.device
        
        # Initialize target tensors
        obj_mask = torch.zeros(batch_size, height, width, num_anchors, device=device)
        noobj_mask = torch.ones(batch_size, height, width, num_anchors, device=device)
        tx = torch.zeros(batch_size, height, width, num_anchors, device=device)
        ty = torch.zeros(batch_size, height, width, num_anchors, device=device)
        tw = torch.zeros(batch_size, height, width, num_anchors, device=device)
        th = torch.zeros(batch_size, height, width, num_anchors, device=device)
        tconf = torch.zeros(batch_size, height, width, num_anchors, device=device)
        tcls = torch.zeros(batch_size, height, width, num_anchors, self.num_classes, device=device)
        
        # Process each image in batch
        for b in range(batch_size):
            if targets[b] is None or len(targets[b]['boxes']) == 0:
                continue
            
            gt_boxes = targets[b]['boxes']  # (N, 4) [x1, y1, x2, y2]
            gt_labels = targets[b]['labels']  # (N,)
            
            # Convert boxes to center format
            gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
            gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
            gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            
            # Find grid cell for each ground truth
            grid_x = (gt_cx / stride).long().clamp(0, width - 1)
            grid_y = (gt_cy / stride).long().clamp(0, height - 1)
            
            # Find best anchor for each ground truth
            scaled_anchors = anchors / stride
            
            for idx in range(len(gt_boxes)):
                gx, gy = grid_x[idx], grid_y[idx]
                gw, gh = gt_w[idx], gt_h[idx]
                label = gt_labels[idx]
                
                # Find best anchor based on IoU
                gt_wh = torch.tensor([[gw, gh]], device=device)
                anchor_wh = scaled_anchors
                
                # Compute IoU between gt box and anchors (both at same location)
                min_wh = torch.min(gt_wh, anchor_wh)
                intersection = min_wh[:, 0] * min_wh[:, 1]
                gt_area = gw * gh
                anchor_areas = anchor_wh[:, 0] * anchor_wh[:, 1]
                union = gt_area + anchor_areas - intersection
                ious = intersection / union
                
                best_anchor = ious.argmax()
                
                # Set targets
                obj_mask[b, gy, gx, best_anchor] = 1
                noobj_mask[b, gy, gx, best_anchor] = 0
                
                # Bounding box targets (offset from grid cell)
                tx[b, gy, gx, best_anchor] = gt_cx[idx] / stride - gx.float()
                ty[b, gy, gx, best_anchor] = gt_cy[idx] / stride - gy.float()
                tw[b, gy, gx, best_anchor] = torch.log(gw / scaled_anchors[best_anchor, 0] + 1e-6)
                th[b, gy, gx, best_anchor] = torch.log(gh / scaled_anchors[best_anchor, 1] + 1e-6)
                
                # Confidence target
                tconf[b, gy, gx, best_anchor] = 1
                
                # Class target
                tcls[b, gy, gx, best_anchor, label] = 1
        
        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls
    
    def forward(self, predictions, targets, anchors, stride):
        """
        Compute YOLO detection loss
        
        Args:
            predictions: (B, H, W, num_anchors, 5 + num_classes)
            targets: List of dictionaries with 'boxes' and 'labels'
            anchors: (num_anchors, 2)
            stride: Feature map stride
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Extract predictions
        pred_x = torch.sigmoid(predictions[..., 0])
        pred_y = torch.sigmoid(predictions[..., 1])
        pred_w = predictions[..., 2]
        pred_h = predictions[..., 3]
        pred_conf = torch.sigmoid(predictions[..., 4])
        pred_cls = torch.sigmoid(predictions[..., 5:])
        
        # Build targets
        obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(
            predictions, targets, anchors, stride
        )
        
        # Localization loss (only for cells with objects)
        loss_x = self.mse_loss(pred_x * obj_mask, tx * obj_mask)
        loss_y = self.mse_loss(pred_y * obj_mask, ty * obj_mask)
        loss_w = self.mse_loss(pred_w * obj_mask, tw * obj_mask)
        loss_h = self.mse_loss(pred_h * obj_mask, th * obj_mask)
        loss_loc = (loss_x + loss_y + loss_w + loss_h) / batch_size
        
        # Objectness loss
        loss_conf_obj = self.bce_loss(pred_conf * obj_mask, tconf * obj_mask)
        loss_conf_noobj = self.bce_loss(pred_conf * noobj_mask, tconf * noobj_mask)
        loss_conf = (loss_conf_obj + self.lambda_noobj * loss_conf_noobj) / batch_size
        
        # Classification loss (only for cells with objects)
        obj_mask_expanded = obj_mask.unsqueeze(-1).expand_as(pred_cls)
        loss_cls = self.bce_loss(pred_cls * obj_mask_expanded, tcls * obj_mask_expanded)
        loss_cls = loss_cls / batch_size
        
        # Total loss
        total_loss = self.lambda_coord * loss_loc + loss_conf + loss_cls
        
        return {
            'total': total_loss,
            'localization': loss_loc,
            'confidence': loss_conf,
            'classification': loss_cls
        }


class MultiScaleDetectionLoss(nn.Module):
    """
    Multi-scale detection loss for YOLO
    """
    def __init__(self, num_classes, anchors_dict, lambda_coord=5.0, lambda_noobj=0.5):
        super(MultiScaleDetectionLoss, self).__init__()
        
        self.loss_small = YOLODetectionLoss(
            num_classes, anchors_dict['small'], lambda_coord, lambda_noobj
        )
        self.loss_medium = YOLODetectionLoss(
            num_classes, anchors_dict['medium'], lambda_coord, lambda_noobj
        )
        self.loss_large = YOLODetectionLoss(
            num_classes, anchors_dict['large'], lambda_coord, lambda_noobj
        )
    
    def forward(self, predictions, targets):
        """
        Compute multi-scale detection loss
        
        Args:
            predictions: Dictionary with 'predictions_small', 'predictions_medium', 'predictions_large'
            targets: List of target dictionaries
        """
        # Small scale (stride=2)
        loss_small = self.loss_small(
            predictions['predictions_small'],
            targets,
            predictions['anchors']['small'],
            stride=2
        )
        
        # Medium scale (stride=4)
        loss_medium = self.loss_medium(
            predictions['predictions_medium'],
            targets,
            predictions['anchors']['medium'],
            stride=4
        )
        
        # Large scale (stride=8)
        loss_large = self.loss_large(
            predictions['predictions_large'],
            targets,
            predictions['anchors']['large'],
            stride=8
        )
        
        # Combine losses
        total_loss = (
            loss_small['total'] +
            loss_medium['total'] +
            loss_large['total']
        )
        
        return {
            'total': total_loss,
            'small': loss_small,
            'medium': loss_medium,
            'large': loss_large
        }


if __name__ == "__main__":
    # Test detection loss
    num_classes = 12
    anchors = {
        'small': torch.tensor([[10, 13], [16, 30], [33, 23]]).float(),
        'medium': torch.tensor([[30, 61], [62, 45], [59, 119]]).float(),
        'large': torch.tensor([[116, 90], [156, 198], [373, 326]]).float()
    }
    
    loss_fn = MultiScaleDetectionLoss(num_classes, anchors)
    
    predictions = {
        'predictions_small': torch.randn(2, 208, 208, 3, 17),
        'predictions_medium': torch.randn(2, 104, 104, 3, 17),
        'predictions_large': torch.randn(2, 52, 52, 3, 17),
        'anchors': anchors
    }
    
    targets = [
        {'boxes': torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]]).float(),
         'labels': torch.tensor([0, 5])},
        {'boxes': torch.tensor([[20, 30, 80, 90]]).float(),
         'labels': torch.tensor([3])}
    ]
    
    losses = loss_fn(predictions, targets)
    print("Total loss:", losses['total'].item())
