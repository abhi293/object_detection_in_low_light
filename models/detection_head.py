"""
YOLO-style detection head for object detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionBlock(nn.Module):
    """Single scale detection block"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(DetectionBlock, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Each anchor predicts: [x, y, w, h, objectness, class1, class2, ..., classN]
        self.num_outputs = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels * 2, self.num_outputs, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W)
        
        Returns:
            Reshaped predictions (B, H, W, num_anchors, 5 + num_classes)
        """
        out = self.conv(x)
        batch_size, _, height, width = out.shape
        
        # Reshape to (B, num_anchors, 5 + num_classes, H, W)
        out = out.view(batch_size, self.num_anchors, 5 + self.num_classes, height, width)
        
        # Permute to (B, H, W, num_anchors, 5 + num_classes)
        out = out.permute(0, 3, 4, 1, 2).contiguous()
        
        return out


class YOLODetectionHead(nn.Module):
    """
    Multi-scale YOLO-style detection head
    Processes restored features to detect objects
    """
    def __init__(self, num_classes=12, num_anchors=3, base_channels=32):
        super(YOLODetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.base_channels = base_channels
        
        # Feature adaptation layers for different scales (adapt to base_channels)
        self.adapt_small = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.adapt_medium = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.adapt_large = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Detection heads for different scales
        self.detect_small = DetectionBlock(base_channels * 2, num_classes, num_anchors)   # For small objects
        self.detect_medium = DetectionBlock(base_channels * 4, num_classes, num_anchors)  # For medium objects
        self.detect_large = DetectionBlock(base_channels * 8, num_classes, num_anchors)   # For large objects
        
        # Predefined anchors (will be scaled based on feature map size)
        self.register_buffer('anchors_small', torch.tensor([
            [10, 13], [16, 30], [33, 23]
        ]).float())
        
        self.register_buffer('anchors_medium', torch.tensor([
            [30, 61], [62, 45], [59, 119]
        ]).float())
        
        self.register_buffer('anchors_large', torch.tensor([
            [116, 90], [156, 198], [373, 326]
        ]).float())
    
    def forward(self, restored_features, decoder_features):
        """
        Args:
            restored_features: Final restored features (B, 32, H, W)
            decoder_features: List of decoder features at different scales
        
        Returns:
            Dictionary with multi-scale predictions
        """
        # Extract features at different scales from decoder
        feat_small = decoder_features[-1]    # Highest resolution (B, 32, H, W)
        feat_medium = decoder_features[-2]   # Medium resolution (B, 64, H/2, W/2)
        feat_large = decoder_features[-3]    # Lower resolution (B, 128, H/4, W/4)
        
        # Adapt features for detection
        feat_small = self.adapt_small(feat_small)
        feat_medium = self.adapt_medium(feat_medium)
        feat_large = self.adapt_large(feat_large)
        
        # Multi-scale predictions
        pred_small = self.detect_small(feat_small)
        pred_medium = self.detect_medium(feat_medium)
        pred_large = self.detect_large(feat_large)
        
        return {
            'predictions_small': pred_small,
            'predictions_medium': pred_medium,
            'predictions_large': pred_large,
            'anchors': {
                'small': self.anchors_small,
                'medium': self.anchors_medium,
                'large': self.anchors_large
            }
        }
    
    def decode_predictions(self, predictions, anchors, stride, conf_threshold=0.25):
        """
        Decode raw predictions to bounding boxes
        
        Args:
            predictions: (B, H, W, num_anchors, 5 + num_classes)
            anchors: (num_anchors, 2)
            stride: Feature map stride
            conf_threshold: Confidence threshold
        
        Returns:
            boxes, scores, labels for each image in batch
        """
        batch_size, height, width, num_anchors, _ = predictions.shape
        device = predictions.device
        
        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).unsqueeze(3)  # (1, H, W, 1, 2)
        
        # Scale anchors
        scaled_anchors = anchors / stride
        scaled_anchors = scaled_anchors.view(1, 1, 1, num_anchors, 2)
        
        # Decode predictions
        xy = (torch.sigmoid(predictions[..., 0:2]) + grid) * stride
        wh = (torch.exp(predictions[..., 2:4]) * scaled_anchors)
        objectness = torch.sigmoid(predictions[..., 4:5])
        class_probs = torch.sigmoid(predictions[..., 5:])
        
        # Convert to [x1, y1, x2, y2]
        boxes = torch.zeros_like(predictions[..., :4])
        boxes[..., 0] = xy[..., 0] - wh[..., 0] / 2  # x1
        boxes[..., 1] = xy[..., 1] - wh[..., 1] / 2  # y1
        boxes[..., 2] = xy[..., 0] + wh[..., 0] / 2  # x2
        boxes[..., 3] = xy[..., 1] + wh[..., 1] / 2  # y2
        
        # Get class scores
        class_scores = objectness * class_probs
        
        # Process each image in batch
        batch_boxes = []
        batch_scores = []
        batch_labels = []
        
        for b in range(batch_size):
            # Flatten predictions
            boxes_b = boxes[b].view(-1, 4)
            scores_b = class_scores[b].view(-1, self.num_classes)
            
            # Get best class for each prediction
            max_scores, labels_b = scores_b.max(dim=1)
            
            # Filter by confidence
            mask = max_scores > conf_threshold
            
            batch_boxes.append(boxes_b[mask])
            batch_scores.append(max_scores[mask])
            batch_labels.append(labels_b[mask])
        
        return batch_boxes, batch_scores, batch_labels


if __name__ == "__main__":
    # Test detection head
    base_channels = 32
    model = YOLODetectionHead(num_classes=12, base_channels=base_channels)
    
    restored_features = torch.randn(2, base_channels, 416, 416)
    decoder_features = [
        torch.randn(2, base_channels * 4, 52, 52),   # Large scale
        torch.randn(2, base_channels * 2, 104, 104),  # Medium scale
        torch.randn(2, base_channels, 208, 208)   # Small scale
    ]
    
    output = model(restored_features, decoder_features)
    print("Small predictions shape:", output['predictions_small'].shape)
    print("Medium predictions shape:", output['predictions_medium'].shape)
    print("Large predictions shape:", output['predictions_large'].shape)
