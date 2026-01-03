"""
Unified model integrating restoration and detection
"""
import torch
import torch.nn as nn
from .restoration_encoder import RestorationEncoder
from .enhancement_modules import MultiObjectiveRestoration
from .detection_head import YOLODetectionHead


class LowLightObjectDetector(nn.Module):
    """
    End-to-end low-light object detection model
    
    Pipeline:
    1. Shared Restoration Encoder (feature extraction)
    2. Multi-Objective Restoration (illumination + denoise + deblur)
    3. YOLO Detection Head (object detection)
    """
    def __init__(self, num_classes=12, base_channels=32, num_anchors=3, use_gradient_checkpointing=False):
        super(LowLightObjectDetector, self).__init__()
        
        self.num_classes = num_classes
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Reduce curve iterations for smaller models (memory constrained devices)
        num_curve_iterations = 4 if base_channels <= 16 else 8
        
        # Stage 1: Shared encoder for feature extraction
        self.encoder = RestorationEncoder(
            in_channels=3,
            base_channels=base_channels
        )
        
        # Stage 2: Multi-objective restoration
        self.restoration = MultiObjectiveRestoration(
            feature_channels=base_channels,
            num_curve_iterations=num_curve_iterations
        )
        
        # Stage 3: Detection head
        self.detector = YOLODetectionHead(
            num_classes=num_classes,
            num_anchors=num_anchors,
            base_channels=base_channels
        )
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage"""
        # Store the original bottleneck forward
        if hasattr(self.encoder, 'bottleneck'):
            original_bottleneck_forward = self.encoder.bottleneck.forward
            
            def checkpointed_forward(x):
                """Checkpointed forward pass for bottleneck"""
                return torch.utils.checkpoint.checkpoint(
                    original_bottleneck_forward, 
                    x, 
                    use_reentrant=False
                )
            
            # Replace forward method with checkpointed version
            self.encoder.bottleneck.forward = checkpointed_forward
    
    def forward(self, x, return_all=True):
        """
        Forward pass through the complete pipeline
        
        Args:
            x: Input low-light images (B, 3, H, W), normalized to [0, 1]
            return_all: If True, return all intermediate results
        
        Returns:
            Dictionary containing:
                - restored_image: Final restored image
                - predictions: Multi-scale detection predictions
                - (optional) intermediate results for loss computation
        """
        # Ensure input is in correct range
        x = torch.clamp(x, 0, 1)
        
        # Stage 1: Feature extraction
        encoder_output = self.encoder(x)
        final_features = encoder_output['final_features']
        decoder_features = encoder_output['decoder_features']
        
        # Stage 2: Multi-objective restoration
        restoration_output = self.restoration(final_features, x)
        restored_image = restoration_output['restored_image']
        restoration_features = restoration_output['final_features']
        
        # Stage 3: Object detection
        detection_output = self.detector(restoration_features, decoder_features)
        
        # Prepare output
        output = {
            'restored_image': restored_image,
            'predictions': detection_output
        }
        
        # Include intermediate results if requested (for training/visualization)
        if return_all:
            output.update({
                'enhanced_image': restoration_output['enhanced_image'],
                'denoised_image': restoration_output['denoised_image'],
                'curve_params': restoration_output['curve_params'],
                'encoder_output': encoder_output
            })
        
        return output
    
    def predict(self, x, conf_threshold=0.25, iou_threshold=0.45):
        """
        Inference mode: Get final detection results
        
        Args:
            x: Input images (B, 3, H, W)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        
        Returns:
            For each image: (boxes, scores, labels)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_all=False)
            predictions = output['predictions']
            
            # Decode predictions at all scales
            all_boxes = []
            all_scores = []
            all_labels = []
            
            # Small scale (stride=2)
            boxes_s, scores_s, labels_s = self.detector.decode_predictions(
                predictions['predictions_small'],
                predictions['anchors']['small'],
                stride=2,
                conf_threshold=conf_threshold
            )
            
            # Medium scale (stride=4)
            boxes_m, scores_m, labels_m = self.detector.decode_predictions(
                predictions['predictions_medium'],
                predictions['anchors']['medium'],
                stride=4,
                conf_threshold=conf_threshold
            )
            
            # Large scale (stride=8)
            boxes_l, scores_l, labels_l = self.detector.decode_predictions(
                predictions['predictions_large'],
                predictions['anchors']['large'],
                stride=8,
                conf_threshold=conf_threshold
            )
            
            # Combine all scales
            batch_size = x.size(0)
            final_results = []
            
            for b in range(batch_size):
                # Concatenate all scales
                boxes = torch.cat([boxes_s[b], boxes_m[b], boxes_l[b]], dim=0)
                scores = torch.cat([scores_s[b], scores_m[b], scores_l[b]], dim=0)
                labels = torch.cat([labels_s[b], labels_m[b], labels_l[b]], dim=0)
                
                # Apply NMS
                if len(boxes) > 0:
                    keep_indices = self.nms(boxes, scores, iou_threshold)
                    boxes = boxes[keep_indices]
                    scores = scores[keep_indices]
                    labels = labels[keep_indices]
                
                final_results.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                    'restored_image': output['restored_image'][b]
                })
            
            return final_results
    
    def nms(self, boxes, scores, iou_threshold):
        """
        Non-Maximum Suppression
        
        Args:
            boxes: (N, 4) tensor
            scores: (N,) tensor
            iou_threshold: IoU threshold
        
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            # Compute IoU
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


if __name__ == "__main__":
    # Test the complete model
    model = LowLightObjectDetector(num_classes=12)
    x = torch.rand(2, 3, 416, 416)
    
    # Training mode
    output = model(x)
    print("Restored image shape:", output['restored_image'].shape)
    print("Predictions keys:", output['predictions'].keys())
    
    # Inference mode
    results = model.predict(x)
    print(f"\nInference results for {len(results)} images")
    for i, res in enumerate(results):
        print(f"Image {i}: {len(res['boxes'])} detections")
