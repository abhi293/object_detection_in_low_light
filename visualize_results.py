"""
Script to visualize training results from saved checkpoints and logs
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils.visualization import (
    plot_training_curves,
    plot_loss_breakdown,
    plot_confusion_matrix,
    create_training_summary,
    visualize_predictions,
    plot_enhancement_comparison
)
from models.unified_model import LowLightObjectDetector
from data.dataset import ExDarkAnnotatedDataset
from data.transforms import get_transforms
from config.default_config import DEFAULT_CONFIG
import torch.nn.functional as F


def load_checkpoint_history(checkpoint_path):
    """Load training history from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'history' in checkpoint:
        return checkpoint['history']
    else:
        print("Warning: No history found in checkpoint")
        return None


def visualize_from_checkpoint(checkpoint_path, output_dir='visualizations'):
    """Generate all visualizations from a checkpoint file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load history
    if 'history' not in checkpoint:
        print("No training history found in checkpoint!")
        return
    
    history = checkpoint['history']
    print(f"Found {len(history.get('epochs', []))} epochs of training data")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    # 1. Training curves
    plot_training_curves(history, save_path=output_dir / 'training_curves.png')
    
    # 2. Loss breakdown
    plot_loss_breakdown(history, save_path=output_dir / 'loss_breakdown.png')
    
    # 3. Training summary (comprehensive)
    create_training_summary(history, save_path=output_dir / 'training_summary.png')
    
    # 4. Save metrics as JSON
    from utils.visualization import save_metrics_json
    save_metrics_json(history, save_path=output_dir / 'training_metrics.json')
    
    # 5. Print final statistics
    print("\n" + "="*60)
    print("Training Statistics:")
    print("="*60)
    if 'train_loss' in history:
        print(f"Final Training Loss:   {history['train_loss'][-1]:.4f}")
        print(f"Best Training Loss:    {min(history['train_loss']):.4f}")
    if 'val_loss' in history:
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
        print(f"Best Validation Loss:  {min(history['val_loss']):.4f}")
    if 'val_mAP' in history:
        print(f"Final mAP:             {history['val_mAP'][-1]:.4f}")
        print(f"Best mAP:              {max(history['val_mAP']):.4f}")
    print("="*60)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


def visualize_predictions_from_model(checkpoint_path, data_root, num_samples=5, 
                                     output_dir='predictions', device='cpu',
                                     image_size=416, conf_threshold=0.5):
    """
    Load model and visualize predictions on sample images
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Root directory of ExDark dataset
        num_samples: Number of samples to visualize
        output_dir: Output directory for visualizations
        device: Device to run inference on
        image_size: Image size for inference
        conf_threshold: Confidence threshold for predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🔮 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model_config = checkpoint.get('model_config', {})
    base_channels = model_config.get('base_channels', 32)
    num_classes = model_config.get('num_classes', 12)
    
    model = LowLightObjectDetector(
        num_classes=num_classes,
        base_channels=base_channels
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load dataset
    print(f"Loading dataset from: {data_root}")
    _, val_transforms = get_transforms(image_size=image_size)
    dataset = ExDarkAnnotatedDataset(
        root_dir=data_root,
        transform=val_transforms,
        annotation_dir=None  # Will use dummy boxes
    )
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    print(f"\n📸 Visualizing {len(indices)} predictions...")
    
    class_names = DEFAULT_CONFIG['class_names']
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, targets = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(image_tensor)
            restored_image = outputs['restored_image'][0]
            
            # Get predictions
            predictions = []
            for scale_idx, pred in enumerate(outputs['predictions']):
                # pred shape: [1, 3, H, W, num_classes+5]
                batch_size, num_anchors, grid_h, grid_w, _ = pred.shape
                
                # Reshape predictions
                pred = pred.view(batch_size, -1, num_classes + 5)
                
                # Extract components
                conf = torch.sigmoid(pred[..., 4])
                cls_pred = torch.softmax(pred[..., 5:], dim=-1)
                
                # Filter by confidence
                mask = conf[0] > conf_threshold
                if mask.sum() == 0:
                    continue
                
                # Get boxes, scores, labels
                boxes = pred[0, mask, :4]
                scores = conf[0, mask]
                labels = torch.argmax(cls_pred[0, mask], dim=-1)
                
                predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels
                })
            
            # Combine predictions from all scales
            if predictions:
                all_boxes = torch.cat([p['boxes'] for p in predictions], dim=0)
                all_scores = torch.cat([p['scores'] for p in predictions], dim=0)
                all_labels = torch.cat([p['labels'] for p in predictions], dim=0)
                
                # Simple NMS (keep top predictions)
                top_k = min(10, len(all_scores))
                top_indices = torch.topk(all_scores, top_k).indices
                
                final_predictions = {
                    'boxes': all_boxes[top_indices],
                    'scores': all_scores[top_indices],
                    'labels': all_labels[top_indices]
                }
            else:
                final_predictions = {
                    'boxes': torch.zeros((0, 4)),
                    'scores': torch.zeros(0),
                    'labels': torch.zeros(0, dtype=torch.long)
                }
            
            # Visualize
            save_path = output_dir / f'prediction_{i+1}.png'
            visualize_predictions(
                image=image,
                predictions=final_predictions,
                class_names=class_names,
                save_path=save_path,
                show_enhanced=True,
                enhanced_image=restored_image
            )
            
            print(f"  ✓ Saved: {save_path} ({len(final_predictions['boxes'])} detections)")
    
    print(f"\n✅ All predictions saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, default='metrics',
                       choices=['metrics', 'predictions', 'both'],
                       help='Visualization mode')
    parser.add_argument('--data_root', type=str, default='ExDark_Dataset/ExDark',
                       help='Root directory of ExDark dataset (for predictions mode)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (predictions mode)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for inference')
    parser.add_argument('--image_size', type=int, default=416,
                       help='Image size for inference')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                       help='Confidence threshold for predictions')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Low-Light Object Detection - Results Visualization")
    print("="*60)
    
    if args.mode in ['metrics', 'both']:
        print("\n📊 Generating training metrics visualizations...")
        visualize_from_checkpoint(args.checkpoint, output_dir=args.output_dir)
    
    if args.mode in ['predictions', 'both']:
        print("\n🔮 Generating prediction visualizations...")
        pred_dir = Path(args.output_dir) / 'predictions'
        visualize_predictions_from_model(
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            num_samples=args.num_samples,
            output_dir=pred_dir,
            device=args.device,
            image_size=args.image_size,
            conf_threshold=args.conf_threshold
        )
    
    print("\n" + "="*60)
    print("✨ Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
