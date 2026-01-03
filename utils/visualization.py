"""
Visualization utilities for training metrics and results
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import json
import torch


def plot_training_curves(history, save_path='training_curves.png'):
    """
    Plot training and validation loss curves
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'epochs'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = history['epochs']
    
    # Total Loss
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Component Losses
    ax = axes[0, 1]
    if 'train_illum_loss' in history:
        ax.plot(epochs, history['train_illum_loss'], label='Illumination', linewidth=2)
    if 'train_resto_loss' in history:
        ax.plot(epochs, history['train_resto_loss'], label='Restoration', linewidth=2)
    if 'train_detect_loss' in history:
        ax.plot(epochs, history['train_detect_loss'], label='Detection', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Component Losses', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # mAP
    ax = axes[1, 0]
    if 'val_mAP' in history:
        ax.plot(epochs, history['val_mAP'], 'g-', label='mAP', linewidth=2, marker='o')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mAP', fontsize=12)
    ax.set_title('Mean Average Precision', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate
    ax = axes[1, 1]
    if 'learning_rate' in history:
        ax.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_loss_breakdown(history, save_path='loss_breakdown.png'):
    """
    Plot detailed breakdown of all loss components
    
    Args:
        history: Dictionary with loss components
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Detailed Loss Breakdown', fontsize=16, fontweight='bold')
    
    epochs = history['epochs']
    
    # Illumination losses
    ax = axes[0, 0]
    if 'train_spatial_loss' in history:
        ax.plot(epochs, history['train_spatial_loss'], label='Spatial', linewidth=2)
    if 'train_exposure_loss' in history:
        ax.plot(epochs, history['train_exposure_loss'], label='Exposure', linewidth=2)
    if 'train_color_loss' in history:
        ax.plot(epochs, history['train_color_loss'], label='Color', linewidth=2)
    ax.set_title('Illumination Components', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Detection losses
    ax = axes[0, 1]
    if 'train_loc_loss' in history:
        ax.plot(epochs, history['train_loc_loss'], label='Localization', linewidth=2)
    if 'train_conf_loss' in history:
        ax.plot(epochs, history['train_conf_loss'], label='Confidence', linewidth=2)
    if 'train_cls_loss' in history:
        ax.plot(epochs, history['train_cls_loss'], label='Classification', linewidth=2)
    ax.set_title('Detection Components', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overall comparison
    ax = axes[0, 2]
    if 'train_illum_loss' in history:
        ax.plot(epochs, history['train_illum_loss'], label='Illumination', linewidth=2)
    if 'train_resto_loss' in history:
        ax.plot(epochs, history['train_resto_loss'], label='Restoration', linewidth=2)
    if 'train_detect_loss' in history:
        ax.plot(epochs, history['train_detect_loss'], label='Detection', linewidth=2)
    ax.set_title('Loss Components Comparison', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class AP (if available)
    ax = axes[1, 0]
    if 'val_AP_per_class' in history and len(history['val_AP_per_class']) > 0:
        # Take the last epoch's per-class AP
        last_ap = history['val_AP_per_class'][-1]
        class_names = history.get('class_names', [f'Class {i}' for i in range(len(last_ap))])
        ax.barh(range(len(last_ap)), last_ap, color='steelblue')
        ax.set_yticks(range(len(last_ap)))
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Average Precision')
        ax.set_title('Per-Class AP (Final Epoch)', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No per-class AP data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    # Train vs Val comparison
    ax = axes[1, 1]
    if 'train_loss' in history and 'val_loss' in history:
        width = 0.35
        x = np.arange(len(epochs))
        ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, marker='o')
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2, marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Validation Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Metrics summary
    ax = axes[1, 2]
    metrics_text = "Training Summary\n" + "="*30 + "\n"
    if 'train_loss' in history:
        metrics_text += f"Final Train Loss: {history['train_loss'][-1]:.4f}\n"
    if 'val_loss' in history:
        metrics_text += f"Final Val Loss: {history['val_loss'][-1]:.4f}\n"
    if 'val_mAP' in history:
        metrics_text += f"Best mAP: {max(history['val_mAP']):.4f}\n"
        metrics_text += f"Final mAP: {history['val_mAP'][-1]:.4f}\n"
    
    ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss breakdown saved to {save_path}")


def plot_confusion_matrix(confusion_matrix, class_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix for object detection
    
    Args:
        confusion_matrix: NxN numpy array
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, int(confusion_matrix[i, j]),
                          ha="center", va="center",
                          color="white" if confusion_matrix[i, j] > thresh else "black",
                          fontsize=8)
    
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def visualize_predictions(image, predictions, class_names, save_path='predictions.png', 
                         show_enhanced=False, enhanced_image=None):
    """
    Visualize detection predictions on image
    
    Args:
        image: Original image tensor (C, H, W) or numpy array
        predictions: Dictionary with 'boxes', 'scores', 'labels'
        class_names: List of class names
        save_path: Path to save the plot
        show_enhanced: Whether to show enhanced image comparison
        enhanced_image: Enhanced image tensor if available
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().permute(1, 2, 0).numpy()
    if torch.is_tensor(enhanced_image):
        enhanced_image = enhanced_image.cpu().permute(1, 2, 0).numpy()
    
    # Ensure image is in [0, 1] range
    image = np.clip(image, 0, 1)
    if enhanced_image is not None:
        enhanced_image = np.clip(enhanced_image, 0, 1)
    
    # Create figure
    if show_enhanced and enhanced_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        ax2 = None
    
    # Plot original image with detections
    ax1.imshow(image)
    ax1.set_title('Detections on Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw bounding boxes
    boxes = predictions['boxes'].cpu().numpy() if torch.is_tensor(predictions['boxes']) else predictions['boxes']
    scores = predictions['scores'].cpu().numpy() if torch.is_tensor(predictions['scores']) else predictions['scores']
    labels = predictions['labels'].cpu().numpy() if torch.is_tensor(predictions['labels']) else predictions['labels']
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[int(label)]
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # Add label
        label_text = f"{class_names[int(label)]}: {score:.2f}"
        ax1.text(x1, y1 - 5, label_text, color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # Plot enhanced image if available
    if ax2 is not None:
        ax2.imshow(enhanced_image)
        ax2.set_title('Enhanced Image', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Draw same boxes on enhanced image
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            color = colors[int(label)]
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2,
                                    edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Predictions visualization saved to {save_path}")


def plot_enhancement_comparison(original, enhanced, denoised, restored, save_path='enhancement_comparison.png'):
    """
    Plot comparison of enhancement stages
    
    Args:
        original: Original low-light image
        enhanced: Illumination-enhanced image
        denoised: Denoised image
        restored: Final restored image
        save_path: Path to save the plot
    """
    # Convert tensors to numpy
    images = [original, enhanced, denoised, restored]
    titles = ['Original (Low-Light)', 'Illumination Enhanced', 'Denoised', 'Final Restored']
    
    processed_images = []
    for img in images:
        if torch.is_tensor(img):
            img = img.cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        processed_images.append(img)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, img, title in zip(axes, processed_images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Image Enhancement Pipeline', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Enhancement comparison saved to {save_path}")


def save_metrics_json(history, save_path='training_metrics.json'):
    """
    Save training metrics to JSON file
    
    Args:
        history: Dictionary with training history
        save_path: Path to save JSON file
    """
    # Convert numpy arrays and tensors to lists
    clean_history = {}
    for key, value in history.items():
        if isinstance(value, (np.ndarray, list)):
            clean_history[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
        elif isinstance(value, (np.floating, float)):
            clean_history[key] = float(value)
        else:
            clean_history[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(clean_history, f, indent=4)
    
    print(f"Metrics saved to {save_path}")


def create_training_summary(history, save_path='training_summary.png'):
    """
    Create a comprehensive training summary visualization
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    epochs = history['epochs']
    
    # Main loss curve (large)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2.5, marker='o', markersize=4)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2.5, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # mAP curve
    ax2 = fig.add_subplot(gs[0, 2])
    if 'val_mAP' in history:
        ax2.plot(epochs, history['val_mAP'], 'g-', linewidth=2.5, marker='D', markersize=4)
        ax2.fill_between(epochs, history['val_mAP'], alpha=0.3, color='green')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('mAP', fontsize=11)
    ax2.set_title('Mean Average Precision', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Component losses
    ax3 = fig.add_subplot(gs[1, 0])
    if 'train_illum_loss' in history:
        ax3.plot(epochs, history['train_illum_loss'], label='Illumination', linewidth=2)
    if 'train_resto_loss' in history:
        ax3.plot(epochs, history['train_resto_loss'], label='Restoration', linewidth=2)
    if 'train_detect_loss' in history:
        ax3.plot(epochs, history['train_detect_loss'], label='Detection', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss', fontsize=11)
    ax3.set_title('Loss Components', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Learning rate
    ax4 = fig.add_subplot(gs[1, 1])
    if 'learning_rate' in history:
        ax4.plot(epochs, history['learning_rate'], color='purple', linewidth=2)
        ax4.fill_between(epochs, history['learning_rate'], alpha=0.3, color='purple')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Learning Rate', fontsize=11)
    ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Statistics box
    ax5 = fig.add_subplot(gs[1, 2])
    stats_text = "Training Statistics\n" + "="*35 + "\n\n"
    if 'train_loss' in history:
        stats_text += f"Initial Train Loss:  {history['train_loss'][0]:.4f}\n"
        stats_text += f"Final Train Loss:    {history['train_loss'][-1]:.4f}\n"
        stats_text += f"Best Train Loss:     {min(history['train_loss']):.4f}\n\n"
    if 'val_loss' in history:
        stats_text += f"Initial Val Loss:    {history['val_loss'][0]:.4f}\n"
        stats_text += f"Final Val Loss:      {history['val_loss'][-1]:.4f}\n"
        stats_text += f"Best Val Loss:       {min(history['val_loss']):.4f}\n\n"
    if 'val_mAP' in history:
        stats_text += f"Initial mAP:         {history['val_mAP'][0]:.4f}\n"
        stats_text += f"Final mAP:           {history['val_mAP'][-1]:.4f}\n"
        stats_text += f"Best mAP:            {max(history['val_mAP']):.4f}\n"
    
    ax5.text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
             verticalalignment='top', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.axis('off')
    
    # Per-class AP bar chart
    ax6 = fig.add_subplot(gs[2, :])
    if 'val_AP_per_class' in history and len(history['val_AP_per_class']) > 0:
        last_ap = history['val_AP_per_class'][-1]
        class_names = history.get('class_names', [f'C{i}' for i in range(len(last_ap))])
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(last_ap)))
        bars = ax6.bar(class_names, last_ap, color=colors_bar, edgecolor='black', linewidth=1.5)
        ax6.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Average Precision', fontsize=12, fontweight='bold')
        ax6.set_title('Per-Class Average Precision (Final Epoch)', fontsize=14, fontweight='bold')
        ax6.grid(True, axis='y', alpha=0.3)
        ax6.set_ylim([0, 1.0])
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'No per-class AP data available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.axis('off')
    
    plt.suptitle('Low-Light Object Detection - Training Summary', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training summary saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - plot_training_curves()")
    print("  - plot_loss_breakdown()")
    print("  - plot_confusion_matrix()")
    print("  - visualize_predictions()")
    print("  - plot_enhancement_comparison()")
    print("  - create_training_summary()")
