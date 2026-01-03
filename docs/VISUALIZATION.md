# Visualization Tools Documentation

## Overview

The Low-Light Object Detection project includes comprehensive visualization tools for analyzing training progress, model performance, and prediction results.

## Available Visualization Functions

### 1. Training Curves (`plot_training_curves`)

**Purpose:** Display training and validation loss progression over epochs

**Features:**
- Total loss comparison (train vs validation)
- Component losses (illumination, restoration, detection)
- Mean Average Precision (mAP) progression
- Learning rate schedule

**Usage:**
```python
from utils.visualization import plot_training_curves

history = {
    'epochs': [1, 2, 3, ...],
    'train_loss': [...],
    'val_loss': [...],
    'val_mAP': [...],
    'learning_rate': [...]
}

plot_training_curves(history, save_path='training_curves.png')
```

---

### 2. Loss Breakdown (`plot_loss_breakdown`)

**Purpose:** Detailed analysis of all loss components

**Features:**
- Illumination losses (spatial, exposure, color)
- Detection losses (localization, confidence, classification)
- Component comparison over time
- Per-class Average Precision bar chart
- Numerical metrics summary

**Usage:**
```python
from utils.visualization import plot_loss_breakdown

plot_loss_breakdown(history, save_path='loss_breakdown.png')
```

---

### 3. Training Summary (`create_training_summary`)

**Purpose:** Comprehensive overview of entire training session

**Features:**
- Large training/validation loss curves
- mAP progression with area fill
- All component losses
- Learning rate schedule (log scale)
- Statistics box with key metrics
- Per-class AP bar chart with values

**Usage:**
```python
from utils.visualization import create_training_summary

create_training_summary(history, save_path='training_summary.png')
```

---

### 4. Confusion Matrix (`plot_confusion_matrix`)

**Purpose:** Visualize classification performance across classes

**Features:**
- Color-coded matrix (blue scale)
- Count annotations on each cell
- Automatic color thresholding for readability
- All 12 ExDark classes

**Usage:**
```python
from utils.visualization import plot_confusion_matrix
import numpy as np

confusion_matrix = np.array([[...]])  # Shape: (12, 12)
class_names = ['Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 
               'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table']

plot_confusion_matrix(confusion_matrix, class_names, 
                     save_path='confusion_matrix.png')
```

---

### 5. Prediction Visualization (`visualize_predictions`)

**Purpose:** Display detection results on images

**Features:**
- Bounding boxes with class labels
- Confidence scores
- Side-by-side original and enhanced images
- Color-coded boxes per class

**Usage:**
```python
from utils.visualization import visualize_predictions

predictions = {
    'boxes': torch.tensor([[x1, y1, x2, y2], ...]),  # Shape: (N, 4)
    'scores': torch.tensor([0.9, 0.85, ...]),        # Shape: (N,)
    'labels': torch.tensor([0, 2, ...])              # Shape: (N,)
}

visualize_predictions(
    image=image_tensor,                    # (C, H, W)
    predictions=predictions,
    class_names=class_names,
    save_path='predictions.png',
    show_enhanced=True,
    enhanced_image=enhanced_tensor         # (C, H, W)
)
```

---

### 6. Enhancement Comparison (`plot_enhancement_comparison`)

**Purpose:** Show progression through enhancement stages

**Features:**
- Original low-light image
- Illumination-enhanced image
- Denoised image
- Final restored image

**Usage:**
```python
from utils.visualization import plot_enhancement_comparison

plot_enhancement_comparison(
    original=original_tensor,      # (C, H, W)
    enhanced=enhanced_tensor,
    denoised=denoised_tensor,
    restored=restored_tensor,
    save_path='enhancement_comparison.png'
)
```

---

### 7. Metrics JSON Export (`save_metrics_json`)

**Purpose:** Save all metrics in machine-readable format

**Usage:**
```python
from utils.visualization import save_metrics_json

save_metrics_json(history, save_path='training_metrics.json')
```

**Output format:**
```json
{
  "epochs": [1, 2, 3],
  "train_loss": [2.5, 2.3, 2.1],
  "val_loss": [2.6, 2.4, 2.2],
  "val_mAP": [0.15, 0.22, 0.28],
  "learning_rate": [0.0001, 0.000095, 0.00009],
  "class_names": ["Bicycle", "Boat", ...]
}
```

---

## Command-Line Tools

### visualize_results.py

**Purpose:** Generate all visualizations from saved checkpoints

**Modes:**
1. **metrics** - Generate training metric visualizations only
2. **predictions** - Generate prediction visualizations only
3. **both** - Generate both types

**Examples:**

```bash
# Generate all metrics plots
python visualize_results.py \
  --checkpoint checkpoints/best_model.pth \
  --mode metrics

# Generate prediction visualizations
python visualize_results.py \
  --checkpoint checkpoints/best_model.pth \
  --mode predictions \
  --data_root ExDark_Dataset/ExDark \
  --num_samples 10 \
  --device cpu \
  --conf_threshold 0.3

# Generate everything
python visualize_results.py \
  --checkpoint checkpoints/best_model.pth \
  --mode both \
  --data_root ExDark_Dataset/ExDark \
  --output_dir my_visualizations
```

**Arguments:**
- `--checkpoint`: Path to checkpoint file (required)
- `--mode`: Visualization mode (metrics/predictions/both)
- `--data_root`: Dataset root directory
- `--output_dir`: Output directory for plots
- `--num_samples`: Number of prediction samples to visualize
- `--device`: Device for inference (cpu/cuda/directml)
- `--image_size`: Image size for inference (default: 416)
- `--conf_threshold`: Confidence threshold for predictions (default: 0.3)

---

### visualize.bat (Windows)

**Purpose:** Convenient batch script for Windows users

**Usage:**
```bash
# Metrics only
visualize.bat checkpoints/best_model.pth metrics

# Both metrics and predictions
visualize.bat checkpoints/best_model.pth both

# Default mode (both)
visualize.bat checkpoints/best_model.pth
```

---

## Automatic Visualization During Training

The training script automatically generates visualizations when training completes:

**Generated files:**
```
checkpoints/
└── visualizations/
    ├── training_curves.png      # Loss and mAP curves
    ├── loss_breakdown.png       # Component losses
    ├── training_summary.png     # Comprehensive overview
    └── training_metrics.json    # JSON metrics
```

**No additional action needed** - just run training normally:
```bash
python train.py --data_root ExDark_Dataset/ExDark --epochs 20
```

---

## Testing Visualization Tools

Generate sample plots without training:

```bash
python generate_sample_plots.py
```

This creates demonstration plots with synthetic data:
- `sample_training_curves.png`
- `sample_loss_breakdown.png`
- `sample_training_summary.png`
- `sample_confusion_matrix.png`

---

## Integration with Training

The visualization system is fully integrated with the training pipeline:

1. **History Tracking:** All metrics are automatically tracked during training
2. **Checkpoint Saving:** History is saved in checkpoint files
3. **Auto-Generation:** Plots are created automatically at training completion
4. **Manual Generation:** Use `visualize_results.py` to regenerate anytime

**Checkpoint structure:**
```python
checkpoint = {
    'epoch': 20,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'best_map': 0.65,
    'history': {
        'epochs': [1, 2, ..., 20],
        'train_loss': [...],
        'val_loss': [...],
        'val_mAP': [...],
        ...
    }
}
```

---

## Customization

### Custom Plot Styling

Modify visualization parameters in `utils/visualization.py`:

```python
# Change figure size
fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # Default: (15, 10)

# Change DPI
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Default: 150

# Change color schemes
colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
```

### Adding New Metrics

1. Track metric in training loop:
```python
history['my_metric'].append(metric_value)
```

2. Add plot in visualization function:
```python
ax.plot(epochs, history['my_metric'], label='My Metric')
```

### Custom Visualizations

Create custom visualization functions following the same pattern:

```python
def plot_my_custom_viz(history, save_path='custom.png'):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Your plotting code here
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Custom visualization saved to {save_path}")
```

---

## Tips & Best Practices

### 1. High-Resolution Plots
```python
# Increase DPI for publications
plot_training_curves(history, save_path='plot.png')
# Then manually edit to increase dpi=300 in the function
```

### 2. Viewing Plots
- Windows: Use default image viewer or VS Code
- Python: Use matplotlib's interactive backend
```python
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode
```

### 3. Multiple Experiments
```bash
# Compare different runs
python visualize_results.py --checkpoint run1/best_model.pth --output_dir viz_run1
python visualize_results.py --checkpoint run2/best_model.pth --output_dir viz_run2
```

### 4. Batch Processing
Create a script to visualize multiple checkpoints:
```python
import glob
from visualize_results import visualize_from_checkpoint

for ckpt in glob.glob('experiments/*/best_model.pth'):
    output = ckpt.replace('best_model.pth', 'visualizations')
    visualize_from_checkpoint(ckpt, output_dir=output)
```

---

## Troubleshooting

### Issue: "No module named matplotlib"
```bash
pip install matplotlib
```

### Issue: Plots not showing (headless environment)
Already handled - all functions use `plt.savefig()` and `plt.close()`

### Issue: Out of memory during visualization
- Reduce `--num_samples` for prediction visualization
- Process in batches
- Use lower resolution images

### Issue: No history in checkpoint
- Old checkpoints may not have history
- Retrain or manually create history dict

---

## Examples Gallery

### Training Curves
Shows loss progression and mAP improvement over time. Useful for:
- Detecting overfitting (train/val divergence)
- Monitoring convergence
- Evaluating learning rate schedules

### Loss Breakdown
Detailed view of individual loss components. Useful for:
- Identifying problematic components
- Balancing loss weights
- Understanding model behavior

### Training Summary
Complete overview of training session. Useful for:
- Presentations and reports
- Quick assessment of training quality
- Comparing experiments

### Confusion Matrix
Classification performance visualization. Useful for:
- Identifying commonly confused classes
- Class-specific performance analysis
- Dataset imbalance detection

### Predictions
Visual validation of model performance. Useful for:
- Qualitative assessment
- Debugging detection issues
- Demonstrating results

---

## Advanced Usage

### Creating Custom Reports

Combine multiple visualizations in a single report:

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('training_report.pdf') as pdf:
    # Page 1: Training curves
    plot_training_curves(history, save_path='temp.png')
    img = plt.imread('temp.png')
    plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    pdf.savefig()
    plt.close()
    
    # Page 2: Loss breakdown
    plot_loss_breakdown(history, save_path='temp.png')
    # ... repeat
```

### Real-time Monitoring

For long training sessions, update plots periodically:

```python
# In training loop
if epoch % 5 == 0:
    plot_training_curves(history, save_path='latest_curves.png')
```

---

## Dependencies

Required packages:
- `matplotlib >= 3.7.0` - Plotting library
- `numpy >= 1.24.0` - Numerical operations
- `torch >= 2.0.0` - For tensor handling
- `Pillow >= 9.5.0` - Image processing

Install all:
```bash
pip install matplotlib numpy torch pillow
```

---

## License & Attribution

These visualization tools are part of the Low-Light Object Detection project. Feel free to adapt and extend for your own projects.
