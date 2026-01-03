# Visualization Quick Reference

## 🎨 One-Line Commands

### Generate All Visualizations from Checkpoint
```bash
python visualize_results.py --checkpoint checkpoints/best_model.pth --mode both
```

### Generate Sample Plots (No Training Required)
```bash
python generate_sample_plots.py
```

### Windows Batch Script
```bash
visualize.bat checkpoints/best_model.pth both
```

---

## 📊 Available Visualizations

| Visualization | Purpose | Auto-Generated |
|---------------|---------|----------------|
| **Training Curves** | Loss & mAP over time | ✅ |
| **Loss Breakdown** | Component loss analysis | ✅ |
| **Training Summary** | Comprehensive overview | ✅ |
| **Metrics JSON** | Machine-readable data | ✅ |
| **Confusion Matrix** | Classification performance | Manual |
| **Predictions** | Visual detection results | Manual |
| **Enhancement Comparison** | Enhancement pipeline stages | Manual |

---

## 🚀 Quick Python API

```python
from utils.visualization import (
    plot_training_curves,
    plot_loss_breakdown,
    create_training_summary,
    visualize_predictions,
    plot_confusion_matrix
)

# Basic usage
history = {...}  # From checkpoint or training
plot_training_curves(history, save_path='curves.png')
plot_loss_breakdown(history, save_path='breakdown.png')
create_training_summary(history, save_path='summary.png')
```

---

## 📁 Output Structure

After training completes:
```
checkpoints/
├── best_model.pth
└── visualizations/
    ├── training_curves.png      # Main loss curves
    ├── loss_breakdown.png       # Detailed components
    ├── training_summary.png     # Complete overview
    └── training_metrics.json    # Raw data
```

After running visualize_results.py:
```
visualizations/
├── training_curves.png
├── loss_breakdown.png
├── training_summary.png
├── training_metrics.json
└── predictions/
    ├── prediction_1.png
    ├── prediction_2.png
    └── ...
```

---

## ⚙️ Command-Line Options

### visualize_results.py

```bash
python visualize_results.py \
  --checkpoint PATH              # Required: checkpoint file
  --mode {metrics,predictions,both}  # Default: metrics
  --data_root PATH              # For predictions mode
  --output_dir DIR              # Default: visualizations
  --num_samples N               # Default: 5
  --device {cpu,cuda,directml}  # Default: cpu
  --image_size SIZE             # Default: 416
  --conf_threshold CONF         # Default: 0.3
```

### Examples

**Metrics only (fastest):**
```bash
python visualize_results.py --checkpoint checkpoints/best_model.pth
```

**Predictions only:**
```bash
python visualize_results.py \
  --checkpoint checkpoints/best_model.pth \
  --mode predictions \
  --data_root ExDark_Dataset/ExDark \
  --num_samples 10
```

**Everything with custom output:**
```bash
python visualize_results.py \
  --checkpoint checkpoints/best_model.pth \
  --mode both \
  --output_dir my_results
```

---

## 🔍 Interpreting Plots

### Training Curves
- **Train/Val Gap** → Overfitting if val loss increases
- **Flat Loss** → May need higher learning rate or longer training
- **Increasing mAP** → Good sign, model is learning
- **Oscillating Loss** → Try reducing learning rate

### Loss Breakdown
- **High Illumination Loss** → Enhancement not converging
- **High Detection Loss** → May need more training or better features
- **Balanced Components** → Good training progress

### Training Summary
- **Check Per-Class AP** → Identify weak classes
- **Learning Rate Decay** → Should show smooth decrease
- **Final Statistics** → Compare across experiments

---

## 🛠️ Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| No matplotlib | `pip install matplotlib` |
| No history in checkpoint | Retrain with updated code |
| Out of memory | Reduce `--num_samples` |
| Slow prediction viz | Use `--device cpu` |
| Empty plots | Check history has data |

---

## 💡 Pro Tips

1. **Compare Experiments:** Use different output directories
   ```bash
   python visualize_results.py --checkpoint exp1/best.pth --output_dir viz_exp1
   python visualize_results.py --checkpoint exp2/best.pth --output_dir viz_exp2
   ```

2. **High-Res for Papers:** Edit `dpi=150` → `dpi=300` in visualization.py

3. **Quick Check:** Run `generate_sample_plots.py` to test installation

4. **Batch Processing:** Loop through multiple checkpoints
   ```bash
   for file in checkpoints/*.pth; do
       python visualize_results.py --checkpoint "$file" --output_dir "viz_$(basename $file)"
   done
   ```

5. **Real-time Monitoring:** Save plots every N epochs during training

---

## 📚 Full Documentation

See [VISUALIZATION.md](VISUALIZATION.md) for complete documentation including:
- Detailed function APIs
- Customization options
- Advanced usage examples
- Troubleshooting guide

---

## ✅ Checklist

Before running visualizations:
- [ ] Training completed successfully
- [ ] Checkpoint file exists
- [ ] matplotlib installed (`pip install matplotlib`)
- [ ] Dataset available (for predictions mode)
- [ ] Sufficient disk space for output images

Quick test:
```bash
python -c "import matplotlib; print('✅ matplotlib OK')"
python -c "import torch; print('✅ torch OK')"
```

---

## 🎯 Recommended Workflow

1. **Train model:**
   ```bash
   python train.py --data_root ExDark_Dataset/ExDark --epochs 20
   ```

2. **Auto-generated plots appear in:**
   ```
   checkpoints/visualizations/
   ```

3. **Generate predictions (optional):**
   ```bash
   python visualize_results.py \
     --checkpoint checkpoints/best_model.pth \
     --mode predictions \
     --num_samples 10
   ```

4. **Review and analyze results**

---

**Last Updated:** January 2026
**Version:** 1.0
