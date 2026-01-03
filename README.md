# Low-Light Object Detection

A conference-grade deep learning pipeline for object detection in low-light, noisy images using the ExDark dataset.

## 🎯 Architecture Overview

This implementation follows a **task-aware restoration + detection** pipeline:

```
Input Low-Light Image
        ↓
Shared Restoration Encoder (Lightweight U-Net)
        ↓
Multi-Objective Restoration
  • Illumination Correction (Zero-DCE inspired)
  • Noise Suppression (RIDNet-style)
  • Deblur Refinement (MPRNet-style)
        ↓
Restored Feature Maps
        ↓
YOLO-based Detection Head
        ↓
Object Classification & Localization
```

### Key Features

✅ **No GAN/ESRGAN** - Computationally efficient, stable training  
✅ **Multi-objective restoration** - Illumination + Denoise + Deblur in one pipeline  
✅ **Task-aware learning** - Detection loss influences restoration  
✅ **Multi-scale detection** - YOLO-style detection at 3 scales  
✅ **Device optimization** - Auto-adapts to CUDA, DirectML, or CPU  
✅ **Modular design** - Clean, organized codebase  

## 📁 Project Structure

```
Object_Detection_in_low_lights/
├── train.py                    # Main training script
├── config/
│   └── default_config.py      # Default configuration
├── models/
│   ├── restoration_encoder.py # Shared U-Net encoder
│   ├── enhancement_modules.py # Zero-DCE, denoise, deblur
│   ├── detection_head.py      # YOLO-style detector
│   └── unified_model.py       # Complete pipeline
├── losses/
│   ├── zero_dce_loss.py       # Zero-DCE losses
│   ├── restoration_loss.py    # Restoration losses
│   └── detection_loss.py      # YOLO detection losses
├── data/
│   ├── dataset.py             # ExDark dataset loader
│   └── transforms.py          # Data augmentation
└── utils/
    ├── device_optimizer.py    # Device optimization
    └── metrics.py             # Evaluation metrics
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
pip install torch torchvision tqdm pillow numpy
```

**Optional (for DirectML support on Intel GPUs):**
```bash
pip install torch-directml
```

### 2. Prepare Dataset

Ensure your ExDark dataset is structured as:
```
ExDark_Dataset/
  ExDark/
    Bicycle/
      2015_00001.jpg
      ...
    Boat/
    Bottle/
    ...
```

### 3. Basic Training

**Quick start with auto-detected settings:**
```bash
python train.py --data_root ExDark_Dataset/ExDark
```

The system will automatically:
- Detect your device (CUDA/DirectML/CPU)
- Optimize batch size and workers
- Use appropriate gradient accumulation

## 🎮 Training Options

### Dataset Configuration
```bash
python train.py \
  --data_root ExDark_Dataset/ExDark \
  --train_split 0.8 \
  --image_size 416
```

### Model Architecture
```bash
python train.py \
  --num_classes 12 \
  --base_channels 32
```

### Training Hyperparameters
```bash
python train.py \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --optimizer adam \
  --scheduler cosine
```

**Optimizers:**
- `adam` - Default, stable convergence
- `adamw` - Adam with decoupled weight decay
- `sgd` - SGD with momentum (0.9)

**Schedulers:**
- `cosine` - Cosine annealing (recommended)
- `step` - Step decay every 30 epochs
- `plateau` - Reduce on validation plateau
- `none` - No scheduler

### Loss Weights (Tuning)
```bash
python train.py \
  --lambda_illumination 1.0 \
  --lambda_denoise 0.5 \
  --lambda_deblur 0.5 \
  --lambda_detection 2.0
```

### Checkpointing
```bash
python train.py \
  --save_dir checkpoints \
  --save_interval 5 \
  --resume checkpoints/best_model.pth
```

### Complete Example
```bash
python train.py \
  --data_root ExDark_Dataset/ExDark \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --optimizer adam \
  --scheduler cosine \
  --lambda_illumination 1.0 \
  --lambda_denoise 0.5 \
  --lambda_deblur 0.5 \
  --lambda_detection 2.0 \
  --save_dir checkpoints \
  --save_interval 5
```

## 💡 Device-Specific Recommendations

### NVIDIA GPU (RTX 4060)
```bash
python train.py --batch_size 32 --num_workers 4
```
**Expected:** Fast training, ~20-30 seconds/epoch

### Intel Integrated GPU (DirectML)
```bash
python train.py --batch_size 4 --num_workers 0
```
**Expected:** Slower training, uses gradient accumulation (8x) to maintain effective batch size of 32

### CPU Only
```bash
python train.py --batch_size 8 --num_workers 0
```
**Expected:** Slow training, gradient accumulation (4x) for effective batch size of 32

## 📊 Monitoring Training

The training script provides real-time monitoring:

```
Epoch 1/100: 100%|████████| 150/150 [01:23<00:00, 1.80it/s, 
  loss=2.456, illum=0.234, resto=0.122, detect=2.100, lr=0.0001]

Epoch 1/100 - Train Loss: 2.4560
Validation Loss: 2.3120, mAP: 0.3456
```

**Metrics:**
- `loss` - Total combined loss
- `illum` - Illumination enhancement loss
- `resto` - Restoration loss (denoise + deblur)
- `detect` - Detection loss (localization + confidence + classification)
- `lr` - Current learning rate
- `mAP` - Mean Average Precision @ IoU=0.5

## 📈 Checkpoints

Checkpoints are saved in the specified directory:

```
checkpoints/
├── best_model.pth              # Best model (highest mAP)
├── checkpoint_epoch_5.pth      # Periodic checkpoints
├── checkpoint_epoch_10.pth
└── train_config.json           # Training configuration
```

### Resume Training
```bash
python train.py --resume checkpoints/best_model.pth
```

## 🔬 Architecture Details

### 1. Restoration Encoder
- Lightweight U-Net architecture
- Multi-scale feature extraction
- Efficient residual blocks
- ~2M parameters for base_channels=32

### 2. Enhancement Modules

**Illumination Correction (Zero-DCE):**
- Iterative curve adjustment (8 iterations)
- Pixel-wise enhancement
- Preserves spatial consistency

**Noise Suppression (RIDNet-style):**
- Enhanced residual blocks
- Channel attention
- Self-supervised learning

**Deblur Refinement (MPRNet-style):**
- Multi-scale processing
- Feature fusion
- Residual sharpening

### 3. Detection Head
- YOLO-style multi-scale detection
- 3 detection scales (small/medium/large objects)
- Predefined anchors optimized for ExDark
- Non-Maximum Suppression (NMS) post-processing

## 🎓 Loss Functions

### Illumination Loss (Zero-DCE)
- **Spatial Consistency:** Preserve image structure
- **Exposure Control:** Target average intensity
- **Color Constancy:** Maintain color balance
- **TV Regularization:** Smooth enhancement curves

### Restoration Loss (Self-Supervised)
- **Noise Variance:** Reduce local variance
- **Sharpness:** Encourage edge clarity

### Detection Loss (YOLO)
- **Localization:** Bounding box regression
- **Objectness:** Object confidence
- **Classification:** Class probabilities

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch_size 4

# Or reduce image size
python train.py --image_size 320
```

### Slow Training
```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# Increase workers if CPU/RAM allows
python train.py --num_workers 4
```

### Poor Detection Performance
```bash
# Increase detection loss weight
python train.py --lambda_detection 3.0

# Or adjust other loss weights
python train.py \
  --lambda_illumination 0.5 \
  --lambda_detection 2.5
```

## 📝 Notes

### Dataset Annotations
The current implementation uses **image-level labels** (folder-based) as ExDark doesn't include bounding box annotations by default.

For **full object detection** with proper bounding boxes:
1. Add annotation files in YOLO format (`.txt` files)
2. Use `ExDarkAnnotatedDataset` in `data/dataset.py`
3. Organize as:
   ```
   ExDark/
     images/
       Bicycle/
     labels/
       Bicycle/
         2015_00001.txt  # class_id x_center y_center width height
   ```

### Conference Submission
This architecture is designed for conference-grade research:
- ✅ Clean and efficient pipeline
- ✅ No GAN complexity
- ✅ Clear ablation components
- ✅ Task-aware design
- ✅ Easy to implement and reproduce

### Performance Expectations
- **mAP improvement:** Expected 10-15% over baseline in extreme low-light
- **Convergence:** ~50-80 epochs for good results
- **Training time:** 
  - RTX 4060: ~2-3 hours for 100 epochs
  - CPU: ~10-15 hours for 100 epochs

## 📚 References

This implementation is inspired by:
- **Zero-DCE:** Zero-Reference Deep Curve Estimation
- **RIDNet:** Real Image Denoising
- **MPRNet:** Multi-Stage Progressive Image Restoration
- **YOLOv3:** Object Detection Architecture

## 🤝 Contributing

For improvements or bug reports, please ensure:
- Code follows existing structure
- All modules are properly documented
- Training scripts remain functional

## 📄 License

This project is for research and educational purposes.

---

**Happy Training! 🚀**
