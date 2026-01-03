"""
Training script for Low-Light Object Detection
Run with: python train.py --help to see all options
"""
import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
from datetime import datetime

from models import LowLightObjectDetector
from data import ExDarkDataset, get_train_transforms, get_val_transforms, collate_fn
from losses import ZeroDCELoss, SelfSupervisedRestorationLoss, MultiScaleDetectionLoss
from utils import get_device, optimize_for_device, print_device_info, DetectionMetrics
from utils.visualization import (
    plot_training_curves,
    plot_loss_breakdown,
    create_training_summary,
    save_metrics_json
)
from config.default_config import Config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Low-Light Object Detection Model')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='ExDark_Dataset/ExDark',
                       help='Path to ExDark dataset root directory')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio (default: 0.8)')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=12,
                       help='Number of object classes (default: 12 for ExDark)')
    parser.add_argument('--base_channels', type=int, default=None,
                       help='Base number of channels in encoder (default: 32, auto-reduced for DirectML)')
    parser.add_argument('--image_size', type=int, default=416,
                       help='Input image size (default: 416)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type (default: adam)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    
    # Loss weights
    parser.add_argument('--lambda_illumination', type=float, default=1.0,
                       help='Weight for illumination loss (default: 1.0)')
    parser.add_argument('--lambda_denoise', type=float, default=0.5,
                       help='Weight for denoising loss (default: 0.5)')
    parser.add_argument('--lambda_deblur', type=float, default=0.5,
                       help='Weight for deblurring loss (default: 0.5)')
    parser.add_argument('--lambda_detection', type=float, default=2.0,
                       help='Weight for detection loss (default: 2.0)')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loading workers (auto-detected if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Evaluate every N epochs (default: 1)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def create_optimizer(model, args):
    """Create optimizer based on arguments"""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler"""
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * steps_per_epoch
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    else:
        scheduler = None
    
    return scheduler


def train_one_epoch(model, dataloader, criterion_dict, optimizer, scheduler, device, 
                   args, epoch, grad_accum_steps):
    """Train for one epoch"""
    model.train()
    
    # Loss trackers
    total_loss = 0
    illumination_loss = 0
    restoration_loss = 0
    detection_loss = 0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        try:
            # Move to device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images, return_all=True)
            
            # Compute losses
            # 1. Illumination enhancement loss (Zero-DCE)
            loss_illum = criterion_dict['illumination'](
                outputs['enhanced_image'],
                images,
                outputs['curve_params']
            )
            
            # 2. Restoration loss (self-supervised)
            loss_resto = criterion_dict['restoration'](
                outputs['enhanced_image'],
                outputs['denoised_image'],
                outputs['restored_image']
            )
            
            # 3. Detection loss
            loss_detect = criterion_dict['detection'](
                outputs['predictions'],
                targets
            )
            
            # Combined loss
            loss = (
                args.lambda_illumination * loss_illum['total'] +
                (args.lambda_denoise + args.lambda_deblur) * 
                (loss_resto['noise_variance'] + loss_resto['sharpness']) +
                args.lambda_detection * loss_detect['total']
            )
            
            # Backward pass with gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
        except RuntimeError as e:
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                print(f"\n❌ Out of Memory Error at batch {batch_idx}")
                print("💡 Suggestions:")
                print("   1. Use smaller image size: --image_size 256 or --image_size 224")
                print("   2. Reduce base channels: --base_channels 16 or --base_channels 8")
                print("   3. The system will auto-configure for DirectML next run")
                print(f"\n   Try: python train.py --image_size 256 --base_channels 16")
                raise
            else:
                raise
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None and args.scheduler == 'cosine':
                scheduler.step()
            
            # Clear cache for DirectML to prevent memory accumulation
            if 'DirectML' in str(device):
                try:
                    import torch_directml
                    torch_directml.device(device).empty_cache()
                except:
                    pass
        
        # Update metrics
        total_loss += loss.item() * grad_accum_steps
        illumination_loss += loss_illum['total'].item()
        restoration_loss += (loss_resto['noise_variance'] + loss_resto['sharpness']).item()
        detection_loss += loss_detect['total'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'illum': illumination_loss / num_batches,
            'resto': restoration_loss / num_batches,
            'detect': detection_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return {
        'total_loss': total_loss / num_batches,
        'illumination_loss': illumination_loss / num_batches,
        'restoration_loss': restoration_loss / num_batches,
        'detection_loss': detection_loss / num_batches
    }


def validate(model, dataloader, criterion_dict, device, args):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    num_batches = 0
    
    # Detection metrics
    metrics = DetectionMetrics(num_classes=args.num_classes)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        
        for images, targets in pbar:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images, return_all=True)
            
            # Compute losses
            loss_illum = criterion_dict['illumination'](
                outputs['enhanced_image'],
                images,
                outputs['curve_params']
            )
            
            loss_resto = criterion_dict['restoration'](
                outputs['enhanced_image'],
                outputs['denoised_image'],
                outputs['restored_image']
            )
            
            loss_detect = criterion_dict['detection'](
                outputs['predictions'],
                targets
            )
            
            loss = (
                args.lambda_illumination * loss_illum['total'] +
                (args.lambda_denoise + args.lambda_deblur) * 
                (loss_resto['noise_variance'] + loss_resto['sharpness']) +
                args.lambda_detection * loss_detect['total']
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions for metrics
            results = model.predict(images, conf_threshold=0.25)
            
            for i, (result, target) in enumerate(zip(results, targets)):
                metrics.update(
                    result['boxes'],
                    result['labels'],
                    result['scores'],
                    target['boxes'].to(device),
                    target['labels'].to(device)
                )
    
    # Compute mAP
    detection_metrics = metrics.compute()
    
    return {
        'total_loss': total_loss / num_batches,
        'mAP': detection_metrics['mAP']
    }


def save_checkpoint(model, optimizer, scheduler, epoch, args, best_map, save_path, history=None):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_map': best_map,
        'args': vars(args)
    }
    
    if history is not None:
        checkpoint['history'] = history
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device and optimize settings
    device, device_type = get_device()
    device_params = optimize_for_device(device_type)
    
    # Override batch_size and num_workers if not specified
    if args.batch_size is None:
        args.batch_size = device_params['batch_size']
    if args.num_workers is None:
        args.num_workers = device_params['num_workers']
    
    grad_accum_steps = device_params['gradient_accumulation_steps']
    use_gradient_checkpointing = device_params.get('use_gradient_checkpointing', False)
    
    # Apply device-specific image size if not user-specified
    if args.image_size == 416:  # Default value, not user-specified
        if device_params.get('reduce_image_size', False):
            args.image_size = device_params.get('default_image_size', 256)
            print(f"\n⚠️  Auto-adjusted image size to {args.image_size} for {device_type}")
        elif 'CUDA' in device_type:
            print(f"\n✅ Using full resolution ({args.image_size}px) for {device_type}")
    
    # Auto-set base_channels based on device if not specified
    if args.base_channels is None:
        args.base_channels = device_params.get('default_base_channels', 32)
        if 'CUDA' in device_type:
            print(f"✅ Using full model capacity (base_channels={args.base_channels}) for {device_type}")
        elif 'DirectML' in device_type:
            print(f"⚠️  Using reduced model (base_channels={args.base_channels}) for {device_type}")
        else:
            print(f"ℹ️  Using moderate model (base_channels={args.base_channels}) for {device_type}")
    
    # Print device info
    print_device_info(device, device_type, device_params)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(args.save_dir, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {args.data_root}")
    print(f"Image Size: {args.image_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation: {grad_accum_steps}")
    print(f"Effective Batch Size: {args.batch_size * grad_accum_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print("="*60 + "\n")
    
    # Create datasets
    print("Loading dataset...")
    full_dataset = ExDarkDataset(
        root_dir=args.data_root,
        transform=get_train_transforms(args.image_size),
        image_size=args.image_size
    )
    
    # Split into train and val
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update val dataset transform
    val_dataset.dataset.transform = get_val_transforms(args.image_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device_params['pin_memory'],
        prefetch_factor=device_params['prefetch_factor'],
        persistent_workers=device_params['persistent_workers'] if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device_params['pin_memory'],
        prefetch_factor=device_params['prefetch_factor'],
        persistent_workers=device_params['persistent_workers'] if args.num_workers > 0 else False
    )
    
    # Create model
    print("\nCreating model...")
    model = LowLightObjectDetector(
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        use_gradient_checkpointing=use_gradient_checkpointing
    ).to(device)
    
    model_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {model_params:.2f}M")
    
    if 'CUDA' in device_type:
        print(f"🚀 Full-capacity model ready for GPU acceleration!")
    elif 'DirectML' in device_type:
        print(f"💡 Lightweight model optimized for integrated GPU")
    
    if use_gradient_checkpointing:
        print("✅ Gradient checkpointing enabled for memory efficiency")
    
    # Create loss functions
    criterion_dict = {
        'illumination': ZeroDCELoss(
            lambda_spatial=Config.lambda_spatial_consistency,
            lambda_exposure=Config.lambda_exposure,
            lambda_color=Config.lambda_color,
            lambda_tvA=Config.lambda_tvA
        ).to(device),
        'restoration': SelfSupervisedRestorationLoss().to(device),
        'detection': MultiScaleDetectionLoss(
            num_classes=args.num_classes,
            anchors_dict={
                'small': model.detector.anchors_small,
                'medium': model.detector.anchors_medium,
                'large': model.detector.anchors_large
            }
        ).to(device)
    }
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, len(train_loader))
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_map = 0.0
    
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint['best_map']
        print(f"Resumed from epoch {checkpoint['epoch']}, best mAP: {best_map:.4f}")
    
    # Training loop
    print("\nStarting training...\n")
    
    # Initialize training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_illum_loss': [],
        'train_resto_loss': [],
        'train_detect_loss': [],
        'val_loss': [],
        'val_mAP': [],
        'learning_rate': [],
        'class_names': Config.CLASS_NAMES
    }
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion_dict, optimizer, scheduler,
            device, args, epoch, grad_accum_steps
        )
        
        print(f"\nEpoch {epoch}/{args.epochs} - Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Validate
        if epoch % args.eval_interval == 0:
            val_metrics = validate(model, val_loader, criterion_dict, device, args)
            print(f"Validation Loss: {val_metrics['total_loss']:.4f}, mAP: {val_metrics['mAP']:.4f}")
            
            # Update history
            history['epochs'].append(epoch)
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_illum_loss'].append(train_metrics['illumination_loss'])
            history['train_resto_loss'].append(train_metrics['restoration_loss'])
            history['train_detect_loss'].append(train_metrics['detection_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['val_mAP'].append(val_metrics['mAP'])
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Update scheduler if using plateau
            if scheduler and args.scheduler == 'plateau':
                scheduler.step(val_metrics['total_loss'])
            
            # Save best model
            if val_metrics['mAP'] > best_map:
                best_map = val_metrics['mAP']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, args, best_map,
                    os.path.join(args.save_dir, 'best_model.pth'),
                    history=history
                )
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, args, best_map,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'),
                history=history
            )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best mAP: {best_map:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    
    # Generate visualizations
    if len(history['epochs']) > 0:
        print("\n📊 Generating training visualizations...")
        viz_dir = os.path.join(args.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            plot_training_curves(history, save_path=os.path.join(viz_dir, 'training_curves.png'))
            plot_loss_breakdown(history, save_path=os.path.join(viz_dir, 'loss_breakdown.png'))
            create_training_summary(history, save_path=os.path.join(viz_dir, 'training_summary.png'))
            save_metrics_json(history, save_path=os.path.join(viz_dir, 'training_metrics.json'))
            print(f"✅ Visualizations saved to: {viz_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not generate visualizations: {e}")
    
    print("="*60)


if __name__ == "__main__":
    main()
