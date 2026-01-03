"""
Default configuration for Low-Light Object Detection
"""

class Config:
    # Dataset paths
    dataset_root = "ExDark_Dataset/ExDark"
    
    # Model architecture
    backbone = "resnet18"  # Options: resnet18, resnet34, mobilenetv2
    num_classes = 12  # ExDark has 12 object classes
    
    # Training hyperparameters
    epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    # Image settings
    image_size = 416  # Standard YOLO input size
    
    # Loss weights (can be tuned)
    lambda_illumination = 1.0
    lambda_denoise = 0.5
    lambda_deblur = 0.5
    lambda_detection = 2.0
    lambda_spatial_consistency = 1.0
    lambda_exposure = 10.0
    lambda_color = 5.0
    lambda_tvA = 200.0
    
    # Detection parameters
    conf_threshold = 0.25
    iou_threshold = 0.45
    num_anchors = 3
    
    # Optimizer
    optimizer = "adam"  # Options: adam, adamw, sgd
    
    # Scheduler
    scheduler = "cosine"  # Options: cosine, step, plateau
    
    # Early stopping
    patience = 15
    
    # Checkpointing
    save_dir = "checkpoints"
    save_interval = 5
    
    # Class names for ExDark dataset
    class_names = [
        'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
        'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
    ]
