"""
Device optimization utilities
"""
import torch


def get_device():
    """
    Detect and return the best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "MPS (Apple Silicon)"
    else:
        try:
            import torch_directml
            device = torch_directml.device()
            device_type = "DirectML"
        except ImportError:
            device = torch.device("cpu")
            device_type = "CPU"
    
    return device, device_type


def optimize_for_device(device_type):
    """
    Return optimized hyperparameters based on device type
    """
    if 'CUDA' in device_type:
        return {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True,
            'gradient_accumulation_steps': 1,
            'use_gradient_checkpointing': False,
            'reduce_image_size': False,
            'default_image_size': 416,  # Full resolution for CUDA
            'default_base_channels': 32  # Full model capacity for CUDA
        }
    elif 'DirectML' in device_type:
        # Very conservative settings for integrated GPU (Intel Iris Xe)
        return {
            'batch_size': 1,  # Minimum to prevent OOM on integrated GPU
            'num_workers': 0,  # Reduced to save memory
            'pin_memory': False,
            'prefetch_factor': None,
            'persistent_workers': False,
            'gradient_accumulation_steps': 16,  # Increased to maintain effective batch size of 16
            'use_gradient_checkpointing': True,  # Enable gradient checkpointing
            'reduce_image_size': True,  # Use smaller images
            'default_image_size': 256,  # Default to 256 for DirectML
            'default_base_channels': 8  # Even smaller model
        }
    else:  # CPU
        return {
            'batch_size': 8,
            'num_workers': 0,
            'pin_memory': False,
            'prefetch_factor': None,
            'persistent_workers': False,
            'gradient_accumulation_steps': 4,
            'use_gradient_checkpointing': False,
            'reduce_image_size': False,
            'default_image_size': 320,  # Moderate size for CPU
            'default_base_channels': 24  # Moderate capacity for CPU
        }


def print_device_info(device, device_type, optimized_params):
    """
    Print device information and optimized parameters
    """
    print("\n" + "="*60)
    print("DEVICE CONFIGURATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Device Type: {device_type}")
    print(f"Batch Size: {optimized_params['batch_size']}")
    print(f"Gradient Accumulation Steps: {optimized_params['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {optimized_params['batch_size'] * optimized_params['gradient_accumulation_steps']}")
    print(f"Num Workers: {optimized_params['num_workers']}")
    print(f"Pin Memory: {optimized_params['pin_memory']}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("="*60 + "\n")
