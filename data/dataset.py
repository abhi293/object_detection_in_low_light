"""
ExDark Dataset Loader
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import glob


class ExDarkDataset(Dataset):
    """
    ExDark Dataset for low-light object detection
    
    The ExDark dataset structure:
    ExDark/
        Bicycle/
            2015_00001.jpg
            2015_00002.jpg
            ...
        Boat/
        ...
        
    For this implementation, we'll use image-level labels based on the folder name.
    For full object detection, you would need to add annotation files (YOLO format, COCO format, etc.)
    """
    def __init__(self, root_dir, transform=None, image_size=416):
        """
        Args:
            root_dir: Path to ExDark root directory
            transform: Transforms to apply
            image_size: Target image size
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        
        # ExDark class names
        self.class_names = [
            'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
            'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load all images
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images from the dataset"""
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Get all images in this class directory
            image_paths = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                         glob.glob(os.path.join(class_dir, '*.png'))
            
            for img_path in image_paths:
                self.samples.append({
                    'image_path': img_path,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name]
                })
        
        print(f"Loaded {len(self.samples)} images from ExDark dataset")
        print(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Returns:
            image: Tensor (C, H, W)
            target: Dictionary with 'boxes', 'labels'
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = TF.to_tensor(image)
        
        # For this simple version, we create a dummy bounding box
        # that covers most of the image (as we don't have annotations)
        # In a real scenario, you would load actual annotations
        _, h, w = image.shape
        
        # Create a dummy box (you should replace this with actual annotations)
        boxes = torch.tensor([[
            w * 0.1,  # x1
            h * 0.1,  # y1
            w * 0.9,  # x2
            h * 0.9   # y2
        ]], dtype=torch.float32)
        
        labels = torch.tensor([sample['class_idx']], dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_path': sample['image_path']
        }
        
        return image, target


class ExDarkAnnotatedDataset(Dataset):
    """
    ExDark Dataset with actual annotations
    This version expects annotation files in YOLO format
    
    Expected structure:
    ExDark/
        images/
            Bicycle/
                2015_00001.jpg
            Boat/
                ...
        labels/
            Bicycle/
                2015_00001.txt  # YOLO format: class_id x_center y_center width height (normalized)
            Boat/
                ...
    """
    def __init__(self, root_dir, transform=None, image_size=416, split='train'):
        """
        Args:
            root_dir: Path to ExDark root directory
            transform: Transforms to apply
            image_size: Target image size
            split: 'train' or 'val'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.split = split
        
        # ExDark class names
        self.class_names = [
            'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
            'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load samples
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset with annotations"""
        # Check if we have annotations directory
        labels_dir = os.path.join(self.root_dir, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory not found at {labels_dir}")
            print("Using simple dataset without proper annotations...")
            # Fall back to simple version
            return
        
        for class_name in self.class_names:
            class_image_dir = os.path.join(self.root_dir, 'images', class_name)
            class_label_dir = os.path.join(labels_dir, class_name)
            
            if not os.path.exists(class_image_dir):
                class_image_dir = os.path.join(self.root_dir, class_name)  # Try old structure
            
            if not os.path.exists(class_image_dir):
                continue
            
            # Get all images
            image_paths = glob.glob(os.path.join(class_image_dir, '*.jpg')) + \
                         glob.glob(os.path.join(class_image_dir, '*.png'))
            
            for img_path in image_paths:
                # Check if annotation exists
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(class_label_dir, f"{img_name}.txt")
                
                self.samples.append({
                    'image_path': img_path,
                    'label_path': label_path if os.path.exists(label_path) else None,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name]
                })
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")
    
    def _parse_yolo_annotation(self, label_path, img_width, img_height):
        """
        Parse YOLO format annotation
        
        Args:
            label_path: Path to annotation file
            img_width: Image width
            img_height: Image height
        
        Returns:
            boxes: (N, 4) tensor [x1, y1, x2, y2]
            labels: (N,) tensor
        """
        if not os.path.exists(label_path):
            return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long)
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert to [x1, y1, x2, y2]
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
        
        if len(boxes) == 0:
            return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long)
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get item from dataset"""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = TF.to_tensor(image)
        _, h, w = image.shape
        
        # Load annotations
        if sample['label_path'] and os.path.exists(sample['label_path']):
            boxes, labels = self._parse_yolo_annotation(sample['label_path'], w, h)
        else:
            # Fallback: use image-level label with dummy box
            boxes = torch.tensor([[w * 0.1, h * 0.1, w * 0.9, h * 0.9]], dtype=torch.float32)
            labels = torch.tensor([sample['class_idx']], dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_path': sample['image_path']
        }
        
        return image, target


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles variable number of objects per image
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


if __name__ == "__main__":
    from transforms import get_train_transforms
    
    # Test dataset
    root_dir = "ExDark_Dataset/ExDark"
    transform = get_train_transforms(416)
    
    dataset = ExDarkDataset(root_dir, transform=transform)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")
