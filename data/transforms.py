"""
Data augmentation and transforms for low-light images
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np


class RandomHorizontalFlip:
    """Random horizontal flip for both image and bounding boxes"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            image = TF.hflip(image)
            if len(boxes) > 0:
                width = image.shape[-1]
                boxes = boxes.clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, boxes, labels


class RandomCrop:
    """Random crop for both image and bounding boxes"""
    def __init__(self, min_scale=0.8):
        self.min_scale = min_scale
    
    def __call__(self, image, boxes, labels):
        if random.random() < 0.5:
            _, h, w = image.shape
            scale = random.uniform(self.min_scale, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            
            image = TF.crop(image, top, left, new_h, new_w)
            
            if len(boxes) > 0:
                # Adjust boxes
                boxes = boxes.clone()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] - left
                boxes[:, [1, 3]] = boxes[:, [1, 3]] - top
                
                # Filter boxes that are outside the crop
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, new_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, new_h)
                
                # Keep boxes with valid area
                keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[keep]
                labels = labels[keep]
            
            # Resize back to original size
            image = TF.resize(image, [h, w])
            if len(boxes) > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (w / new_w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (h / new_h)
        
        return image, boxes, labels


class ColorJitter:
    """Color jittering (be careful with low-light images)"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(self, image, boxes, labels):
        if random.random() < 0.5:
            image = self.transform(image)
        return image, boxes, labels


class Normalize:
    """Normalize image to [0, 1] range"""
    def __call__(self, image, boxes, labels):
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        return image, boxes, labels


class Resize:
    """Resize image and adjust boxes"""
    def __init__(self, size):
        self.size = size if isinstance(size, (list, tuple)) else (size, size)
    
    def __call__(self, image, boxes, labels):
        _, old_h, old_w = image.shape
        new_h, new_w = self.size
        
        # Resize image
        image = TF.resize(image, self.size)
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / old_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / old_h)
        
        return image, boxes, labels


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


def get_train_transforms(image_size=416):
    """Get training transforms"""
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(p=0.5),
        RandomCrop(min_scale=0.8),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        Normalize()
    ])


def get_val_transforms(image_size=416):
    """Get validation transforms"""
    return Compose([
        Resize(image_size),
        Normalize()
    ])


if __name__ == "__main__":
    # Test transforms
    import torch
    
    image = torch.rand(3, 640, 480)
    boxes = torch.tensor([[50, 50, 150, 150], [200, 200, 300, 350]], dtype=torch.float32)
    labels = torch.tensor([0, 5], dtype=torch.long)
    
    transform = get_train_transforms(416)
    image_t, boxes_t, labels_t = transform(image, boxes, labels)
    
    print("Original image shape:", image.shape)
    print("Transformed image shape:", image_t.shape)
    print("Original boxes:", boxes)
    print("Transformed boxes:", boxes_t)
    print("Labels:", labels_t)
