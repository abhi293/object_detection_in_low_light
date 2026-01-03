"""
Restoration losses for denoising and deblurring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RestorationLoss(nn.Module):
    """
    Combined loss for image restoration (denoising + deblurring)
    """
    def __init__(self, lambda_denoise=0.5, lambda_deblur=0.5):
        super(RestorationLoss, self).__init__()
        self.lambda_denoise = lambda_denoise
        self.lambda_deblur = lambda_deblur
    
    def perceptual_loss(self, restored, target):
        """
        Simple perceptual loss using L1 distance
        For a full implementation, you could use VGG features
        """
        return F.l1_loss(restored, target)
    
    def edge_loss(self, restored, target):
        """
        Edge-aware loss to preserve structural details
        """
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=restored.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=restored.device).view(1, 1, 3, 3)
        
        # Apply to each channel
        def compute_edges(img):
            channels = []
            for c in range(img.shape[1]):
                img_c = img[:, c:c+1, :, :]
                edge_x = F.conv2d(img_c, sobel_x, padding=1)
                edge_y = F.conv2d(img_c, sobel_y, padding=1)
                edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
                channels.append(edge)
            return torch.cat(channels, dim=1)
        
        restored_edges = compute_edges(restored)
        target_edges = compute_edges(target)
        
        return F.mse_loss(restored_edges, target_edges)
    
    def gradient_loss(self, restored, target):
        """
        Gradient difference loss for sharpness
        """
        def gradient(img):
            grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
            grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
            return grad_x, grad_y
        
        restored_gx, restored_gy = gradient(restored)
        target_gx, target_gy = gradient(target)
        
        loss_x = F.l1_loss(restored_gx, target_gx)
        loss_y = F.l1_loss(restored_gy, target_gy)
        
        return loss_x + loss_y
    
    def forward(self, denoised, deblurred, target):
        """
        Compute restoration losses
        
        Args:
            denoised: Denoised image
            deblurred: Deblurred (final restored) image
            target: Ground truth clean image (if available)
        
        Note: For ExDark, we don't have clean ground truth images.
        In practice, we can use the enhanced image as a pseudo-target,
        or rely only on the enhancement losses.
        """
        # Since ExDark doesn't have clean images, we'll use self-supervised losses
        # that don't require ground truth
        
        # Denoising loss: encourage smooth regions
        denoise_loss = self.gradient_loss(denoised, deblurred)
        
        # Deblurring loss: encourage sharp edges
        deblur_loss = self.edge_loss(deblurred, target)
        
        total_loss = (
            self.lambda_denoise * denoise_loss +
            self.lambda_deblur * deblur_loss
        )
        
        return {
            'total': total_loss,
            'denoise': denoise_loss,
            'deblur': deblur_loss
        }


class SelfSupervisedRestorationLoss(nn.Module):
    """
    Self-supervised restoration loss that doesn't require clean ground truth
    Useful for ExDark dataset where clean images are not available
    """
    def __init__(self):
        super(SelfSupervisedRestorationLoss, self).__init__()
    
    def noise_variance_loss(self, denoised, enhanced):
        """
        Encourage denoised image to have lower variance in smooth regions
        """
        # Compute local variance
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                          device=denoised.device) / (kernel_size ** 2)
        
        def local_variance(img):
            # Mean of each channel
            mean = F.conv2d(img, kernel.repeat(img.shape[1], 1, 1, 1), 
                          padding=kernel_size//2, groups=img.shape[1])
            # Mean of squares
            mean_sq = F.conv2d(img ** 2, kernel.repeat(img.shape[1], 1, 1, 1),
                             padding=kernel_size//2, groups=img.shape[1])
            # Variance
            variance = mean_sq - mean ** 2
            return variance
        
        denoised_var = local_variance(denoised)
        enhanced_var = local_variance(enhanced)
        
        # Denoised should have lower variance
        loss = torch.mean(F.relu(denoised_var - enhanced_var + 0.01))
        
        return loss
    
    def sharpness_loss(self, deblurred):
        """
        Encourage sharp edges using Laplacian
        """
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                dtype=torch.float32, device=deblurred.device).view(1, 1, 3, 3)
        
        # Apply Laplacian to each channel
        sharpness = 0
        for c in range(deblurred.shape[1]):
            img_c = deblurred[:, c:c+1, :, :]
            lap = F.conv2d(img_c, laplacian, padding=1)
            sharpness += torch.mean(torch.abs(lap))
        
        # Negative because we want to maximize sharpness
        return -sharpness / deblurred.shape[1]
    
    def forward(self, enhanced, denoised, deblurred):
        """
        Compute self-supervised restoration losses
        
        Args:
            enhanced: Illumination-enhanced image
            denoised: Denoised image
            deblurred: Final deblurred image
        """
        noise_loss = self.noise_variance_loss(denoised, enhanced)
        sharp_loss = self.sharpness_loss(deblurred)
        
        return {
            'noise_variance': noise_loss,
            'sharpness': sharp_loss
        }


if __name__ == "__main__":
    # Test restoration loss
    loss_fn = SelfSupervisedRestorationLoss()
    
    enhanced = torch.rand(2, 3, 256, 256)
    denoised = torch.rand(2, 3, 256, 256)
    deblurred = torch.rand(2, 3, 256, 256)
    
    losses = loss_fn(enhanced, denoised, deblurred)
    print("Noise variance loss:", losses['noise_variance'].item())
    print("Sharpness loss:", losses['sharpness'].item())
