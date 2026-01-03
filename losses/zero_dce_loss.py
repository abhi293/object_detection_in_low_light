"""
Zero-DCE inspired losses for illumination enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroDCELoss(nn.Module):
    """
    Combined loss for Zero-DCE style enhancement
    Includes: spatial consistency, exposure control, color constancy, and illumination smoothness
    """
    def __init__(self, lambda_spatial=1.0, lambda_exposure=10.0, 
                 lambda_color=5.0, lambda_tvA=200.0):
        super(ZeroDCELoss, self).__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_exposure = lambda_exposure
        self.lambda_color = lambda_color
        self.lambda_tvA = lambda_tvA
    
    def spatial_consistency_loss(self, enhanced, original):
        """
        Preserve spatial consistency between original and enhanced images
        """
        # Compute gradients
        original_mean = torch.mean(original, dim=1, keepdim=True)
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)
        
        # Gradient in x direction
        original_dx = original_mean[:, :, :, :-1] - original_mean[:, :, :, 1:]
        enhanced_dx = enhanced_mean[:, :, :, :-1] - enhanced_mean[:, :, :, 1:]
        
        # Gradient in y direction
        original_dy = original_mean[:, :, :-1, :] - original_mean[:, :, 1:, :]
        enhanced_dy = enhanced_mean[:, :, :-1, :] - enhanced_mean[:, :, 1:, :]
        
        # Spatial consistency loss
        loss_dx = torch.mean(torch.abs(original_dx - enhanced_dx))
        loss_dy = torch.mean(torch.abs(original_dy - enhanced_dy))
        
        return loss_dx + loss_dy
    
    def exposure_control_loss(self, enhanced, target_value=0.6):
        """
        Control the exposure level to avoid over/under enhancement
        """
        # Average intensity per patch
        patch_size = 16
        batch, channels, height, width = enhanced.shape
        
        # Ensure dimensions are divisible by patch_size
        h_pad = (patch_size - height % patch_size) % patch_size
        w_pad = (patch_size - width % patch_size) % patch_size
        
        if h_pad > 0 or w_pad > 0:
            enhanced = F.pad(enhanced, (0, w_pad, 0, h_pad), mode='reflect')
        
        # Reshape to patches
        enhanced_mean = torch.mean(enhanced, dim=1, keepdim=True)
        patches = F.unfold(enhanced_mean, kernel_size=patch_size, stride=patch_size)
        
        # Compute average intensity per patch
        avg_intensity = torch.mean(patches, dim=1)
        
        # Loss: deviation from target exposure
        loss = torch.mean((avg_intensity - target_value) ** 2)
        
        return loss
    
    def color_constancy_loss(self, enhanced):
        """
        Preserve color consistency across channels
        """
        # Mean of each channel
        mean_r = torch.mean(enhanced[:, 0, :, :])
        mean_g = torch.mean(enhanced[:, 1, :, :])
        mean_b = torch.mean(enhanced[:, 2, :, :])
        
        # Differences between channels
        loss_rg = (mean_r - mean_g) ** 2
        loss_rb = (mean_r - mean_b) ** 2
        loss_gb = (mean_g - mean_b) ** 2
        
        return loss_rg + loss_rb + loss_gb
    
    def illumination_smoothness_loss(self, curve_params):
        """
        Total variation loss on curve parameters for smoothness
        
        Args:
            curve_params: (B, iterations, 3, H, W)
        """
        batch, iterations, channels, h, w = curve_params.shape
        
        # Flatten iterations and channels
        params = curve_params.view(batch, iterations * channels, h, w)
        
        # Total variation in x and y
        tv_h = torch.mean(torch.abs(params[:, :, 1:, :] - params[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(params[:, :, :, 1:] - params[:, :, :, :-1]))
        
        return tv_h + tv_w
    
    def forward(self, enhanced, original, curve_params):
        """
        Compute total Zero-DCE loss
        
        Args:
            enhanced: Enhanced image
            original: Original image
            curve_params: Curve parameters from enhancement
        """
        loss_spa = self.spatial_consistency_loss(enhanced, original)
        loss_exp = self.exposure_control_loss(enhanced)
        loss_col = self.color_constancy_loss(enhanced)
        loss_tv = self.illumination_smoothness_loss(curve_params)
        
        total_loss = (
            self.lambda_spatial * loss_spa +
            self.lambda_exposure * loss_exp +
            self.lambda_color * loss_col +
            self.lambda_tvA * loss_tv
        )
        
        return {
            'total': total_loss,
            'spatial': loss_spa,
            'exposure': loss_exp,
            'color': loss_col,
            'tv': loss_tv
        }


if __name__ == "__main__":
    # Test Zero-DCE loss
    loss_fn = ZeroDCELoss()
    
    original = torch.rand(2, 3, 256, 256)
    enhanced = torch.rand(2, 3, 256, 256)
    curve_params = torch.rand(2, 8, 3, 256, 256)
    
    losses = loss_fn(enhanced, original, curve_params)
    print("Total loss:", losses['total'].item())
    print("Spatial loss:", losses['spatial'].item())
    print("Exposure loss:", losses['exposure'].item())
    print("Color loss:", losses['color'].item())
    print("TV loss:", losses['tv'].item())
