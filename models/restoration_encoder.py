"""
Shared Restoration Encoder
Lightweight U-Net style encoder for feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for feature refinement"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual
        return F.relu(out)


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        return self.conv2(self.conv1(x))


class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class RestorationEncoder(nn.Module):
    """
    Lightweight U-Net encoder for low-light image restoration
    Extracts multi-scale features for enhancement and detection
    """
    def __init__(self, in_channels=3, base_channels=32):
        super(RestorationEncoder, self).__init__()
        
        # Initial convolution
        self.init_conv = ConvBlock(in_channels, base_channels)
        
        # Encoder path
        self.down1 = DownBlock(base_channels, base_channels * 2)      # 64
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)  # 256
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
        
        # Decoder path
        self.up1 = UpBlock(base_channels * 8, base_channels * 4)  # 128
        self.up2 = UpBlock(base_channels * 4, base_channels * 2)  # 64
        self.up3 = UpBlock(base_channels * 2, base_channels)      # 32
        
        # Final refinement
        self.final_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        )
    
    def forward(self, x):
        """
        Forward pass returns both restored features and intermediate features
        """
        # Encoder
        x1 = self.init_conv(x)      # Skip 1
        x2 = self.down1(x1)          # Skip 2
        x3 = self.down2(x2)          # Skip 3
        x4 = self.down3(x3)          # Bottleneck input
        
        # Bottleneck
        x_bottleneck = self.bottleneck(x4)
        
        # Decoder
        x_up1 = self.up1(x_bottleneck, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        
        # Final features
        features = self.final_conv(x_up3)
        
        # Return features at multiple scales for detection
        return {
            'final_features': features,
            'bottleneck': x_bottleneck,
            'decoder_features': [x_up1, x_up2, x_up3],
            'encoder_features': [x1, x2, x3, x4]
        }


if __name__ == "__main__":
    # Test the encoder
    model = RestorationEncoder(in_channels=3, base_channels=32)
    x = torch.randn(2, 3, 416, 416)
    output = model(x)
    print("Final features shape:", output['final_features'].shape)
    print("Bottleneck shape:", output['bottleneck'].shape)
    print("Decoder features shapes:", [f.shape for f in output['decoder_features']])
