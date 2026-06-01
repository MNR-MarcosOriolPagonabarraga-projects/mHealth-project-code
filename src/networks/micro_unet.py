import torch
import torch.nn as nn

class MultiScaleBlock(nn.Module):
    """
    Parallel branches to capture different frequency signatures 
    at the same resolution.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Wide kernel: Low-freq (Delta/Theta)
        self.wide = nn.Conv1d(in_ch, out_ch // 2, kernel_size=31, padding=15)
        # Narrow kernel: High-freq (Alpha/Beta)
        self.narrow = nn.Conv1d(in_ch, out_ch // 2, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat([self.wide(x), self.narrow(x)], dim=1)
        return self.relu(self.bn(x))

class MicroUNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # Encoder
        self.enc1 = MultiScaleBlock(in_channels, 16)
        self.pool1 = nn.MaxPool1d(2) # 3000 -> 1500
        self.enc2 = MultiScaleBlock(16, 32)
        self.pool2 = nn.MaxPool1d(2) # 1500 -> 750
        
        # Bottleneck
        self.bottleneck = MultiScaleBlock(32, 64)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec1 = MultiScaleBlock(64 + 32, 32) # Skip connection
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec2 = MultiScaleBlock(32 + 16, 16)
        
        # Output: Probability mask (3000 samples)
        self.head = nn.Conv1d(16, 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Encoder + Skips
        s1 = self.enc1(x)
        x = self.pool1(s1)
        s2 = self.enc2(x)
        x = self.pool2(s2)
        
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec1(torch.cat([self.up1(x), s2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))

        x = self.head(x)
        
        return self.pool(x).squeeze(-1)