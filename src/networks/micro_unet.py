import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileDilatedBlock(nn.Module):
    """
    Ultra-lightweight depthwise separable convolution block with dilation.
    """
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        # Depthwise 1D Conv with dilation
        padding = ((7 - 1) * dilation) // 2
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=7, padding=padding, 
            dilation=dilation, groups=in_ch, bias=False
        )
        # Pointwise 1D Conv to change channel dimension
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(0.05)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(self.bn(x))
        return self.dropout(x)


class MicroSleepArousalUNet(nn.Module):
    def __init__(self, bottleneck, in_channels=2):
        super().__init__()
        s = int(bottleneck / 8)
        m = int(s * 2)
        l = int(m * 2)
        xl = int(l + 1/3*l)

        
        # --- Encoder Layers ---
        self.enc1 = MobileDilatedBlock(in_channels, s, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        self.enc2 = MobileDilatedBlock(s, m, dilation=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        self.enc3 = MobileDilatedBlock(m, l, dilation=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        self.enc4 = MobileDilatedBlock(l, xl, dilation=8)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) 

        # --- Bottleneck Layer ---
        self.bottleneck = MobileDilatedBlock(xl, bottleneck, dilation=8)
        
        # --- Decoder Layers ---
        self.up1 = nn.ConvTranspose1d(bottleneck, xl, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = MobileDilatedBlock(xl + xl, xl, dilation=4)
        
        self.up2 = nn.ConvTranspose1d(xl, l, kernel_size=2, stride=2, output_padding=1)
        self.dec2 = MobileDilatedBlock(l + l, l, dilation=2)
        
        self.up3 = nn.ConvTranspose1d(l, m, kernel_size=2, stride=2, output_padding=0)
        self.dec3 = MobileDilatedBlock(m + m, m, dilation=1)
        
        self.up4 = nn.ConvTranspose1d(m, s, kernel_size=2, stride=2, output_padding=0)
        self.dec4 = MobileDilatedBlock(s + s, s, dilation=1)
        
        # --- Output Head ---
        self.head = nn.Conv1d(s, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        x = self.pool1(s1)
        
        s2 = self.enc2(x)
        x = self.pool2(s2)
        
        s3 = self.enc3(x)
        x = self.pool3(s3)
        
        s4 = self.enc4(x)
        x = self.pool4(s4)
        
        x = self.bottleneck(x)
        
        x = self.dec1(torch.cat([self.up1(x), s4], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))
        x = self.dec4(torch.cat([self.up4(x), s1], dim=1))

        logits = self.head(x)
        return logits.squeeze(1)
    

class BCEDiceLoss(nn.Module):
    """
    Combines sample-wise BCE with sequence-wide Dice optimization.
    Extremely effective for capturing the precise boundaries of brief micro-arousals.
    """
    def __init__(self, pos_weight_scalar=1.0):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight_scalar], dtype=torch.float32))
        
    def forward(self, logits, targets):
        targets = targets.to(logits.dtype)
        
        # Binary Cross Entropy over the dense sequence
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='mean'
        )
        
        # Soft Dice Loss evaluation
        preds = torch.sigmoid(logits)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        denominator = preds_flat.sum() + targets_flat.sum()
        
        # Epsilon machine tolerance replaces 1.0 to ensure precise gradient tracking
        eps = 1e-6 
        dice_coef = (2. * intersection + eps) / (denominator + eps)
        dice_loss = 1.0 - dice_coef

        return (2.0 * bce) + dice_loss

