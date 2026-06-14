import torch
from torch import nn
import torch.nn.functional as F

class ArousalNet(nn.Module):
    def __init__(self, temporal_in_ch=2, ctx_in_ch=10):
        super(ArousalNet, self).__init__()
        
        # --- TEMPORAL BRANCH (Raw EEG) ---
        # Multi-scale entry: one narrow kernel (k=10), one wide (k=50)
        self.temp_conv_fast = nn.Conv1d(temporal_in_ch, 8, kernel_size=10, stride=5, padding=5)
        self.temp_conv_slow = nn.Conv1d(temporal_in_ch, 8, kernel_size=50, stride=5, padding=25)
        self.temp_bn_init = nn.BatchNorm1d(16)
        
        self.temporal_deep = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Dropout1d(0.3),
            
            # Widened channels: 16 -> 32
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout1d(0.3),
            
            # Widened channels: 32 -> 64
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),

            SEBlock1D(in_channels=64, reduction=4)
        )
        
        # --- CONTEXT BRANCH (Rolling Bandpower) ---
        self.context_branch = nn.Sequential(
            nn.Conv1d(in_channels=ctx_in_ch, out_channels=16, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.01),
            nn.Dropout1d(0.3),
            
            # Widened channels: 16 -> 32
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),

            SEBlock1D(in_channels=32, reduction=4)
        )
        
        # --- LATE FUSION ---
        # Temporal yields 128 features (64 Max + 64 Avg). Context yields 32 features (Avg). Total = 160
        self.fusion = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=160, out_features=64),
            nn.BatchNorm1d(64), 
            nn.LeakyReLU(0.01), 
            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=1) 
        )

    def forward(self, signals, context):
        # Multi-scale Temporal Extraction
        x_fast = self.temp_conv_fast(signals)
        x_slow = self.temp_conv_slow(signals)
        x_temp = torch.cat((x_fast, x_slow), dim=1) # Shape: (B, 16, T)
        x_temp = self.temp_bn_init(x_temp)
        
        # Deep Temporal Processing
        x_temp = self.temporal_deep(x_temp) # Shape: (B, 64, T)
        
        # Dual Pooling (Max + Avg)
        temp_max = nn.functional.adaptive_max_pool1d(x_temp, 1).squeeze(-1)
        temp_avg = nn.functional.adaptive_avg_pool1d(x_temp, 1).squeeze(-1)
        x_temp_flat = torch.cat([temp_max, temp_avg], dim=1) # Shape: (B, 128)
        
        # Context Processing
        x_ctx = self.context_branch(context)
        x_ctx_flat = nn.functional.adaptive_avg_pool1d(x_ctx, 1).squeeze(-1) # Shape: (B, 32)
        
        x_fused = torch.cat((x_temp_flat, x_ctx_flat), dim=1) # Shape: (B, 160)
        logits = self.fusion(x_fused)
        
        return logits.squeeze(-1)

class SEBlock1D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock1D, self).__init__()
        # Squeeze: Summarize the whole time window into a single value per channel
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: A tiny 2-layer network to learn which channels matter right now
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid() # Outputs a strict "volume" weight between 0.0 and 1.0
        )

    def forward(self, x):
        b, c, t = x.size()
        
        # Squeeze: (Batch, Channels, Time) -> (Batch, Channels)
        y = self.squeeze(x).view(b, c)
        
        # Excite: Calculate the volume knob for each channel
        y = self.excitation(y)
        
        # Multiply: Apply the volume knob back to the original time-series data
        y = y.view(b, c, 1)
        return x * y
    
class WeightedFocalWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float, gamma: float = 2.0):
        super(WeightedFocalWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Sstandard BCE loss per element (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Logits to probabilities to calculate the difficulty multiplier
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate Focal modulation factor
        focal_weight = (1 - p_t) ** self.gamma
        
        # Re-apply class balance weight (pos_weight)
        class_weight = targets * self.pos_weight + (1 - targets)
        
        loss = class_weight * focal_weight * bce_loss
        return loss.mean()