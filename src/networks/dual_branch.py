import torch
from torch import nn

class FragmentSleepNet(nn.Module):
    def __init__(self, temporal_in_ch=2, psd_bins=159):
        super(FragmentSleepNet, self).__init__()
        
        # =================================================================
        # BRANCH 1: Temporal (Raw EEG)
        # Input shape: (Batch, 2, 3000)
        # =================================================================
        self.temporal_branch = nn.Sequential(
            # Conv 1: Wide kernel to capture slow delta waves, stride 5 drops 80% of data instantly
            nn.Conv1d(in_channels=temporal_in_ch, out_channels=8, kernel_size=50, stride=5, padding=25),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: (Batch, 8, 300)
            nn.Dropout1d(0.2),
            
            
            # Conv 2: Smaller kernel to catch rapid beta/spindle bursts
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=10, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: (Batch, 16, 75)
            nn.Dropout1d(0.2),
            
            # Conv 3: Final feature extraction
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(30), # Force spatial dimension to exactly 30
            nn.Flatten() # Output: 16 * 30 = 480 features
        )
        
        # =================================================================
        # BRANCH 2: Spectral (PSD)
        # Input shape: (Batch, 2, 159)
        # =================================================================
        self.spectral_branch = nn.Sequential(
            nn.Flatten(), # 2 * 159 = 318 features
            nn.Linear(in_features=psd_bins * 2, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU()
            # Output: 64 features
        )
        
        # =================================================================
        # FUSION & CLASSIFICATION
        # Concatenated Input: 480 + 64 = 544 features
        # =================================================================
        self.fusion = nn.Sequential(
            nn.Dropout(p=0.5), # Heavy dropout to prevent overfitting on small datasets
            nn.Linear(in_features=544, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=1) 
            # NO Sigmoid here because we will use BCEWithLogitsLoss
        )

    def forward(self, signals, psd):
        x_temp = self.temporal_branch(signals)
        x_spec = self.spectral_branch(psd)
        
        x_fused = torch.cat((x_temp, x_spec), dim=1)
        
        logits = self.fusion(x_fused)
        return logits.squeeze(-1)
    
