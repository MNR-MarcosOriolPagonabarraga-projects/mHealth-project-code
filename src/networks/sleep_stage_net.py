from torch import nn

class SleepStageNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (Batch, 30 bands/context, 30 steps)
        self.conv1 = nn.Sequential(
            nn.Conv1d(60, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2) # Reduces time steps from 30 to 15
        )
        
        # Pattern extraction
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Squeezes time down to 1
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        # Permute input to (Batch, Features, Time)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

