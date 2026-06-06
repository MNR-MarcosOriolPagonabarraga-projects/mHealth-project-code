from torch import nn

class LowPowerSleepMLP(nn.Module):
    def __init__(self, num_time_steps=30, num_features=5, num_classes=5):
        super().__init__()
        # 30 * 5 = 150 input features
        input_dim = num_time_steps * num_features
        
        # Extremely lightweight bottleneck layers
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Expects input shape: (Batch, 30, 5)
        return self.network(x)

