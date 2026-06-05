import torch
import torch.nn as nn

class MobileConv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=5, stride=stride, padding=2, 
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class StftArousalNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        # Standard 2D Conv to expand channels without depthwise grouping
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Depthwise Separable blocks
        self.stage2 = MobileConv2dBlock(48, 96, stride=2)
        self.stage3 = MobileConv2dBlock(96, 144, stride=2)
        self.stage4 = MobileConv2dBlock(144, 192, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(192, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        
        return logits.squeeze(1)