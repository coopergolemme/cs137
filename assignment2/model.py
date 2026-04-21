import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if self.use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_residual:
            out += self.shortcut(identity)
            
        out = self.relu2(out)
        out = self.dropout(out)
        return out


class WeatherCNN(nn.Module):
    """
    Spatially-aware CNN for gridded weather forecasting.

    Design goals:
    - Preserve location information longer than a VGG-style global classifier.
    - Keep memory use reasonable for large 2D inputs.
    - Produce 7 outputs: 5 base targets, 1 precipitation target, 1 binary logit.
    """

    def __init__(self, in_channels: int = 35, out_channels: int = 7, use_residual: bool = False):
        super().__init__()
        if out_channels != 7:
            raise ValueError("This model is configured for exactly 7 outputs.")

        self.input_bn = nn.BatchNorm2d(in_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.encoder1 = ConvBlock(32, 64, dropout=0.15, use_residual=use_residual)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock(64, 128, dropout=0.20, use_residual=use_residual)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock(128, 256, dropout=0.25, use_residual=use_residual)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(256, 256, dropout=0.0, use_residual=use_residual)

        # Reduce channels while keeping coarse spatial layout.
        self.head_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Pool only at the end after learning location-aware features.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.reg_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.45),
        )

        self.bin_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.45),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        self.apcp_fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        self.base_head = nn.Linear(128, 5)
        self.apcp_head = nn.Linear(32, 1)
        self.bin_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.stem(x)

        x = self.encoder1(x)
        x = self.down1(x)

        x = self.encoder2(x)
        x = self.down2(x)

        x = self.encoder3(x)
        x = self.down3(x)

        x = self.bottleneck(x)
        x = self.head_conv(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        reg_feat = self.reg_fc(x)
        apcp_feat = self.apcp_fc(x)
        bin_feat = self.bin_fc(x)

        out_base = self.base_head(reg_feat)
        out_apcp = self.apcp_head(apcp_feat)

        # Raw logit for BCEWithLogitsLoss.
        out_bin = self.bin_head(bin_feat)

        return torch.cat([out_base, out_apcp, out_bin], dim=1)


if __name__ == "__main__":
    model = WeatherCNN(in_channels=35, out_channels=7, use_residual=True)
    x = torch.randn(2, 35, 450, 449)
    out = model(x)
    print("Output shape:", out.shape)
