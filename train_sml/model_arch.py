from typing import Any, Dict, List

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Model Definition: Simple UNet for Source Separation
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    """
    A simplified UNet architecture for source separation.
    This model takes a mixture spectrogram (B, 1, F, T) and outputs a dictionary with 
    separated source spectrograms.
    """
    def __init__(self, in_channels: int = 1, features: List[int] = [16, 32, 64]) -> None:
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1] * 2)

        # Decoder (reverse)
        rev_features = features[::-1]
        for feature in rev_features:
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self.double_conv(feature * 2, feature))

        # Final output conv layers for each source.
        # For source separation, we output one channel per source.
        # These are initialized externally after model instantiation.
        self.final_convs = nn.ModuleDict()

    def double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](x)
        
        # Generate individual source spectrograms via separate final conv layers.
        outputs: Dict[str, torch.Tensor] = {}
        for key, conv in self.final_convs.items():
            outputs[key] = conv(x)
        return outputs
