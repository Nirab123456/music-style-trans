from typing import Any, Dict, List

import torch
import torch.nn as nn
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio

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


class LiteResUNet(nn.Module):
    """
    A lightweight UNet with ResNet or MobileNet backbones for source separation.
    Inputs: mixture spectrogram (B, C_in, F, T)
    Outputs: Dict[str, Tensor] of separated spectrograms, one per source.
    """
    def __init__(
        self,
        backbone: str = 'mobilenet_v2',
        source_names: List[str] = ['source_0', 'source_1'],
        in_channels: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.source_names = source_names
        self.in_channels = in_channels

        # Build encoder backbone
        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.enc2 = base.layer1
            self.enc3 = base.layer2
            self.enc4 = base.layer3
            self.enc5 = base.layer4
            enc_channels = [64, 64, 128, 256, 512]

        elif backbone == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            features = list(base.features)
            # Adapt first block to accept in_channels
            first = features[0]
            features[0] = nn.Sequential(
                nn.Conv2d(in_channels, first[0].out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(first[0].out_channels),
                nn.ReLU6(inplace=True)
            )
            # Split into encoder stages
            self.enc1 = nn.Sequential(*features[:4])   # output channels: 24
            self.enc2 = nn.Sequential(*features[4:7])  # output channels: 32
            self.enc3 = nn.Sequential(*features[7:14]) # output channels: 96
            self.enc4 = nn.Sequential(*features[14:])  # output channels: 1280
            self.enc5 = nn.Identity()                  # same as enc4
            # Manually set channel counts for MobileNetV2
            enc_channels = [24, 32, 96, 1280, 1280]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Decoder setup: reverse of encoder channels
        dec_channels = enc_channels[::-1]  # [bottleneck, e4, e3, e2, e1]
        # Transposed convs for upsampling
        self.up_convs = nn.ModuleList([
            nn.ConvTranspose2d(dec_channels[i], dec_channels[i+1], kernel_size=2, stride=2)
            for i in range(len(dec_channels) - 1)
        ])
        # Convs after merging skip + upsample: each has 2*channels
        self.dec_convs = nn.ModuleList([
            self.double_conv(2 * dec_channels[i+1], dec_channels[i+1])
            for i in range(len(dec_channels) - 1)
        ])

        # Final 1x1 convs: one per source, output in_channels channels
        self.final_convs = nn.ModuleDict({
            name: nn.Conv2d(enc_channels[0], in_channels, kernel_size=1)
            for name in source_names
        })

    def double_conv(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Save original spatial dims
        orig_size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottleneck = self.enc5(e4)

        # Decoder with skip connections
        dec = bottleneck
        skips = [e4, e3, e2, e1]
        for up, conv, skip in zip(self.up_convs, self.dec_convs, skips):
            dec = up(dec)
            # Align spatial dims if needed
            if dec.shape[2:] != skip.shape[2:]:
                dec = nn.functional.interpolate(dec, size=skip.shape[2:], mode='bilinear', align_corners=False)
            dec = torch.cat([skip, dec], dim=1)
            dec = conv(dec)

        # Predict and restore original size
        outputs: Dict[str, torch.Tensor] = {}
        for name, conv in self.final_convs.items():
            out = conv(dec)
            if out.shape[2:] != orig_size:
                out = nn.functional.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
            outputs[name] = out
        return outputs
