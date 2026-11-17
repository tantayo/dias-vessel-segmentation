"""Lightweight 5-frame U-Net for DIAS vessel segmentation.

Input:  (B, 5, H, W), where H and W are multiples of 16.
Output: (B, 1, H, W) vessel logits (no sigmoid).

Changes relative to the proposed model:
- The four same-channel, stride-2 convolutions are depthwise-separable
  (3x3 depthwise + 1x1 pointwise, each followed by BatchNorm and ReLU).
- Encoder/decoder widths, four downsampling stages, additive skips,
  nearest-neighbor upsampling, and output head are otherwise unchanged.
No pretrained weights or additional processing are used.
"""

import torch
from torch import nn


def depthwise_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, kernel_size=3, padding=1,
                  groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def downsample(channels: int) -> nn.Sequential:
    """Spatial downsampling with a depthwise-separable convolution."""
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=2,
                  padding=1, groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class VesselUNet(nn.Module):
    def __init__(self, in_channels: int = 5):
        super().__init__()

        self.encoder_stage1 = depthwise_conv(in_channels, 48)
        self.down1 = downsample(48)
        self.encoder_stage2 = depthwise_conv(48, 96)
        self.down2 = downsample(96)
        self.encoder_stage3 = depthwise_conv(96, 160)
        self.down3 = downsample(160)
        self.encoder_stage4 = depthwise_conv(160, 256)
        self.down4 = downsample(256)

        self.bottleneck = depthwise_conv(256, 320)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            depthwise_conv(320, 192),
        )
        self.skip_conn4 = nn.Conv2d(256, 192, kernel_size=1)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            depthwise_conv(192, 128),
        )
        self.skip_conn3 = nn.Conv2d(160, 128, kernel_size=1)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            depthwise_conv(128, 80),
        )
        self.skip_conn2 = nn.Conv2d(96, 80, kernel_size=1)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            depthwise_conv(80, 40),
        )
        self.skip_conn1 = nn.Conv2d(48, 40, kernel_size=1)

        self.output_head = nn.Conv2d(40, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.encoder_stage1[0].in_channels:
            raise ValueError("Expected input shaped (batch, 5, height, width)")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Image height and width must be divisible by 16")

        feat1 = self.encoder_stage1(x)
        feat2 = self.encoder_stage2(self.down1(feat1))
        feat3 = self.encoder_stage3(self.down2(feat2))
        feat4 = self.encoder_stage4(self.down3(feat3))
        bottleneck_out = self.bottleneck(self.down4(feat4))

        upsample1 = self.up1(bottleneck_out) + self.skip_conn4(feat4)
        upsample2 = self.up2(upsample1) + self.skip_conn3(feat3)
        upsample3 = self.up3(upsample2) + self.skip_conn2(feat2)
        upsample4 = self.up4(upsample3) + self.skip_conn1(feat1)

        return self.output_head(upsample4)


if __name__ == "__main__":
    net = VesselUNet()
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    with torch.no_grad():
        test = net(torch.randn(1, 5, 128, 128))
    print("Input: (1, 5, 128, 128); output:", tuple(test.shape))
