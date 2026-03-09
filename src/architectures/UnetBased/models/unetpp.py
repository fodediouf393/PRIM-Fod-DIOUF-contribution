# src/architectures/UnetBased/models/unetpp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) for binary/multi-class segmentation.

    Args:
        in_channels: number of input channels (3 for your case).
        n_classes: number of output channels (1 for binary logits).
        base: base number of filters (e.g., 42).
        deep_supervision: if True, returns list of outputs [x0_1, x0_2, x0_3, x0_4].
                          if False, returns a single output x0_4.
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base: int = 42, deep_supervision: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base = base
        self.deep_supervision = deep_supervision

        # channel sizes
        nb = [base, base * 2, base * 4, base * 8, base * 16]

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder convs: x0_0, x1_0, x2_0, x3_0, x4_0
        self.conv0_0 = ConvBlock(in_channels, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4])

        # Nested decoder convs
        self.conv0_1 = ConvBlock(nb[0] + nb[1], nb[0])
        self.conv1_1 = ConvBlock(nb[1] + nb[2], nb[1])
        self.conv2_1 = ConvBlock(nb[2] + nb[3], nb[2])
        self.conv3_1 = ConvBlock(nb[3] + nb[4], nb[3])

        self.conv0_2 = ConvBlock(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = ConvBlock(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = ConvBlock(nb[2] * 2 + nb[3], nb[2])

        self.conv0_3 = ConvBlock(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = ConvBlock(nb[1] * 3 + nb[2], nb[1])

        self.conv0_4 = ConvBlock(nb[0] * 4 + nb[1], nb[0])

        # Output heads
        self.final0_1 = nn.Conv2d(nb[0], n_classes, kernel_size=1)
        self.final0_2 = nn.Conv2d(nb[0], n_classes, kernel_size=1)
        self.final0_3 = nn.Conv2d(nb[0], n_classes, kernel_size=1)
        self.final0_4 = nn.Conv2d(nb[0], n_classes, kernel_size=1)

    @staticmethod
    def _up(x, ref):
        # Bilinear upsample to match spatial size of ref
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Level 0..3, stage 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))

        # stage 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))

        # stage 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))

        # stage 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))

        if self.deep_supervision:
            out1 = self.final0_1(x0_1)
            out2 = self.final0_2(x0_2)
            out3 = self.final0_3(x0_3)
            out4 = self.final0_4(x0_4)
            return [out1, out2, out3, out4]

        return self.final0_4(x0_4)
