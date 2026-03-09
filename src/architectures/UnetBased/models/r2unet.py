# src/architectures/UnetBased/models/r2unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentBlock(nn.Module):
    def __init__(self, ch, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = x
        for _ in range(self.t):
            out = self.conv(out)
        return out


class RRCNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.rcnn = nn.Sequential(
            RecurrentBlock(out_ch, t=t),
            RecurrentBlock(out_ch, t=t),
        )

    def forward(self, x):
        x = self.conv_1x1(x)
        out = self.rcnn(x)
        return out + x  # residual


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = RRCNNBlock(in_ch, out_ch, t=t)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, t=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.block = RRCNNBlock(in_ch, out_ch, t=t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)


class R2UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, base=42, t=2):
        super().__init__()
        ch = [base, base*2, base*4, base*8, base*16]

        self.inb = RRCNNBlock(in_channels, ch[0], t=t)
        self.d1 = Down(ch[0], ch[1], t=t)
        self.d2 = Down(ch[1], ch[2], t=t)
        self.d3 = Down(ch[2], ch[3], t=t)
        self.d4 = Down(ch[3], ch[4], t=t)

        self.u1 = Up(ch[4], ch[3], t=t)
        self.u2 = Up(ch[3], ch[2], t=t)
        self.u3 = Up(ch[2], ch[1], t=t)
        self.u4 = Up(ch[1], ch[0], t=t)

        self.out = nn.Conv2d(ch[0], n_classes, 1)

    def forward(self, x):
        x1 = self.inb(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out(x)
