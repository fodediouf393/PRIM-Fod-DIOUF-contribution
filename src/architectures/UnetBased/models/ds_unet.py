# src/architectures/UnetBased/models/ds_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DSUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, base=42):
        super().__init__()
        ch = [base, base*2, base*4, base*8, base*16]

        self.inc = DoubleConv(in_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.up1 = Up(ch[4], ch[3])
        self.up2 = Up(ch[3], ch[2])
        self.up3 = Up(ch[2], ch[1])
        self.up4 = Up(ch[1], ch[0])

        # Deep supervision heads (logits)
        self.out4 = nn.Conv2d(ch[3], n_classes, 1)  # after up1
        self.out3 = nn.Conv2d(ch[2], n_classes, 1)  # after up2
        self.out2 = nn.Conv2d(ch[1], n_classes, 1)  # after up3
        self.out1 = nn.Conv2d(ch[0], n_classes, 1)  # final

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        o4 = self.out4(d4)
        o3 = self.out3(d3)
        o2 = self.out2(d2)
        o1 = self.out1(d1)

        # Upsample all to input size and average logits
        o4 = F.interpolate(o4, size=x.shape[2:], mode="bilinear", align_corners=False)
        o3 = F.interpolate(o3, size=x.shape[2:], mode="bilinear", align_corners=False)
        o2 = F.interpolate(o2, size=x.shape[2:], mode="bilinear", align_corners=False)
        # o1 already at input resolution
        return (o1 + o2 + o3 + o4) / 4.0
