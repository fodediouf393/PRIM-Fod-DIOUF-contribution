import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    # Deux convolutions 3x3 + BatchNorm + ReLU
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    # Bloc de downsampling : MaxPool2d + DoubleConv
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    # Bloc d'upsampling :
    # - ConvTranspose2d sur le chemin du haut
    # - concat avec le skip
    # - DoubleConv
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()
        # up : on réduit les canaux de "in_channels" vers "out_channels"
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        # après concat : out_channels (up) + skip_channels
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1 : features venant d'en bas (à upsampler)
        # x2 : skip connection de l'encodeur
        x1 = self.up(x1)

        # Ajustement de taille si besoin
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        # Concat sur les canaux
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    # Dernière convolution 1x1 pour produire les logits
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    # U-Net 2D classique adapté à ton contexte
    # in_channels=3 (Sato, Meijering, Gabor)
    # n_classes=1 (logits pour BCEWithLogits + Dice)
    # base_channels=32 pour limiter la mémoire
    def __init__(self, in_channels=3, n_classes=1, base_channels=32):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        # Chemin descendant
        self.inc = DoubleConv(in_channels, base_channels)              # -> 32
        self.down1 = Down(base_channels, base_channels * 2)            # 32 -> 64
        self.down2 = Down(base_channels * 2, base_channels * 4)        # 64 -> 128
        self.down3 = Down(base_channels * 4, base_channels * 8)        # 128 -> 256
        self.down4 = Down(base_channels * 8, base_channels * 8)        # 256 -> 256

        # Chemin montant
        # x5 : 256 canaux, x4 : 256
        self.up1 = Up(in_channels=base_channels * 8,
                      skip_channels=base_channels * 8,
                      out_channels=base_channels * 4)                  # 256 -> 128, concat avec 256 -> 384, conv -> 128

        # x : 128 canaux, x3 : 128
        self.up2 = Up(in_channels=base_channels * 4,
                      skip_channels=base_channels * 4,
                      out_channels=base_channels * 2)                  # 128 -> 64, concat avec 128 -> 192, conv -> 64

        # x : 64 canaux, x2 : 64
        self.up3 = Up(in_channels=base_channels * 2,
                      skip_channels=base_channels * 2,
                      out_channels=base_channels)                      # 64 -> 32, concat avec 64 -> 96, conv -> 32

        # x : 32 canaux, x1 : 32
        self.up4 = Up(in_channels=base_channels,
                      skip_channels=base_channels,
                      out_channels=base_channels)                      # 32 -> 32, concat avec 32 -> 64, conv -> 32

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
