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


class AttentionBlock(nn.Module):
    # Attention Gate (Oktay et al. 2018)
    # g : feature venant du décodeur (gating)
    # x : skip connection de l'encodeur
    def __init__(self, in_channels_g, in_channels_x, inter_channels):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g : gating signal (décodeur)
        # x : skip de l'encodeur
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ajustement de taille si besoin
        if g1.shape[-2:] != x1.shape[-2:]:
            diff_y = x1.size(2) - g1.size(2)
            diff_x = x1.size(3) - g1.size(3)
            g1 = F.pad(
                g1,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UpAttention(nn.Module):
    # Bloc upsampling avec Attention Gate :
    # upsample, attention sur le skip, concat, DoubleConv
    def __init__(self, in_channels, out_channels, att_channels_skip, att_channels_gate):
        super(UpAttention, self).__init__()

        # upsample : on divise par 2 le nb de canaux
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )

        # Attention : g = feature upsamplé, x = skip
        self.attention = AttentionBlock(
            in_channels_g=att_channels_gate,
            in_channels_x=att_channels_skip,
            inter_channels=att_channels_skip // 2
        )

        # Après concat : [x_att (skip) , x_up]
        # canaux = att_channels_skip + (in_channels // 2)
        conv_in_channels = att_channels_skip + (in_channels // 2)
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x_up, x_skip):
        # Upsample du tenseur venant du décodeur
        x_up = self.up(x_up)

        # Ajustement de taille
        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        x_up = F.pad(
            x_up,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2]
        )

        # Gate = x_up (décodeur), skip = x_skip (encodeur)
        x_att = self.attention(g=x_up, x=x_skip)

        # Concat + DoubleConv
        x = torch.cat([x_att, x_up], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    # Dernière convolution 1x1 pour produire les logits
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    # Attention U-Net adapté à ton contexte
    # in_channels=3, n_classes=1, base_channels=32
    def __init__(self, in_channels=3, n_classes=1, base_channels=32):
        super(AttentionUNet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_channels = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)            # 3  -> 32
        self.down1 = Down(base_channels, base_channels * 2)          # 32 -> 64
        self.down2 = Down(base_channels * 2, base_channels * 4)      # 64 -> 128
        self.down3 = Down(base_channels * 4, base_channels * 8)      # 128 -> 256
        self.down4 = Down(base_channels * 8, base_channels * 8)      # 256 -> 256 (bottleneck)

        # Decoder avec Attention
        # x5 : 256 canaux
        # x4 : 256 canaux
        self.up1 = UpAttention(
            in_channels=base_channels * 8,        # 256 (x5)
            out_channels=base_channels * 4,       # 128
            att_channels_skip=base_channels * 8,  # 256 (x4)
            att_channels_gate=base_channels * 4,  # 128 (x_up après upsample)
        )
        # ensuite :
        # x : 128 canaux
        # x3 : 128 canaux
        self.up2 = UpAttention(
            in_channels=base_channels * 4,        # 128
            out_channels=base_channels * 2,       # 64
            att_channels_skip=base_channels * 4,  # 128 (x3)
            att_channels_gate=base_channels * 2,  # 64 (x_up)
        )
        # x : 64 canaux
        # x2 : 64 canaux
        self.up3 = UpAttention(
            in_channels=base_channels * 2,        # 64
            out_channels=base_channels,           # 32
            att_channels_skip=base_channels * 2,  # 64 (x2)
            att_channels_gate=base_channels,      # 32 (x_up)
        )
        # x : 32 canaux
        # x1 : 32 canaux
        self.up4 = UpAttention(
            in_channels=base_channels,            # 32
            out_channels=base_channels,           # 32
            att_channels_skip=base_channels,      # 32 (x1)
            att_channels_gate=base_channels // 2, # 16 (x_up)
        )

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)   # 32
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 128
        x4 = self.down3(x3)  # 256
        x5 = self.down4(x4)  # 256

        # Decoder avec attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
