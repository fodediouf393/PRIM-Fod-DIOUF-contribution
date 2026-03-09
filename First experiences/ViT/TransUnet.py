import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    # Conv 3x3 + BN + ReLU (x2)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    # MaxPool + DoubleConv
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    # ConvTranspose2d + concat skip + DoubleConv
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        diff_y = x_skip.size(2) - x_up.size(2)
        diff_x = x_skip.size(3) - x_up.size(3)
        if diff_y != 0 or diff_x != 0:
            x_up = F.pad(
                x_up,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        x = torch.cat([x_skip, x_up], dim=1)
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    # Block Transformer Encoder
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        a, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + self.drop1(a)

        x2 = self.norm2(x)
        x = x + self.drop2(self.mlp(x2))
        return x


class TransUNet(nn.Module):
    """
    TransUNet simple : UNet encoder + Transformer au bottleneck + UNet decoder.
    Compatible avec ton pipeline (logits (B,1,H,W)).

    Param important:
    - max_tokens doit être >= (H/16)*(W/16)
      Exemple: patch 256 -> 16x16=256 tokens (OK).
               patch 512 -> 32x32=1024 tokens (OK).
    """

    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        base_channels=32,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        max_tokens=4096,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim

        # Encoder UNet (4 downsamples -> bottleneck H/16)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 8)

        # Projection bottleneck -> tokens
        self.to_embed = nn.Conv2d(base_channels * 8, embed_dim, kernel_size=1)

        # Positional embedding appris
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
             for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Tokens -> feature map bottleneck channels
        self.from_embed = nn.Conv2d(embed_dim, base_channels * 8, kernel_size=1)

        # Decoder UNet
        self.up1 = Up(base_channels * 8, base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels * 2, base_channels)
        self.up4 = Up(base_channels, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # (B, 8C, H/16, W/16)

        # Transformer bottleneck
        feat = self.to_embed(x5)               # (B, D, h, w)
        B, D, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, D)
        N = tokens.size(1)

        if N > self.max_tokens:
            raise RuntimeError(f"N tokens={N} > max_tokens={self.max_tokens}. Augmente max_tokens.")

        tokens = tokens + self.pos_embed[:, :N, :]

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        feat_t = tokens.transpose(1, 2).reshape(B, D, h, w)
        x5t = self.from_embed(feat_t)  # (B, 8C, h, w)

        # Decoder
        x = self.up1(x5t, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits
