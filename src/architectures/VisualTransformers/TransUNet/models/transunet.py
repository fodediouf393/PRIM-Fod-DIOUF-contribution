import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small CNN encoder (UNet-like)
# -------------------------
class ConvBlock(nn.Module):
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
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


# -------------------------
# Transformer encoder (ViT)
# -------------------------
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # x: (B, N, C)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, embed_dim, depth, num_heads,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid

        # Patch embedding: conv with stride=patch_size
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, E, Gh, Gw) -> (B, N, E)
        x = self.patch_embed(x)
        B, E, Gh, Gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)  # (B, N, E)
        # reshape back to feature map
        x = x.transpose(1, 2).reshape(B, E, Gh, Gw)  # (B, E, Gh, Gw)
        return x


# -------------------------
# Decoder blocks
# -------------------------
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -------------------------
# TransUNet (CNN encoder + ViT bottleneck + UNet decoder)
# -------------------------
class TransUNet(nn.Module):
    """
    Input:  (B, 3, 448, 448)
    Output: (B, 1, 448, 448) logits
    """
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        img_size=448,
        patch_size=16,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
        base_channels=32,
    ):
        super().__init__()
        self.img_size = img_size

        # CNN encoder (4 downsamplings)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)       # 448
        self.enc2 = Down(c1, c2)                     # 224
        self.enc3 = Down(c2, c3)                     # 112
        self.enc4 = Down(c3, c4)                     # 56

        # ViT bottleneck operates at 56x56 tokens when patch_size=8, but we apply it on the feature map itself.
        # We feed enc4 feature map into a ViT with patch embedding.
        # For simplicity: ViT expects img_size=56 and in_ch=c4
        vit_img_size = img_size // 8  # because enc4 is downsampled by 2^3 = 8 -> 56
        # patch_size here is on that 56x56 feature map
        # choose p=1,2,4,7 etc. but keep divisible
        vit_patch = max(1, patch_size // 16)  # heuristic: if patch_size=16 on input -> vit_patch=1 on 56
        if vit_img_size % vit_patch != 0:
            vit_patch = 1

        self.vit = ViTEncoder(
            img_size=vit_img_size,
            patch_size=vit_patch,
            in_ch=c4,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=dropout,
            attn_drop=attn_dropout,
        )

        # Project vit features to decoder channels and decode back
        self.proj = nn.Conv2d(embed_dim, c4, kernel_size=1)

        self.up3 = Up(c4, c3, c3)   # 56->112
        self.up2 = Up(c3, c2, c2)   # 112->224
        self.up1 = Up(c2, c1, c1)   # 224->448

        self.head = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)      # (B,c1,448,448)
        s2 = self.enc2(s1)     # (B,c2,224,224)
        s3 = self.enc3(s2)     # (B,c3,112,112)
        s4 = self.enc4(s3)     # (B,c4,56,56)

        b = self.vit(s4)       # (B,embed_dim,56,56) (if vit_patch=1)
        b = self.proj(b)       # (B,c4,56,56)

        x = self.up3(b, s3)    # (B,c3,112,112)
        x = self.up2(x, s2)    # (B,c2,224,224)
        x = self.up1(x, s1)    # (B,c1,448,448)

        return self.head(x)    # logits
