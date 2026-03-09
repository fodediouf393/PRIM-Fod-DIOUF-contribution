# src/architectures/VisualTransformers/SSW_Dual/models/ssw_dual.py

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.swin_transformer import SwinTransformerBlock


# -------------------------
# Basic conv helpers
# -------------------------
class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Conv2(nn.Module):
    """2x conv (like UNet block)."""
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


# -------------------------
# Patch Merging (B,C,H,W) -> (B,2C,H/2,W/2)
# -------------------------
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim * 4)
        self.reduction = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if (H % 2) != 0 or (W % 2) != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            B, C, H, W = x.shape

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # (B,4C,H/2,W/2)
        x = self.norm(x)
        x = self.reduction(x)                   # (B,2C,H/2,W/2)
        return x


# -------------------------
# Swin wrapper (B,C,H,W) <-> (B,H,W,C) with lazy init
# -------------------------
class SwinLayer(nn.Module):
    """
    timm compatibility:
    - Some versions require input_resolution at init.
    - We build the SwinTransformerBlock lazily on first forward when H,W are known.
    """
    def __init__(self, channels, window_size=7, shift=False, drop=0.0):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.shift = shift
        self.drop = drop

        self.block = None  # IMPORTANT: no SwinTransformerBlock in __init__

    def _build_block(self, H, W, device):
        shift_size = self.window_size // 2 if self.shift else 0
        num_heads = max(1, self.channels // 32)

        # kwargs across timm versions
        kwargs = dict(
            dim=self.channels,
            input_resolution=(H, W),  # REQUIRED in your timm
            num_heads=num_heads,
            window_size=(self.window_size, self.window_size),
            shift_size=(shift_size, shift_size),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=0.0,

            # dropout args vary across timm versions
            drop=self.drop,
            proj_drop=self.drop,
            attn_drop=self.drop,
            mlp_drop=self.drop,
        )

        sig = inspect.signature(SwinTransformerBlock.__init__)
        supported = set(sig.parameters.keys())
        supported.discard("self")
        filtered = {k: v for k, v in kwargs.items() if k in supported}

        self.block = SwinTransformerBlock(**filtered).to(device)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.block is None:
            self._build_block(H, W, x.device)

        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        return x


# -------------------------
# DSConv (dynamic snake conv) - simplified but faithful spirit
# -------------------------
def _coordinate_map_scaling(coordinate_map: torch.Tensor, origin: list, target: list = [-1, 1]):
    mn, mx = origin
    a, b = target
    coordinate_map_scaled = torch.clamp(coordinate_map, mn, mx)
    scale_factor = (b - a) / (mx - mn + 1e-6)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - mn)
    return coordinate_map_scaled


def get_coordinate_map_2D(offset: torch.Tensor, morph: int, extend_scope: float, device):
    B, twoK, H, W = offset.shape
    K = twoK // 2

    y_base = torch.arange(H, device=device).view(1, H, 1).repeat(B, 1, W).unsqueeze(1).float()
    x_base = torch.arange(W, device=device).view(1, 1, W).repeat(B, H, 1).unsqueeze(1).float()

    off_y = offset[:, :K]
    off_x = offset[:, K:]
    off_y = torch.cumsum(off_y, dim=1) * extend_scope
    off_x = torch.cumsum(off_x, dim=1) * extend_scope

    xs, ys = [], []
    center = (K - 1) // 2

    if morph == 0:
        for c in range(K):
            dx = (c - center)
            xs.append(x_base + dx)
            ys.append(y_base + off_y[:, c:c+1])
    else:
        for c in range(K):
            dy = (c - center)
            ys.append(y_base + dy)
            xs.append(x_base + off_x[:, c:c+1])

    x_map = torch.cat(xs, dim=1)  # (B,K,H,W)
    y_map = torch.cat(ys, dim=1)
    return y_map, x_map


def get_interpolated_feature(input_feature, y_coordinate_map, x_coordinate_map):
    B, C, H, W = input_feature.shape
    K = y_coordinate_map.shape[1]

    y_scaled = _coordinate_map_scaling(y_coordinate_map, [0, H - 1], [-1, 1])
    x_scaled = _coordinate_map_scaling(x_coordinate_map, [0, W - 1], [-1, 1])

    # (B,H,W,K)
    y_scaled = y_scaled.permute(0, 2, 3, 1)
    x_scaled = x_scaled.permute(0, 2, 3, 1)

    grid = torch.stack([x_scaled, y_scaled], dim=-1)  # (B,H,W,K,2)
    grid = grid.reshape(B, H, W * K, 2)

    feat = F.grid_sample(
        input=input_feature,
        grid=grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B,C,H,W*K)

    feat = feat.reshape(B, C, H, W, K).permute(0, 1, 4, 2, 3)  # (B,C,K,H,W)
    return feat


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, extend_scope=3.0, morph=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.tanh = nn.Tanh()

        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(kernel_size, 1, 1))
        self.gn = nn.GroupNorm(max(1, out_channels // 4), out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.tanh(self.gn_offset(self.offset_conv(x)))
        y_map, x_map = get_coordinate_map_2D(offset, self.morph, self.extend_scope, x.device)
        feat = get_interpolated_feature(x, y_map, x_map)   # (B,C,K,H,W)
        out = self.proj(feat).squeeze(2)                   # (B,out,H,W)
        out = self.relu(self.gn(out))
        return out


class MultiView_DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, extend_scope=3.0):
        super().__init__()
        self.conv = Conv(in_channels, out_channels)
        self.dsconv_x = DSConv(in_channels, out_channels, kernel_size, extend_scope, morph=0)
        self.dsconv_y = DSConv(in_channels, out_channels, kernel_size, extend_scope, morph=1)
        self.fuse = Conv(out_channels * 3, out_channels)

    def forward(self, x):
        a = self.conv(x)
        b = self.dsconv_x(x)
        c = self.dsconv_y(x)
        return self.fuse(torch.cat([a, b, c], dim=1))


# -------------------------
# SSW Dual-Branch (rate=48)
# -------------------------
class SSW_Dual(nn.Module):
    """
    Dual branch inspired by paper:
      - snake branch: MultiView_DSConv + maxpool
      - swin branch:  SwinLayer + patch merging
      - decoder: fuse skips from both branches (concat)
    Returns logits (no sigmoid).
    """
    def __init__(
        self,
        img_ch=3,
        output_ch=1,
        rate=48,
        layer_depth=4,
        kernel_size=9,
        extend_scope=3.0,
        window_size=7,
        dropout=0.0,
        repeat_n=1,
    ):
        super().__init__()
        self.layer_depth = layer_depth
        self.maxpool = nn.MaxPool2d(2)

        # channels schedule
        enc_ch = [rate * (2 ** i) for i in range(layer_depth)]

        # snake blocks (stage0 uses input)
        self.snake_blocks = nn.ModuleList()
        self.snake_blocks.append(MultiView_DSConv(img_ch, enc_ch[0], kernel_size, extend_scope))
        for i in range(1, layer_depth):
            self.snake_blocks.append(MultiView_DSConv(enc_ch[i - 1], enc_ch[i], kernel_size, extend_scope))

        # swin "stem" to map to enc_ch[0] (so we can keep both branches aligned)
        self.swin_stem = Conv2(img_ch, enc_ch[0])

        # swin blocks per stage (operate BEFORE merging)
        self.swin_blocks = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        for i in range(0, layer_depth - 1):
            # at stage i, channels = enc_ch[i]
            blk = []
            for r in range(repeat_n):
                blk.append(SwinLayer(enc_ch[i], window_size=window_size, shift=(r % 2 == 1), drop=dropout))
            self.swin_blocks.append(nn.Sequential(*blk))
            self.patch_merging.append(PatchMerging(enc_ch[i]))  # -> enc_ch[i+1]

        # decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # decoder blocks: for i = depth-1..1, combine:
        #   x (current), snake_skip[i], swin_skip[i]
        self.dec_blocks = nn.ModuleList()
        for i in range(layer_depth - 1, 0, -1):
        # At decoder step i:
        # x_dec has enc_ch[i] channels
        # snake_skip and swin_skip are from stage i-1 -> enc_ch[i-1]
            in_c = enc_ch[i] + 2 * enc_ch[i - 1]
            out_c = enc_ch[i - 1]
            self.dec_blocks.append(Conv2(in_c, out_c))


        self.head = nn.Conv2d(enc_ch[0], output_ch, kernel_size=1)

    def forward(self, x):
        d = self.layer_depth

        # ---- Encoder: snake
        snake_skips = []
        xs = self.snake_blocks[0](x)
        snake_skips.append(xs)
        xs = self.maxpool(xs)

        for i in range(1, d):
            xs = self.snake_blocks[i](xs)
            snake_skips.append(xs)
            if i < d - 1:
                xs = self.maxpool(xs)

        # ---- Encoder: swin
        swin_skips = []
        xw = self.swin_stem(x)
        # stage 0
        xw = self.swin_blocks[0](xw)
        swin_skips.append(xw)
        xw = self.patch_merging[0](xw)

        # stages 1..d-2
        for i in range(1, d - 1):
            xw = self.swin_blocks[i](xw)
            swin_skips.append(xw)
            xw = self.patch_merging[i](xw)

        # last stage (deepest) has channels enc_ch[d-1] already in xw
        # align resolution with snake deepest skip (same stage index)
        # Note: snake deepest skip is snake_skips[d-1]
        # xw is at same stage depth d-1 after last merging
        # create a swin "skip" for deepest stage too
        # (no additional swin block beyond d-2 in this simplified design)
        swin_skips.append(xw)  # stage d-1

        # ---- Bottleneck fusion (simple sum after channel-align if needed)
        # Here channels match enc_ch[d-1]
        x_dec = xs + swin_skips[d - 1]

        # ---- Decoder: from stage d-1 down to 1
        dec_idx = 0
        for stage in range(d - 1, 0, -1):
            x_dec = self.up(x_dec)

            snake_skip = snake_skips[stage - 1]
            swin_skip = swin_skips[stage - 1]

            x_dec = F.interpolate(x_dec, size=snake_skip.shape[-2:], mode="bilinear", align_corners=False)
            x_dec = torch.cat([x_dec, snake_skip, swin_skip], dim=1)

            x_dec = self.dec_blocks[dec_idx](x_dec)
            dec_idx += 1

        logits = self.head(x_dec)
        return logits
