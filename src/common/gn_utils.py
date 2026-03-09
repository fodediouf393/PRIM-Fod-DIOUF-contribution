import torch
import torch.nn as nn


def _choose_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """
    Choose a number of groups that divides num_channels.
    Prefer the largest possible up to max_groups.
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def convert_bn_to_gn(module: nn.Module, max_groups: int = 32) -> nn.Module:
    """
    Recursively replace nn.BatchNorm2d with nn.GroupNorm.

    - GroupNorm has no running stats (more stable for batch_size=1).
    - We copy BN affine params (weight/bias) into GN when available.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            g = _choose_num_groups(c, max_groups=max_groups)
            gn = nn.GroupNorm(num_groups=g, num_channels=c, affine=True)

            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)

            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, max_groups=max_groups)
    return module
