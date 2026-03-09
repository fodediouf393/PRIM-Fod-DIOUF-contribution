# src/architectures/VisualTransformers/UNETR/models/unetr.py
import inspect
import torch.nn as nn
from monai.networks.nets import UNETR


def _build_unetr_with_supported_kwargs(**kwargs):
    """
    Build UNETR while filtering kwargs that are not supported by the installed MONAI version.
    This avoids errors like unexpected keyword argument 'pos_embed'.
    """
    sig = inspect.signature(UNETR.__init__)
    supported = set(sig.parameters.keys())
    # remove 'self'
    supported.discard("self")

    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return UNETR(**filtered)


class UNETR2D(nn.Module):
    """
    UNETR in 2D, compatible across MONAI versions by filtering kwargs.
    Input : (B, C, H, W)
    Output: (B, 1, H, W) logits
    """
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        img_size=448,
        patch_size=8,
        hidden_size=384,
        mlp_dim=1536,
        num_layers=12,
        num_heads=6,
        dropout_rate=0.0,
        use_checkpoint=False,
    ):
        super().__init__()

        # Common kwargs (we will filter non-supported ones)
        kwargs = dict(
            in_channels=in_channels,
            out_channels=n_classes,
            img_size=(img_size, img_size),
            feature_size=16,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            patch_size=(patch_size, patch_size),

            # Optional / version-dependent
            spatial_dims=2,
            use_checkpoint=use_checkpoint,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )

        self.net = _build_unetr_with_supported_kwargs(**kwargs)

    def forward(self, x):
        return self.net(x)
