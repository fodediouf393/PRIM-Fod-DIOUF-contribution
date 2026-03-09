import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNet(nn.Module):
    """
    MONAI SwinUNETR with broad compatibility across MONAI versions.
    We do NOT pass img_size because some versions don't accept it.
    Input size handling is done by padding to 448 in the dataset.
    """
    def __init__(
        self,
        in_channels=3,
        n_classes=1,
        feature_size=48,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        use_checkpoint=False,
    ):
        super().__init__()

        # Try newer signature (with spatial_dims)
        try:
            self.net = SwinUNETR(
                in_channels=in_channels,
                out_channels=n_classes,
                feature_size=feature_size,
                depths=depths,
                num_heads=num_heads,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
                use_checkpoint=use_checkpoint,
                spatial_dims=2,
            )
        except TypeError:
            # Fallback older signature (no spatial_dims)
            self.net = SwinUNETR(
                in_channels=in_channels,
                out_channels=n_classes,
                feature_size=feature_size,
                depths=depths,
                num_heads=num_heads,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
                use_checkpoint=use_checkpoint,
            )

    def forward(self, x):
        return self.net(x)
