from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
from .stitch import stitch_binary_from_patches
from .io import write_mask_binary


def reconstruct_and_save(
    prob_cache: dict[int, "np.ndarray"],
    stitching_yaml: str,
    out_dir_full_bin: str,
    threshold: float,
):
    import numpy as np

    with open(stitching_yaml, "r") as f:
        st = yaml.safe_load(f)

    full_size = int(st["full_size"])
    patch_size = int(st["patch_size"])
    positions = st["positions"]
    patches_per_image = int(st["patches_per_image"])

    full_masks = stitch_binary_from_patches(
        patch_preds=prob_cache,
        full_size=full_size,
        patch_size=patch_size,
        positions=positions,
        patches_per_image=patches_per_image,
        threshold=threshold,
    )

    Path(out_dir_full_bin).mkdir(parents=True, exist_ok=True)
    for image_id, m01 in full_masks.items():
        write_mask_binary(str(Path(out_dir_full_bin) / f"{image_id}.png"), m01)

    return full_masks
