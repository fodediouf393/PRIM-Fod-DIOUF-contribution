from __future__ import annotations
import numpy as np


def patch_id_to_image_and_slot(patch_id: int, patches_per_image: int = 4):
    """
    patch_id in [1..1200]
    returns:
      image_id in [1..300]
      slot in [0..3]
    """
    idx0 = patch_id - 1
    image_id = idx0 // patches_per_image + 1
    slot = idx0 % patches_per_image
    return image_id, slot


def stitch_binary_from_patches(
    patch_preds: dict[int, np.ndarray],
    full_size: int,
    patch_size: int,
    positions: list[list[int]],
    patches_per_image: int = 4,
    threshold: float = 0.5,
):
    """
    patch_preds: {patch_id: prob_map float32 in [0,1], shape (H,W) = (256,256)}
    returns:
      full_masks: {image_id: binary mask (full_size, full_size) uint8 {0,1}}
    """
    n_images = max((patch_id_to_image_and_slot(pid, patches_per_image)[0] for pid in patch_preds.keys()), default=0)
    full_sum = {i: np.zeros((full_size, full_size), dtype=np.float32) for i in range(1, n_images + 1)}
    full_cnt = {i: np.zeros((full_size, full_size), dtype=np.float32) for i in range(1, n_images + 1)}

    for pid, prob in patch_preds.items():
        image_id, slot = patch_id_to_image_and_slot(pid, patches_per_image)
        x, y = positions[slot]
        full_sum[image_id][y:y+patch_size, x:x+patch_size] += prob.astype(np.float32)
        full_cnt[image_id][y:y+patch_size, x:x+patch_size] += 1.0

    full_masks = {}
    for image_id in full_sum.keys():
        avg = full_sum[image_id] / np.maximum(full_cnt[image_id], 1e-6)
        full_masks[image_id] = (avg >= threshold).astype(np.uint8)

    return full_masks
