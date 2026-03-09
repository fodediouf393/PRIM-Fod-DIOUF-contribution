from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import torch
import cv2

from .io import read_gray, write_mask_binary


def load_norm(norm_yaml: str):
    if norm_yaml is None:
        return None
    with open(norm_yaml, "r") as f:
        d = yaml.safe_load(f)
    mean = np.array(d["mean"], dtype=np.float32)  # len=3
    std = np.array(d["std"], dtype=np.float32)
    return mean, std


def load_3ch_patch(p1: str, p2: str, p3: str) -> np.ndarray:
    """
    returns float32 image in [0,1], shape (H,W,3)
    """
    a = read_gray(p1).astype(np.float32) / 255.0
    b = read_gray(p2).astype(np.float32) / 255.0
    c = read_gray(p3).astype(np.float32) / 255.0
    x = np.stack([a, b, c], axis=-1)
    return x


def apply_norm(x: np.ndarray, norm):
    if norm is None:
        return x
    mean, std = norm
    return (x - mean.reshape(1, 1, 3)) / (std.reshape(1, 1, 3) + 1e-8)


@torch.no_grad()
def predict_test_patches_binary(
    model,
    device: str,
    test_ids: list[int],
    in_dirs: dict,
    out_dir_patches_bin: str,
    threshold: float,
    norm_yaml: str | None,
):
    """
    Predict binary masks for test patch IDs, save per-patch binary masks (0/255 png)
    Returns dict patch_id -> prob_map (float32 [0,1]) for stitching.
    """
    Path(out_dir_patches_bin).mkdir(parents=True, exist_ok=True)

    norm = load_norm(norm_yaml) if norm_yaml else None
    model.eval()

    prob_cache = {}

    for pid in test_ids:
        # file name: "1.png" or "1.bmp" etc.
        # We'll try common extensions.
        def find_file(d: str, pid: int):
            base = Path(d)
            for ext in [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]:
                p = base / f"{pid}{ext}"
                if p.exists():
                    return str(p)
            # if no extension match, try raw direct (rare)
            p = base / str(pid)
            if p.exists():
                return str(p)
            raise FileNotFoundError(f"Cannot find patch file for id={pid} in {d}")

        pA = find_file(in_dirs["c1"], pid)
        pB = find_file(in_dirs["c2"], pid)
        pC = find_file(in_dirs["c3"], pid)

        x = load_3ch_patch(pA, pB, pC)
        x = apply_norm(x, norm)
        # to tensor BCHW
        xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)  # 1,3,H,W

        out = model(xt)
        # deep supervision case: list -> take last for inference
        if isinstance(out, (list, tuple)):
            out = out[-1]

        prob = torch.sigmoid(out)[0, 0].detach().cpu().numpy().astype(np.float32)  # H,W

        bin01 = (prob >= threshold).astype(np.uint8)
        write_mask_binary(str(Path(out_dir_patches_bin) / f"{pid}.png"), bin01)

        prob_cache[pid] = prob

    return prob_cache
