import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def robust_minmax01(x: np.ndarray, p_low=1.0, p_high=99.0) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def pad_reflect_to(img: np.ndarray, target: int = 832) -> np.ndarray:
    """Pad H,W up to target with reflect101. Works for 2D or HxWxC."""
    h, w = img.shape[:2]
    pad_h = max(0, target - h)
    pad_w = max(0, target - w)
    if pad_h == 0 and pad_w == 0:
        return img
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    border = cv2.BORDER_REFLECT_101
    return cv2.copyMakeBorder(img, top, bottom, left, right, border)


def load_norm_yaml(norm_yaml: Path):
    cfg = yaml.safe_load(norm_yaml.read_text())
    mean = np.array(cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg["std"], dtype=np.float32).reshape(1, 1, 3)
    return mean, std


def load_tif_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def load_png_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def load_mask01(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read {mask_path}")
    m = (m.astype(np.float32) / 255.0)
    return (m > 0.5).astype(np.uint8)  # 0/1


# ----------------------------
# GroupNorm replacement (must match how you fine-tuned)
# ----------------------------
def _choose_gn_groups(ch: int, gn_groups_max: int) -> int:
    g = min(gn_groups_max, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return g


def replace_bn_with_gn(module: nn.Module, gn_groups_max: int = 32):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            ch = child.num_features
            g = _choose_gn_groups(ch, gn_groups_max)
            gn = nn.GroupNorm(num_groups=g, num_channels=ch, eps=child.eps, affine=True)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, gn_groups_max)


# ----------------------------
# Models (adjust imports if paths differ in your repo)
# ----------------------------
from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


def build_model(arch: str):
    a = arch.lower()
    if a == "unet":
        return UNet(in_channels=3, n_classes=1, base=42)
    if a == "unetpp":
        return UNetPlusPlus(in_channels=3, n_classes=1, base=42, deep_supervision=False)
    if a == "unetpp_ds":
        return UNetPlusPlus(in_channels=3, n_classes=1, base=42, deep_supervision=True)
    if a == "unet3plus":
        return UNet3Plus(in_channels=3, n_classes=1, base=42)
    if a == "attention_unet":
        return AttentionUNet(in_channels=3, n_classes=1, base=42)
    if a == "resunet":
        return ResUNet(in_channels=3, n_classes=1, base=42)
    if a == "r2unet":
        return R2UNet(in_channels=3, n_classes=1, base=42, t=2)
    raise ValueError(f"Unknown arch: {arch}")


def load_best_model(ft_root: Path, arch: str, device: str, gn_groups_max: int = 32) -> nn.Module:
    model = build_model(arch)
    replace_bn_with_gn(model, gn_groups_max=gn_groups_max)
    model.to(device)

    best_path = ft_root / arch / "best_model" / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best model for {arch}: {best_path}")

    ckpt = torch.load(best_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_one(model: nn.Module, x_chw: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_chw: (3,H,W) float32 normalized
    Returns: probs (H,W) float32, pred (H,W) uint8 0/1
    """
    x = torch.from_numpy(x_chw).unsqueeze(0).to(device)  # (1,3,H,W)
    out = model(x)
    logits = out[-1] if isinstance(out, (list, tuple)) else out
    probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    pred = (probs >= 0.5).astype(np.uint8)
    return probs, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_root", default="experiments/finetune_newdomain_gn_pu_seed0")

    ap.add_argument("--img_dir", default="data/ens_tif")
    ap.add_argument("--clahe_dir", default="data/CLAHE")
    ap.add_argument("--dog_dir", default="data/DOG")
    ap.add_argument("--mask_dir", default="data/ens_mask_png")

    ap.add_argument("--norm_yaml", default="configs/norm_newdomain.yaml")
    ap.add_argument("--split_json", default="data/splits/split_newdomain_v1.json")

    ap.add_argument("--out_dir", default="experiments/inference_newdomain")
    ap.add_argument("--pad_to", type=int, default=832)
    ap.add_argument("--gn_groups_max", type=int, default=32)

    # fixed mosaic stems (default as requested)
    ap.add_argument("--mosaic_stems", nargs="+", default=["Test1", "Test5", "Test8", "Test11", "Test18"])

    args = ap.parse_args()

    ft_root = Path(args.ft_root)
    img_dir = Path(args.img_dir)
    clahe_dir = Path(args.clahe_dir)
    dog_dir = Path(args.dog_dir)
    mask_dir = Path(args.mask_dir)
    norm_yaml = Path(args.norm_yaml)
    split_json = Path(args.split_json)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = load_norm_yaml(norm_yaml)

    # Order of models in mosaic
    archs = ["unet", "unetpp", "unetpp_ds", "resunet", "r2unet", "unet3plus", "attention_unet"]

    # Load split -> train filter for mosaic
    split = json.loads(split_json.read_text())
    train_set = set(split["train"])

    # Load all models
    models: Dict[str, nn.Module] = {}
    for a in archs:
        models[a] = load_best_model(ft_root, a, device=device, gn_groups_max=args.gn_groups_max)
        print(f"[OK] loaded {a}")

    # Collect all valid stems (tif + clahe + dog + mask)
    tifs = sorted(img_dir.glob("*.tif"), key=lambda p: natural_key(p.name))
    stems = []
    for p in tifs:
        stem = p.stem
        if not (clahe_dir / f"{stem}.png").exists():
            continue
        if not (dog_dir / f"{stem}.png").exists():
            continue
        if not (mask_dir / f"{stem}_mask.png").exists():
            continue
        stems.append(stem)

    if not stems:
        raise RuntimeError("No valid stems found with tif+clahe+dog+mask present.")

    # Prepare mosaic stems: must be in TRAIN and have all files
    mosaic_stems = []
    for stem in args.mosaic_stems:
        if stem not in train_set:
            print(f"[WARN] {stem} is NOT in train split -> skipped")
            continue
        if not (img_dir / f"{stem}.tif").exists():
            print(f"[WARN] Missing tif for {stem} -> skipped")
            continue
        if not (clahe_dir / f"{stem}.png").exists():
            print(f"[WARN] Missing CLAHE for {stem} -> skipped")
            continue
        if not (dog_dir / f"{stem}.png").exists():
            print(f"[WARN] Missing DOG for {stem} -> skipped")
            continue
        if not (mask_dir / f"{stem}_mask.png").exists():
            print(f"[WARN] Missing mask for {stem} -> skipped")
            continue
        mosaic_stems.append(stem)

    if len(mosaic_stems) == 0:
        raise RuntimeError("No valid mosaic stems after filtering by TRAIN split.")

    print("\n[MOSAIC] Using train stems:", mosaic_stems)

    # Output dirs
    preds_root = out_dir / "predictions"
    probs_root = out_dir / "probabilities"
    preds_root.mkdir(parents=True, exist_ok=True)
    probs_root.mkdir(parents=True, exist_ok=True)
    for a in archs:
        (preds_root / a).mkdir(parents=True, exist_ok=True)
        (probs_root / a).mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Inference on ALL images
    # ----------------------------
    print(f"\n[INFER] Running inference on {len(stems)} images...")
    for stem in tqdm(stems, desc="infer all"):
        # channels
        x0 = robust_minmax01(load_tif_gray(img_dir / f"{stem}.tif"))  # [0,1]
        x1 = load_png_gray(clahe_dir / f"{stem}.png").astype(np.float32) / 255.0
        x2 = load_png_gray(dog_dir / f"{stem}.png").astype(np.float32) / 255.0

        # pad to 832
        x0 = pad_reflect_to(x0, args.pad_to)
        x1 = pad_reflect_to(x1, args.pad_to)
        x2 = pad_reflect_to(x2, args.pad_to)

        img = np.stack([x0, x1, x2], axis=-1)  # (H,W,3)

        # normalize per channel
        img = (img - mean) / std
        x_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

        for a in archs:
            probs, pred = predict_one(models[a], x_chw, device=device)
            np.save(probs_root / a / f"{stem}.npy", probs)
            cv2.imwrite(str(preds_root / a / f"{stem}.png"), (pred * 255).astype(np.uint8))

    print("[OK] Saved predictions to:", preds_root)

    # ----------------------------
    # Mosaic generation (fixed train stems)
    # ----------------------------
    n_rows = len(mosaic_stems)
    n_cols = 1 + len(archs) + 1  # input + preds + GT

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["INPUT"] + [a.upper() for a in archs] + ["GT"]
    for j in range(n_cols):
        axes[0, j].set_title(titles[j], fontsize=12)

    for i, stem in enumerate(mosaic_stems):
        # Input (normalized for display)
        x0 = robust_minmax01(load_tif_gray(img_dir / f"{stem}.tif"))
        x0 = pad_reflect_to(x0, args.pad_to)
        axes[i, 0].imshow(x0, cmap="gray")
        axes[i, 0].axis("off")

        # Predictions (binary)
        for j, a in enumerate(archs, start=1):
            p = load_png_gray(preds_root / a / f"{stem}.png")  # 0/255
            axes[i, j].imshow(p, cmap="gray")
            axes[i, j].axis("off")

        # GT binary (no green)
        gt = load_mask01(mask_dir / f"{stem}_mask.png")
        gt = pad_reflect_to(gt.astype(np.uint8), args.pad_to)
        axes[i, n_cols - 1].imshow(gt * 255, cmap="gray")
        axes[i, n_cols - 1].axis("off")

    plt.tight_layout()
    mosaic_path = out_dir / "mosaic_train_fixed.png"
    plt.savefig(mosaic_path, dpi=200)
    plt.close()
    print("[OK] Mosaic saved to:", mosaic_path)


if __name__ == "__main__":
    main()