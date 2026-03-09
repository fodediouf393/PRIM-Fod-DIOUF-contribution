import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


RUN_ROOTS = [
    "experiments/runs/unet_3ch_norm_2026-02-08_12-28-02",
    "experiments/runs/unetpp_3ch_norm_2026-02-08_20-22-43",
    "experiments/runs/unet3plus_3ch_norm_2026-02-09_20-35-04",
    "experiments/runs/resunet_3ch_norm_2026-02-10_13-52-54",
    "experiments/runs/r2unet_3ch_norm_2026-02-10_20-02-47",
    "experiments/runs/attention_unet_3ch_norm_2026-02-09_12-45-34",
    "experiments/runs/unetpp_DS_3ch_norm_2026-02-11_12-56-57",
]


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_yaml(p: Path) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)


def get_logits(outputs):
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


def build_model_from_cfg(model_cfg: dict, device: str):
    name = str(model_cfg.get("model", "")).lower()
    in_ch = int(model_cfg.get("in_channels", 3))
    n_classes = int(model_cfg.get("n_classes", 1))
    base = int(model_cfg.get("base", 42))

    deep_sup = False
    if isinstance(model_cfg.get("extra", None), dict):
        deep_sup = bool(model_cfg["extra"].get("deep_supervision", False))

    t_r2 = 2
    if isinstance(model_cfg.get("extra", None), dict) and "t" in model_cfg["extra"]:
        t_r2 = int(model_cfg["extra"]["t"])

    if "unet3plus" in name:
        m = UNet3Plus(in_channels=in_ch, n_classes=n_classes, base=base)
    elif "attention" in name:
        m = AttentionUNet(in_channels=in_ch, n_classes=n_classes, base=base)
    elif "resunet" in name:
        m = ResUNet(in_channels=in_ch, n_classes=n_classes, base=base)
    elif "r2unet" in name:
        m = R2UNet(in_channels=in_ch, n_classes=n_classes, base=base, t=t_r2)
    elif "unetpp" in name:
        m = UNetPlusPlus(in_channels=in_ch, n_classes=n_classes, base=base, deep_supervision=deep_sup)
    else:
        m = UNet(in_channels=in_ch, n_classes=n_classes, base=base)

    return m.to(device)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img


def per_patch_zscore(x01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    flat = x01.reshape(-1, 3)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd = np.maximum(sd, eps)
    return (x01 - mu.reshape(1, 1, 3)) / sd.reshape(1, 1, 3)


def load_patch_3ch(raw_dir: Path, clahe_dir: Path, dog_dir: Path, patch_name: str, norm_mode: str) -> torch.Tensor:
    r = read_gray(raw_dir / patch_name)
    c = read_gray(clahe_dir / patch_name)
    d = read_gray(dog_dir / patch_name)

    x = np.stack([r, c, d], axis=-1).astype(np.float32) / 255.0
    if norm_mode == "per_image_zscore":
        x = per_patch_zscore(x)
    elif norm_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    x = np.transpose(x, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(x)


def parse_manifest(manifest_csv: Path) -> Dict[str, List[Tuple[str, int, int]]]:
    groups: Dict[str, List[Tuple[str, int, int]]] = {}
    with open(manifest_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            patch_name = row["patch_name"]
            y = int(row["y"])
            x = int(row["x"])
            groups.setdefault(img_id, []).append((patch_name, y, x))
    for k in groups:
        groups[k].sort(key=lambda t: (t[1], t[2], t[0]))
    return groups


@torch.no_grad()
def infer_and_reconstruct(
    model,
    device: str,
    groups: Dict[str, List[Tuple[str, int, int]]],
    raw_dir: Path,
    clahe_dir: Path,
    dog_dir: Path,
    out_dir: Path,
    norm_mode: str,
    image_size: int,
    patch_size: int,
    batch_size: int,
    thr: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, plist in sorted(groups.items(), key=lambda kv: natural_key(kv[0])):
        acc = np.zeros((image_size, image_size), dtype=np.float32)
        cnt = np.zeros((image_size, image_size), dtype=np.float32)

        i = 0
        while i < len(plist):
            chunk = plist[i:i + batch_size]
            xs = []
            coords = []

            for (pname, y, x) in chunk:
                t = load_patch_3ch(raw_dir, clahe_dir, dog_dir, pname, norm_mode=norm_mode)
                xs.append(t)
                coords.append((y, x))

            x_tensor = torch.stack(xs, dim=0).to(device)
            out = model(x_tensor)
            logits = get_logits(out)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            for b in range(probs.shape[0]):
                y, x = coords[b]
                p = probs[b, 0]
                acc[y:y+patch_size, x:x+patch_size] += p
                cnt[y:y+patch_size, x:x+patch_size] += 1.0

            i += batch_size

        avg = acc / np.maximum(cnt, 1.0)
        mask = (avg >= thr).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{img_id}.png"), mask)
        print(f"[OK] saved {out_dir / f'{img_id}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_root", default="data/capillaire_langevin_512_pseudo3dirs")
    ap.add_argument("--out_root", default="experiments/target_inference_832_final")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--image_size", type=int, default=832)
    ap.add_argument("--patch_size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    troot = Path(args.target_root)
    manifest_csv = troot / "manifest.csv"
    raw_dir = troot / "patches_raw"
    clahe_dir = troot / "patches_clahe"
    dog_dir = troot / "patches_dog"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"manifest.csv not found: {manifest_csv}")

    groups = parse_manifest(manifest_csv)
    print(f"[INFO] Found {len(groups)} images in manifest.")

    for rr in RUN_ROOTS:
        run_root = Path(rr)
        model_yaml = run_root / "configs" / "model.yaml"
        if not model_yaml.exists():
            print(f"[SKIP] missing model.yaml: {model_yaml}")
            continue

        model_cfg = load_yaml(model_yaml)
        model = build_model_from_cfg(model_cfg, device=device)

        # -------- UDA1: student_final --------
        uda1_dir = Path("experiments/uda_self_training") / f"{run_root.name}_seed_0_to_target"
        ckpt1 = uda1_dir / "student_final.pth"
        if ckpt1.exists():
            ck = torch.load(ckpt1, map_location=device)
            model.load_state_dict(ck["model_state"])
            model.eval()

            out_dir = Path(args.out_root) / "UDA1_student_final" / run_root.name
            infer_and_reconstruct(
                model=model,
                device=device,
                groups=groups,
                raw_dir=raw_dir,
                clahe_dir=clahe_dir,
                dog_dir=dog_dir,
                out_dir=out_dir,
                norm_mode="per_image_zscore",  # UDA1 training used zscore
                image_size=args.image_size,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                thr=args.thr,
            )
        else:
            print(f"[WARN] missing UDA1 student_final: {ckpt1}")

        # -------- UDA2: model_final --------
        uda2_dir = Path("experiments/uda_consistency_entropy") / f"{run_root.name}_seed_0_to_target"
        ckpt2 = uda2_dir / "model_final.pth"
        if ckpt2.exists():
            ck = torch.load(ckpt2, map_location=device)
            model.load_state_dict(ck["model_state"])
            model.eval()

            out_dir = Path(args.out_root) / "UDA2_model_final" / run_root.name
            infer_and_reconstruct(
                model=model,
                device=device,
                groups=groups,
                raw_dir=raw_dir,
                clahe_dir=clahe_dir,
                dog_dir=dog_dir,
                out_dir=out_dir,
                norm_mode="none",  # UDA2 training used no zscore
                image_size=args.image_size,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                thr=args.thr,
            )
        else:
            print(f"[WARN] missing UDA2 model_final: {ckpt2}")

    print("[DONE] Inference + reconstruction (final checkpoints) completed.")


if __name__ == "__main__":
    main()