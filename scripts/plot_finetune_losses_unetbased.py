import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.common.losses import combined_paper_loss
from src.common.datamodule_finetune import build_loaders_newdomain_3ch
from src.common.transforms_newdomain import build_eval_transforms_newdomain

# modèles
from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


def _epoch_num(p: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pth$", p.name)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def eval_loss(model, loader, device: str, skel_iters: int = 10) -> float:
    model.eval()
    losses = []
    for imgs, masks, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(imgs)
        logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        loss = combined_paper_loss(logits, masks, skel_iters=skel_iters)
        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses)) if losses else float("nan")


def plot_curve(epochs, train_vals, val_vals, title, out_png: Path):
    plt.figure()
    # demandé : bleu et orange
    plt.plot(epochs, train_vals, label="train_loss", color="tab:blue")
    plt.plot(epochs, val_vals, label="val_loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()

    # grille
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.6)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_history_curve(history_json: Path):
    hist = json.loads(history_json.read_text())
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    val_loss = [h["val_loss"] for h in hist]
    return epochs, train_loss, val_loss


def build_model_for_arch(arch: str):
    arch = arch.lower()
    if arch == "r2unet":
        return R2UNet(in_channels=3, n_classes=1, base=42, t=2)
    if arch == "resunet":
        return ResUNet(in_channels=3, n_classes=1, base=42)
    if arch == "unet3plus":
        return UNet3Plus(in_channels=3, n_classes=1, base=42)
    if arch == "unet":
        return UNet(in_channels=3, n_classes=1, base=42)
    if arch == "unetpp":
        return UNetPlusPlus(in_channels=3, n_classes=1, base=42, deep_supervision=False)
    raise ValueError(f"Unknown arch={arch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/finetune_newdomain_gn_pu_seed0")
    ap.add_argument("--split_json", default="data/splits/split_newdomain_v1.json")
    ap.add_argument("--norm_yaml", default="configs/norm_newdomain.yaml")
    ap.add_argument("--pad_to", type=int, default=832)
    ap.add_argument("--batch_size_eval", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--skel_iters", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loaders (eval only transforms -> padding)
    eval_tf = build_eval_transforms_newdomain(pad_to=args.pad_to)
    train_loader, val_loader, _, _ = build_loaders_newdomain_3ch(
        split_json=args.split_json,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        train_transform=eval_tf,
        val_transform=eval_tf,
        test_transform=eval_tf,
        norm_yaml=args.norm_yaml,
    )

    # 1) Courbes exactes : UNET + UNET++
    for arch in ["unet", "unetpp"]:
        hist_path = root / arch / "history.json"
        if not hist_path.exists():
            print(f"[SKIP] No history.json for {arch}: {hist_path}")
            continue

        epochs, tr, va = load_history_curve(hist_path)
        out_png = out_dir / f"{arch}_train_val_loss.png"
        plot_curve(epochs, tr, va, f"{arch.upper()} — Train/Val Loss", out_png)
        print("[OK]", out_png)

    # 2) Courbes approximatives via checkpoints : R2UNET / RESUNET / UNET3PLUS
    approx_archs = ["r2unet", "resunet", "unet3plus"]
    for arch in approx_archs:
        ckpt_dir = root / arch / "checkpoints"
        if not ckpt_dir.exists():
            print(f"[SKIP] No checkpoints dir for {arch}: {ckpt_dir}")
            continue

        ckpts = sorted(list(ckpt_dir.glob("epoch_*.pth")), key=_epoch_num)
        if not ckpts:
            print(f"[SKIP] No epoch_*.pth found for {arch}: {ckpt_dir}")
            continue

        model = build_model_for_arch(arch).to(device)

        epochs = []
        train_losses = []
        val_losses = []

        for ck in ckpts:
            epoch = _epoch_num(ck)
            state = torch.load(ck, map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"], strict=False)
            else:
                model.load_state_dict(state, strict=False)

            tr_loss = eval_loss(model, train_loader, device=device, skel_iters=args.skel_iters)
            va_loss = eval_loss(model, val_loader, device=device, skel_iters=args.skel_iters)

            epochs.append(epoch)
            train_losses.append(tr_loss)
            val_losses.append(va_loss)

            print(f"[{arch}] epoch={epoch:03d} train_loss~={tr_loss:.4f} val_loss~={va_loss:.4f}")

        out_png = out_dir / f"{arch}_train_val_loss_APPROX.png"
        plot_curve(epochs, train_losses, val_losses, f"{arch.upper()} — Train/Val Loss (approx from checkpoints)", out_png)
        print("[OK]", out_png)

    print("\nDone. Plots are in:", out_dir)


if __name__ == "__main__":
    main()