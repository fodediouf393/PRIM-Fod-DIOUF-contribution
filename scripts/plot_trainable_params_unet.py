import csv
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.architectures.UnetBased.models.unet import UNet

# --- même logique que ton trainer ---
def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True

def apply_freeze_encoder(m: nn.Module):
    freeze_all(m)
    decoder_keys = ["up", "dec", "out", "final", "outc"]
    for name, p in m.named_parameters():
        if any(k in name.lower() for k in decoder_keys):
            p.requires_grad = True
    params = list(m.parameters())
    for p in params[-2:]:
        p.requires_grad = True

def apply_progressive_unfreeze(m: nn.Module, epoch0: int, e1: int, e2: int):
    if epoch0 < e1:
        apply_freeze_encoder(m); return
    if epoch0 < e2:
        apply_freeze_encoder(m)
        deep_keys = ["down3", "down4", "enc4", "enc5", "bottleneck", "bridge"]
        for name, p in m.named_parameters():
            if any(k in name.lower() for k in deep_keys):
                p.requires_grad = True
        return
    unfreeze_all(m)

def main():
    out_dir = Path("experiments/finetune_newdomain_gn_pu_seed0/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = 90
    e1, e2 = 10, 25

    model = UNet(in_channels=3, n_classes=1, base=42)

    total = sum(p.numel() for p in model.parameters())

    rows = []
    for ep in range(1, epochs + 1):
        apply_progressive_unfreeze(model, epoch0=ep-1, e1=e1, e2=e2)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append({"epoch": ep, "trainable_params": trainable, "total_params": total})

    csv_path = out_dir / "unet_trainable_params.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "trainable_params", "total_params"])
        w.writeheader()
        w.writerows(rows)

    # plot
    plt.figure()
    plt.plot([r["epoch"] for r in rows], [r["trainable_params"] for r in rows], label="trainable_params")
    plt.xlabel("Epoch")
    plt.ylabel("# params")
    plt.title("UNet — Trainable parameters over epochs (progressive unfreeze)")
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.6)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "unet_trainable_params.png", dpi=200)
    plt.close()

    print("[OK] wrote:", csv_path)
    print("[OK] saved:", out_dir / "unet_trainable_params.png")

if __name__ == "__main__":
    main()