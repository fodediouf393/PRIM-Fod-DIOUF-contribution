import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.common.seed import set_seed
from src.common.losses import combined_paper_loss
from src.common.metrics import compute_metrics_from_probs
from src.common.datamodule_finetune import build_loaders_newdomain_3ch
from src.common.transforms_newdomain import (
    build_train_transforms_newdomain,
    build_eval_transforms_newdomain,
)

# Models (adjust imports if your filenames/classes differ)
from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


# ---------- clDice metric (needs scikit-image) ----------
try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize = None


def cldice_score_batch(probs: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-7) -> float:
    """
    Mean clDice over a batch (computed on CPU, per image).
    probs: (B,1,H,W) in [0,1]
    targets: (B,1,H,W) in {0,1}
    """
    if skeletonize is None:
        return float("nan")

    probs_np = probs.detach().cpu().numpy()
    targ_np = targets.detach().cpu().numpy()

    clds = []
    for i in range(probs_np.shape[0]):
        p = (probs_np[i, 0] >= thr)
        g = (targ_np[i, 0] >= 0.5)

        sp = skeletonize(p).astype(bool)
        sg = skeletonize(g).astype(bool)

        sp_sum = sp.sum()
        sg_sum = sg.sum()

        tprec = (np.logical_and(sp, g).sum() / (sp_sum + eps)) if sp_sum > 0 else 0.0
        tsens = (np.logical_and(sg, p).sum() / (sg_sum + eps)) if sg_sum > 0 else 0.0

        denom = tprec + tsens
        cld = (2.0 * tprec * tsens / denom) if denom > 0 else 0.0
        clds.append(float(cld))

    return float(np.mean(clds)) if clds else float("nan")


# ---------- GroupNorm replacement ----------
def _choose_gn_groups(ch: int, gn_groups_max: int) -> int:
    g = min(gn_groups_max, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return g


def replace_bn_with_gn(module: nn.Module, gn_groups_max: int = 32):
    """
    Recursively replace BatchNorm2d -> GroupNorm.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            ch = child.num_features
            g = _choose_gn_groups(ch, gn_groups_max)
            gn = nn.GroupNorm(num_groups=g, num_channels=ch, eps=child.eps, affine=True)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, gn_groups_max)


# ---------- model builder ----------
def build_model(arch: str, in_ch: int = 3, n_classes: int = 1, base: int = 42):
    a = arch.lower()
    if a == "unet":
        return UNet(in_channels=in_ch, n_classes=n_classes, base=base)
    if a == "unetpp":
        return UNetPlusPlus(in_channels=in_ch, n_classes=n_classes, base=base, deep_supervision=False)
    if a == "unetpp_ds":
        return UNetPlusPlus(in_channels=in_ch, n_classes=n_classes, base=base, deep_supervision=True)
    if a == "unet3plus":
        return UNet3Plus(in_channels=in_ch, n_classes=n_classes, base=base)
    if a == "attention_unet":
        return AttentionUNet(in_channels=in_ch, n_classes=n_classes, base=base)
    if a == "resunet":
        return ResUNet(in_channels=in_ch, n_classes=n_classes, base=base)
    if a == "r2unet":
        return R2UNet(in_channels=in_ch, n_classes=n_classes, base=base, t=2)
    raise ValueError(f"Unknown arch={arch}")


def find_pretrained_best(pretrained_run_dir: Path) -> Path:
    """
    Prefer seed_0/best_model/best_model.pth, otherwise best_model/best_model.pth.
    """
    p1 = pretrained_run_dir / "seed_0" / "best_model" / "best_model.pth"
    if p1.exists():
        return p1
    p2 = pretrained_run_dir / "best_model" / "best_model.pth"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find best_model.pth under {pretrained_run_dir}")


# ---------- progressive unfreeze ----------
def freeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_all(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = True


def apply_freeze_encoder(m: nn.Module):
    """
    Freeze everything then unfreeze decoder/head by name heuristic.
    """
    freeze_all(m)
    decoder_keys = ["up", "dec", "out", "final", "outc"]
    for name, p in m.named_parameters():
        if any(k in name.lower() for k in decoder_keys):
            p.requires_grad = True

    # fallback: keep last two tensors trainable
    params = list(m.parameters())
    for p in params[-2:]:
        p.requires_grad = True


def apply_progressive_unfreeze(m: nn.Module, epoch0: int, e1: int, e2: int):
    """
    epoch0 is 0-based.
      <e1 : freeze encoder
      [e1,e2) : unfreeze deeper encoder blocks + decoder
      >=e2 : unfreeze all
    """
    if epoch0 < e1:
        apply_freeze_encoder(m)
        return

    if epoch0 < e2:
        apply_freeze_encoder(m)
        deep_keys = ["down3", "down4", "enc4", "enc5", "bottleneck", "conv4_0", "conv3_0", "bridge"]
        for name, p in m.named_parameters():
            if any(k in name.lower() for k in deep_keys):
                p.requires_grad = True
        return

    unfreeze_all(m)


@torch.no_grad()
def eval_loader(model: nn.Module, loader, device: str, skel_iters: int = 10) -> Dict[str, float]:
    model.eval()
    all_probs, all_targets = [], []
    losses = []

    for imgs, masks, _ in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(imgs)
        logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
        loss = combined_paper_loss(logits, masks, skel_iters=skel_iters)

        probs = torch.sigmoid(logits)
        losses.append(loss.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_targets.append(masks.detach().cpu())

    probs_cat = torch.cat(all_probs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    metrics = compute_metrics_from_probs(probs_cat, targets_cat, thr=0.5)
    metrics["loss"] = float(torch.stack(losses).mean().item())
    metrics["cldice"] = cldice_score_batch(probs_cat, targets_cat, thr=0.5)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    help="unet|unetpp|unetpp_ds|unet3plus|attention_unet|resunet|r2unet")
    ap.add_argument("--pretrained_run_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--split_json", default="data/splits/split_newdomain_v1.json")
    ap.add_argument("--norm_yaml", default="configs/norm_newdomain.yaml")

    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--pad_to", type=int, default=832)
    ap.add_argument("--pu_e1", type=int, default=10)
    ap.add_argument("--pu_e2", type=int, default=25)
    ap.add_argument("--gn_groups_max", type=int, default=32)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "best_model").mkdir(parents=True, exist_ok=True)

    # loaders
    train_tf = build_train_transforms_newdomain(pad_to=args.pad_to)
    eval_tf = build_eval_transforms_newdomain(pad_to=args.pad_to)

    train_loader, val_loader, test_loader, _ = build_loaders_newdomain_3ch(
        split_json=args.split_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_tf,
        val_transform=eval_tf,
        test_transform=eval_tf,
        norm_yaml=args.norm_yaml,
    )

    # model
    model = build_model(args.arch, in_ch=3, n_classes=1, base=42)
    replace_bn_with_gn(model, gn_groups_max=args.gn_groups_max)
    model.to(device)

    # load pretrained best
    pretrained_best = find_pretrained_best(Path(args.pretrained_run_dir))
    ckpt = torch.load(pretrained_best, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    print("[PRETRAIN] loaded:", pretrained_best)

    best_val = float("inf")
    best_epoch = -1

    # AMP (new API)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    history = []

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"{args.arch} epochs"):
        apply_progressive_unfreeze(model, epoch0=epoch - 1, e1=args.pu_e1, e2=args.pu_e2)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        model.train()
        train_losses = []

        for imgs, masks, _ in tqdm(train_loader, desc=f"{args.arch} train e{epoch}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                outputs = model(imgs)
                logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                loss = combined_paper_loss(logits, masks, skel_iters=10)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.detach().cpu())

        train_loss = float(torch.stack(train_losses).mean().item())
        val_metrics = eval_loader(model, val_loader, device=device, skel_iters=10)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_metrics["loss"]),
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if epoch == 1 or (epoch % 10) == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       out_dir / "checkpoints" / f"epoch_{epoch:03d}.pth")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       out_dir / "best_model" / "best_model.pth")

        print(
            f"[{args.arch}] Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} cldice={val_metrics['cldice']:.4f}"
        )

    # load best + final eval on train/test
    best_ckpt = torch.load(out_dir / "best_model" / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    train_metrics = eval_loader(model, train_loader, device=device, skel_iters=10)
    test_metrics = eval_loader(model, test_loader, device=device, skel_iters=10)

    summary = {
        "arch": args.arch,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val),
        "train": train_metrics,
        "test": test_metrics,
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "pad_to": args.pad_to,
            "pu_e1": args.pu_e1,
            "pu_e2": args.pu_e2,
            "gn_groups_max": args.gn_groups_max,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "split_json": args.split_json,
            "norm_yaml": args.norm_yaml,
            "pretrained_best": str(pretrained_best),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[OK] wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()