import argparse
from pathlib import Path
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.target.target_dataset_3dirs import TargetPseudo3DirsDataset

from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def find_cfg_dir(source_seed_dir: Path) -> Path:
    p = source_seed_dir.parent / "configs"
    if p.exists():
        return p
    raise FileNotFoundError(f"configs/ not found at expected location: {p}")


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


def get_logits(outputs):
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


def masked_bce_with_logits(logits, targets, mask):
    bce = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    bce = bce * mask
    denom = mask.sum().clamp_min(1.0)
    return bce.sum() / denom


def save_ckpt(path: Path, epoch: int, model, optimizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_ckpt(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run_dir", required=True)  # seed_0 dir
    ap.add_argument("--target_root", default="data/capillaire_langevin_512_pseudo3dirs")
    ap.add_argument("--out_dir", default="experiments/uda_self_training")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--pos_thr", type=float, default=0.9)
    ap.add_argument("--neg_thr", type=float, default=0.1)
    ap.add_argument("--save_every", type=int, default=5)

    # NEW: resume
    ap.add_argument("--resume_ckpt", default=None, help="Path to student_epoch_XXX.pth to resume from")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_seed = Path(args.source_run_dir)
    cfg_dir = find_cfg_dir(source_seed)
    model_cfg = load_yaml(cfg_dir / "model.yaml")

    # teacher (fixed)
    teacher = build_model_from_cfg(model_cfg, device=device)
    base_ckpt = torch.load(source_seed / "best_model" / "best_model.pth", map_location=device)
    teacher.load_state_dict(base_ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # student
    student = build_model_from_cfg(model_cfg, device=device)
    student.load_state_dict(base_ckpt["model_state"])
    student.train()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    # output folder for this adaptation
    out_dir = Path(args.out_dir) / (source_seed.parent.name + "_" + source_seed.name + "_to_target")
    out_dir.mkdir(parents=True, exist_ok=True)

    # RESUME if requested
    start_epoch = 1
    if args.resume_ckpt is not None:
        resume_path = Path(args.resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_ckpt not found: {resume_path}")
        start_epoch = load_ckpt(resume_path, student, optimizer=opt, device=device)
        print(f"[RESUME] loaded {resume_path} -> starting at epoch {start_epoch}")

    # target loader
    troot = Path(args.target_root)
    ds = TargetPseudo3DirsDataset(
        str(troot / "patches_raw"),
        str(troot / "patches_clahe"),
        str(troot / "patches_dog"),
        norm_mode="per_image_zscore",
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        steps = 0

        for x, _ in dl:
            x = x.to(device)

            with torch.no_grad():
                t_out = teacher(x)
                t_logits = get_logits(t_out)
                t_probs = torch.sigmoid(t_logits)

                pos_mask = (t_probs >= args.pos_thr).float()
                neg_mask = (t_probs <= args.neg_thr).float()
                conf_mask = (pos_mask + neg_mask).clamp_max(1.0)

                pseudo = (t_probs >= 0.5).float()

            s_out = student(x)
            s_logits = get_logits(s_out)

            loss = masked_bce_with_logits(s_logits, pseudo, conf_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"[Epoch {epoch}/{args.epochs}] masked_BCE={avg:.6f}")

        if (epoch % args.save_every) == 0 or epoch == args.epochs:
            save_path = out_dir / f"student_epoch_{epoch:03d}.pth"
            save_ckpt(save_path, epoch, student, opt)
            print("[OK] saved:", save_path)

    final_path = out_dir / "student_final.pth"
    save_ckpt(final_path, args.epochs, student, opt)
    print("[OK] final saved:", final_path)


if __name__ == "__main__":
    main()