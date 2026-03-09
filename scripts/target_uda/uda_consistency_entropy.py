import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from src.target.target_dataset_3dirs import TargetPseudo3DirsDataset

from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


# --- Numerical safety constants ---
EPS = 1e-4            # for log + clamp probs
LOGIT_CLAMP = 20.0    # clamp logits before sigmoid/log
HUBER_BETA = 0.05     # SmoothL1 beta (smaller = more robust)
GRAD_CLIP = 1.0       # gradient norm clip


def load_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def find_cfg_dir(source_seed_dir: Path) -> Path:
    """
    Your structure:
      experiments/runs/<run_name>/configs/
      experiments/runs/<run_name>/seed_0/
    so configs are in parent/configs when source_seed_dir is seed_0.
    """
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
    # deep supervision -> list/tuple, take last output
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


# ------------------ augmentations (SAFE) ------------------

def weak_aug(x: torch.Tensor) -> torch.Tensor:
    # weak: horizontal flip only
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    return x


def strong_aug(x: torch.Tensor) -> torch.Tensor:
    """
    Strong but safe:
    - flips
    - very mild gaussian noise
    NO intensity jitter (it can destabilize)
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])

    x = x + torch.randn_like(x) * 0.005
    return x


# ------------------ stable losses ------------------

def safe_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts logits -> probabilities in (0,1) with full safety:
    - replace nan/inf
    - clamp logits
    - sigmoid
    - clamp probs
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP)
    logits = logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
    p = torch.sigmoid(logits)
    p = p.clamp(EPS, 1 - EPS)
    return p


def entropy_loss_from_probs(p: torch.Tensor) -> torch.Tensor:
    """
    Binary entropy: -p log p - (1-p) log(1-p)
    p must already be clamped in (eps, 1-eps).
    """
    return (-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_run_dir", required=True)  # seed_0 dir
    ap.add_argument("--target_root", default="data/capillaire_langevin_512_pseudo3dirs")
    ap.add_argument("--out_dir", default="experiments/uda_consistency_entropy")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lambda_cons", type=float, default=1.0)
    ap.add_argument("--lambda_ent", type=float, default=0.01)
    ap.add_argument("--save_every", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_seed = Path(args.source_run_dir)
    cfg_dir = find_cfg_dir(source_seed)
    model_cfg = load_yaml(cfg_dir / "model.yaml")

    model = build_model_from_cfg(model_cfg, device=device)
    ckpt = torch.load(source_seed / "best_model" / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Target dataset: IMPORTANT -> norm_mode="none" for stability
    troot = Path(args.target_root)
    ds = TargetPseudo3DirsDataset(
        str(troot / "patches_raw"),
        str(troot / "patches_clahe"),
        str(troot / "patches_dog"),
        norm_mode="none",
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    out_dir = Path(args.out_dir) / (source_seed.parent.name + "_" + source_seed.name + "_to_target")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        steps = 0
        skipped = 0

        # debug stats (optional)
        max_abs_logit_seen = 0.0

        for x, _ in dl:
            x = x.to(device)

            xw = weak_aug(x.clone())
            xs = strong_aug(x.clone())

            out_w = model(xw)
            out_s = model(xs)

            logits_w = get_logits(out_w).detach()
            logits_s = get_logits(out_s)

            # track logits magnitude before clamp (debug)
            if torch.isfinite(logits_s).any():
                max_abs_logit_seen = max(max_abs_logit_seen, float(torch.nan_to_num(logits_s).abs().max().item()))

            # probabilities (SAFE)
            pw = safe_probs_from_logits(logits_w).detach()
            ps = safe_probs_from_logits(logits_s)

            # robust consistency on probabilities (bounded)
            cons = F.smooth_l1_loss(ps, pw, beta=HUBER_BETA)

            # entropy (bounded)
            ent = entropy_loss_from_probs(ps)

            loss = args.lambda_cons * cons + args.lambda_ent * ent

            if not torch.isfinite(loss):
                skipped += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            opt.step()

            total += float(loss.item())
            steps += 1

        avg = total / max(1, steps)
        print(
            f"[Epoch {epoch}/{args.epochs}] loss={avg:.6f} "
            f"(steps={steps}, skipped={skipped}, max_abs_logit~{max_abs_logit_seen:.2f})"
        )

        if (epoch % args.save_every) == 0 or epoch == args.epochs:
            save_path = out_dir / f"model_epoch_{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, save_path)
            print("[OK] saved:", save_path)

    final_path = out_dir / "model_final.pth"
    torch.save({"epoch": args.epochs, "model_state": model.state_dict()}, final_path)
    print("[OK] final saved:", final_path)


if __name__ == "__main__":
    main()