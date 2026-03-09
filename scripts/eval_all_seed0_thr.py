import argparse
from pathlib import Path
import csv
import json

import torch
import yaml

from src.common.datamodule import build_loaders_3ch
from src.common.transforms import build_eval_transforms
from src.common.metrics import compute_metrics_from_probs

from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.unet3plus import UNet3Plus


def load_yaml(p: Path):
    return yaml.safe_load(p.read_text())


def infer_arch(run_name: str) -> str:
    n = run_name.lower()
    if n.startswith("attention_unet"):
        return "attention_unet"
    if n.startswith("unet3plus"):
        return "unet3plus"
    if n.startswith("r2unet"):
        return "r2unet"
    if n.startswith("resunet"):
        return "resunet"
    if n.startswith("unetpp_ds"):
        return "unetpp_ds"
    if n.startswith("unetpp"):
        return "unetpp"
    if n.startswith("unet_"):
        return "unet"
    return ""


def build_model_from_arch(arch: str, in_channels: int, n_classes: int, base: int):
    # IMPORTANT: ces choix doivent correspondre à tes trainings (base=42, 3ch)
    if arch == "unet":
        return UNet(in_channels=in_channels, n_classes=n_classes, base=base)

    if arch == "unetpp":
        return UNetPlusPlus(in_channels=in_channels, n_classes=n_classes, base=base, deep_supervision=False)

    if arch == "unetpp_ds":
        return UNetPlusPlus(in_channels=in_channels, n_classes=n_classes, base=base, deep_supervision=True)

    if arch == "resunet":
        return ResUNet(in_channels=in_channels, n_classes=n_classes, base=base)

    if arch == "r2unet":
        # si tu as entraîné avec t=2 (souvent le cas)
        return R2UNet(in_channels=in_channels, n_classes=n_classes, base=base, t=2)

    if arch == "attention_unet":
        return AttentionUNet(in_channels=in_channels, n_classes=n_classes, base=base)

    if arch == "unet3plus":
        return UNet3Plus(in_channels=in_channels, n_classes=n_classes, base=base)

    raise ValueError(f"Unknown arch inferred from folder name: {arch}")


@torch.no_grad()
def eval_best_seed0(run_dir: Path, test_loader, device: str, thr: float,
                    arch: str, in_channels: int, n_classes: int, base: int):

    best_path = run_dir / "seed_0" / "best_model" / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing: {best_path}")

    model = build_model_from_arch(arch, in_channels=in_channels, n_classes=n_classes, base=base).to(device)
    ckpt = torch.load(best_path, map_location=device)

    # checkpoints saved as {"epoch":..., "model_state":...}
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)

    model.eval()
    all_probs, all_targets = [], []

    for imgs, masks, _ in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out = model(imgs)
        if isinstance(out, (list, tuple)):  # deep supervision -> last
            out = out[-1]

        probs = torch.sigmoid(out).detach().cpu()
        all_probs.append(probs)
        all_targets.append(masks.detach().cpu())

    probs_cat = torch.cat(all_probs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    return compute_metrics_from_probs(probs_cat, targets_cat, thr=thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="experiments/runs")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--norm", default="configs/norm.yaml")

    # on lit data.yaml global (car pas de configs sauvegardées dans les runs)
    ap.add_argument("--data", default="configs/data.yaml")

    # hyperparams modèle (doivent matcher tes trainings)
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--n_classes", type=int, default=1)
    ap.add_argument("--base", type=int, default=42)

    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs_root = Path(args.runs_root)

    data_cfg = load_yaml(Path(args.data))
    eval_tf = build_eval_transforms()

    # on construit le test_loader UNE SEULE FOIS (même test pour toutes les archis)
    _, _, test_loader, _ = build_loaders_3ch(
        split_json=data_cfg["split_json"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_transform=None,
        val_transform=eval_tf,
        test_transform=eval_tf,
        norm_yaml=args.norm,
    )

    if args.only and len(args.only) > 0:
        run_dirs = [runs_root / name for name in args.only]
    else:
        run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    rows = []
    for rd in run_dirs:
        arch = infer_arch(rd.name)
        if not arch:
            print(f"[SKIP] {rd.name}: cannot infer architecture from folder name")
            continue

        try:
            m = eval_best_seed0(
                rd, test_loader, device, args.thr,
                arch=arch,
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                base=args.base,
            )
            row = {
                "architecture": rd.name,
                "thr": args.thr,
                "dice": m["dice"],
                "iou": m["iou"],
                "mcc": m["mcc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "auc": m["auc"],
                "accuracy": m["accuracy"],
            }
            rows.append(row)

            # json per run
            (rd / "seed_0" / f"test_metrics_thr_{args.thr:.2f}_seed0.json").write_text(json.dumps(row, indent=2))

            print(f"[OK] {rd.name} -> dice={row['dice']:.4f} iou={row['iou']:.4f}")
        except Exception as e:
            print(f"[SKIP/ERROR] {rd.name}: {e}")

    out_csv = Path("experiments") / f"summary_seed0_test_thr_{args.thr:.2f}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["architecture", "thr", "dice", "iou", "mcc", "precision", "recall", "auc", "accuracy"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[OK] Wrote table: {out_csv}")


if __name__ == "__main__":
    main()
