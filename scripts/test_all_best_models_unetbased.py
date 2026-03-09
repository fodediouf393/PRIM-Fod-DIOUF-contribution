import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

from src.common.datamodule import build_loaders_3ch
from src.common.transforms import build_eval_transforms

# ---- models (adjust if names differ)
from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet


# ---------- clDice ----------
try:
    from skimage.morphology import skeletonize
except Exception as e:
    skeletonize = None
    _SKIMAGE_ERR = e


def _skeletonize_bool(mask_bool: np.ndarray) -> np.ndarray:
    return skeletonize(mask_bool).astype(bool)


def cldice_score(pred_bool: np.ndarray, gt_bool: np.ndarray, eps: float = 1e-7) -> float:
    if skeletonize is None:
        raise RuntimeError(
            "scikit-image is required for clDice. Install with: pip install scikit-image\n"
            f"Original error: {_SKIMAGE_ERR}"
        )

    sp = _skeletonize_bool(pred_bool)
    sg = _skeletonize_bool(gt_bool)

    sp_sum = sp.sum()
    sg_sum = sg.sum()

    tprec = (np.logical_and(sp, gt_bool).sum() / (sp_sum + eps)) if sp_sum > 0 else 0.0
    tsens = (np.logical_and(sg, pred_bool).sum() / (sg_sum + eps)) if sg_sum > 0 else 0.0

    denom = (tprec + tsens)
    if denom <= 0:
        return 0.0
    return float(2.0 * tprec * tsens / denom)


# ---------- helpers ----------
def load_yaml(p: Path) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)


def mean_ci95(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(sum(values) / n)
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci = 1.96 * std / math.sqrt(n)
    return mean, float(ci)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_seed_dirs(run_dir: Path) -> List[Path]:
    seeds = sorted([p for p in run_dir.glob("seed_*") if p.is_dir()], key=lambda p: natural_key(p.name))
    return seeds if seeds else [run_dir]


def find_configs_dir(seed_dir: Path) -> Path:
    """
    Look for configs in:
    1) seed_dir/configs
    2) seed_dir.parent/configs  (typical multi-run)
    3) project_root/configs     (fallback)
    """
    if (seed_dir / "configs").exists():
        return seed_dir / "configs"

    if seed_dir.name.startswith("seed_") and (seed_dir.parent / "configs").exists():
        return seed_dir.parent / "configs"

    # fallback: repository configs/
    if Path("configs").exists():
        return Path("configs")

    raise FileNotFoundError(f"Could not find configs/ for seed_dir={seed_dir}")


def _get_logits(outputs):
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


@torch.no_grad()
def eval_one_seed(seed_dir: Path, device: str, thr: float = 0.5) -> Dict[str, float]:
    cfg_dir = find_configs_dir(seed_dir)

    data_cfg = load_yaml(cfg_dir / "data.yaml")
    train_cfg = load_yaml(cfg_dir / "train_paper.yaml")
    model_cfg = load_yaml(cfg_dir / "model.yaml")

    norm_path = (cfg_dir / "norm_path.txt").read_text().strip() if (cfg_dir / "norm_path.txt").exists() else "configs/norm.yaml"

    eval_tf = build_eval_transforms()
    _, _, test_loader, _ = build_loaders_3ch(
        split_json=data_cfg["split_json"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_transform=None,
        val_transform=eval_tf,
        test_transform=eval_tf,
        norm_yaml=norm_path,
    )

    model_name = str(model_cfg.get("model", "")).lower()
    in_ch = int(model_cfg["in_channels"])
    n_classes = int(model_cfg["n_classes"])
    base = int(model_cfg["base"])

    deep_sup = False
    if isinstance(model_cfg.get("extra", None), dict):
        deep_sup = bool(model_cfg["extra"].get("deep_supervision", False))
    t_r2 = None
    if isinstance(model_cfg.get("extra", None), dict) and "t" in model_cfg["extra"]:
        t_r2 = int(model_cfg["extra"]["t"])

    if "unet3plus" in model_name or "unet3+" in model_name:
        model = UNet3Plus(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
    elif "attention" in model_name:
        model = AttentionUNet(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
    elif "resunet" in model_name:
        model = ResUNet(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
    elif "r2unet" in model_name:
        model = R2UNet(in_channels=in_ch, n_classes=n_classes, base=base, t=(t_r2 or 2)).to(device)
    elif "unetpp" in model_name or "unet++" in model_name:
        model = UNetPlusPlus(in_channels=in_ch, n_classes=n_classes, base=base, deep_supervision=deep_sup).to(device)
    elif "unet" in model_name:
        model = UNet(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
    else:
        # fallback by run folder name
        run_name = (seed_dir.parent.name if seed_dir.name.startswith("seed_") else seed_dir.name).lower()
        if "unetpp" in run_name:
            model = UNetPlusPlus(in_channels=in_ch, n_classes=n_classes, base=base, deep_supervision=deep_sup).to(device)
        elif "unet3plus" in run_name or "unet3+" in run_name:
            model = UNet3Plus(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
        elif "attention" in run_name:
            model = AttentionUNet(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
        elif "resunet" in run_name:
            model = ResUNet(in_channels=in_ch, n_classes=n_classes, base=base).to(device)
        elif "r2unet" in run_name:
            model = R2UNet(in_channels=in_ch, n_classes=n_classes, base=base, t=(t_r2 or 2)).to(device)
        else:
            model = UNet(in_channels=in_ch, n_classes=n_classes, base=base).to(device)

    best_path = seed_dir / "best_model" / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best model: {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dice_list, iou_list, cldice_list = [], [], []
    y_true_all, y_prob_all, y_pred_all = [], [], []
    eps = 1e-7

    for imgs, masks, _ in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(imgs)
        logits = _get_logits(outputs)
        probs = torch.sigmoid(logits)

        probs_np = probs.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        preds_np = (probs_np >= thr).astype(np.uint8)
        gts_np = (masks_np >= 0.5).astype(np.uint8)

        B = preds_np.shape[0]
        for i in range(B):
            p = preds_np[i, 0].astype(bool)
            g = gts_np[i, 0].astype(bool)

            inter = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            p_sum = p.sum()
            g_sum = g.sum()

            dice_list.append(float((2.0 * inter + eps) / (p_sum + g_sum + eps)))
            iou_list.append(float((inter + eps) / (union + eps)))
            cldice_list.append(cldice_score(p, g))

        y_true_all.append(gts_np.reshape(-1))
        y_prob_all.append(probs_np.reshape(-1))
        y_pred_all.append(preds_np.reshape(-1))

    y_true = np.concatenate(y_true_all).astype(np.uint8)
    y_prob = np.concatenate(y_prob_all).astype(np.float32)
    y_pred = np.concatenate(y_pred_all).astype(np.uint8)

    dice = float(np.mean(dice_list)) if dice_list else float("nan")
    iou = float(np.mean(iou_list)) if iou_list else float("nan")
    cld = float(np.mean(cldice_list)) if cldice_list else float("nan")

    mcc = float(matthews_corrcoef(y_true, y_pred)) if y_true.size else float("nan")
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")

    return {
        "dice": dice, "iou": iou, "cldice": cld,
        "mcc": mcc, "precision": prec, "recall": rec, "auc": auc, "accuracy": acc
    }


def format4(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.4f}"


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="experiments/runs")
    ap.add_argument("--out_dir", default="experiments")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    run_dirs = [
        runs_root / "attention_unet_3ch_norm_2026-02-09_12-45-34",
        runs_root / "r2unet_3ch_norm_2026-02-10_20-02-47",
        runs_root / "resunet_3ch_norm_2026-02-10_13-52-54",
        runs_root / "unet_3ch_norm_2026-02-08_12-28-02",
        runs_root / "unet3plus_3ch_norm_2026-02-09_20-35-04",
        runs_root / "unetpp_3ch_norm_2026-02-08_20-22-43",
        runs_root / "unetpp_DS_3ch_norm_2026-02-11_12-56-57",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    arch_to_seed_metrics: Dict[str, List[Dict[str, float]]] = {}

    for rd in run_dirs:
        if not rd.exists():
            print(f"[WARN] missing run dir, skip: {rd}")
            continue

        arch_name = rd.name.split("_3ch_norm_")[0]
        seed_dirs = find_seed_dirs(rd)

        print(f"\n[ARCH] {arch_name}  seeds={len(seed_dirs)}  ({rd})")
        arch_to_seed_metrics.setdefault(arch_name, [])

        for sd in seed_dirs:
            best_path = sd / "best_model" / "best_model.pth"
            if not best_path.exists():
                print(f"  [SKIP] {sd.name}: missing best_model.pth")
                continue
            try:
                m = eval_one_seed(sd, device=device, thr=0.5)
                arch_to_seed_metrics[arch_name].append(m)
                print(f"  [OK] {sd.name}: dice={m['dice']:.4f} iou={m['iou']:.4f} cldice={m['cldice']:.4f}")
            except Exception as e:
                print(f"  [ERR] {sd.name}: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_csv = out_dir / "results_unetbased_core.csv"
    other_csv = out_dir / "results_unetbased_other.csv"

    core_rows, other_rows = [], []

    for arch, lst in arch_to_seed_metrics.items():
        if not lst:
            continue

        n = len(lst)

        dice_vals = [x["dice"] for x in lst]
        iou_vals = [x["iou"] for x in lst]
        cld_vals = [x["cldice"] for x in lst]

        mcc_vals = [x["mcc"] for x in lst]
        prec_vals = [x["precision"] for x in lst]
        rec_vals = [x["recall"] for x in lst]
        auc_vals = [x["auc"] for x in lst if not (isinstance(x["auc"], float) and math.isnan(x["auc"]))]  # drop nan
        acc_vals = [x["accuracy"] for x in lst]

        dice_mean, dice_ci = mean_ci95(dice_vals)
        iou_mean, iou_ci = mean_ci95(iou_vals)
        cld_mean, cld_ci = mean_ci95(cld_vals)

        mcc_mean, mcc_ci = mean_ci95(mcc_vals)
        prec_mean, prec_ci = mean_ci95(prec_vals)
        rec_mean, rec_ci = mean_ci95(rec_vals)
        auc_mean, auc_ci = mean_ci95(auc_vals) if auc_vals else (float("nan"), float("nan"))
        acc_mean, acc_ci = mean_ci95(acc_vals)

        core_rows.append({
            "architecture": arch,
            "n_seeds": n,
            "dice_mean": format4(dice_mean), "dice_ci95": format4(dice_ci),
            "iou_mean": format4(iou_mean), "iou_ci95": format4(iou_ci),
            "cldice_mean": format4(cld_mean), "cldice_ci95": format4(cld_ci),
        })

        other_rows.append({
            "architecture": arch,
            "n_seeds": n,
            "mcc_mean": format4(mcc_mean), "mcc_ci95": format4(mcc_ci),
            "precision_mean": format4(prec_mean), "precision_ci95": format4(prec_ci),
            "recall_mean": format4(rec_mean), "recall_ci95": format4(rec_ci),
            "auc_mean": format4(auc_mean), "auc_ci95": format4(auc_ci),
            "accuracy_mean": format4(acc_mean), "accuracy_ci95": format4(acc_ci),
        })

    write_csv(
        core_csv,
        sorted(core_rows, key=lambda r: r["architecture"]),
        ["architecture", "n_seeds",
         "dice_mean", "dice_ci95",
         "iou_mean", "iou_ci95",
         "cldice_mean", "cldice_ci95"],
    )

    write_csv(
        other_csv,
        sorted(other_rows, key=lambda r: r["architecture"]),
        ["architecture", "n_seeds",
         "mcc_mean", "mcc_ci95",
         "precision_mean", "precision_ci95",
         "recall_mean", "recall_ci95",
         "auc_mean", "auc_ci95",
         "accuracy_mean", "accuracy_ci95"],
    )

    print("\n[OK] Wrote:")
    print(f"  {core_csv}")
    print(f"  {other_csv}")


if __name__ == "__main__":
    main()
