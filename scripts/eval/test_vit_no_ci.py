#!/usr/bin/env python3
import re
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm  # pip install tqdm if missing

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ==========================
# CONFIG
# ==========================
RUNS_ROOT = Path("/home/infres/diouf-25/prim-project/experiments/runs")
OUT_DIR = Path("/home/infres/diouf-25/prim-project/experiments/summary_vit_no_ci")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5
SKEL_ITERS = 10

# ViT run prefixes
VIT_KEYS = {"swinunetr", "unetr", "transunet", "sswdual", "swinunet"}  # allow both spellings

# Show batch tqdm for each run (can be verbose)
SHOW_BATCH_TQDM = True


# ==========================
# DATA LOADER (Full images 3ch)
# ==========================
def get_fullimg_test_loader():
    from src.common_vit.utils import load_yaml
    from src.common_vit.transforms import build_train_transforms, build_eval_transforms
    from src.common_vit.dataset_fullimg_3ch import build_loaders_fullimg_3ch

    data_cfg = load_yaml("configs/data_fullimg.yaml")
    train_tf = build_train_transforms()
    eval_tf = build_eval_transforms()

    _, _, test_loader = build_loaders_fullimg_3ch(
        data_cfg=data_cfg,
        split_json=data_cfg["split_json"],
        norm_yaml="configs/norm_fullimg.yaml",
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_tf=train_tf,
        eval_tf=eval_tf,
    )
    return test_loader


# ==========================
# clDice (soft skeleton)
# ==========================
def _soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)

def _soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def _soft_open(img):
    return _soft_dilate(_soft_erode(img))

def soft_skel(img, iters=10):
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def cldice_from_bin(pred_bin, gt_bin, iters=10, eps=1e-6):
    pred = pred_bin.float()
    gt = gt_bin.float()
    skel_p = soft_skel(pred, iters=iters)
    skel_g = soft_skel(gt, iters=iters)
    tprec = (skel_p * gt).sum(dim=(1, 2, 3)) / (skel_p.sum(dim=(1, 2, 3)) + eps)
    tsens = (skel_g * pred).sum(dim=(1, 2, 3)) / (skel_g.sum(dim=(1, 2, 3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return float(cl.mean().item())


# ==========================
# Metrics
# ==========================
def compute_metrics(probs, targets, thr=0.5):
    """
    probs, targets are torch tensors on CPU: (N,1,H,W)
    """
    p = (probs >= thr).to(torch.uint8)
    y = (targets >= 0.5).to(torch.uint8)

    p_f = p.view(-1).numpy()
    y_f = y.view(-1).numpy()
    probs_f = probs.view(-1).numpy()

    tp = int(((p_f == 1) & (y_f == 1)).sum())
    tn = int(((p_f == 0) & (y_f == 0)).sum())
    fp = int(((p_f == 1) & (y_f == 0)).sum())
    fn = int(((p_f == 0) & (y_f == 1)).sum())

    eps = 1e-8
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = ((tp * tn) - (fp * fn)) / mcc_den

    auc = float("nan")
    if roc_auc_score is not None:
        try:
            auc = float(roc_auc_score(y_f, probs_f))
        except Exception:
            auc = float("nan")

    cldice = cldice_from_bin(p, y, iters=SKEL_ITERS)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "cldice": float(cldice),
        "mcc": float(mcc),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "accuracy": float(acc),
    }


# ==========================
# Model loading (no manual factory)
# ==========================
def load_model_from_key(model_key: str):
    """
    Uses your known ViT modules.
    """
    if model_key in ("swinunetr", "swinunet"):
        # Your file shows class name SwinUNet wrapping MONAI SwinUNETR
        from src.architectures.VisualTransformers.SwinUnet.models.swin_unet import SwinUNet
        return SwinUNet(in_channels=3, n_classes=1, feature_size=48)

    if model_key == "unetr":
        from src.architectures.VisualTransformers.UNETR.models.unetr import UNETR2D
        return UNETR2D(in_channels=3, n_classes=1, img_size=448, patch_size=8)

    if model_key == "transunet":
        from src.architectures.VisualTransformers.TransUNet.models.transunet import TransUNet
        return TransUNet(in_channels=3, n_classes=1, img_size=448)

    if model_key == "sswdual":
        from src.architectures.VisualTransformers.SSW_Dual.models.ssw_dual import SSW_Dual
        return SSW_Dual(
            img_ch=3, output_ch=1,
            rate=48, layer_depth=4,
            kernel_size=9, extend_scope=3.0,
            window_size=7, dropout=0.0, repeat_n=1
        )

    raise KeyError(f"Unknown model_key: {model_key}")


def _extract_state(ckpt):
    return ckpt.get("model_state", ckpt)


@torch.no_grad()
def _warmup_build_lazy_modules_if_needed(model_key: str, model: torch.nn.Module, test_loader):
    """
    For SSW_Dual, Swin blocks are lazy-built on first forward.
    We MUST forward once before strict loading, so keys exist.
    """
    if model_key != "sswdual":
        return

    model.eval()
    batch = next(iter(test_loader))
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, _, _ = batch
    else:
        x, _ = batch
    x = x.to(DEVICE, non_blocking=True)
    _ = model(x)  # create lazy submodules


@torch.no_grad()
def eval_best_model(model_key: str, ckpt_path: Path, test_loader):
    model = load_model_from_key(model_key).to(DEVICE)

    # Warmup for SSW_Dual so that lazy Swin blocks exist before strict loading
    _warmup_build_lazy_modules_if_needed(model_key, model, test_loader)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = _extract_state(ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    all_probs, all_targets = [], []

    iterator = test_loader
    if SHOW_BATCH_TQDM:
        iterator = tqdm(test_loader, desc="  test", leave=False)

    for batch in iterator:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(y.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_metrics(probs, targets, thr=THRESH)


# ==========================
# CSV helpers
# ==========================
def fmt4(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.4f}"

def write_csv(path: Path, header: List[str], rows: List[List[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


# ==========================
# Run parsing
# ==========================
def detect_model_key(run_name: str) -> str:
    n = run_name.lower()
    for k in sorted(VIT_KEYS, key=len, reverse=True):
        if n.startswith(k):
            return k
    return "unknown"

def detect_loss(run_name: str) -> str:
    m = re.search(r"(loss[123])", run_name.lower())
    return m.group(1) if m else "loss?"


def collect_vit_runs() -> List[Tuple[str, str, Path, Path]]:
    """
    Returns list of (model_key, loss, run_dir, ckpt_path).
    """
    entries = []
    for run_dir in sorted([p for p in RUNS_ROOT.iterdir() if p.is_dir()]):
        model_key = detect_model_key(run_dir.name)
        if model_key == "unknown":
            continue
        ckpt = run_dir / "best_model" / "best_model.pth"
        if not ckpt.exists():
            continue
        loss = detect_loss(run_dir.name)
        entries.append((model_key, loss, run_dir, ckpt))
    return entries


def main():
    print("[INFO] Building full-image test loader...")
    test_loader = get_fullimg_test_loader()

    entries = collect_vit_runs()
    if not entries:
        print("No ViT runs with best_model/best_model.pth found.")
        return

    core_header = ["Model", "loss", "DICE", "IOU", "CLDICE"]
    other_header = ["Model", "loss", "MCC", "Precision", "Recall", "AUC", "Accuracy"]

    rows_core = []
    rows_other = []

    for model_key, loss, run_dir, ckpt_path in tqdm(entries, desc="Evaluating runs"):
        print(f"\n[EVAL] {run_dir.name}")
        mets = eval_best_model(model_key, ckpt_path, test_loader)

        rows_core.append([
            model_key.upper(), loss,
            fmt4(mets["dice"]), fmt4(mets["iou"]), fmt4(mets["cldice"])
        ])

        rows_other.append([
            model_key.upper(), loss,
            fmt4(mets["mcc"]), fmt4(mets["precision"]), fmt4(mets["recall"]),
            fmt4(mets["auc"]), fmt4(mets["accuracy"])
        ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "vit_core_metrics.csv", core_header, rows_core)
    write_csv(OUT_DIR / "vit_other_metrics.csv", other_header, rows_other)

    print("\n✅ CSV saved in:", OUT_DIR)
    print(" - vit_core_metrics.csv")
    print(" - vit_other_metrics.csv")


if __name__ == "__main__":
    main()
