import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

from src.architectures.VisualTransformers.DualBranchDINO.models.dual_branch_dino import DualBranchDINO


# -------------------------
# Losses (Dice, BCE, Tversky, clDice)
# -------------------------
import torch.nn.functional as F

def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iters=10):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        den = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.eps) / (den + self.eps)
        return 1 - dice.mean()

class clDiceLoss(nn.Module):
    def __init__(self, iters=10, eps=1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        skel_p = soft_skel(probs, iters=self.iters)
        skel_g = soft_skel(targets, iters=self.iters)
        tprec = (skel_p * targets).sum(dim=(1,2,3)) / (skel_p.sum(dim=(1,2,3)) + self.eps)
        tsens = (skel_g * probs).sum(dim=(1,2,3)) / (skel_g.sum(dim=(1,2,3)) + self.eps)
        cl = (2 * tprec * tsens) / (tprec + tsens + self.eps)
        return 1 - cl.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p = p.view(p.size(0), -1)
        t = targets.view(targets.size(0), -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        tv = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tv.mean()

def build_loss(loss_name: str):
    dice = DiceLoss()
    cld = clDiceLoss(iters=10)
    bce = nn.BCEWithLogitsLoss()
    tv  = TverskyLoss(alpha=0.7, beta=0.3)

    if loss_name == "loss1":
        # 0.8 Dice + 0.2 clDice
        def fn(logits, y):
            return 0.8 * dice(logits, y) + 0.2 * cld(logits, y)
        return fn

    if loss_name == "loss2":
        # 0.5 BCE + 0.5 Dice + 0.2 clDice
        def fn(logits, y):
            return 0.5 * bce(logits, y) + 0.5 * dice(logits, y) + 0.2 * cld(logits, y)
        return fn

    if loss_name == "loss3":
        # Tversky 0.5 + Dice 0.3 + clDice 0.2 (tu peux ajuster)
        def fn(logits, y):
            return 0.5 * tv(logits, y) + 0.3 * dice(logits, y) + 0.2 * cld(logits, y)
        return fn

    raise ValueError(f"Unknown loss_name={loss_name}. Use loss1|loss2|loss3")


# -------------------------
# Metrics
# -------------------------
def cldice_metric_from_bin(pred_bin, gt_bin, iters=10, eps=1e-6):
    pred = pred_bin.float()
    gt = gt_bin.float()
    skel_p = soft_skel(pred, iters=iters)
    skel_g = soft_skel(gt, iters=iters)
    tprec = (skel_p * gt).sum(dim=(1,2,3)) / (skel_p.sum(dim=(1,2,3)) + eps)
    tsens = (skel_g * pred).sum(dim=(1,2,3)) / (skel_g.sum(dim=(1,2,3)) + eps)
    cl = (2 * tprec * tsens) / (tprec + tsens + eps)
    return float(cl.mean().item())

@torch.no_grad()
def compute_metrics(logits, y, thr=0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    p = (probs >= thr).to(torch.uint8)
    t = (y >= 0.5).to(torch.uint8)

    p_f = p.view(-1).cpu().numpy()
    t_f = t.view(-1).cpu().numpy()
    probs_f = probs.view(-1).cpu().numpy()

    tp = int(((p_f == 1) & (t_f == 1)).sum())
    tn = int(((p_f == 0) & (t_f == 0)).sum())
    fp = int(((p_f == 1) & (t_f == 0)).sum())
    fn = int(((p_f == 0) & (t_f == 1)).sum())

    eps = 1e-8
    dice = (2*tp) / (2*tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + eps)
    mcc = ((tp*tn) - (fp*fn)) / mcc_den

    auc = float("nan")
    if roc_auc_score is not None:
        try:
            auc = float(roc_auc_score(t_f, probs_f))
        except Exception:
            auc = float("nan")

    cld = cldice_metric_from_bin(p, t, iters=10)

    return dict(
        dice=float(dice),
        iou=float(iou),
        cldice=float(cld),
        mcc=float(mcc),
        precision=float(precision),
        recall=float(recall),
        auc=float(auc),
        accuracy=float(acc),
    )


# -------------------------
# Utils / config
# -------------------------
def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Data loader (reuse your existing fullimg 3ch pipeline)
# -------------------------
def get_loaders(data_cfg: Dict):
    from src.common_vit.transforms import build_train_transforms, build_eval_transforms
    from src.common_vit.dataset_fullimg_3ch import build_loaders_fullimg_3ch

    train_tf = build_train_transforms()
    eval_tf = build_eval_transforms()

    # norm_yaml optional: fallback to standard location
    norm_yaml = data_cfg.get("norm_yaml", str(Path("/home/infres/diouf-25/prim-project/configs/norm_fullimg.yaml")))

    train_loader, val_loader, test_loader = build_loaders_fullimg_3ch(
        data_cfg=data_cfg,
        split_json=data_cfg["split_json"],
        norm_yaml=norm_yaml,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_tf=train_tf,
        eval_tf=eval_tf,
    )
    return train_loader, val_loader, test_loader



# -------------------------
# Train loop
# -------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    losses = []
    for x, y, *_ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device) -> Tuple[float, Dict[str,float]]:
    model.eval()
    losses = []
    mets_accum = []
    for x, y, *_ in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))
        mets_accum.append(compute_metrics(logits, y))

    loss_mean = float(np.mean(losses)) if losses else float("nan")
    if not mets_accum:
        return loss_mean, {}

    # average metrics
    keys = mets_accum[0].keys()
    mets = {k: float(np.mean([m[k] for m in mets_accum])) for k in keys}
    return loss_mean, mets


def save_ckpt(path: Path, model, optimizer, epoch: int, best_val_loss: float, best_val_dice: float):
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_val_dice": best_val_dice,
        },
        path
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to yaml config")
    ap.add_argument("--loss", required=True, choices=["loss1","loss2","loss3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(cfg["data_yaml"])

    set_seed(args.seed)

    run_name = args.run_name or f"{cfg['run_prefix']}_{args.loss}_seed{args.seed}"
    run_dir = Path(cfg["runs_root"]) / run_name
    ensure_dir(run_dir)

    (run_dir / "best_model").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # loaders
    train_loader, val_loader, test_loader = get_loaders(data_cfg)

    # model
    model = DualBranchDINO(
        vit_name=cfg["vit_name"],
        base_ch=cfg["base_ch"],
        in_chans=cfg["in_chans"],
        out_ch=1,
        pretrained_vit=cfg["pretrained_vit"],
        vit_drop=cfg["vit_drop"],
        vit_freeze=cfg["vit_freeze"],
        use_gating=cfg["use_gating"],
    ).to(cfg["device"])

    # loss
    loss_fn = build_loss(args.loss)

    # optimizer
    lr = cfg["lr"]
    vit_lr = cfg.get("vit_lr", lr * 0.1)

    # different LR groups (ViT usually smaller LR)
    vit_params = list(model.vit.parameters())
    other_params = [p for n,p in model.named_parameters() if not n.startswith("vit.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr},
            {"params": vit_params, "lr": vit_lr},
        ],
        weight_decay=cfg["weight_decay"],
    )

    # scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg.get("min_lr", 1e-6),
    )

    best_val_dice = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        # Optional: unfreeze ViT after some epochs
        if cfg.get("unfreeze_epoch", None) is not None:
            if epoch == int(cfg["unfreeze_epoch"]) and cfg["vit_freeze"]:
                for p in model.vit.parameters():
                    p.requires_grad = True
                cfg["vit_freeze"] = False  # prevent repeated
                print(f"[INFO] Unfroze ViT at epoch {epoch}")

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg["device"])
        val_loss, val_mets = eval_epoch(model, val_loader, loss_fn, cfg["device"])

        val_dice = val_mets.get("dice", float("nan"))

        # selection rule: max val_dice; if tie -> min val_loss
        better = False
        if val_dice > best_val_dice + 1e-12:
            better = True
        elif abs(val_dice - best_val_dice) <= 1e-12 and val_loss < best_val_loss:
            better = True

        # save epoch ckpt (optional)
        save_ckpt(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best_val_loss, best_val_dice)

        if better:
            best_val_dice = float(val_dice)
            best_val_loss = float(val_loss)
            best_epoch = epoch
            save_ckpt(run_dir / "best_model" / "best_model.pth", model, optimizer, epoch, best_val_loss, best_val_dice)

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_mets.get("dice", None),
            "val_iou": val_mets.get("iou", None),
            "val_cldice": val_mets.get("cldice", None),
            "val_mcc": val_mets.get("mcc", None),
            "val_precision": val_mets.get("precision", None),
            "val_recall": val_mets.get("recall", None),
            "val_auc": val_mets.get("auc", None),
            "val_accuracy": val_mets.get("accuracy", None),
            "lr_main": optimizer.param_groups[0]["lr"],
            "lr_vit": optimizer.param_groups[1]["lr"],
            "best_epoch": best_epoch,
        }
        history.append(row)

        print(f"[E{epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} best_epoch={best_epoch}")

    # load best and test
    ckpt = torch.load(run_dir / "best_model" / "best_model.pth", map_location=cfg["device"])
    model.load_state_dict(ckpt["model_state"], strict=True)
    test_loss, test_mets = eval_epoch(model, test_loader, loss_fn, cfg["device"])

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "test_loss": float(test_loss),
        **{f"test_{k}": float(v) for k,v in test_mets.items()}
    }

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[TEST]", summary)


if __name__ == "__main__":
    main()
