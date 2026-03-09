# src/common/engine.py
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import combined_paper_loss
from .metrics import compute_metrics_from_probs, cldice_from_probs


def _compute_ds_loss(outputs, masks, skel_iters: int):
    if isinstance(outputs, (list, tuple)):
        losses = [combined_paper_loss(o, masks, skel_iters=skel_iters) for o in outputs]
        return sum(losses) / len(losses)
    return combined_paper_loss(outputs, masks, skel_iters=skel_iters)


def _select_logits_for_metrics(outputs):
    if isinstance(outputs, (list, tuple)):
        return outputs[-1]
    return outputs


def run_one_epoch_train(model, loader, optimizer, device, skel_iters: int, desc: str = "train") -> float:
    model.train()
    total_loss = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = _compute_ds_loss(outputs, masks, skel_iters=skel_iters)

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, n)


@torch.no_grad()
def run_eval(model, loader, device, skel_iters: int, desc: str = "eval") -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    all_probs = []
    all_targets = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = _compute_ds_loss(outputs, masks, skel_iters=skel_iters)

        logits = _select_logits_for_metrics(outputs)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach())
        all_targets.append(masks.detach())

        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    probs_cat = torch.cat(all_probs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    metrics = compute_metrics_from_probs(probs_cat, targets_cat, thr=0.5)
    cld = cldice_from_probs(probs_cat, targets_cat, thr=0.5, skel_iters=skel_iters)

    return {
        "loss": total_loss / max(1, n),
        **metrics,
        "cldice": float(cld),
    }