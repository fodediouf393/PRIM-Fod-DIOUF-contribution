# src/common_vit/engine.py
import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from .losses import combined_loss
from .metrics import compute_metrics_from_logits


def _atomic_save(obj, path: Path):
    """
    Safer save: write tmp then rename (atomic).
    Helps on network FS / quota edge cases.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    device,
    out_dir: str,
    epochs: int,
    log_every: int,
    loss_cfg: dict,
    save_every: int = 10,
    resume_ckpt: str | None = None,
):
    """
    Train + select best checkpoint by:
      1) max val_dice
      2) tie-breaker: min val_loss
    Saves:
      - checkpoints/last.pth each epoch
      - checkpoints/epoch_XXX.pth every save_every epochs (and epoch 1)
      - best_model/best_model.pth (best by rule above)
      - metrics.json (history + summary)
      - TensorBoard logs
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    best_dir = out_dir / "best_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    # ---- BEST selection: maximize val_dice, tie-break on val_loss
    best_val_dice = float("-inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_path = best_dir / "best_model.pth"

    history = []

    # ---- Resume (optional)
    start_epoch = 1
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        try:
            scheduler.last_epoch = start_epoch - 2  # because we call scheduler.step() per epoch
        except Exception:
            pass
        print(f"[RESUME] {resume_ckpt} -> start_epoch={start_epoch}")

    def run_eval(loader):
        model.eval()
        total_loss = 0.0
        n = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = combined_loss(logits, y, **loss_cfg)

                bs = x.size(0)
                total_loss += float(loss.item()) * bs
                n += bs

                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        mets = compute_metrics_from_logits(logits_cat, targets_cat, thr=0.5)

        return {"loss": total_loss / max(1, n), **mets}

    for epoch in range(start_epoch, epochs + 1):
        # ---- Train
        model.train()
        total_train = 0.0
        n = 0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = combined_loss(logits, y, **loss_cfg)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_train += float(loss.item()) * bs
            n += bs

        train_loss = total_train / max(1, n)

        # ---- Val
        val = run_eval(val_loader)

        # ---- Step scheduler
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # ---- Always save last
        _atomic_save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val["loss"],
                "val_dice": val["dice"],
                "lr": lr,
            },
            ckpt_dir / "last.pth",
        )

        # ---- Periodic checkpoint
        if epoch == 1 or (epoch % save_every) == 0:
            _atomic_save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val["loss"],
                    "val_dice": val["dice"],
                    "lr": lr,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pth",
            )

        # ---- BEST checkpoint selection:
        # 1) higher dice wins
        # 2) if dice equal (within eps), lower val_loss wins
        eps = 1e-12
        is_better = False
        if val["dice"] > best_val_dice + eps:
            is_better = True
        elif abs(val["dice"] - best_val_dice) <= eps and val["loss"] < best_val_loss:
            is_better = True

        if is_better:
            best_val_dice = float(val["dice"])
            best_val_loss = float(val["loss"])
            best_epoch = epoch
            _atomic_save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_val_dice": best_val_dice, "best_val_loss": best_val_loss},
                best_path,
            )

        # ---- TensorBoard
        if epoch == 1 or (epoch % log_every) == 0:
            writer.add_scalar("train/loss", train_loss, epoch)
            for k in ["loss", "dice", "iou", "mcc", "precision", "recall", "auc", "accuracy"]:
                writer.add_scalar(f"val/{k}", val[k], epoch)
            writer.add_scalar("lr", lr, epoch)

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "lr": lr, **{f"val_{k}": v for k, v in val.items()}}
        )

        print(
            f"[{epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val['loss']:.4f} "
            f"val_dice={val['dice']:.6f} "
            f"val_iou={val['iou']:.4f} "
            f"val_mcc={val['mcc']:.4f} "
            f"val_auc={val['auc']} "
            f"lr={lr:.3g}"
        )

    writer.close()

    # ---- Test best
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test = run_eval(test_loader)

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_loss": best_val_loss,

        "test_loss": test["loss"],
        "test_dice": test["dice"],
        "test_iou": test["iou"],
        "test_mcc": test["mcc"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_auc": test["auc"],
        "test_accuracy": test["accuracy"],
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"history": history, "summary": summary}, f, indent=2)

    print("[BEST] epoch=", best_epoch, " val_dice=", best_val_dice, " val_loss=", best_val_loss)
    print("[TEST]", summary)
    return summary
