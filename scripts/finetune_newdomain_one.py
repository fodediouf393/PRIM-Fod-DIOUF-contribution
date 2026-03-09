import argparse
from pathlib import Path
import yaml
import torch

from src.common.seed import set_seed
from src.common.optim import build_optimizer_and_scheduler
from src.common.engine import train_model
from src.common.datamodule_finetune import build_loaders_newdomain_3ch
from src.common.transforms_newdomain import build_train_transforms_newdomain, build_eval_transforms_newdomain

# models
from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet
from src.architectures.UnetBased.models.ds_unet import DSUNet


def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def build_model(arch: str, in_channels: int, n_classes: int, base: int):
    a = arch.lower()
    if a == "unet":
        return UNet(in_channels=in_channels, n_classes=n_classes, base=base)
    if a == "unetpp":
        return UNetPlusPlus(in_channels=in_channels, n_classes=n_classes, base=base, deep_supervision=False)
    if a == "unetpp_ds":
        return UNetPlusPlus(in_channels=in_channels, n_classes=n_classes, base=base, deep_supervision=True)
    if a == "unet3plus":
        return UNet3Plus(in_channels=in_channels, n_classes=n_classes, base=base)
    if a == "attention_unet":
        return AttentionUNet(in_channels=in_channels, n_classes=n_classes, base=base)
    if a == "resunet":
        return ResUNet(in_channels=in_channels, n_classes=n_classes, base=base)
    if a == "r2unet":
        return R2UNet(in_channels=in_channels, n_classes=n_classes, base=base, t=2)
    if a == "ds_unet":
        return DSUNet(in_channels=in_channels, n_classes=n_classes, base=base)
    raise ValueError(f"Unknown arch: {arch}")


def freeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def set_bn_eval(module: torch.nn.Module):
    # useful when batch is small and encoder is frozen
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()


def apply_freeze_encoder(model: torch.nn.Module, arch: str):
    """
    Best-effort encoder freeze for our implementations.
    We freeze known encoder modules per arch; decoder/head remain trainable.
    """
    arch = arch.lower()

    # First freeze everything, then unfreeze decoder/head
    freeze_all(model)

    # UNet / DSUNet / AttentionUNet usually have up* and out* as decoder/head
    decoder_keys = ["up", "dec", "out", "final", "outc"]

    # Unfreeze decoder/head parameters by name heuristic
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in decoder_keys):
            p.requires_grad = True

    # Also keep last conv head trainable even if name differs
    # (fallback: unfreeze last 2 parameter tensors)
    params = list(model.parameters())
    for p in params[-2:]:
        p.requires_grad = True

    # Put BN in encoder to eval to avoid BN instability with small batch
    model.apply(set_bn_eval)


def apply_progressive_unfreeze(model: torch.nn.Module, arch: str, epoch: int, e1: int, e2: int):
    """
    Progressive scheme:
      epoch < e1 : freeze encoder
      e1 <= epoch < e2 : unfreeze "deep encoder" (bottleneck / last enc blocks) + decoder
      epoch >= e2 : unfreeze all
    """
    if epoch < e1:
        apply_freeze_encoder(model, arch)
        return

    if epoch < e2:
        # start from freeze_encoder then unfreeze deeper layers by name heuristic
        apply_freeze_encoder(model, arch)
        # unfreeze layers that look "deep"
        deep_keys = ["down3", "down4", "enc4", "enc5", "bottleneck", "conv4_0", "conv3_0"]
        for name, p in model.named_parameters():
            if any(k in name.lower() for k in deep_keys):
                p.requires_grad = True
        return

    unfreeze_all(model)


def load_pretrained_weights(model: torch.nn.Module, pretrained_best_pth: Path):
    ckpt = torch.load(pretrained_best_pth, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    help="unet|unetpp|unetpp_ds|unet3plus|attention_unet|resunet|r2unet|ds_unet")
    ap.add_argument("--strategy", required=True, choices=["freeze_encoder", "full", "progressive_unfreeze"])

    ap.add_argument("--pretrained_run_dir", required=True,
                    help="Path to the OCTA run directory (the one that contains seed_0/best_model/best_model.pth)")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--split_json", default="data/splits/split_newdomain_v1.json")
    ap.add_argument("--norm_yaml", default="configs/norm_newdomain.yaml")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--lr_start", type=float, default=1e-5)
    ap.add_argument("--lr_max", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--pad_to", type=int, default=832)

    # progressive unfreeze milestones
    ap.add_argument("--pu_e1", type=int, default=10)
    ap.add_argument("--pu_e2", type=int, default=25)

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # build model (always 3ch)
    model = build_model(args.arch, in_channels=3, n_classes=1, base=42).to(device)

    # load pretrained seed_0 best
    pretrained_best = Path(args.pretrained_run_dir) / "seed_0" / "best_model" / "best_model.pth"
    if not pretrained_best.exists():
        # some of your runs might have best directly under run_dir (no seed_0)
        alt = Path(args.pretrained_run_dir) / "best_model" / "best_model.pth"
        pretrained_best = alt if alt.exists() else pretrained_best
    if not pretrained_best.exists():
        raise FileNotFoundError(f"Cannot find pretrained best_model.pth under {args.pretrained_run_dir}")

    missing, unexpected = load_pretrained_weights(model, pretrained_best)
    print("[PRETRAIN] loaded:", pretrained_best)
    if missing:
        print("[PRETRAIN] missing keys (ok with strict=False):", len(missing))
    if unexpected:
        print("[PRETRAIN] unexpected keys:", len(unexpected))

    # strategy initial apply
    if args.strategy == "freeze_encoder":
        apply_freeze_encoder(model, args.arch)
    elif args.strategy == "full":
        unfreeze_all(model)
    elif args.strategy == "progressive_unfreeze":
        apply_progressive_unfreeze(model, args.arch, epoch=0, e1=args.pu_e1, e2=args.pu_e2)

    # optimizer/scheduler only on trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr_start, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: min(1.0, (ep + 1) / max(1, args.warmup_epochs))  # warmup then constant
    )
    # after warmup, set to lr_max
    for pg in optimizer.param_groups:
        pg["lr"] = args.lr_start

    # custom training loop to handle progressive unfreeze
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We reuse train_model, but progressive unfreeze needs intervention per epoch.
    # So for progressive_unfreeze we run epochs manually (simple and robust).
    if args.strategy != "progressive_unfreeze":
        # Use existing train_model (handles best ckpt + test eval + metrics.json)
        # Build a scheduler that warms up to lr_max then stays constant:
        def build_warmup_to_max(opt):
            class WarmupToMax(torch.optim.lr_scheduler._LRScheduler):
                def __init__(self, optimizer, lr_start, lr_max, warmup_epochs, last_epoch=-1):
                    self.lr_start = lr_start
                    self.lr_max = lr_max
                    self.warmup_epochs = max(1, warmup_epochs)
                    super().__init__(optimizer, last_epoch)

                def get_lr(self):
                    e = self.last_epoch + 1
                    if e <= self.warmup_epochs:
                        t = e / self.warmup_epochs
                        lr = self.lr_start + t * (self.lr_max - self.lr_start)
                    else:
                        lr = self.lr_max
                    return [lr for _ in self.base_lrs]

            return WarmupToMax(opt, args.lr_start, args.lr_max, args.warmup_epochs)

        scheduler = build_warmup_to_max(optimizer)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            out_dir=str(out_dir),
            epochs=args.epochs,
            log_every=5,
            skel_iters=10,
        )
        return

    # progressive_unfreeze manual loop (still uses same loss/metrics logic inside engine functions)
    # We’ll import needed helpers from engine
    from src.common.engine import run_one_epoch_train, run_eval

    best_val = float("inf")
    best_epoch = -1
    (out_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    (out_dir / "best_model").mkdir(exist_ok=True, parents=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        apply_progressive_unfreeze(model, args.arch, epoch=epoch-1, e1=args.pu_e1, e2=args.pu_e2)

        # rebuild optimizer when new params become trainable (simple approach)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr_start, weight_decay=args.weight_decay)

        # warmup schedule
        class WarmupToMax(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, lr_start, lr_max, warmup_epochs, last_epoch=-1):
                self.lr_start = lr_start
                self.lr_max = lr_max
                self.warmup_epochs = max(1, warmup_epochs)
                super().__init__(optimizer, last_epoch)
            def get_lr(self):
                e = self.last_epoch + 1
                if e <= self.warmup_epochs:
                    t = e / self.warmup_epochs
                    lr = self.lr_start + t * (self.lr_max - self.lr_start)
                else:
                    lr = self.lr_max
                return [lr for _ in self.base_lrs]

        scheduler = WarmupToMax(optimizer, args.lr_start, args.lr_max, args.warmup_epochs, last_epoch=epoch-2)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        train_loss = run_one_epoch_train(model, train_loader, optimizer, device, skel_iters=10)
        val_metrics = run_eval(model, val_loader, device, skel_iters=10)

        # save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                out_dir / "checkpoints" / f"epoch_{epoch:03d}.pth"
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       out_dir / "best_model" / "best_model.pth")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], **{f"val_{k}": v for k, v in val_metrics.items()}})

        print(f"[PU][{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} lr={lr:.2e}")

    # final test on best
    best_ckpt = torch.load(out_dir / "best_model" / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = run_eval(model, test_loader, device, skel_iters=10)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
        "test_mcc": test_metrics["mcc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
    }

    with open(out_dir / "metrics.json", "w") as f:
        yaml.safe_dump({"history": history, "summary": summary}, f)

    print("[DONE] summary:", summary)


if __name__ == "__main__":
    main()