# src/architectures/UnetBased/train.py
import argparse
from pathlib import Path
import yaml
import torch

from src.common.seed import set_seed
from src.common.datamodule import build_loaders_3ch
from src.common.transforms import build_train_transforms, build_eval_transforms
from src.common.optim import build_optimizer_and_scheduler
from src.common.engine import train_model

from src.architectures.UnetBased.models.unetpp import UNetPlusPlus


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--train", default="configs/train_paper.yaml")
    ap.add_argument("--model", default="configs/unetpp.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", default="configs/norm.yaml")

    # NEW: resume support
    ap.add_argument("--run_dir", default=None, help="Existing run directory to continue writing into")
    ap.add_argument("--resume_ckpt", default=None, help="Checkpoint path (.pth) to resume from")

    args = ap.parse_args()

    set_seed(args.seed)

    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.train)
    model_cfg = load_yaml(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = build_train_transforms()
    eval_tf = build_eval_transforms()

    train_loader, val_loader, test_loader, _ = build_loaders_3ch(
        split_json=data_cfg["split_json"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_transform=train_tf,
        val_transform=eval_tf,
        test_transform=eval_tf,
        norm_yaml=args.norm,
    )

    deep_sup = False
    if "extra" in model_cfg and isinstance(model_cfg["extra"], dict):
        deep_sup = bool(model_cfg["extra"].get("deep_supervision", False))

    model = UNetPlusPlus(
        in_channels=model_cfg["in_channels"],
        n_classes=model_cfg["n_classes"],
        base=model_cfg["base"],
        deep_supervision=deep_sup,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_start=train_cfg["lr_start"],
        lr_max=train_cfg["lr_max"],
        warmup_epochs=train_cfg["warmup_epochs"],
        weight_decay=train_cfg["weight_decay"],
    )

    if args.run_dir is None:
        raise RuntimeError("Pour reprendre un run, tu dois fournir --run_dir (dossier existant du seed).")

    out_dir = Path(args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # configs (ne pas écraser si déjà présents)
    (out_dir / "configs").mkdir(exist_ok=True)
    if not (out_dir / "configs" / "data.yaml").exists():
        (out_dir / "configs" / "data.yaml").write_text(yaml.safe_dump(data_cfg))
        (out_dir / "configs" / "train_paper.yaml").write_text(yaml.safe_dump(train_cfg))
        (out_dir / "configs" / "model.yaml").write_text(yaml.safe_dump(model_cfg))
        (out_dir / "configs" / "norm_path.txt").write_text(str(args.norm) + "\n")

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        out_dir=str(out_dir),
        epochs=train_cfg["epochs"],
        log_every=train_cfg["log_every"],
        skel_iters=train_cfg["skel_iters"],
        resume_ckpt=args.resume_ckpt,
    )

    print("\nRun saved to:", out_dir)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
