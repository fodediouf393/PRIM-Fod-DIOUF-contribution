# src/architectures/UnetBased/test.py
import argparse
from pathlib import Path
import yaml
import torch

from src.common.datamodule import build_loaders_3ch
from src.common.transforms import build_eval_transforms
from src.common.engine import run_eval

from src.architectures.UnetBased.models.unetpp import UNetPlusPlus


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--norm", default="configs/norm.yaml")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_cfg = load_yaml(run_dir / "configs" / "data.yaml")
    train_cfg = load_yaml(run_dir / "configs" / "train_paper.yaml")
    model_cfg = load_yaml(run_dir / "configs" / "model.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_tf = build_eval_transforms()
    _, _, test_loader, _ = build_loaders_3ch(
        split_json=data_cfg["split_json"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_transform=None,
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

    best_path = run_dir / "best_model" / "best_model.pth"
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    metrics = run_eval(model, test_loader, device, skel_iters=train_cfg["skel_iters"])
    print(
        "[TEST] "
        f"loss={metrics['loss']:.4f} "
        f"dice={metrics['dice']:.4f} "
        f"iou={metrics['iou']:.4f} "
        f"mcc={metrics['mcc']:.4f} "
        f"prec={metrics['precision']:.4f} "
        f"rec={metrics['recall']:.4f} "
        f"auc={metrics['auc']:.4f} "
        f"acc={metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
