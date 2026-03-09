# scripts/train_multi.py
import argparse
import csv
import math
import re
from datetime import datetime
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


def mean_ci95(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci = 1.96 * std / math.sqrt(n)
    return mean, ci


def _latest_ckpt(seed_dir: Path) -> str | None:
    ckpt_dir = seed_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    # Match epoch_XXX.pth
    pats = list(ckpt_dir.glob("epoch_*.pth"))
    if not pats:
        return None

    def epoch_num(p: Path) -> int:
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    pats.sort(key=epoch_num)
    return str(pats[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--train", default="configs/train_paper.yaml")
    ap.add_argument("--model", default="configs/unetpp.yaml")
    ap.add_argument("--norm", default="configs/norm.yaml")
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--run_prefix", default="unetpp_DS_3ch_norm")

    # NEW: continue an existing multi-run folder
    ap.add_argument(
        "--root_run_dir",
        default=None,
        help="Existing root runs folder (e.g., experiments/runs/<prefix>_YYYY-MM-DD_HH-MM-SS) to resume/finish."
    )

    args = ap.parse_args()

    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.train)
    model_cfg = load_yaml(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # choose root_out
    if args.root_run_dir is not None:
        root_out = Path(args.root_run_dir)
        if not root_out.exists():
            raise FileNotFoundError(f"--root_run_dir not found: {root_out}")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root_out = Path("experiments") / "runs" / f"{args.run_prefix}_{ts}"
        root_out.mkdir(parents=True, exist_ok=True)

    # snapshot configs (only if missing)
    (root_out / "configs").mkdir(parents=True, exist_ok=True)
    cfg_files = {
        "data.yaml": yaml.safe_dump(data_cfg),
        "train_paper.yaml": yaml.safe_dump(train_cfg),
        "model.yaml": yaml.safe_dump(model_cfg),
        "norm_path.txt": str(args.norm) + "\n",
    }
    for name, content in cfg_files.items():
        p = root_out / "configs" / name
        if not p.exists():
            p.write_text(content)

    deep_sup = False
    if "extra" in model_cfg and isinstance(model_cfg["extra"], dict):
        deep_sup = bool(model_cfg["extra"].get("deep_supervision", False))

    results = []

    for seed in range(args.n_runs):
        seed_dir = root_out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = seed_dir / "metrics.json"
        if metrics_path.exists():
            # already completed (we assume finished)
            print(f"[SKIP] seed_{seed}: metrics.json exists -> already done.")
            # Optionally load summary for CSV
            try:
                import json
                data = json.loads(metrics_path.read_text())
                summary = data.get("summary", {})
                if summary:
                    results.append({"seed": seed, **summary})
            except Exception:
                pass
            continue

        # resume if ckpt exists
        resume_ckpt = _latest_ckpt(seed_dir)
        if resume_ckpt is not None:
            print(f"[RESUME] seed_{seed}: using {resume_ckpt}")
        else:
            print(f"[START] seed_{seed}: no checkpoint found, starting fresh")

        set_seed(seed)

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

        summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            out_dir=str(seed_dir),
            epochs=train_cfg["epochs"],
            log_every=train_cfg["log_every"],
            skel_iters=train_cfg["skel_iters"],
            resume_ckpt=resume_ckpt,
        )

        results.append({"seed": seed, **summary})

    # ---- write global summary (only for seeds we have)
    keys = [
        "test_loss",
        "test_dice",
        "test_iou",
        "test_mcc",
        "test_precision",
        "test_recall",
        "test_auc",
        "test_accuracy",
    ]

    # compute mean/ci from available results
    means, cis = {}, {}
    for k in keys:
        vals = []
        for r in results:
            v = r.get(k, None)
            if v is None:
                continue
            vals.append(float(v))
        if vals:
            means[k], cis[k] = mean_ci95(vals)
        else:
            means[k], cis[k] = float("nan"), float("nan")

    summary_csv = Path("experiments") / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seed",
        "best_epoch",
        "best_val_loss",
        "test_loss",
        "test_dice",
        "test_iou",
        "test_mcc",
        "test_precision",
        "test_recall",
        "test_auc",
        "test_accuracy",
    ]

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # sort by seed (ints first)
        for r in sorted(results, key=lambda x: x["seed"] if isinstance(x["seed"], int) else 10**9):
            w.writerow({
                "seed": r["seed"],
                "best_epoch": r.get("best_epoch", ""),
                "best_val_loss": r.get("best_val_loss", ""),

                "test_loss": r.get("test_loss", ""),
                "test_dice": r.get("test_dice", ""),
                "test_iou": r.get("test_iou", ""),
                "test_mcc": r.get("test_mcc", ""),
                "test_precision": r.get("test_precision", ""),
                "test_recall": r.get("test_recall", ""),
                "test_auc": r.get("test_auc", ""),
                "test_accuracy": r.get("test_accuracy", ""),
            })

        w.writerow({"seed": "MEAN", **{k: means[k] for k in keys}})
        w.writerow({"seed": "CI95", **{k: cis[k] for k in keys}})

    print(f"\n[OK] Wrote global summary: {summary_csv}")
    print("All runs are under:", root_out)


if __name__ == "__main__":
    main()
