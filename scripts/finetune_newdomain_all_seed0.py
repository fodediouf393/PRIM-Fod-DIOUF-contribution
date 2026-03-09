import argparse
import csv
import json
from pathlib import Path
import subprocess


RUNS = {
    "unet": "experiments/runs/unet_3ch_norm_2026-02-08_12-28-02",
    "unetpp": "experiments/runs/unetpp_3ch_norm_2026-02-08_20-22-43",
    "unetpp_ds": "experiments/runs/unetpp_DS_3ch_norm_2026-02-11_12-56-57",
    "unet3plus": "experiments/runs/unet3plus_3ch_norm_2026-02-09_20-35-04",
    "attention_unet": "experiments/runs/attention_unet_3ch_norm_2026-02-09_12-45-34",
    "resunet": "experiments/runs/resunet_3ch_norm_2026-02-10_13-52-54",
    "r2unet": "experiments/runs/r2unet_3ch_norm_2026-02-10_20-02-47",
}

STRATEGIES = ["freeze_encoder", "full", "progressive_unfreeze"]


def read_metrics(out_dir: Path):
    p = out_dir / "metrics.json"
    if not p.exists():
        return None
    try:
        # metrics.json from engine is json; from progressive_unfreeze branch we wrote yaml.safe_dump
        txt = p.read_text()
        if txt.strip().startswith("{"):
            data = json.loads(txt)
        else:
            import yaml
            data = yaml.safe_load(txt)
        return data.get("summary", None)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out_root", default="experiments/finetune_newdomain_seed0")
    ap.add_argument("--split_json", default="data/splits/split_newdomain_v1.json")
    ap.add_argument("--norm_yaml", default="configs/norm_newdomain.yaml")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Run all
    for arch, run_dir in RUNS.items():
        for strat in STRATEGIES:
            out_dir = out_root / arch / strat
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3", "scripts/finetune_newdomain_one.py",
                "--arch", arch,
                "--strategy", strat,
                "--pretrained_run_dir", run_dir,
                "--seed", "0",
                "--split_json", args.split_json,
                "--norm_yaml", args.norm_yaml,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--out_dir", str(out_dir),
            ]
            print("\n[RUN]", " ".join(cmd))
            subprocess.run(cmd, check=True)

    # Collect CSV
    rows = []
    for arch in RUNS.keys():
        for strat in STRATEGIES:
            out_dir = out_root / arch / strat
            summ = read_metrics(out_dir)
            if not summ:
                continue
            rows.append({
                "arch": arch,
                "strategy": strat,
                "test_dice": summ.get("test_dice", ""),
                "test_iou": summ.get("test_iou", ""),
                "test_mcc": summ.get("test_mcc", ""),
                "test_precision": summ.get("test_precision", ""),
                "test_recall": summ.get("test_recall", ""),
                "test_auc": summ.get("test_auc", ""),
                "test_accuracy": summ.get("test_accuracy", ""),
                "best_epoch": summ.get("best_epoch", ""),
                "best_val_loss": summ.get("best_val_loss", ""),
            })

    csv_path = out_root / "summary_finetune.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\n[OK] wrote", csv_path)


if __name__ == "__main__":
    main()