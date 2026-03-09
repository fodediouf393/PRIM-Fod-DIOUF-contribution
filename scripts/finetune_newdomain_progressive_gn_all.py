# scripts/finetune_newdomain_progressive_gn_all.py
import argparse
import csv
import json
import os
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


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="experiments/finetune_newdomain_gn_pu_seed0")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--split_json", default="data/splits/split_newdomain_v1.json")
    ap.add_argument("--norm_yaml", default="configs/norm_newdomain.yaml")
    ap.add_argument("--pad_to", type=int, default=832)
    ap.add_argument("--pu_e1", type=int, default=10)
    ap.add_argument("--pu_e2", type=int, default=25)
    ap.add_argument("--gn_groups_max", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    for arch, run_dir in RUNS.items():
        out_dir = out_root / arch
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            print(f"[SKIP] {arch}: already has {summary_path}")
            continue

        cmd = [
            "python3",
            "scripts/finetune_newdomain_progressive_gn.py",
            "--arch", arch,
            "--pretrained_run_dir", run_dir,
            "--seed", "0",
            "--split_json", args.split_json,
            "--norm_yaml", args.norm_yaml,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--pad_to", str(args.pad_to),
            "--pu_e1", str(args.pu_e1),
            "--pu_e2", str(args.pu_e2),
            "--gn_groups_max", str(args.gn_groups_max),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
            "--out_dir", str(out_dir),
        ]

        print("\n[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env, cwd=str(project_root))

    train_rows, test_rows = [], []

    for arch in RUNS.keys():
        summary_path = out_root / arch / "summary.json"
        if not summary_path.exists():
            print(f"[WARN] Missing summary.json for {arch}: {summary_path}")
            continue

        data = json.loads(summary_path.read_text())
        train = data.get("train", {})
        test = data.get("test", {})

        base_cols = {
            "arch": arch,
            "best_epoch": data.get("best_epoch", ""),
            "best_val_loss": data.get("best_val_loss", ""),
        }

        train_rows.append({
            **base_cols,
            "loss": train.get("loss", ""),
            "dice": train.get("dice", ""),
            "iou": train.get("iou", ""),
            "cldice": train.get("cldice", ""),
            "mcc": train.get("mcc", ""),
            "precision": train.get("precision", ""),
            "recall": train.get("recall", ""),
            "auc": train.get("auc", ""),
            "accuracy": train.get("accuracy", ""),
        })

        test_rows.append({
            **base_cols,
            "loss": test.get("loss", ""),
            "dice": test.get("dice", ""),
            "iou": test.get("iou", ""),
            "cldice": test.get("cldice", ""),
            "mcc": test.get("mcc", ""),
            "precision": test.get("precision", ""),
            "recall": test.get("recall", ""),
            "auc": test.get("auc", ""),
            "accuracy": test.get("accuracy", ""),
        })

    train_csv = out_root / "summary_train.csv"
    test_csv = out_root / "summary_test.csv"
    write_csv(train_csv, train_rows)
    write_csv(test_csv, test_rows)

    print("\n[OK] wrote:")
    print(" ", train_csv)
    print(" ", test_csv)


if __name__ == "__main__":
    main()