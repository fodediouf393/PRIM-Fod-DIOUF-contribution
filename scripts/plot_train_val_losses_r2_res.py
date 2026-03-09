# scripts/plot_train_val_losses_r2_res.py
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


R2_RUN = Path("experiments/runs/r2unet_3ch_norm_2026-02-10_20-02-47")
RES_RUN = Path("experiments/runs/resunet_3ch_norm_2026-02-10_13-52-54")
OUT_DIR = Path("experiments/plots")


def _find_seed_dirs(run_dir: Path):
    seeds = sorted([p for p in run_dir.glob("seed_*") if p.is_dir()])
    return seeds if seeds else [run_dir]


def _load_history(seed_dir: Path):
    metrics_path = seed_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {seed_dir}")
    data = json.loads(metrics_path.read_text())
    hist = data.get("history", [])
    if not hist:
        raise ValueError(f"Empty history in {metrics_path}")
    return hist


def _extract_losses(history):
    epochs, train_losses, val_losses = [], [], []
    for row in history:
        epochs.append(int(row["epoch"]))
        train_losses.append(float(row.get("train_loss", np.nan)))

        # Most common storage in your history:
        if "val_loss" in row:
            val_losses.append(float(row["val_loss"]))
        elif "val_metrics" in row and isinstance(row["val_metrics"], dict) and "loss" in row["val_metrics"]:
            val_losses.append(float(row["val_metrics"]["loss"]))
        else:
            val_losses.append(np.nan)

    return np.array(epochs), np.array(train_losses), np.array(val_losses)


def _mean_curve(curves):
    mat = np.vstack(curves)
    return np.nanmean(mat, axis=0)


def plot_run(run_dir: Path, title: str, out_png: Path):
    seed_dirs = _find_seed_dirs(run_dir)

    epochs_ref = None
    train_curves, val_curves = [], []
    used = 0

    for sd in seed_dirs:
        try:
            hist = _load_history(sd)
            epochs, tr, va = _extract_losses(hist)

            if epochs_ref is None:
                epochs_ref = epochs
            else:
                if len(epochs) != len(epochs_ref) or not np.all(epochs == epochs_ref):
                    print(f"[WARN] Epoch grid mismatch in {sd.name}, skipping it.")
                    continue

            train_curves.append(tr)
            val_curves.append(va)
            used += 1
        except Exception as e:
            print(f"[WARN] skipping {sd}: {e}")

    if used == 0:
        raise RuntimeError(f"No usable seeds found in {run_dir}")

    train_mean = _mean_curve(train_curves)
    val_mean = _mean_curve(val_curves)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epochs_ref, train_mean, label="train_loss (mean)")
    plt.plot(epochs_ref, val_mean, label="val_loss (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} — Train vs Val Loss (mean over {used} seeds)")
    plt.legend()

    # ✅ GRID (major + minor)
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.6)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] Saved: {out_png}")


def main():
    plot_run(R2_RUN, "R2UNet", OUT_DIR / "r2unet_train_val_loss.png")
    plot_run(RES_RUN, "ResUNet", OUT_DIR / "resunet_train_val_loss.png")


if __name__ == "__main__":
    main()
