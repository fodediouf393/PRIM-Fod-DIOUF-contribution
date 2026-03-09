# scripts/plot_history_curve.py
import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_one(run_dir: Path):
    hist = json.loads((run_dir / "history.json").read_text())
    epochs = [h["epoch"] for h in hist]
    train = [h["train_loss"] for h in hist]
    val = [h["val_loss"] for h in hist]

    plt.figure()
    plt.plot(epochs, train, label="train_loss", color="tab:blue")
    plt.plot(epochs, val, label="val_loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(run_dir.name)
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.6)
    plt.grid(True, which="minor", linestyle="--", linewidth=0.4, alpha=0.35)
    plt.tight_layout()
    plt.savefig(run_dir / f"{run_dir.name}_train_val_loss.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    root = Path("experiments/finetune_newdomain_gn_pu_seed0_curves")
    plot_one(root / "r2unet")
    plot_one(root / "resunet")
    print("[OK] saved in each run folder")