import cmd
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/infres/diouf-25/prim-project")
import sys
PYTHON = sys.executable

TARGET_ROOT = "data/capillaire_langevin_512_pseudo3dirs"

RUNS = [
    "experiments/runs/unet_3ch_norm_2026-02-08_12-28-02",
    "experiments/runs/unetpp_3ch_norm_2026-02-08_20-22-43",
    "experiments/runs/unet3plus_3ch_norm_2026-02-09_20-35-04",
    "experiments/runs/resunet_3ch_norm_2026-02-10_13-52-54",
    "experiments/runs/r2unet_3ch_norm_2026-02-10_20-02-47",
    "experiments/runs/attention_unet_3ch_norm_2026-02-09_12-45-34",
    "experiments/runs/unetpp_DS_3ch_norm_2026-02-11_12-56-57",
]

#  20 epochs
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4
POS_THR = 0.9
NEG_THR = 0.1


def main():
    for r in RUNS:
        run_root = PROJECT_ROOT / r
        seed0 = run_root / "seed_0"

        if not seed0.exists():
            print(f"[SKIP] seed_0 not found: {seed0}")
            continue

        best = seed0 / "best_model" / "best_model.pth"
        if not best.exists():
            print(f"[SKIP] best_model not found: {best}")
            continue

        cmd = [
            PYTHON, "scripts/target_uda/uda_self_training.py",
            "--source_run_dir", str(seed0),
            "--target_root", TARGET_ROOT,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--pos_thr", str(POS_THR),
            "--neg_thr", str(NEG_THR),
        ]

        print("\n[RUN]", " ".join(cmd))
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)

    print("\n[OK] Finished UDA self-training for 7 models.")


if __name__ == "__main__":
    main()