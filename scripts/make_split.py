# scripts/make_split.py
import json
import re
from pathlib import Path

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    project_root = Path(__file__).resolve().parents[1]

    # RENAMED
    raw_dir = project_root / "data" / "overlap_patches_ilm_opl"
    lab_dir = project_root / "data" / "overlap_patches_labels"
    out_path = project_root / "data" / "splits" / "split_v1.json"

    assert raw_dir.exists(), f"Missing: {raw_dir}"
    assert lab_dir.exists(), f"Missing: {lab_dir}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exts = {".bmp", ".png", ".jpg", ".jpeg"}
    raw_files = [p for p in raw_dir.iterdir() if p.suffix.lower() in exts]
    raw_files = sorted(raw_files, key=lambda p: natural_key(p.name))

    paired = []
    for rp in raw_files:
        if (lab_dir / rp.name).exists():
            paired.append(rp.name)

    if len(paired) == 0:
        raise RuntimeError("No paired raw/label files found (same filename expected).")

    n = len(paired)
    train_ratio, val_ratio = 0.7, 0.1
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = paired[:n_train]
    val = paired[n_train:n_train + n_val]
    test = paired[n_train + n_val:]

    split = {
        "raw_dir": str(raw_dir),
        "label_dir": str(lab_dir),
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train": train,
        "val": val,
        "test": test,
    }

    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"[OK] Wrote split: {out_path}")
    print(f"Total={n} Train={len(train)} Val={len(val)} Test={len(test)}")

if __name__ == "__main__":
    main()
