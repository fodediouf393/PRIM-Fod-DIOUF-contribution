import json
import random
from pathlib import Path
import yaml

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_yaml("configs/data_fullimg.yaml")
    full_dir = Path(cfg["full_dir"])
    out = Path(cfg["split_json"])
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in full_dir.iterdir() if p.is_file()])
    # keep only typical image files
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

    random.seed(0)
    random.shuffle(files)

    n = len(files)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    split = {"train": train, "val": val, "test": test}
    out.write_text(json.dumps(split, indent=2))
    print(f"[OK] Wrote split -> {out} (n={n}, train={len(train)}, val={len(val)}, test={len(test)})")

if __name__ == "__main__":
    main()
