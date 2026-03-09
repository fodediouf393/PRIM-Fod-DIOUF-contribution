import json
import re
from pathlib import Path

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def main():
    project_root = Path(__file__).resolve().parents[1]

    img_dir = project_root / "data" / "ens_tif"          # <-- adapte si besoin
    mask_dir = project_root / "data" / "ens_mask_png"    # <-- tu l’as donné

    out_path = project_root / "data" / "splits" / "split_newdomain_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tifs = sorted(list(img_dir.glob("*.tif")), key=lambda p: natural_key(p.name))
    if not tifs:
        raise RuntimeError(f"No .tif in {img_dir}")

    paired = []
    for p in tifs:
        stem = p.stem
        if (mask_dir / f"{stem}_mask.png").exists():
            paired.append(stem)

    if not paired:
        raise RuntimeError("No (image, mask) pairs found.")

    n = len(paired)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    split = {
        "img_dir": str(img_dir),
        "clahe_dir": str(project_root / "data" / "CLAHE"),
        "dog_dir": str(project_root / "data" / "DOG"),
        "mask_dir": str(mask_dir),
        "n_total": n,
        "train": paired[:n_train],
        "val": paired[n_train:n_train+n_val],
        "test": paired[n_train+n_val:],
    }

    out_path.write_text(json.dumps(split, indent=2))
    print(f"[OK] wrote {out_path}")
    print(f"Total={n}, Train={len(split['train'])}, Val={len(split['val'])}, Test={len(split['test'])}")

if __name__ == "__main__":
    main()