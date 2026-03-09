import json
import re
from pathlib import Path

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_ids_from_dir(folder: Path):
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    names = sorted([p.name for p in files], key=natural_key)
    return names

def main():
    root = Path(__file__).resolve().parents[1]

    full_dir   = root / "data" / "OCTA(FULL)"
    ilm_dir    = root / "data" / "OCTA(ILM_OPL)"
    opl_dir    = root / "data" / "OCTA(OPL_BM)"
    label_dir  = root / "data" / "GT_Capillary"

    out_path = root / "data" / "splits" / "split_fullimg_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    assert full_dir.exists(),  f"Missing: {full_dir}"
    assert ilm_dir.exists(),   f"Missing: {ilm_dir}"
    assert opl_dir.exists(),   f"Missing: {opl_dir}"
    assert label_dir.exists(), f"Missing: {label_dir}"

    # on prend les IDs depuis FULL, puis on vérifie la présence partout
    candidates = list_ids_from_dir(full_dir)

    paired = []
    for name in candidates:
        if (ilm_dir / name).exists() and (opl_dir / name).exists() and (label_dir / name).exists():
            paired.append(name)

    if len(paired) == 0:
        raise RuntimeError("No paired full-image triplets found (FULL/ILM_OPL/OPL_BM + label).")

    # Split simple 70/10/20 (tu pourras remplacer par split paper-like si besoin)
    n = len(paired)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    train = paired[:n_train]
    val   = paired[n_train:n_train + n_val]
    test  = paired[n_train + n_val:]

    split = {
        "full_dir": str(full_dir),
        "ilm_opl_dir": str(ilm_dir),
        "opl_bm_dir": str(opl_dir),
        "label_dir": str(label_dir),

        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),

        "train": train,
        "val": val,
        "test": test,
    }

    out_path.write_text(json.dumps(split, indent=2))
    print(f"[OK] Wrote {out_path}")
    print(f"Total={n} Train={len(train)} Val={len(val)} Test={len(test)}")

if __name__ == "__main__":
    main()
