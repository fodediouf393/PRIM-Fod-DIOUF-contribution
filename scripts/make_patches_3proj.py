# scripts/make_patches_3proj.py
import re
from pathlib import Path
import cv2
import numpy as np

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def list_images(folder: Path):
    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    return sorted(files, key=lambda p: natural_key(p.name))

def sliding_positions(L: int, window: int, stride: int):
    """
    For L=400, window=256, stride=144 -> [0, 144]
    General: last patch is forced to end at L if needed (safe).
    """
    if L < window:
        return [0]
    pos = list(range(0, L - window + 1, stride))
    if pos[-1] != L - window:
        pos.append(L - window)
    return pos

def make_patches_for_folder(
    src_folder: Path,
    dst_folder: Path,
    ref_count: int,
    patch_size: int = 256,
    stride: int = 144,
    force_gray: bool = True,
):
    dst_folder.mkdir(parents=True, exist_ok=True)

    src_files = list_images(src_folder)
    if len(src_files) == 0:
        raise RuntimeError(f"No images found in {src_folder}")

    out_idx = 1
    for img_path in src_files:
        # read
        if force_gray:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Could not read: {img_path}")

        H, W = img.shape[:2]
        ys = sliding_positions(H, patch_size, stride)
        xs = sliding_positions(W, patch_size, stride)

        # order: top->bottom, left->right
        for y in ys:
            for x in xs:
                patch = img[y:y+patch_size, x:x+patch_size]
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    # Shouldn't happen with positions() but safety
                    continue

                out_path = dst_folder / f"{out_idx}.bmp"
                cv2.imwrite(str(out_path), patch)
                out_idx += 1

                if out_idx > ref_count:
                    # stop if we reached reference
                    break
            if out_idx > ref_count:
                break
        if out_idx > ref_count:
            break

    produced = out_idx - 1
    if produced != ref_count:
        raise RuntimeError(
            f"Produced {produced} patches in {dst_folder}, expected {ref_count}. "
            f"Check number of source images or sliding params."
        )

    print(f"[OK] {dst_folder} -> {produced} patches (1..{ref_count})")

def main():
    project_root = Path(__file__).resolve().parents[1]

    # Reference patches (already exist): ILM_OPL patches 256
    ref_dir = project_root / "data" / "overlap_patches_raw"
    ref_files = list_images(ref_dir)
    ref_count = len(ref_files)

    if ref_count == 0:
        raise RuntimeError(f"No reference patches found in {ref_dir}")

    # New projections (400x400)
    src_opl_bm = project_root / "data" / "OCTA(OPL_BM)"
    src_full   = project_root / "data" / "OCTA(FULL)"

    # Output patch folders
    dst_opl_bm = project_root / "data" / "overlap_patches_opl_bm"
    dst_full   = project_root / "data" / "overlap_patches_full"

    print(f"Reference count (ILM_OPL patches): {ref_count}")

    # Generate patches
    make_patches_for_folder(src_opl_bm, dst_opl_bm, ref_count, patch_size=256, stride=144, force_gray=True)
    make_patches_for_folder(src_full, dst_full, ref_count, patch_size=256, stride=144, force_gray=True)

if __name__ == "__main__":
    main()
