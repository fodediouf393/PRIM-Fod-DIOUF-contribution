from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import re
import yaml
import torch

from src.inference.predict_patches import predict_test_patches_binary
from src.inference.reconstruct_full import reconstruct_and_save
from src.inference.io import ensure_dir

# Modern models
from src.architectures.UnetBased.models.unet import UNet
from src.architectures.UnetBased.models.unetpp import UNetPlusPlus
from src.architectures.UnetBased.models.resunet import ResUNet
from src.architectures.UnetBased.models.r2unet import R2UNet
from src.architectures.UnetBased.models.unet3plus import UNet3Plus
from src.architectures.UnetBased.models.attention_unet import AttentionUNet

# Legacy UNet3+ (exact old architecture)
from src.architectures.UnetBased.models.unet3plus_legacy import UNet3PlusLegacy


def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def load_split_test_ids(split_json: str):
    """
    split_json['test'] can contain:
      - ints: 961
      - numeric strings: "961"
      - filenames: "961.bmp" / "961.png"
      - possibly paths: "some/path/961.bmp"
    Returns list[int] patch_ids
    """
    with open(split_json, "r") as f:
        d = json.load(f)

    test_list = d["test"]
    ids = []
    for x in test_list:
        if isinstance(x, int):
            ids.append(x)
            continue

        s = str(x)
        s = os.path.basename(s)
        root, _ext = os.path.splitext(s)
        m = re.search(r"\d+", root)
        if m is None:
            raise ValueError(f"Cannot parse numeric id from test entry: {x}")
        ids.append(int(m.group(0)))

    return ids


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_ckpt_into_model(model, ckpt_path: str, device: str):
    """
    Supports:
      - dict with 'model_state'
      - dict with 'state_dict'
      - raw state_dict
    Strips optional 'module.' prefix.
    Loads with strict=False but prints missing/unexpected (so you see if something is wrong).
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[WARN] load_state_dict(strict=False)")
        if missing:
            print("  Missing keys (first 20):", missing[:20])
        if unexpected:
            print("  Unexpected keys (first 20):", unexpected[:20])

    return model


def build_model(entry: dict):
    arch = entry["arch"].lower()
    extra = entry.get("extra", {}) or {}

    in_ch = int(entry.get("in_channels", 3))
    n_classes = int(entry.get("n_classes", 1))
    base = int(entry.get("base", 42))

    if arch == "unet":
        return UNet(in_channels=in_ch, n_classes=n_classes, base=base)

    if arch == "unetpp":
        ds = bool(extra.get("deep_supervision", False))
        return UNetPlusPlus(in_channels=in_ch, n_classes=n_classes, base=base, deep_supervision=ds)

    if arch == "resunet":
        return ResUNet(in_channels=in_ch, n_classes=n_classes, base=base)

    if arch == "r2unet":
        t = int(extra.get("t", 2))
        return R2UNet(in_channels=in_ch, n_classes=n_classes, base=base, t=t)

    if arch == "unet3plus":
        decoder_ch = int(extra.get("decoder_ch", 128))
        deep_supervision = bool(extra.get("deep_supervision", False))
        return UNet3Plus(
            in_channels=in_ch,
            n_classes=n_classes,
            base=base,
            decoder_ch=decoder_ch,
            deep_supervision=deep_supervision,
        )

    if arch == "attention_unet":
        return AttentionUNet(in_channels=in_ch, n_classes=n_classes, base=base)

    if arch == "unet3plus_legacy":
        base_channels = int(extra.get("base_channels", 32))
        mid_channels = int(extra.get("mid_channels", 32))
        return UNet3PlusLegacy(
            in_channels=in_ch,
            n_classes=n_classes,
            base_channels=base_channels,
            mid_channels=mid_channels,
        )

    raise ValueError(f"Unknown arch: {arch}")


def patch_id_to_image_id(patch_id: int, patches_per_image: int) -> int:
    # patch ids are 1..1200
    return (patch_id - 1) // patches_per_image + 1


def image_id_to_patch_ids(image_id: int, patches_per_image: int) -> list[int]:
    # image_id 1..300 -> patches (image-1)*4 + [1..4]
    start = (image_id - 1) * patches_per_image + 1
    return list(range(start, start + patches_per_image))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", default="configs/data.yaml")
    ap.add_argument("--inf_yaml", default="configs/inference/inference.yaml")
    ap.add_argument("--models_yaml", default="configs/inference/models_registry.yaml")
    ap.add_argument("--stitch_yaml", default="configs/inference/stitching.yaml")
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_yaml)
    inf_cfg = load_yaml(args.inf_yaml)
    models_cfg = load_yaml(args.models_yaml)
    stitch_cfg = load_yaml(args.stitch_yaml)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # test patches from split
    test_patch_ids = load_split_test_ids(data_cfg["split_json"])

    # stitching params
    patches_per_image = int(stitch_cfg.get("patches_per_image", 4))

    # IMPORTANT FIX:
    # derive test image_ids from test patches, then expand to all patches per image
    test_image_ids = sorted({patch_id_to_image_id(pid, patches_per_image) for pid in test_patch_ids})
    expanded_patch_ids = []
    for img_id in test_image_ids:
        expanded_patch_ids.extend(image_id_to_patch_ids(img_id, patches_per_image))
    expanded_patch_ids = sorted(set(expanded_patch_ids))

    print(f"[INFO] test patches in split: {len(test_patch_ids)}")
    print(f"[INFO] derived test images: {len(test_image_ids)}")
    print(f"[INFO] expanded patches to predict (full coverage): {len(expanded_patch_ids)}")

    out_root = inf_cfg["out_root"]
    ensure_dir(out_root)

    threshold = float(inf_cfg.get("threshold", 0.5))
    norm_yaml_default = inf_cfg.get("norm_yaml", None)

    # New approach 3ch projections
    raw_dirs = {
        "c1": "data/overlap_patches_ilm_opl",
        "c2": "data/overlap_patches_opl_bm",
        "c3": "data/overlap_patches_full",
    }

    # Legacy filtered 3ch
    filt_dirs = {
        "c1": "data/overlap_sato",
        "c2": "data/overlap_meijering",
        "c3": "data/overlap_gabor",
    }

    for entry in models_cfg["models"]:
        name = entry["name"]
        arch = entry["arch"]
        ckpt_path = entry["ckpt_path"]
        input_mode = entry.get("input_mode", "raw")  # raw / filtered
        norm_yaml = entry.get("norm_yaml", norm_yaml_default)  # can be null for legacy

        print(f"\n=== Inference: {name} ({arch}) ===")
        print("ckpt:", ckpt_path)
        print("input_mode:", input_mode)
        print("norm_yaml:", norm_yaml)
        print("threshold:", threshold)

        model = build_model(entry).to(device)
        model = load_ckpt_into_model(model, ckpt_path, device)
        model.eval()

        out_dir_model = Path(out_root) / name / "seed_0"
        out_patches_bin = out_dir_model / "patches_bin"
        out_full_bin = out_dir_model / "full_400_bin"
        ensure_dir(str(out_patches_bin))
        ensure_dir(str(out_full_bin))

        in_dirs = filt_dirs if input_mode == "filtered" else raw_dirs

        # Predict patches that cover the full test images (not only random test patches)
        prob_cache = predict_test_patches_binary(
            model=model,
            device=device,
            test_ids=expanded_patch_ids,
            in_dirs=in_dirs,
            out_dir_patches_bin=str(out_patches_bin),
            threshold=threshold,
            norm_yaml=norm_yaml,  # None => no mean/std
        )

        # Reconstruct full masks, then only save the images that belong to the test set
        full_masks = reconstruct_and_save(
            prob_cache=prob_cache,
            stitching_yaml=args.stitch_yaml,
            out_dir_full_bin=str(out_full_bin),
            threshold=threshold,
        )

        # OPTIONAL: if reconstruct_and_save saves all, we can clean by keeping only test_image_ids
        # But our reconstruct currently will reconstruct only images present in prob_cache anyway.
        # For safety, keep only those:
        for p in Path(out_full_bin).glob("*.png"):
            try:
                img_id = int(p.stem)
                if img_id not in set(test_image_ids):
                    p.unlink(missing_ok=True)
            except Exception:
                pass

        print("Saved patch binary masks (.png):", out_patches_bin)
        print("Saved reconstructed test full 400 binary masks (.png):", out_full_bin)

    print("\nDone.")


if __name__ == "__main__":
    main()
