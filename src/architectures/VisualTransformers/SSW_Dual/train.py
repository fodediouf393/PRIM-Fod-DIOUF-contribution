import argparse
from datetime import datetime
from pathlib import Path
import torch

from src.common_vit.seed import set_seed
from src.common_vit.utils import load_yaml
from src.common_vit.transforms import build_train_transforms, build_eval_transforms
from src.common_vit.dataset_fullimg_3ch import build_loaders_fullimg_3ch
from src.common_vit.optim import build_optimizer_and_scheduler
from src.common_vit.engine import train_model

from src.architectures.VisualTransformers.SSW_Dual.models.ssw_dual import SSW_Dual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="configs/data_fullimg.yaml")
    ap.add_argument("--train", default="configs/train_vit_dice_cldice.yaml")
    ap.add_argument("--model", default="configs/ssw_dual.yaml")
    ap.add_argument("--norm", default="configs/norm_fullimg.yaml")
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()

    set_seed(args.seed)

    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.train)
    model_cfg = load_yaml(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = build_train_transforms()
    eval_tf = build_eval_transforms()

    train_loader, val_loader, test_loader = build_loaders_fullimg_3ch(
        data_cfg=data_cfg,
        split_json=data_cfg["split_json"],
        norm_yaml=args.norm,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_tf=train_tf,
        eval_tf=eval_tf,
    )

    model = SSW_Dual(
        img_ch=int(model_cfg["in_channels"]),
        output_ch=int(model_cfg["n_classes"]),
        rate=int(model_cfg["rate"]),
        layer_depth=int(model_cfg["layer_depth"]),
        kernel_size=int(model_cfg["kernel_size"]),
        extend_scope=float(model_cfg["extend_scope"]),
        window_size=int(model_cfg["window_size"]),
        dropout=float(model_cfg["dropout"]),
        repeat_n=int(model_cfg["repeat_n"]),
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_start=train_cfg["lr_start"],
        lr_max=train_cfg["lr_max"],
        warmup_epochs=train_cfg["warmup_epochs"],
        weight_decay=train_cfg["weight_decay"],
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"ssw_dual_r{model_cfg['rate']}_seed{args.seed}_{ts}"
    out_dir = Path("experiments") / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_cfg = dict(
        w_bce=float(train_cfg.get("w_bce", 0.0)),
        w_dice=float(train_cfg.get("w_dice", 0.0)),
        w_cldice=float(train_cfg.get("w_cldice", 0.0)),
        w_tversky=float(train_cfg.get("w_tversky", 0.0)),
        tversky_alpha=float(train_cfg.get("tversky_alpha", 0.7)),
        tversky_beta=float(train_cfg.get("tversky_beta", 0.3)),
        skel_iters=int(train_cfg.get("skel_iters", 10)),
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        out_dir=str(out_dir),
        epochs=int(train_cfg["epochs"]),
        log_every=int(train_cfg["log_every"]),
        loss_cfg=loss_cfg,
        save_every=int(train_cfg["save_every"]),
        resume_ckpt=None,
    )


if __name__ == "__main__":
    main()
