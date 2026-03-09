import argparse, csv, math, json
from datetime import datetime
from pathlib import Path
import torch

from src.common_vit.seed import set_seed
from src.common_vit.utils import load_yaml
from src.common_vit.transforms import build_train_transforms, build_eval_transforms
from src.common_vit.dataset_fullimg_3ch import build_loaders_fullimg_3ch
from src.common_vit.optim import build_optimizer_and_scheduler
from src.common_vit.engine import train_model

from src.architectures.VisualTransformers.SwinUnet.models.swin_unet import SwinUNet


def mean_ci95(values):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(n)
    return m, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data_fullimg.yaml")
    ap.add_argument("--train", default="configs/train_vit.yaml")
    ap.add_argument("--model", default="configs/swin_unet.yaml")
    ap.add_argument("--norm", default="configs/norm_fullimg.yaml")
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--run_prefix", default="swinunet_fullimg_3ch")
    args = ap.parse_args()

    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.train)
    model_cfg = load_yaml(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path("experiments") / "runs" / f"{args.run_prefix}_{ts}"
    root.mkdir(parents=True, exist_ok=True)

    loss_cfg = dict(
        w_bce=train_cfg["w_bce"],
        w_dice=train_cfg["w_dice"],
        w_cldice=train_cfg["w_cldice"],
        skel_iters=train_cfg["skel_iters"],
    )

    results = []

    for seed in range(args.n_runs):
        set_seed(seed)

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

        model = SwinUNet(
            in_channels=model_cfg["in_channels"],
            n_classes=model_cfg["n_classes"],
            encoder_name=model_cfg["encoder_name"],
            encoder_weights=model_cfg["encoder_weights"],
            decoder_channels=tuple(model_cfg["decoder_channels"]),
        ).to(device)

        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            lr_start=train_cfg["lr_start"],
            lr_max=train_cfg["lr_max"],
            warmup_epochs=train_cfg["warmup_epochs"],
            weight_decay=train_cfg["weight_decay"],
        )

        out_dir = root / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            out_dir=str(out_dir),
            epochs=train_cfg["epochs"],
            log_every=train_cfg["log_every"],
            loss_cfg=loss_cfg,
            save_every=train_cfg["save_every"],
            resume_ckpt=None,
        )

        results.append({"seed": seed, **summary})

    keys = ["test_loss","test_dice","test_iou","test_mcc","test_precision","test_recall","test_auc","test_accuracy"]
    means, cis = {}, {}
    for k in keys:
        vals = [float(r[k]) for r in results if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
        means[k], cis[k] = mean_ci95(vals) if vals else (float("nan"), float("nan"))

    summary_csv = Path("experiments") / "summary_vit.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed","best_epoch","best_val_loss"] + keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in ["seed","best_epoch","best_val_loss"] + keys})
        w.writerow({"seed": "MEAN", **{k: means[k] for k in keys}})
        w.writerow({"seed": "CI95", **{k: cis[k] for k in keys}})

    print("[OK] wrote", summary_csv)

if __name__ == "__main__":
    main()
