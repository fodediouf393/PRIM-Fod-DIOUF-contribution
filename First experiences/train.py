# train_little_unet.py

import os
import numpy as np
import pandas as pd

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from LittleUnet import LittleUNet
from Unet import UNet
from AttentionUnet import AttentionUNet
from TransUnet import TransUNet
from Unet3plus import UNet3Plus
from DataGenerator import build_dataloaders_3ch
from utils import (
    set_seed,
    combined_loss,
    evaluate,
    mean_and_ci,
)


#  Racine de l'expérience pour Little U-Net
#  → chaque modèle aura son sous-dossier : experiment/Little_Unet

PROJECT_ROOT = "/home/infres/diouf-25/PRIM-Project"
EXP_ROOT = os.path.join(PROJECT_ROOT, "experiment", "Unet3plus")

BEST_MODEL_DIR   = os.path.join(EXP_ROOT, "best_model")
CHECKPOINT_DIR   = os.path.join(EXP_ROOT, "checkpoints")
TEST_RESULTS_DIR = os.path.join(EXP_ROOT, "test_results")
TEST_VISUALS_DIR = os.path.join(EXP_ROOT, "test_visuals")
LOG_DIR          = os.path.join(EXP_ROOT, "logs")

for d in [BEST_MODEL_DIR, CHECKPOINT_DIR, TEST_RESULTS_DIR, TEST_VISUALS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


def train_one_epoch(model, loader, optimizer, device):
    # Entraîne le modèle sur une époque
    model.train()
    running_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def run_single_experiment(
    device,
    image_folders,
    label_folder,
    run_idx: int,
    writer: SummaryWriter,
    batch_size=4,
    n_cycles=2,
    cycle_length=50,
    lr_max=1e-3,
    lr_min=1e-6,
):
    """
    Un run complet :
    - entraînement sur n_cycles * cycle_length epochs
    - sélection du meilleur modèle sur le Dice de validation
    - log complet TensorBoard (loss, metrics, LR)
    """

    train_loader, val_loader, test_loader = build_dataloaders_3ch(
        image_folders,
        label_folder,
        batch_size=batch_size,
    )

    #model = LittleUNet(in_channels=3, n_classes=1).to(device)
    #model = UNet(in_channels=3, n_classes=1).to(device)
    #model = AttentionUNet(in_channels=3, n_classes=1).to(device)
    #model = TransUNet(in_channels=3, n_classes=1).to(device)
    model = UNet3Plus(in_channels=3, n_classes=1).to(device)
    # Ajout du graphe dans TensorBoard (optionnel si ça plante)
    try:
        imgs_example, _, _ = next(iter(train_loader))
        imgs_example = imgs_example.to(device)
        writer.add_graph(model, imgs_example)
        del imgs_example
    except Exception as e:
        print(f"Impossible d'ajouter le graphe dans TensorBoard : {e}")

    # Optimiseur plus "doux" + régularisation
    optimizer = Adam(model.parameters(), lr=lr_max, weight_decay=1e-5)

    num_epochs = n_cycles * cycle_length
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cycle_length,
        eta_min=lr_min,
    )

    # On sélectionne le meilleur modèle sur le Dice de validation
    best_val_dice = -1.0

    for epoch in tqdm(range(num_epochs), desc=f"Run {run_idx} - Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, desc="Val")

        val_dice = val_metrics["dice"]

        # Mise à jour du scheduler (LR cyclique)
        scheduler.step()

        # Sauvegarde du checkpoint à chaque époque
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"run{run_idx}_epoch{epoch+1}.pth",
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
            },
            ckpt_path,
        )

        # Sauvegarde du meilleur modèle sur le Dice de validation
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(BEST_MODEL_DIR, f"best_run{run_idx}.pth")
            torch.save(model.state_dict(), best_path)

        # Logging TensorBoard
        global_step = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", train_loss, global_step)
        writer.add_scalar("Loss/val", val_metrics["loss"], global_step)
        writer.add_scalar("AUC/val", val_metrics["auc"], global_step)
        writer.add_scalar("Dice/val", val_metrics["dice"], global_step)
        writer.add_scalar("MCC/val", val_metrics["mcc"], global_step)
        writer.add_scalar("Precision/val", val_metrics["precision"], global_step)
        writer.add_scalar("Recall/val", val_metrics["recall"], global_step)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], global_step)
        writer.add_scalar("LR", current_lr, global_step)

        print(
            f"[Run {run_idx} | Epoch {epoch+1}/{num_epochs}] "
            f"TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_metrics['loss']:.4f} | "
            f"ValAUC={val_metrics['auc']:.4f} | "
            f"ValDice={val_metrics['dice']:.4f} | "
            f"ValMCC={val_metrics['mcc']:.4f} | "
            f"ValPrec={val_metrics['precision']:.4f} | "
            f"ValRec={val_metrics['recall']:.4f} | "
            f"ValAcc={val_metrics['accuracy']:.4f} | "
            f"LR={current_lr:.6f}"
        )

    # Évaluation finale sur le test avec le dernier modèle du run
    test_metrics = evaluate(model, test_loader, device, desc="Test")

    writer.add_scalar("Test/Loss", test_metrics["loss"], run_idx)
    writer.add_scalar("Test/AUC", test_metrics["auc"], run_idx)
    writer.add_scalar("Test/Dice", test_metrics["dice"], run_idx)
    writer.add_scalar("Test/MCC", test_metrics["mcc"], run_idx)
    writer.add_scalar("Test/Precision", test_metrics["precision"], run_idx)
    writer.add_scalar("Test/Recall", test_metrics["recall"], run_idx)
    writer.add_scalar("Test/Accuracy", test_metrics["accuracy"], run_idx)

    print(
        f"[RUN {run_idx} - TEST] "
        f"Loss={test_metrics['loss']:.4f} | "
        f"AUC={test_metrics['auc']:.4f} | "
        f"Dice={test_metrics['dice']:.4f} | "
        f"MCC={test_metrics['mcc']:.4f} | "
        f"Prec={test_metrics['precision']:.4f} | "
        f"Rec={test_metrics['recall']:.4f} | "
        f"Acc={test_metrics['accuracy']:.4f}"
    )

    return test_metrics


if __name__ == "__main__":
    # Dossiers de patches à adapter si besoin
    image_folders = [
        "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_sato",
        "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_meijering",
        "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_gabor",
    ]
    label_folder = "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_patches_labels"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    # -----------------------------
    # Hyperparamètres entraînement
    # -----------------------------
    batch_size = 4
    lr_max = 1e-3
    lr_min = 1e-6
    cycle_length = 50
    n_cycles = 2          # → 2 cycles de 50 epochs = 100 epochs
    n_runs = 5            # 5 seeds différents

    all_auc = []
    all_dice = []
    all_mcc = []
    all_prec = []
    all_rec = []
    all_acc = []

    for run_idx in range(1, n_runs + 1):
        seed = run_idx
        print("\n" + "=" * 60)
        print(f"RUN {run_idx}/{n_runs}  (seed = {seed})")
        print("=" * 60)
        set_seed(seed)

        run_log_dir = os.path.join(LOG_DIR, f"run_{run_idx}")
        writer = SummaryWriter(log_dir=run_log_dir)

        test_metrics = run_single_experiment(
            device=device,
            image_folders=image_folders,
            label_folder=label_folder,
            run_idx=run_idx,
            writer=writer,
            batch_size=batch_size,
            n_cycles=n_cycles,
            cycle_length=cycle_length,
            lr_max=lr_max,
            lr_min=lr_min,
        )

        writer.close()

        all_auc.append(test_metrics["auc"])
        all_dice.append(test_metrics["dice"])
        all_mcc.append(test_metrics["mcc"])
        all_prec.append(test_metrics["precision"])
        all_rec.append(test_metrics["recall"])
        all_acc.append(test_metrics["accuracy"])

    # ---------------------------------------------
    # Moyennes + intervalles de confiance (IC 95%)
    # ---------------------------------------------
    mean_auc, ci_auc = mean_and_ci(all_auc)
    mean_dice, ci_dice = mean_and_ci(all_dice)
    mean_mcc, ci_mcc = mean_and_ci(all_mcc)
    mean_prec, ci_prec = mean_and_ci(all_prec)
    mean_rec, ci_rec = mean_and_ci(all_rec)
    mean_acc, ci_acc = mean_and_ci(all_acc)

    print("\n===== Résultats finaux sur le TEST (5 runs) =====")
    print(f"AUC     = {mean_auc*100:.2f} ± {ci_auc*100:.2f}")
    print(f"Dice    = {mean_dice*100:.2f} ± {ci_dice*100:.2f}")
    print(f"MCC     = {mean_mcc:.4f} ± {ci_mcc:.4f}")
    print(f"Prec    = {mean_prec*100:.2f} ± {ci_prec*100:.2f}")
    print(f"Recall  = {mean_rec*100:.2f} ± {ci_rec*100:.2f}")
    print(f"Accuracy= {mean_acc*100:.2f} ± {ci_acc*100:.2f}")

    # ---------------------------------------------
    # Tableau pandas récapitulatif (style article)
    # ---------------------------------------------
    summary = pd.DataFrame(
        {
            "Method": [" Unet3plus(3ch)"],
            "AUC":   [f"{mean_auc:.3f} ± {ci_auc:.3f}"],
            "Dice":  [f"{mean_dice:.3f} ± {ci_dice:.3f}"],
            "Precision": [f"{mean_prec:.3f} ± {ci_prec:.3f}"],
            "Recall":    [f"{mean_rec:.3f} ± {ci_rec:.3f}"],
            "Accuracy":  [f"{mean_acc:.3f} ± {ci_acc:.3f}"],
            "MCC":       [f"{mean_mcc:.3f} ± {ci_mcc:.3f}"],
        }
    )

    print("\n===== Tableau récapitulatif (Unet3plus) =====")
    print(summary.to_string(index=False))

    table_path = os.path.join(EXP_ROOT, "Unet3plusmetrics_summary.csv")
    summary.to_csv(table_path, index=False)
    print(f"\nTableau sauvegardé dans : {table_path}")