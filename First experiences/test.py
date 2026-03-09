# test_little_unet.py

import os
import torch

from LittleUnet import LittleUNet              # adapte si besoin
from DataGenerator import build_dataloaders_3ch
from utils import evaluate

if __name__ == "__main__":
    # Dossiers de patches à adapter
    image_folders = [
        "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_sato",
        "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_meijering",
        "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_gabor",
    ]
    label_folder = "/home/infres/diouf-25/PRIM-Project/Data_Preprocessing/overlap_patches_labels"

    # Chemin du modèle à tester à adapter
    checkpoint_path = "/home/infres/diouf-25/PRIM-Project/experiment/best_model/best_run1.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device)

    _, _, test_loader = build_dataloaders_3ch(
        image_folders,
        label_folder,
        batch_size=4,
    )

    model = LittleUNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Checkpoint chargé depuis : {checkpoint_path}")

    test_metrics = evaluate(model, test_loader, device, desc="Test")

    print("\n===== Résultats sur le TEST (modèle chargé) =====")
    print(f"Loss    = {test_metrics['loss']:.4f}")
    print(f"AUC     = {test_metrics['auc']*100:.2f}")
    print(f"Dice    = {test_metrics['dice']*100:.2f}")
    print(f"MCC     = {test_metrics['mcc']:.4f}")
    print(f"Prec    = {test_metrics['precision']*100:.2f}")
    print(f"Recall  = {test_metrics['recall']*100:.2f}")
    print(f"Accuracy= {test_metrics['accuracy']*100:.2f}")
