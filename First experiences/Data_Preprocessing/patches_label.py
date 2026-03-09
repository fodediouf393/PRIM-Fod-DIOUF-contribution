import os
import cv2
from tqdm import tqdm

# Dossier d'entrée
input_dir = r"C:\Users\hp\OneDrive\Bureau\PRIM\Personal Code\Data\labelonlycapillary"

# Dossier de sortie
output_dir = r"C:\Users\hp\OneDrive\Bureau\PRIM\Personal Code\Data\Patches_label"
os.makedirs(output_dir, exist_ok=True)

print("Dossier de sortie :", output_dir)

patch_index = 1  # numérotation de 1 à 1200

start_idx = 10001
end_idx = 10300  # inclus

patch_size = 200
final_size = 256

# Padding (pour passer de 200 → 256)
pad_total = final_size - patch_size  # 56
pad_top = pad_total // 2
pad_bottom = pad_total - pad_top
pad_left = pad_total // 2
pad_right = pad_total - pad_left

for idx in tqdm(range(start_idx, end_idx + 1), desc="Découpage des images"):
    img_path = os.path.join(input_dir, f"{idx}.bmp")

    if not os.path.exists(img_path):
        print(f"[WARN] {img_path} introuvable, ignoré.")
        continue

    # Lecture en niveaux de gris (ou couleur si tu veux garder les 3 canaux)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] Impossible de lire {img_path}, ignoré.")
        continue

    h, w = img.shape[:2]
    if h != 400 or w != 400:
        print(f"[WARN] {img_path} a une taille {w}x{h} au lieu de 400x400.")
        continue

    # Découpage en 4 patches 200x200
    for i in range(2):
        for j in range(2):
            patch = img[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size
            ]

            # Padding à 256x256
            patch_padded = cv2.copyMakeBorder(
                patch,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            # Vérification
            if patch_padded.shape[0] != final_size or patch_padded.shape[1] != final_size:
                print(f"[ERROR] Mauvaise taille pour patch {patch_index}")
                continue

            # Sauvegarde
            out_path = os.path.join(output_dir, f"{patch_index}.bmp")
            cv2.imwrite(out_path, patch_padded)
            patch_index += 1

print(f"Terminé  {patch_index - 1} patches enregistrés dans :\n{output_dir}")
