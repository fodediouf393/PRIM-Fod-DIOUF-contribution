# Import des bibliothèques
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

#.......... Chemins d'entrée et sortie
INPUT_OVERLAP_DIR = Path(r"C:\Users\hp\OneDrive\Bureau\PRIM\PRIM-Project\Data\overlap_patches_raw")
OUTPUT_GABOR_DIR = Path(r"C:\Users\hp\OneDrive\Bureau\PRIM\PRIM-Project\Data\overlap_gabor")

OUTPUT_GABOR_DIR.mkdir(parents=True, exist_ok=True)

#.......... Paramètres du filtre médian
median_kernel = 3

#.......... Paramètres Gabor (mêmes que ton pipeline précédent)
thetas = np.linspace(0, np.pi, 12, endpoint=False)
gabor_lambd = 3.5
gabor_sigma = 1.75
gabor_gamma = 0.5

#.......... Paramètres CLAHE + Normalisation (mêmes que ton pipeline précédent)
clahe_clip = 2.0
clahe_grid = (8, 8)

#.......... Fonction Gabor (petits vaisseaux)
def gabor_vessel_enhancement(img_uint8, thetas, lambd=4.0, sigma=2.0, gamma=0.5):
    #...... Conversion en float32
    img = img_uint8.astype(np.float32)
    responses = []
    for theta in thetas:
        kernel = cv2.getGaborKernel(
            ksize=(9, 9),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=0,
            ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        responses.append(filtered)
    #...... Maximum sur les orientations
    vesselness = np.max(responses, axis=0)
    vesselness = vesselness - np.min(vesselness)
    vmax = np.max(vesselness)
    if vmax > 0:
        vesselness = vesselness / vmax * 255.0
    return vesselness.astype(np.uint8)

#.......... Fonction CLAHE + Normalisation min-max
def enhance_contrast_clahe_norm(img_uint8, clip_limit=2.0, tile_grid_size=(8, 8)):
    #...... CLAHE local
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_uint8)
    #...... Normalisation min-max [0,255]
    img_norm = cv2.normalize(img_clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)

#.......... Liste triée des patches overlappés (1.bmp, 2.bmp, ...)
file_list = sorted(INPUT_OVERLAP_DIR.glob("*.bmp"), key=lambda p: int(p.stem))

print(f"Nombre de patches en entrée: {len(file_list)}")
print(f"Application pipeline Médian 3x3 -> Gabor -> CLAHE+Norm sans padding...\n")

for in_path in tqdm(file_list, desc="Pipeline overlap Gabor", ncols=80):
    #...... Lecture du patch en niveaux de gris
    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Lecture impossible: {in_path}")

    #...... Médian 3x3
    med = cv2.medianBlur(img, median_kernel)

    #...... Gabor avec paramètres capillaires
    gab = gabor_vessel_enhancement(med, thetas, lambd=gabor_lambd, sigma=gabor_sigma, gamma=gabor_gamma)

    #...... CLAHE + Normalisation
    out = enhance_contrast_clahe_norm(gab, clip_limit=clahe_clip, tile_grid_size=clahe_grid)

    #...... Sauvegarde dans overlap_gabor avec le même nom (1.bmp, 2.bmp, ...)
    out_path = OUTPUT_GABOR_DIR / in_path.name
    ok = cv2.imwrite(str(out_path), out.astype(np.uint8))
    if not ok:
        raise IOError(f"Échec d'écriture: {out_path}")

print(f"\nTerminé. Images finales sauvegardées dans: {OUTPUT_GABOR_DIR}")
