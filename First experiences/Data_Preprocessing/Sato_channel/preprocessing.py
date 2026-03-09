import numpy as np
from skimage import io, img_as_float32
from skimage.morphology import white_tophat, disk
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.exposure import rescale_intensity
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi, sato, meijering, apply_hysteresis_threshold, threshold_otsu
from skimage.morphology import binary_opening, binary_closing, remove_small_objects, disk, skeletonize, dilation
from scipy.ndimage import distance_transform_edt as edt, binary_fill_holes
from skimage.transform import resize
from skimage.filters import threshold_sauvola, gaussian, median
from skimage.morphology import binary_opening, binary_closing, dilation, remove_small_objects, disk
from scipy.ndimage import binary_fill_holes
from pathlib import Path
import cv2, numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_local
from skimage.filters import gabor
from skimage.exposure import rescale_intensity

def load_and_normalize(path: str) -> np.ndarray:
    """
    Charge une image OCTA et la normalise entre 0 et 1.
    - Convertit en niveaux de gris si RGB
    - Tronque les extrêmes (1er et 99e percentile)
    """
    img = io.imread(path)

    # --- 1. Conversion en niveaux de gris ---
    if img.ndim == 3:
        img = img[..., :3]  # ignore alpha
        img = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]

    # --- 2. Normalisation [0,1] robuste ---
    img = img_as_float32(img)
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip((img - p1) / (p99 - p1 + 1e-12), 0, 1)
    return img

def flatten_and_clahe(img01: np.ndarray,
                      tophat_radius_frac: float = 1/30,
                      clahe_clip: float = 0.02,
                      tiles: int = 8) -> np.ndarray:
    """Aplanit le fond (white tophat) puis améliore localement le contraste (CLAHE)."""
    r = max(2, int(round(min(img01.shape)*tophat_radius_frac)))
    flat = white_tophat(img01, footprint=disk(r))
    clahe = equalize_adapthist(flat, clip_limit=clahe_clip,
                               nbins=256, kernel_size=(tiles, tiles)).astype(np.float32)
    return clahe

def denoise_light(img01, method="median"):
    """
    Débruitage doux pour calmer le grain avant la vesselness.
    - "bilateral" : lisse le bruit tout en gardant les bords
    - "gaussian"  : plus simple, très rapide
    """
    m = method.lower()
    if m == "bilateral":
        return denoise_bilateral(img01, sigma_color=0.05, sigma_spatial=3, channel_axis=None)
    elif m == "tv":
        return denoise_tv_chambolle(img01, weight=0.08, eps=2e-4)
    elif m == "median":
        return gaussian(img01, sigma=0.6, preserve_range=True)
    else:
        raise ValueError("method must be 'bilateral' | 'tv' | 'median'")

# SITK
try:
    import SimpleITK as sitk
    HAVE_SITK = True
except Exception:
    HAVE_SITK = False

def _eigvals_robust(Hxx, Hxy, Hyy):
    """Compat scikit-image: accepte ancienne/nouvelle API."""
    try:
        ev = hessian_matrix_eigvals((Hxx, Hxy, Hyy))   # new API
    except TypeError:
        ev = hessian_matrix_eigvals(Hxx, Hxy, Hyy)     # old API
    if isinstance(ev, (tuple, list)):
        l1, l2 = ev[0], ev[1]
    else:
        E = np.asarray(ev)
        if E.ndim >= 3 and E.shape[0] >= 2:
            l1, l2 = E[0], E[1]
        elif E.ndim >= 3 and E.shape[-1] >= 2:
            l1, l2 = E[...,0], E[...,1]
        else:
            raise RuntimeError(f"Unexpected eig shape {E.shape}")
    # assurer |l1| <= |l2|
    swap = np.abs(l1) > np.abs(l2)
    l1c = l1.copy()
    l1[swap], l2[swap] = l2[swap], l1c[swap]
    return l1, l2

def vesselness_jerman_numpy(img01, sigmas=(0.8,2.0), alpha=0.5, beta=0.5, gamma=5.0, bright=True):
    """
    Approximation Jerman: Hessien multi-échelle + objectness simplifié.
    bright=True si vaisseaux clairs sur fond sombre.
    """
    smin, smax = sigmas
    scales = np.linspace(smin, smax, num=4)
    eps = 1e-12
    acc = []
    for s in scales:
        Hxx, Hxy, Hyy = hessian_matrix(img01, sigma=float(s), order='rc', use_gaussian_derivatives=False)
        l1, l2 = _eigvals_robust(Hxx, Hxy, Hyy)
        cond = (l2 < 0.0) if bright else (l2 > 0.0)
        Rb = np.abs(l1) / (np.abs(l2) + eps)                  # petit pour structures en ligne
        S  = np.sqrt(l1**2 + l2**2)
        v = (1.0 - np.exp(-(Rb**2)/(2.0*(alpha**2+eps)))) * np.exp(-(S**2)/(2.0*(gamma**2+eps)))
        v *= np.power(s, beta)                                # pondération d’échelle douce
        v = np.where(cond, v, 0.0)
        acc.append(v.astype(np.float32))
    V = np.max(np.stack(acc, axis=0), axis=0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    return rescale_intensity(V, in_range="image", out_range=(0.0,1.0)).astype(np.float32)

def vesselness_jerman(img01, sigmas=(0.8,2.0), alpha=0.5, beta=0.5, gamma=5.0, bright=True):
    if HAVE_SITK:
        itk = sitk.GetImageFromArray(img01.astype(np.float32))
        smin, smax = sigmas
        scales = np.linspace(smin, smax, num=4)
        resp = []
        for s in scales:
            hess = sitk.HessianRecursiveGaussianImageFilter()
            hess.SetSigma(float(s))
            H = hess.Execute(itk)
            obj = sitk.HessianToObjectnessMeasureImageFilter()
            obj.SetBrightObject(bool(bright))
            obj.SetScaleObjectnessMeasure(True)
            obj.SetAlpha(float(alpha))
            obj.SetBeta(float(beta))
            obj.SetGamma(float(gamma))
            V = obj.Execute(H)
            resp.append(sitk.GetArrayFromImage(V))
        V = np.max(np.stack(resp, axis=0), axis=0)
        V = rescale_intensity(V, in_range="image", out_range=(0.0,1.0)).astype(np.float32)
        return V
    else:
        return vesselness_jerman_numpy(img01, sigmas, alpha, beta, gamma, bright)

def vesselness(img01, method="meijering", sigmas=(1.0, 3.0), bright=True):
    """
    method ∈ {'frangi','sato','meijering','jerman'}
    Retourne une carte [0,1].
    """
    m = method.lower()
    smin, smax = sigmas
    scales = np.linspace(smin, smax, num=4)

    if m == "jerman":
        V = vesselness_jerman(img01, sigmas=sigmas, bright=bright)
    elif m == "frangi":
        V = frangi(img01, sigmas=scales, black_ridges=not bright)
    elif m == "sato":
        V = sato(img01, sigmas=scales, black_ridges=not bright)
    elif m == "meijering":
        V = meijering(img01, sigmas=scales, black_ridges=not bright)
    else:
        raise ValueError("method must be frangi|sato|meijering|jerman")

    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    return rescale_intensity(V, in_range="image", out_range=(0.0, 1.0)).astype(np.float32)

def vesselness_fusion_meij_jerman(img01,
                                  sig_small=(0.8,1.6),
                                  sig_large=(1.6,3.0),
                                  mode="weighted",     # "max" | "weighted" | "geom"
                                  w_small=0.6, w_large=0.4,
                                  bright=True):
    """
    Fusionne Meijering (petites échelles) + Jerman (grandes échelles) en une carte unique.
    - 'max': prend le meilleur des deux (robuste, contrasté)
    - 'weighted' : w_small*Meij + w_large*Jerman (lisse et contrôlable)
    - 'geom': sqrt(Meij*Jerman) (renforce intersections)
    """
    V_meij  = vesselness(img01, method="meijering", sigmas=sig_small, bright=bright)
    V_jerm  = vesselness(img01, method="jerman",    sigmas=sig_large, bright=bright)

    if mode == "max":
        V = np.maximum(V_meij, V_jerm)
    elif mode == "weighted":
        V = w_small*V_meij + w_large*V_jerm
    elif mode == "geom":
        V = np.sqrt(np.clip(V_meij,0,1) * np.clip(V_jerm,0,1))
    else:
        raise ValueError("mode must be 'max'|'weighted'|'geom'")

    V = rescale_intensity(V, in_range="image", out_range=(0.0,1.0)).astype(np.float32)
    return V, {"V_meij": V_meij, "V_jerm": V_jerm}

def make_fov_mask(img01: np.ndarray):
    """Masque du champ utile (FOV) dérivé de l'image (ou de la vesselness)."""
    t = threshold_otsu(img01)
    m = img01 > (0.6 * t)
    m = binary_closing(m, disk(5))
    m = binary_fill_holes(m)
    return m

def binarize_V(V: np.ndarray, p_high=94, p_low=82,
               min_obj=300, open_r=1, close_r=1, fov_mask=None):
    """Binarise une carte de vesselness par hystérèse + nettoyage léger."""
    bw = apply_hysteresis_threshold(V, np.percentile(V, p_low), np.percentile(V, p_high))
    if min_obj: bw = remove_small_objects(bw, min_size=min_obj)
    if open_r:  bw = binary_opening(bw, disk(open_r))
    if close_r: bw = binary_closing(bw, disk(close_r))
    if fov_mask is not None: bw = bw & fov_mask
    return bw

def split_by_radius_adaptive(bw: np.ndarray, percentile=65, min_recon_r=1):
    """
    Sépare small/large sur le binaire global via le rayon local au squelette.
    Renvoie: small (capillaires), large, r_th (px).
    """
    dist = edt(bw).astype(np.float32)
    skel = skeletonize(bw)
    rad  = dist * skel
    vals = rad[rad > 0]
    if vals.size == 0:
        return np.zeros_like(bw, bool), np.zeros_like(bw, bool), 0.0
    r_th = np.percentile(vals, percentile)
    large_skel = rad >= r_th
    recon = dilation(large_skel, disk(int(max(min_recon_r, np.ceil(r_th)))))
    large = bw & recon
    small = bw & (~large)
    return small, large, float(r_th)

def rescale_to_3mm(img01):
    # 6mm 512×512 -> 1024×1024 (équivalent 3mm 512×512 en px/mm)
    return resize(img01, (img01.shape[0]*2, img01.shape[1]*2),
                  order=3, preserve_range=True, anti_aliasing=True).astype(img01.dtype)

def dualband_caps_large(den: np.ndarray,
                        sig_small=(0.8, 1.6),
                        sig_large=(1.6, 3.0),
                        large_method="frangi",      # "frangi" ou "jerman"
                        p_high=94, p_low=82,
                        min_obj=250, percentile=65):
    """
    - V_small = Meijering(sig_small)      -> capillaires
    - V_large = Frangi/Jerman(sig_large)  -> gros vaisseaux
    - binarisation séparée (+FOV) + soustraction des grands
    - filet de sécurité par rayon adaptatif
    """
    V_small = vesselness(den, method="meijering", sigmas=sig_small, bright=True)
    V_large = vesselness(den, method=large_method,  sigmas=sig_large, bright=True)

    fov   = make_fov_mask(V_large)
    bw_s  = binarize_V(V_small, p_high=p_high, p_low=p_low, min_obj=min_obj, fov_mask=fov)
    bw_l  = binarize_V(V_large, p_high=p_high, p_low=p_low, min_obj=min_obj, fov_mask=fov)

    # retirer l'empreinte des grands des petits (tampon léger)
    bw_s_clean = bw_s & (~dilation(bw_l, disk(1)))
    bw_l_clean = bw_l

    # sécurité par rayon adaptatif
    caps_auto, large_auto, r_th = split_by_radius_adaptive(bw_s_clean | bw_l_clean, percentile=percentile)

    caps_final  = (bw_s_clean & (~bw_l_clean)) | (caps_auto & (~large_auto))
    large_final = bw_l_clean | large_auto

    debug = {
        "V_small": V_small, "V_large": V_large,
        "bw_small": bw_s_clean, "bw_large": bw_l_clean,
        "fov": fov, "r_th_px": r_th
    }
    return caps_final, large_final, debug

def make_multiclass_label(img01_float):
    """
    Construit un label 0/1/2 à partir de l'image OCTA float [0,1].
    1 = gros vaisseaux (Jerman/Sato), 2 = capillaires (Meijering sans les gros)
    """
    # préproc léger
    pre = flatten_and_clahe(img01_float, tophat_radius_frac=1/25, clahe_clip=0.01, tiles=8)
    den = denoise_light(pre, method="median")

    # Séparation dual-band (tu as déjà dualband_caps_large)
    caps_final, large_final, _ = dualband_caps_large(
        den, sig_small=(0.6,1.8), sig_large=(1.6,3.2),
        large_method="jerman", p_high=92, p_low=75, min_obj=200, percentile=65
    )

    # Label 0/1/2
    lbl = np.zeros(img01_float.shape, np.uint8)
    lbl[large_final] = 1
    lbl[caps_final]  = 2
    return lbl

def gabor_bank_max(img01, freqs=(0.12, 0.18), thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Max des réponses Gabor sur plusieurs directions/fréquences, renvoie [0,1]."""
    img01 = img_as_float32(img01)
    resp = []
    for f in freqs:
        for th in thetas:
            r, _ = gabor(img01, frequency=f, theta=th)
            resp.append(r.astype(np.float32))
    R = np.max(np.stack(resp,0), axis=0)
    return rescale_intensity(R, out_range=(0,1)).astype(np.float32)

def build_channels_triplet(gray_u8):
    """
    gray_u8: patch uint8 (256x256).
    Sortie: (3,H,W) float32 in [0,1] = [Sato, Meijering, Gabor].
    """
    img = img_as_float32(gray_u8)
    pre = flatten_and_clahe(img, tophat_radius_frac=1/25, clahe_clip=0.005, tiles=8)
    den = denoise_light(pre, method="median")

    ch_sato      = vesselness(den, method="sato",      sigmas=(1.0, 2.6), bright=True)
    ch_meijering = vesselness(den, method="meijering", sigmas=(0.6, 1.8), bright=True)
    ch_gabor     = gabor_bank_max(den, freqs=(0.12,0.18), thetas=(0,np.pi/4,np.pi/2,3*np.pi/4))

    X = np.stack([ch_sato, ch_meijering, ch_gabor], axis=0).astype(np.float32)
    return X

