import numpy as np
import matplotlib.pyplot as plt
import os


# Paramètres du scheduler

lr_max = 1e-2       # Learning rate initial
lr_min = 1e-8       # LR minimum
cycle_length = 50   # Longueur d’un cycle
n_cycles = 1       # Nombre de cycles

num_epochs = cycle_length * n_cycles



# Fonction CosineAnnealingLR

def cosine_lr(epoch, lr_min, lr_max, cycle_length):
    t = epoch % cycle_length
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / cycle_length))


# Calcul du LR pour chaque epoch

lrs = [cosine_lr(e, lr_min, lr_max, cycle_length) for e in range(num_epochs)]



# Affichage + sauvegarde

plt.figure(figsize=(10, 4))
plt.plot(lrs, linewidth=2)
plt.title("Évolution du Learning Rate (CosineAnnealingLR)", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.grid(True)
plt.tight_layout()

save_path = "lr_curve.png"
plt.savefig(save_path, dpi=150)
plt.close()

print(f"Courbe sauvegardée dans : {save_path}")
