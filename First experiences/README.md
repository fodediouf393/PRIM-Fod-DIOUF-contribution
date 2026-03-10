# First Experiences
Ce répertoire contient le code des premières expérimentations,principalement avec les CNNS, avec en entrée les images issues des filtres de Sato, Meijering et Gabor.
## Project Structure

```text
.
├── Data_Preprocessing
│   ├── Gabor_channel 
│   │   └── Gabor_Patches
│   ├── Meijering_channel
│   │   └── Meijering_patches
│   ├── overlap_gabor
│   ├── overlap_meijering
│   ├── overlap_patches_labels
│   ├── overlap_patches_raw
│   ├── overlap_sato
│   ├── Patches_label
│   └── Sato_channel
│       └── sato_patches
├── experiment
│   ├── AttentionUnet
│   │   ├── best_model
│   │   └── figures
│   │       ├── all_runs
│   │       └── per_run
│   ├── Little_Unet
│   │   ├── best_model
│   │   └── figures
│   │       ├── all_runs
│   │       └── per_run
│   ├── Unet
│   │   ├── best_model
│   │   └── figures
│   │       ├── all_runs
│   │       └── per_run
│   └── Unet3plus
│       └── best_model
└── ViT