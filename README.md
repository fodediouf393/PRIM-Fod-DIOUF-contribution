# PRIM-Fod-DIOUF-contribution
Personal Code Contribution on PRIM Project ( Segmentation and Quantification of high-resolution Retinal Vasculature Images using Deep-Learning, from OCT/OCTA images to AO-RSO:Large and Small-size blood vessels) at Télécom Paris. 

## Project Structure

```text
.
├── configs/
│   └── inference/
│
├── data/
│   ├── capillaire_langevin/
│   ├── capillaire_langevin_512_pseudo3dirs/
│   │   ├── patches_clahe/
│   │   ├── patches_dog/
│   │   └── patches_raw/
│   ├── capillaire_langevin_832/
│   ├── CLAHE/
│   ├── DOG/
│   ├── ens_mask_png/
│   ├── ens_tif/
│   ├── GT_Capillary/
│   ├── OCTA(FULL)/
│   ├── OCTA(ILM_OPL)/
│   ├── OCTA(OPL_BM)/
│   ├── overlap_gabor/
│   ├── overlap_meijering/
│   ├── overlap_patches_full/
│   ├── overlap_patches_ilm_opl/
│   ├── overlap_patches_labels/
│   ├── overlap_patches_opl_bm/
│   ├── overlap_sato/
│   └── splits/
│
├── experiments/
│   ├── finetune_newdomain_gn_pu_seed0/
│   │   ├── attention_unet/
│   │   ├── r2unet/
│   │   ├── resunet/
│   │   ├── unet/
│   │   ├── unet3plus/
│   │   ├── unetpp/
│   │   └── unetpp_ds/
│   │
│   ├── finetune_newdomain_gn_pu_seed0_curves/
│   │   ├── attention_unet/
│   │   │   ├── best_model/
│   │   │   └── checkpoints/
│   │   ├── r2unet/
│   │   ├── resunet/
│   │   ├── unet/
│   │   ├── unet3plus/
│   │   ├── unetpp/
│   │   └── unetpp_ds/
│   │
│   ├── inference_newdomain/
│   │   ├── predictions/
│   │   └── probabilities/
│   │
│   ├── inference_outputs/
│   │   ├── attention_unet_3ch/
│   │   │   └── seed_0/
│   │   │       ├── full_400_bin/
│   │   │       └── patches_bin/
│   │   ├── r2unet_3ch/
│   │   ├── resunet_3ch/
│   │   ├── unet_3ch/
│   │   ├── unet3plus_3ch/
│   │   ├── unetpp_3ch/
│   │   └── unetpp_DS_3ch/
│   │
│   ├── runs/
│   │   ├── attention_unet_3ch_norm_2026-02-09_12-45-34/
│   │   ├── r2unet_3ch_norm_2026-02-10_20-02-47/
│   │   ├── resunet_3ch_norm_2026-02-10_13-52-54/
│   │   ├── unet_3ch_norm_2026-02-08_12-28-02/
│   │   ├── unet3plus_3ch_norm_2026-02-09_20-35-04/
│   │   ├── unetpp_3ch_norm_2026-02-08_20-22-43/
│   │   ├── unetpp_DS_3ch_norm_2026-02-11_12-56-57/
│   │   ├── sswdual_r48_loss1_seed0/
│   │   ├── sswdual_r48_loss2_bce0.5_dice0.5_cldice0.2_seed0/
│   │   ├── sswdual_r48_loss3_tversky0.5_dice0.3_cldice0.2_seed0/
│   │   ├── swinunetr_loss1_dice0.8_cldice0.2_seed0/
│   │   ├── swinunetr_loss2_bce0.5_dice0.5_cldice0.2_seed0/
│   │   ├── transunet_loss1_dice0.8_cldice0.2_seed0/
│   │   ├── transunet_loss2_bce0.5_dice0.5_cldice0.2_seed0/
│   │   ├── unetr_ps8_loss1_dice0.8_cldice0.2_seed0/
│   │   └── unetr_ps8_loss2_bce0.5_dice0.5_cldice0.2_seed0/
│   │
│   ├── target_inference_832_final/
│   │   ├── UDA1_student_final/
│   │   └── UDA2_model_final/
│   │
│   ├── target_inference_832_post/
│   ├── target_inference_832_post_soft/
│   ├── uda_consistency_entropy/
│   └── uda_self_training/
│
├── First experiences/
│   ├── Data_Preprocessing/
│   │   ├── Gabor_channel/
│   │   ├── Meijering_channel/
│   │   ├── Sato_channel/
│   │   ├── Patches_label/
│   │   └── overlap_*/
│   ├── experiment/
│   │   ├── AttentionUnet/
│   │   ├── Little_Unet/
│   │   ├── Unet/
│   │   └── Unet3plus/
│   └── ViT/
│
├── scripts/
│   ├── eval/
│   ├── infer/
│   ├── target_uda/
│   └── vit/
│
└── src/
    ├── architectures/
    │   ├── UnetBased/
    │   │   └── models/
    │   └── VisualTransformers/
    │       ├── DualBranchDINO/
    │       ├── SSW_Dual/
    │       ├── SwinUnet/
    │       ├── TransUNet/
    │       └── UNETR/
    ├── common/
    ├── common_vit/
    ├── inference/
    └── target/