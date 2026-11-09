import torch
import torch.nn as nn

from models.stage_a_encoder import PerChannelCNNEncoder
from models.stage_c_fusion import RegionGroupedFusion
from models.peripheral_encoder import PeripheralEncoder
from models.hcan_fusion import HCANCrossAttentionFusion
from training.smote import apply_latent_smote
from training.imbalance import DeepSMOTELatent, GAMOLatent


class HCANCIBNet(nn.Module):
    """
    Unified Architecture combining HCAN multimodal fusion (EEG + Peripheral)
    and CIB-Net Stage A/B/C regional invariance & spectral decoupling for AMIGOS (14 EEG + 18 Periph).
    """

    def __init__(
        self,
        n_channels: int = 14,
        n_classes: int = 2,
        d_embed: int = 64,
        n_periph_features: int = 18,
        shared_weights: bool = True,
        region_dropout_p: float = 0.2,
        imbalance_method: str = "none",
        ablate_modality: str = None,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.d_embed = int(d_embed)
        self.n_periph_features = int(n_periph_features)
        self.imbalance_method = str(imbalance_method or "none").lower()
        self.ablate_modality = ablate_modality
        self.last_aux_loss = 0.0

        self.eeg_encoder = PerChannelCNNEncoder(
            d_embed=d_embed, shared_weights=shared_weights, n_channels=n_channels
        )
        self.region_tokenizer = RegionGroupedFusion(
            d_embed=d_embed, n_classes=n_classes, region_dropout_p=region_dropout_p
        )
        self.periph_encoder = PeripheralEncoder(
            in_features=n_periph_features, d_embed=d_embed, n_classes=n_classes
        )
        self.fusion = HCANCrossAttentionFusion(
            d_embed=d_embed, n_heads=4, n_classes=n_classes
        )

        joint_dim = n_channels * d_embed + n_periph_features
        self.deepsmote_module = (
            DeepSMOTELatent(input_dim=joint_dim, hidden_dim=128, latent_dim=64)
            if self.imbalance_method == "deepsmote"
            else None
        )
        self.gamo_module = (
            GAMOLatent(input_dim=joint_dim, noise_dim=16, hidden_dim=128)
            if self.imbalance_method == "gamo"
            else None
        )

    def forward(self, x_eeg, x_periph, y=None, subj_ids=None, smote_k: int = 0):
        self.last_aux_loss = torch.zeros((), device=x_eeg.device, dtype=torch.float32)

        if self.ablate_modality == "eeg":
            x_eeg = torch.zeros_like(x_eeg)
        elif self.ablate_modality == "periph":
            x_periph = torch.zeros_like(x_periph)

        # 1. Stage A: Per-channel EEG encoding -> [B, 14, d_embed]
        channel_embeddings = self.eeg_encoder(x_eeg)

        # 2. Latent oversampling (applied during training before region tokenization)
        if self.training and y is not None and subj_ids is not None:
            method = self.imbalance_method
            if method == "deepsmote" and self.deepsmote_module is not None:
                k_use = smote_k if smote_k > 0 else 5
                channel_embeddings, x_periph, y, subj_ids, aux_loss = self.deepsmote_module.oversample(
                    channel_embeddings, x_periph, y, subj_ids, k_neighbors=k_use
                )
                self.last_aux_loss = 0.1 * aux_loss
            elif method == "gamo" and self.gamo_module is not None:
                channel_embeddings, x_periph, y, subj_ids, aux_loss = self.gamo_module.oversample(
                    channel_embeddings, x_periph, y, subj_ids
                )
                self.last_aux_loss = aux_loss
            elif method in ("v_smote", "smote") or smote_k > 0:
                k_use = smote_k if smote_k > 0 else 5
                var_preserve = method != "smote"
                channel_embeddings, x_periph, y, subj_ids = apply_latent_smote(
                    channel_embeddings,
                    x_periph,
                    y,
                    subj_ids,
                    k_neighbors=k_use,
                    variance_preserve=var_preserve,
                )

        # 3. Stage C: Region tokenization -> list of 4 [B, d_embed] & 4 [B, n_classes]
        group_feats, group_logits = self.region_tokenizer.extract_region_tokens(channel_embeddings)
        region_tokens = torch.stack(group_feats, dim=1)  # [B, 4, d_embed]

        # 4. Peripheral encoding -> [B, 1, d_embed], [B, n_classes]
        periph_token, z_periph = self.periph_encoder(x_periph)

        # 5. HCAN multimodal fusion
        final_logits, z_eeg = self.fusion(periph_token, region_tokens)

        return final_logits, group_logits, z_eeg, z_periph, y, subj_ids

