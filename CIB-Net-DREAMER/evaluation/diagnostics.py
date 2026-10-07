import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy

class DiagnosticsSuite:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device
        
    def _evaluate(self, model=None, modify_batch_fn=None):
        if model is None:
            model = self.model
            
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for eeg, periph, y, _, _ in self.dataloader:
                eeg, periph, y = eeg.to(self.device), periph.to(self.device), y.to(self.device)
                
                if modify_batch_fn:
                    eeg, periph = modify_batch_fn(eeg, periph)
                    
                logits = model(eeg, periph)
                if isinstance(logits, tuple):
                    logits = logits[0]
                    
                preds = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        return roc_auc_score(all_targets, all_preds)

    def run_modality_ablation(self):
        """Zeroes out one modality at a time to check for starvation."""
        print("\n--- Modality Ablation ---")
        base_auroc = self._evaluate()
        print(f"Base AUROC: {base_auroc:.4f}")
        
        def zero_eeg(eeg, periph): return torch.zeros_like(eeg), periph
        auroc_no_eeg = self._evaluate(modify_batch_fn=zero_eeg)
        
        def zero_periph(eeg, periph): return eeg, torch.zeros_like(periph)
        auroc_no_periph = self._evaluate(modify_batch_fn=zero_periph)
        
        print(f"AUROC w/o EEG: {auroc_no_eeg:.4f} (Drop: {base_auroc - auroc_no_eeg:+.4f})")
        print(f"AUROC w/o Periph: {auroc_no_periph:.4f} (Drop: {base_auroc - auroc_no_periph:+.4f})")
        return base_auroc, auroc_no_eeg, auroc_no_periph

    def run_channel_ablation(self):
        """Zeroes out one EEG channel at a time to find dominant channels."""
        print("\n--- Channel Leave-One-Out Ablation ---")
        base_auroc = self._evaluate()
        channel_drops = {}
        
        n_channels = next(iter(self.dataloader))[0].shape[1]
        for ch in range(n_channels):
            def zero_channel(eeg, periph, c=ch):
                eeg_mod = eeg.clone()
                eeg_mod[:, c, :] = 0
                return eeg_mod, periph
                
            auroc = self._evaluate(modify_batch_fn=zero_channel)
            drop = base_auroc - auroc
            channel_drops[ch] = drop
            
        # Top 5 most important channels
        top_channels = sorted(channel_drops.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 Most Important Channels:")
        for ch, drop in top_channels:
            print(f"  Channel {ch}: {drop:+.4f} drop")
            
        return channel_drops

    def run_region_gradient_norm(self, region_fusion_layer):
        """
        Calculates gradient norms for each region in the fusion layer.
        Must be called AFTER a backward pass.
        """
        norms = {}
        for region, classifier in region_fusion_layer.group_classifiers.items():
            norm = 0.0
            for param in classifier.parameters():
                if param.grad is not None:
                    norm += param.grad.norm().item() ** 2
            norms[region] = norm ** 0.5
            
        total_norm = sum(norms.values())
        print("\n--- Region Gradient Norms ---")
        for region, norm in norms.items():
            pct = (norm / total_norm) * 100 if total_norm > 0 else 0
            print(f"  {region.capitalize():<10}: {norm:.4f} ({pct:.1f}%)")
            
        return norms

    def compute_contribution_gini(self, region_fusion_layer):
        """
        Computes the Gini coefficient of the region gradient norms.
        A higher Gini (closer to 1) means the model is starved/dominated by few regions.
        A lower Gini (closer to 0) means contributions are balanced.
        """
        norms = self.run_region_gradient_norm(region_fusion_layer)
        values = list(norms.values())
        
        # Gini coefficient calculation
        if not values or sum(values) == 0:
            return 0.0
            
        values = sorted(values)
        n = len(values)
        mean_val = sum(values) / n
        
        # Gini = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean)
        # Simplified for sorted array: Gini = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
        # where i is 1-indexed
        gini = (2.0 * sum((i + 1) * val for i, val in enumerate(values)) / (n * sum(values))) - (n + 1.0) / n
        return gini

    def compute_embedding_quality(self, model=None):
        """
        Extracts 64-d embeddings from the model and computes clustering metrics:
        - Silhouette score
        - kNN class purity (k=5)
        """
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np
        
        if model is None:
            model = self.model
            
        model.eval()
        all_embeds = []
        all_targets = []
        
        with torch.no_grad():
            for eeg, periph, y, _, _ in self.dataloader:
                eeg, periph, y = eeg.to(self.device), periph.to(self.device), y.to(self.device)
                
                # Extract embeddings (depends on model architecture)
                # For HCANCIBNet, we want the fused_token before the final classifier
                # We can approximate this by calling the encoders and fusion manually
                # Or just use the raw peripheral + EEG pooled if it's B0
                
                # Let's extract from stage A for simplicity to check backbone quality,
                # or if HCAN, we intercept the fusion token.
                # Assuming the model has a fusion layer we can get the feature from:
                if hasattr(model, 'fusion'):
                    channel_embeddings = model.eeg_encoder(eeg)
                    group_feats = []
                    for region, channels in model.region_tokenizer.regions.items():
                        region_x = channel_embeddings[:, channels, :] 
                        attn = model.region_tokenizer.intra_attentions[region](region_x)
                        pooled = torch.sum(region_x * attn, dim=1)
                        group_feats.append(pooled)
                    region_tokens = torch.stack(group_feats, dim=1)
                    periph_token, _ = model.periph_encoder(periph)
                    
                    # Inside HCAN cross attention
                    pooled_eeg = torch.mean(region_tokens, dim=1, keepdim=True)
                    kv = torch.cat([region_tokens, pooled_eeg], dim=1)
                    attn_out, _ = model.fusion.cross_attn(periph_token, kv, kv)
                    embeds = model.fusion.layer_norm(periph_token + attn_out).squeeze(1)
                else:
                    # Fallback for baseline models
                    # If it's StageAModel or similar
                    embeds = model.encoder(eeg).view(eeg.size(0), -1)
                    
                all_embeds.extend(embeds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        all_embeds = np.array(all_embeds)
        all_targets = np.array(all_targets)
        
        if len(np.unique(all_targets)) < 2:
            return {"silhouette": 0.0, "knn_purity": 0.0}
            
        # 1. Silhouette
        sil_score = silhouette_score(all_embeds, all_targets)
        
        # 2. kNN Purity
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(all_embeds, all_targets)
        knn_preds = knn.predict(all_embeds)
        knn_purity = np.mean(knn_preds == all_targets)
        
        return {
            "silhouette": sil_score,
            "knn_purity": knn_purity
        }
