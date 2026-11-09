import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier


class DiagnosticsSuite:
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def _evaluate(self, model=None, modify_batch_fn=None):
        model = model or self.model
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for eeg, periph, y, _, _ in self.dataloader:
                eeg, periph, y = eeg.to(self.device), periph.to(self.device), y.to(self.device)
                if modify_batch_fn:
                    eeg, periph = modify_batch_fn(eeg, periph)
                out = model(eeg, periph)
                logits = out[0] if isinstance(out, tuple) else out
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        if len(np.unique(all_targets)) < 2:
            return float("nan")
        return float(roc_auc_score(all_targets, all_preds))

    def run_modality_ablation(self):
        base_auroc = self._evaluate()
        auroc_no_eeg = self._evaluate(modify_batch_fn=lambda e, p: (torch.zeros_like(e), p))
        auroc_no_periph = self._evaluate(modify_batch_fn=lambda e, p: (e, torch.zeros_like(p)))
        print(f"[Modality Ablation] Base={base_auroc:.4f} | w/o EEG={auroc_no_eeg:.4f} | w/o Periph={auroc_no_periph:.4f}")
        return base_auroc, auroc_no_eeg, auroc_no_periph

    def run_channel_ablation(self):
        base_auroc = self._evaluate()
        n_channels = next(iter(self.dataloader))[0].shape[1]
        drops = {}
        for ch in range(n_channels):
            def _zero_ch(e, p, c=ch):
                e_mod = e.clone()
                e_mod[:, c, :] = 0.0
                return e_mod, p
            drops[ch] = base_auroc - self._evaluate(modify_batch_fn=_zero_ch)
        return drops

    def compute_embedding_quality(self, model=None):
        model = model or self.model
        model.eval()
        all_embeds, all_targets = [], []
        with torch.no_grad():
            for eeg, periph, y, _, _ in self.dataloader:
                eeg, periph = eeg.to(self.device), periph.to(self.device)
                if hasattr(model, "fusion"):
                    ch_emb = model.eeg_encoder(eeg)
                    group_feats, _ = model.region_tokenizer.extract_region_tokens(ch_emb)
                    region_tokens = torch.stack(group_feats, dim=1)
                    periph_token, _ = model.periph_encoder(periph)
                    pooled_eeg = torch.mean(region_tokens, dim=1, keepdim=True)
                    kv = torch.cat([region_tokens, pooled_eeg], dim=1)
                    attn_out, _ = model.fusion.cross_attn(periph_token, kv, kv)
                    embeds = model.fusion.layer_norm(periph_token + attn_out).squeeze(1)
                else:
                    embeds = model.encoder(eeg).view(eeg.size(0), -1)
                all_embeds.append(embeds.cpu().numpy())
                all_targets.append(y.numpy())
        X = np.concatenate(all_embeds, axis=0)
        y_np = np.concatenate(all_targets, axis=0)
        if len(np.unique(y_np)) < 2:
            return {"silhouette": 0.0, "knn_purity": 0.0}
        sil = float(silhouette_score(X, y_np))
        knn = KNeighborsClassifier(n_neighbors=min(5, len(X) - 1)).fit(X, y_np)
        pur = float(np.mean(knn.predict(X) == y_np))
        return {"silhouette": sil, "knn_purity": pur}

