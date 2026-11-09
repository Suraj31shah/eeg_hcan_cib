import os
import torch
import numpy as np
from tqdm import tqdm

from models.cib_net_motor import CIBNetMotor
from models.stage_b_invariance import InvarianceGate
from training.losses import SpectralDecouplingLoss
from evaluation.metrics import compute_classification_metrics, format_metrics_table
from evaluation.plots import (
    plot_training_history, plot_confusion_matrix,
    plot_invariance_gates, plot_tsne_embeddings
)

class Trainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate Model
        self.model = CIBNetMotor(
            n_channels=config['dataset']['n_channels'],
            n_classes=config['model']['n_classes'],
            d_embed=config['model']['d_embed'],
            shared_weights=config['model']['shared_weights'],
            region_dropout_p=config['model']['region_dropout_p']
        ).to(self.device)
        
        # Stage B Invariance Gate
        self.invariance_gate = InvarianceGate(
            n_regions=len(config['regions']),
            decay=config['loss']['ema_decay'],
            tau=config['loss']['tau_gate']
        ).to(self.device)
        
        # Optimizer & Loss
        self.loss_fn = SpectralDecouplingLoss(
            lambda_0=config['loss']['lambda_0'],
            beta_aux=0.2
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.checkpoints_dir = config['paths']['checkpoints_dir']
        self.plots_dir = config['paths']['plots_dir']
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, total_main, total_aux, total_sd = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        
        for x, y, subj_ids, _ in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            subj_ids = subj_ids.to(self.device)
            
            self.optimizer.zero_grad()
            final_logits, group_logits, group_feats, _ = self.model(x)
            
            # Stage B Update
            self.invariance_gate.update(group_logits, y, subj_ids)
            inv_gates = self.invariance_gate.get_gates()
            
            # Stage C Gated Loss
            loss, l_main, l_aux, l_sd = self.loss_fn(final_logits, y, group_logits, inv_gates, group_feats)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_main += l_main
            total_aux += l_aux
            total_sd += l_sd
            n_batches += 1
            
        return {
            'loss': total_loss / max(n_batches, 1),
            'main_loss': total_main / max(n_batches, 1),
            'aux_loss': total_aux / max(n_batches, 1),
            'sd_loss': total_sd / max(n_batches, 1)
        }
        
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        all_embeddings = []
        
        for x, y, _, _ in dataloader:
            x = x.to(self.device)
            final_logits, _, _, ch_emb = self.model(x)
            
            probs = torch.softmax(final_logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeddings.extend(ch_emb.mean(dim=1).cpu().numpy())
            
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)
        embeddings = np.array(all_embeddings)
        
        metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        return metrics, y_true, y_pred, y_probs, embeddings
        
    def fit(self, train_loader, val_loader, epochs=30):
        history = {
            'train_loss': [], 'train_main_loss': [], 'train_sd_loss': [],
            'val_accuracy': [], 'val_balanced_acc': [], 'val_auroc': [], 'val_f1': []
        }
        best_val_score = -1.0
        best_ckpt_path = os.path.join(self.checkpoints_dir, 'best_checkpoint.pth')
        
        print(f"\n[CIB-Net] Beginning Training for {epochs} epochs on device: {self.device}")
        
        for ep in range(1, epochs + 1):
            train_res = self.train_epoch(train_loader)
            val_metrics, _, _, _, _ = self.evaluate(val_loader)
            
            history['train_loss'].append(train_res['loss'])
            history['train_main_loss'].append(train_res['main_loss'])
            history['train_sd_loss'].append(train_res['sd_loss'])
            
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['val_f1'].append(val_metrics['macro_f1'])
            
            print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_res['loss']:.4f} "
                  f"(Main: {train_res['main_loss']:.4f}, SD: {train_res['sd_loss']:.4f}) | "
                  f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                  f"Val BalAcc: {val_metrics['balanced_accuracy']*100:.2f}% | "
                  f"Val AUROC: {val_metrics['auroc']:.4f}")
                  
            # Checkpoint on best balanced accuracy
            if val_metrics['balanced_accuracy'] > best_val_score:
                best_val_score = val_metrics['balanced_accuracy']
                torch.save({
                    'epoch': ep,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'gates': self.invariance_gate.get_gates().cpu().numpy(),
                    'val_metrics': val_metrics,
                    'config': self.config
                }, best_ckpt_path)
                
        # Generate and save all final plots
        plot_training_history(history, save_path=os.path.join(self.plots_dir, 'training_curves.png'))
        
        # Load best weights for final artifacts
        checkpoint = torch.load(best_ckpt_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        test_metrics, y_true, y_pred, y_probs, embeddings = self.evaluate(val_loader)
        
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=['Left Fist', 'Right Fist'] if self.config['model']['n_classes'] == 2 else None,
            save_path=os.path.join(self.plots_dir, 'confusion_matrix.png')
        )
        
        region_names = list(self.config['regions'].keys())
        gates = checkpoint['gates'].tolist()
        plot_invariance_gates(gates, region_names, save_path=os.path.join(self.plots_dir, 'invariance_gates.png'))
        
        plot_tsne_embeddings(
            embeddings, y_true,
            class_names=['Left Fist', 'Right Fist'] if self.config['model']['n_classes'] == 2 else None,
            save_path=os.path.join(self.plots_dir, 'tsne_latent_manifold.png')
        )
        
        print("\n" + "="*50)
        print("FINAL EVALUATION METRICS (Best Checkpoint)")
        print("="*50)
        print(format_metrics_table(test_metrics))
        print("="*50)
        print(f"Artifacts saved to: {self.plots_dir} and {self.checkpoints_dir}")
        return test_metrics, history

