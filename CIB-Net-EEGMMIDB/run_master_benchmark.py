import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.eegmmidb_loader import EEGMMIDBDataset
from run_comparative_benchmark import evaluate_model_pipeline

def generate_grand_comparison_table(results_dict, output_md_path):
    cols = [
        "Model Pipeline", "Accuracy", "Balanced Acc", "AUROC", "Gini AUROC",
        "PR AUC", "Macro F1", "MCC", "Delta Trivial", "Recall Minority", "ECE", "Minority Pred Rate"
    ]
    
    rows = []
    for model_name, res in results_dict.items():
        row = [
            f"**{model_name}**",
            f"`{res['accuracy']['mean']:.4f} ± {res['accuracy']['std']:.4f}`",
            f"`{res['balanced_acc']['mean']:.4f} ± {res['balanced_acc']['std']:.4f}`",
            f"`{res['auroc']['mean']:.4f} ± {res['auroc']['std']:.4f}`",
            f"`{res['gini_auroc']['mean']:+.4f} ± {res['gini_auroc']['std']:.4f}`",
            f"`{res['pr_auc']['mean']:.4f} ± {res['pr_auc']['std']:.4f}`",
            f"`{res['macro_f1']['mean']:.4f} ± {res['macro_f1']['std']:.4f}`",
            f"`{res['mcc']['mean']:+.4f} ± {res['mcc']['std']:.4f}`",
            f"`{res['delta_trivial']['mean']:+.4f} ± {res['delta_trivial']['std']:.4f}`",
            f"`{res['recall_minority']['mean']:.4f} ± {res['recall_minority']['std']:.4f}`",
            f"`{res['ece']['mean']:.4f} ± {res['ece']['std']:.4f}`",
            f"`{res['minority_pred_rate']['mean']:.4f} ± {res['minority_pred_rate']['std']:.4f}`"
        ]
        rows.append(row)
        
    df = pd.DataFrame(rows, columns=cols)
    md_content = "# Comprehensive EEGMMIDB Benchmark: Baselines vs CIB-Net\n\n"
    md_content += "## 1. Metric Comparison\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n---\n\n"
    md_content += "*(Evaluated under strict zero-leakage Leave-One-Subject-Out (LOSO) cross-validation on 64-channel EEGMMIDB Motor Imagery).*\n"
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[SUMMARY] Grand comparison table saved to: {output_md_path}")
    print("\n" + df.to_markdown(index=False))

def main():
    cache_file = "E:/kagglehub_cache/eegmmidb_cache_all109.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[BENCHMARK] Loading dataset from cache: {cache_file}")
    dataset = EEGMMIDBDataset(
        data_root="E:/kagglehub_cache/datasets/utsavsinghal2604/eeg-motorimagery-eegmmidb-v1-0-0/versions/1/physionet.org/files/eegmmidb/1.0.0",
        task="imagery_left_right",
        cache_path=cache_file
    )
    
    # Standardized 23-Fold LOSO benchmark (matching DREAMER scale)
    n_folds = 23
    epochs = 20
    
    models_to_run = [
        "PlainCNN (Baseline)",
        "ACRNN (Baseline)",
        "EEGNet (Baseline)",
        "CIB-Net (Phase 1-3)",
        "CIB-Net (V-SMOTE)",
        "CIB-Net (GAMO)",
        "CIB-Net (Z-score)",
        "CIB-Net (DeepSMOTE)"
    ]
    
    master_results = {}
    results_dir = "CIB-Net-EEGMMIDB/results"
    
    for m in models_to_run:
        model_key = m.replace(" (Baseline)", "").replace(" (Phase 1-3)", "")
        clean_name = model_key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        json_file = os.path.join(results_dir, f"fold_results_eegmmidb_{clean_name}.json")
        cm_file = os.path.join(results_dir, "plots", f"confusion_matrix_{clean_name}.png")
        norm_mode = "zscore" if "Z-score" in m else "none"
        
        # If already evaluated, load from disk
        if os.path.exists(json_file):
            print(f"[BENCHMARK] Loading existing results for {m} from {json_file}")
            with open(json_file) as f:
                saved = json.load(f)
                master_results[m] = saved["summary"]
        else:
            res = evaluate_model_pipeline(
                model_type=model_key,
                dataset=dataset,
                n_folds=n_folds,
                epochs=epochs,
                batch_size=64,
                lr=0.0005,
                device=device,
                normalize=norm_mode
            )
            master_results[m] = res
            
        # Update grand comparison markdown after each model finishes
        generate_grand_comparison_table(master_results, "EEGMMIDB_Comparative_Baselines_Report.md")
        
    print("\n" + "="*80)
    print(" ALL 8 ARCHITECTURES COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Audio chime notification (alerts user when away from screen)
    try:
        import winsound
        print("[NOTIFY] Sounding completion chime...")
        for _ in range(3):
            for freq in [523, 659, 784, 1046]: # C5, E5, G5, C6
                winsound.Beep(freq, 200)
            time.sleep(0.3)
    except Exception as e:
        print(f"[NOTIFY] Could not play sound chime: {e}")

if __name__ == "__main__":
    main()

