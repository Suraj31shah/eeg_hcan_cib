"""
video_prior_baseline.py
No-EEG video-level prior baselines on clean vs altered labels.
Results saved to E:\audit_scratch\video_prior_results.json
"""

import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import labels_clean as lc

# ------------------------------------------------------------------ loaders
def _get_binary_and_mask(source, ratings_arr, target, mode):
    """
    source  : 'clean' | 'altered'
    ratings_arr : (32,40,4) float64 — the ratings to binarise
    Returns labels (32,40) int, mask (32,40) bool
    """
    col = {"valence": 0, "arousal": 1}[target]
    ratings = ratings_arr[:, :, col]
    if mode == "fixed5":
        labels = (ratings > 5.0).astype(int)
        mask   = np.ones((32, 40), dtype=bool)
    elif mode == "median_drop_ties":
        labels = np.zeros((32, 40), dtype=int)
        mask   = np.ones((32, 40), dtype=bool)
        for s in range(32):
            m = float(np.median(ratings[s]))
            labels[s] = (ratings[s] > m).astype(int)
            mask[s]   = (ratings[s] != m)
    else:
        raise ValueError(mode)
    return labels, mask

# ------------------------------------------------------------------ bootstrap
def _bootstrap_ci(values, n_boot=10000, seed=0):
    rng = np.random.RandomState(seed)
    vals = np.array(values)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return lo, hi

# ------------------------------------------------------------------ null SD
def _null_auroc_sd(n1, n2):
    return float(np.sqrt((n1 + n2 + 1) / (12 * n1 * n2)))

# ------------------------------------------------------------------ main
def run_baselines():
    xls_arr = lc.load_ratings()                                     # (32,40,4)
    dat_arr = np.load(r"E:\audit_scratch\labels_from_dat.npy")      # (32,40,4)

    records = {}   # keyed by combo string
    combo_rows = []

    # ---- per-video mean table (clean labels) ----
    xls_val = xls_arr[:,:,0]   # (32,40)
    xls_aro = xls_arr[:,:,1]
    vid_mean_val = xls_val.mean(axis=0)    # (40,)
    vid_mean_aro = xls_aro.mean(axis=0)

    vid_table = []
    for v in range(40):
        vid_table.append({"video_id": v+1,
                          "mean_valence": float(vid_mean_val[v]),
                          "mean_arousal": float(vid_mean_aro[v])})

    # ---- continuous prior + Spearman ----
    spearman_rows = []
    for source, ratings_arr in [("clean", xls_arr), ("altered", dat_arr)]:
        for target in ["valence", "arousal"]:
            col = {"valence": 0, "arousal": 1}[target]
            raw = ratings_arr[:, :, col]   # (32,40)
            spear_vals = []
            for s in range(32):
                # cprior[v] = mean of raw[s', v] for s' != s
                cprior = np.delete(raw, s, axis=0).mean(axis=0)  # (40,)
                r, _ = stats.spearmanr(cprior, raw[s])
                spear_vals.append(float(r))
            spear_vals_arr = np.array(spear_vals)
            mean_s = float(np.nanmean(spear_vals_arr))
            sd_s   = float(np.nanstd(spear_vals_arr, ddof=1))
            n_v    = int(np.sum(np.isfinite(spear_vals_arr)))
            se_s   = sd_s / np.sqrt(n_v) if n_v > 0 else float("nan")
            spearman_rows.append({
                "source": source, "target": target,
                "mean": mean_s, "sd": sd_s, "se": se_s, "n": n_v
            })

    # ---- binary prior baselines ----
    for source, ratings_arr in [("clean", xls_arr), ("altered", dat_arr)]:
        for target in ["valence", "arousal"]:
            for mode in ["fixed5", "median_drop_ties"]:
                key = f"{source}_{target}_{mode}"
                labels, mask = _get_binary_and_mask(source, ratings_arr,
                                                     target, mode)

                auroc_vals, acc_vals, trivial_vals = [], [], []
                n_nan_auroc = 0
                all_probs, all_lbls = [], []

                for s in range(32):
                    # training pool = all other subjects
                    other = [i for i in range(32) if i != s]
                    # prior_score[v] = mean over s' with mask[s',v]==True
                    prior = np.zeros(40, dtype=float)
                    for v in range(40):
                        pool = [labels[i, v] for i in other if mask[i, v]]
                        prior[v] = float(np.mean(pool)) if pool else 0.5

                    # held-out subject masked-in trials
                    s_mask = mask[s]           # (40,)
                    s_lbl  = labels[s][s_mask] # 1-D
                    s_pr   = prior[s_mask]     # 1-D

                    if len(s_lbl) == 0:
                        n_nan_auroc += 1
                        continue

                    # AUROC
                    unique_cls = np.unique(s_lbl)
                    if len(unique_cls) < 2:
                        auroc_vals.append(float("nan"))
                        n_nan_auroc += 1
                    else:
                        auc = float(roc_auc_score(s_lbl, s_pr))
                        auroc_vals.append(auc)
                        all_probs.extend(s_pr.tolist())
                        all_lbls.extend(s_lbl.tolist())

                    # Accuracy: top-half by prior score over all 40 videos
                    rank_40 = np.argsort(-prior, kind="stable")  # descending
                    top_half = np.zeros(40, dtype=int)
                    top_half[rank_40[:20]] = 1
                    acc_v  = float((top_half[s_mask] == s_lbl).mean())
                    acc_vals.append(acc_v)

                    # Trivial accuracy
                    p_pos = float(s_lbl.mean())
                    trivial_vals.append(max(p_pos, 1 - p_pos))

                # Aggregate
                valid_auroc = [v for v in auroc_vals if not np.isnan(v)]
                n_valid     = len(valid_auroc)
                mean_auroc  = float(np.mean(valid_auroc)) if valid_auroc else float("nan")
                sd_auroc    = float(np.std(valid_auroc, ddof=1)) if len(valid_auroc) > 1 else float("nan")
                se_auroc    = sd_auroc / np.sqrt(n_valid) if n_valid > 0 else float("nan")
                ci_lo, ci_hi = _bootstrap_ci(valid_auroc) if len(valid_auroc) > 1 else (float("nan"), float("nan"))

                mean_acc   = float(np.mean(acc_vals))  if acc_vals   else float("nan")
                sd_acc     = float(np.std(acc_vals, ddof=1)) if len(acc_vals)>1 else float("nan")
                se_acc     = sd_acc / np.sqrt(len(acc_vals)) if acc_vals else float("nan")
                ci_lo_acc, ci_hi_acc = _bootstrap_ci(acc_vals) if len(acc_vals) > 1 else (float("nan"), float("nan"))

                mean_triv  = float(np.mean(trivial_vals)) if trivial_vals else float("nan")

                # Pooled AUROC
                if len(set(all_lbls)) == 2:
                    pooled_auc = float(roc_auc_score(all_lbls, all_probs))
                else:
                    pooled_auc = float("nan")

                # n1/n2 for null SD: approximate using mean per-subject counts
                total_pos = sum(int(labels[s, mask[s]].sum()) for s in range(32) if mask[s].sum() > 0)
                total_n   = sum(int(mask[s].sum()) for s in range(32))
                if n_valid > 0:
                    avg_n = total_n / 32.0
                    avg_p = total_pos / total_n
                    n1_approx = max(1, round(avg_n * avg_p))
                    n2_approx = max(1, round(avg_n * (1 - avg_p)))
                    null_sd = _null_auroc_sd(n1_approx, n2_approx)
                else:
                    null_sd = float("nan")

                records[key] = {
                    "source": source, "target": target, "mode": mode,
                    "auroc_mean": mean_auroc, "auroc_sd": sd_auroc,
                    "auroc_se": se_auroc, "auroc_n": n_valid,
                    "auroc_ci95_lo": ci_lo, "auroc_ci95_hi": ci_hi,
                    "auroc_nan_subjects": n_nan_auroc,
                    "acc_mean": mean_acc, "acc_sd": sd_acc,
                    "acc_se": se_acc, "acc_n": len(acc_vals),
                    "acc_ci95_lo": ci_lo_acc, "acc_ci95_hi": ci_hi_acc,
                    "trivial_acc_mean": mean_triv,
                    "pooled_auroc": pooled_auc,
                    "null_auroc_sd_approx": null_sd,
                    "null_sd_n1": n1_approx if not np.isnan(null_sd) else None,
                    "null_sd_n2": n2_approx if not np.isnan(null_sd) else None,
                }

    return records, spearman_rows, vid_table


if __name__ == "__main__":
    records, spearman_rows, vid_table = run_baselines()

    # Save JSON
    out = {
        "combos": records,
        "spearman": spearman_rows,
        "video_means": vid_table,
    }
    json_path = r"E:\audit_scratch\video_prior_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {json_path}")

    # ---- Print results table ----
    print("\n" + "="*100)
    print("AUROC and ACCURACY TABLE (8 combos)")
    print("="*100)
    hdr = (f"{'Source':<8} {'Target':<8} {'Mode':<20} "
           f"{'AUROC_mean':>10} {'AUROC_sd':>9} {'AUROC_se':>9} {'AUROC_n':>7} "
           f"{'CI95_lo':>8} {'CI95_hi':>8} {'NaN_s':>6} "
           f"{'Acc_mean':>9} {'Acc_sd':>8} {'Acc_se':>8} "
           f"{'Trivial':>8} {'Pooled_AUC':>11} "
           f"{'Null_SD':>9} {'n1':>5} {'n2':>5}")
    print(hdr)
    print("-"*100)
    for k, r in records.items():
        print(f"{r['source']:<8} {r['target']:<8} {r['mode']:<20} "
              f"{r['auroc_mean']:>10.4f} {r['auroc_sd']:>9.4f} {r['auroc_se']:>9.4f} "
              f"{r['auroc_n']:>7} "
              f"{r['auroc_ci95_lo']:>8.4f} {r['auroc_ci95_hi']:>8.4f} "
              f"{r['auroc_nan_subjects']:>6} "
              f"{r['acc_mean']:>9.4f} {r['acc_sd']:>8.4f} {r['acc_se']:>8.4f} "
              f"{r['trivial_acc_mean']:>8.4f} {r['pooled_auroc']:>11.4f} "
              f"{r['null_auroc_sd_approx']:>9.4f} "
              f"{r['null_sd_n1']:>5} {r['null_sd_n2']:>5}")

    print("\n" + "="*80)
    print("SPEARMAN CORRELATION (continuous prior vs raw ratings)")
    print("="*80)
    print(f"{'Source':<8} {'Target':<8} {'Mean':>8} {'SD':>8} {'SE':>8} {'N':>5}")
    for row in spearman_rows:
        print(f"{row['source']:<8} {row['target']:<8} "
              f"{row['mean']:>8.4f} {row['sd']:>8.4f} {row['se']:>8.4f} {row['n']:>5}")

    print("\n" + "="*70)
    print("NULL SD of AUROC formula: sqrt((n1+n2+1)/(12*n1*n2))")
    print("="*70)
    for k, r in records.items():
        ns = r.get("null_auroc_sd_approx", float("nan"))
        n1 = r.get("null_sd_n1"); n2 = r.get("null_sd_n2")
        obs_sd = r["auroc_sd"]
        print(f"{r['source']:<8} {r['target']:<8} {r['mode']:<20}  "
              f"null_SD={ns:.4f} (n1={n1},n2={n2})  observed_SD={obs_sd:.4f}")

    print("\n" + "="*70)
    print("VIDEO MEAN RATINGS TABLE (clean labels, all 32 subjects)")
    print("="*70)
    print(f"{'video_id':>8}  {'mean_valence':>12}  {'mean_arousal':>12}")
    for row in vid_table:
        print(f"{row['video_id']:>8}  {row['mean_valence']:>12.4f}  {row['mean_arousal']:>12.4f}")
