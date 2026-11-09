"""
residual_v2.py — Ridge residual decoding grid (Step 5, Task 2).
All labels from labels_clean.py only. Seed 0. Fast models only.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import labels_clean as lc

FEATS   = r"E:\audit_scratch\feats_v2\feats_trial.npz"
OUT_CSV = r"E:\audit_scratch\results_v2\residual_folds.csv"
SEED    = 0
ALPHAS  = [1, 10, 100, 1000, 10000]

# peripheral channel order from windowed_loader.py lines 47-56 (channels 32-39):
# features = [means(8), stds(8), ptp(8)]  →  first 8 = means, next 8 = stds, last 8 = ptp
# channel order (dat indices 32-39): GSR, Respiration, Plethysmograph, Temperature,
#   EMG_Zyg, EMG_Trap, EOG_Horiz, EOG_Vert  (windowed_loader preserves data channel order)
PERIPH_NAMES = ["GSR","Resp","Pleth","Temp","EMG_Zyg","EMG_Trap","EOG_H","EOG_V"]

def bootstrap_ci(v, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    b = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def inner_alpha(Xtr, ytr, gtr, alphas=ALPHAS):
    """Global 5-fold GroupKFold, score = mean per-inner-subj Spearman."""
    gkf = GroupKFold(n_splits=5)
    best_a, best_s = alphas[-1], -1e9    # ties -> larger alpha
    for a in alphas:
        scores = []
        for ti, vi in gkf.split(Xtr, ytr, gtr):
            sc = StandardScaler().fit(Xtr[ti])
            m = Ridge(alpha=a, random_state=SEED)
            m.fit(sc.transform(Xtr[ti]), ytr[ti])
            pred = m.predict(sc.transform(Xtr[vi]))
            gvi = gtr[vi]
            for g in np.unique(gvi):
                idx = gvi == g
                if idx.sum() < 2: continue
                r, _ = stats.spearmanr(pred[idx], ytr[vi][idx])
                if np.isfinite(r): scores.append(r)
        s = np.mean(scores) if scores else -1e9
        if s > best_s + 1e-12:
            best_s, best_a = s, a
    return best_a

def subjz(X_s):
    """Z-score each feature over the 40 videos for one subject."""
    m = X_s.mean(0); sd = X_s.std(0) + 1e-8
    return (X_s - m) / sd

def make_features(feat_mat, variant, subj_ids, vid_ids, T_subjects):
    """
    feat_mat: (1280, D)
    variant: 'raw' | 'subjz' | 'subjz_vc'
    Returns X (1280, D) with the chosen preprocessing.
    """
    n = feat_mat.shape[0]; D = feat_mat.shape[1]
    X = feat_mat.copy().astype(np.float64)
    if variant == "raw":
        return X
    # subjz: per-subject z-score across own 40 videos
    X_sz = np.zeros_like(X)
    for s in np.unique(subj_ids):
        idx = subj_ids == s
        X_sz[idx] = subjz(X[idx])
    if variant == "subjz":
        return X_sz
    # subjz_vc: additionally subtract per-video mean over T subjects
    # For each row, subtract the mean of X_sz at the same video over T_subjects (excluding self)
    X_vc = X_sz.copy()
    all_subjs = np.unique(subj_ids)
    T_set = set(T_subjects)
    for v in np.unique(vid_ids):
        vmask = vid_ids == v
        for s in all_subjs:
            smask = subj_ids == s
            row_idx = np.where(vmask & smask)[0]
            if len(row_idx) == 0: continue
            # mean over T subjects at this video, excluding s itself
            other = [i for i in np.where(vmask)[0] if subj_ids[i] != s and subj_ids[i] in T_set]
            if other:
                mean_v = X_sz[other].mean(0)
            else:
                mean_v = np.zeros(D)
            X_vc[row_idx] -= mean_v
    return X_vc

def run_residual(feat_mat, subj_a, vid_a, ratings, variant, alphas=ALPHAS):
    """LOSO residual Ridge. Returns list of dicts."""
    rows = []
    subjects = np.arange(1, 33)
    T_all = list(subjects)
    for h in subjects:
        T = [s for s in subjects if s != h]
        T_set = set(T)
        # precompute features for the FULL dataset under variant (T-based stats where needed)
        # For vc centering we need T context, so pass T_subjects
        X = make_features(feat_mat, variant, subj_a, vid_a, T)

        tr_idx = np.isin(subj_a, T)
        te_idx = subj_a == h

        Xtr = X[tr_idx]; Xte = X[te_idx]
        str_ = subj_a[tr_idx]; str_v = vid_a[tr_idx]
        vte_ = vid_a[te_idx]

        # residual targets for training subjects
        def res_for(s, v):
            # rating[s,v] - mean over s2 in T, s2!=s of rating[s2,v]
            others = [s2-1 for s2 in T if s2 != s]
            return float(ratings[s-1, v-1]) - float(np.mean(ratings[others, v-1]))

        ytr = np.array([res_for(str_[i], str_v[i]) for i in range(len(Xtr))])

        # true residual for held-out h
        # rating[h,v] - mean over s2 in T of rating[s2,v]
        T_idx = [s2-1 for s2 in T]
        true_res = np.array([float(ratings[h-1, v-1]) - float(ratings[T_idx, v-1].mean())
                              for v in vte_])

        # inner alpha selection
        a = inner_alpha(Xtr, ytr, str_, alphas)

        # fit on all T
        sc = StandardScaler().fit(Xtr)
        clf = Ridge(alpha=a, random_state=SEED)
        clf.fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))

        r = float(stats.spearmanr(pred, true_res)[0]) if len(np.unique(pred)) > 1 and len(np.unique(true_res)) > 1 else np.nan
        rows.append(dict(subject=int(h), alpha=a, spearman=r))
    return rows

def summarize(rows, tag=""):
    v = np.array([r["spearman"] for r in rows if not np.isnan(r["spearman"])])
    n = len(v)
    mu = float(v.mean()) if n else np.nan
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n else np.nan
    ci = bootstrap_ci(v) if n > 1 else (np.nan, np.nan)
    n_pos = int((v > 0).sum())
    _, wx_p = stats.wilcoxon(v) if n >= 10 else (None, np.nan)
    return dict(mean=mu, sd=sd, se=se, ci_lo=ci[0], ci_hi=ci[1],
                n_valid=n, n_pos=n_pos, wilcoxon_p=float(wx_p))

def main():
    t0 = time.time()
    z = np.load(FEATS)
    eeg  = z["eeg_de_trial"].reshape(1280, -1).astype(np.float64)  # (1280,128)
    per  = z["periph_trial"].astype(np.float64)                     # (1280,24)
    subj_a = z["subject"].astype(int)
    vid_a  = z["video"].astype(int)

    PE = np.concatenate([per, eeg], axis=1)  # (1280,152)
    feat_sets = {"P": per, "E": eeg, "PE": PE}

    RV = lc.get_ratings("valence")    # (32,40)
    RA = lc.get_ratings("arousal")

    targets   = ["valence","arousal"]
    variants  = ["raw","subjz","subjz_vc"]
    fsets     = ["P","E","PE"]

    all_rows = []
    combos = [(t, f, v) for t in targets for f in fsets for v in variants]
    print(f"Total combos: {len(combos)}", flush=True)

    for ci, (tgt, fs, var) in enumerate(combos):
        ratings = RV if tgt == "valence" else RA
        rows = run_residual(feat_sets[fs], subj_a, vid_a, ratings, var)
        for r in rows:
            r.update(target=tgt, fset=fs, variant=var)
            all_rows.append(r)
        elapsed = time.time() - t0
        if elapsed > 2400:   # 40-min guard
            print(f"  40-min guard hit at combo {ci+1}/{len(combos)}, stopping.", flush=True)
            break
        print(f"  {ci+1}/{len(combos)} {tgt}/{fs}/{var}  ({elapsed:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}  rows={len(df)}")

    # Summary table
    for tgt in targets:
        sub = df[df["target"] == tgt]
        rows_out = []
        for (fs, var), g in sub.groupby(["fset","variant"]):
            s = summarize(g.to_dict("records"), f"{tgt}/{fs}/{var}")
            s.update(target=tgt, fset=fs, variant=var)
            rows_out.append(s)
        rows_out.sort(key=lambda x: -x["mean"])
        print(f"\n{'='*100}")
        print(f"TARGET={tgt}")
        print(f"{'='*100}")
        print(f"{'fset':<4} {'var':<10} {'mean':>7} {'SD':>7} {'SE':>6} {'N':>4} "
              f"{'CI_lo':>7} {'CI_hi':>7} {'n_pos':>6} {'Wilcox_p':>10}")
        print("-"*100)
        for r in rows_out:
            print(f"{r['fset']:<4} {r['variant']:<10} {r['mean']:>7.4f} {r['sd']:>7.4f} "
                  f"{r['se']:>6.4f} {r['n_valid']:>4} {r['ci_lo']:>7.4f} {r['ci_hi']:>7.4f} "
                  f"{r['n_pos']:>6} {r['wilcoxon_p']:>10.4f}")

    # Identify best per target
    best = {}
    for tgt in targets:
        sub = df[df["target"] == tgt]
        means = sub.groupby(["fset","variant"])["spearman"].mean()
        best_key = means.idxmax()
        best[tgt] = best_key
        print(f"\nBEST {tgt}: fset={best_key[0]} variant={best_key[1]}  mean={means[best_key]:.4f}")

    # Save best info for ablations and perm
    import json
    with open(r"E:\audit_scratch\results_v2\residual_best.json", "w") as f:
        json.dump({t: list(v) for t, v in best.items()}, f)

    print(f"\nTotal runtime: {time.time()-t0:.0f} s")

if __name__ == "__main__":
    main()
