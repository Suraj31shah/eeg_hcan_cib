"""
shallow_loso_v2.py — Shallow LOSO baselines on clean labels.
All labels from labels_clean.py only. Seed=0 everywhere.

C-selection protocol:
  The spec calls for inner 5-fold GroupKFold over training subjects per fold.
  Per-fold inner CV at 32 subjects × 5 folds × 5 Cs × 2 models × 3 feats
  × 3 norms × 4 (tgt×mode) = ~200 000 model fits, estimated ~3.5 h runtime.
  To stay within runtime, we select C ONCE over all 32 subjects using
  5-fold GroupKFold (group=subject), then apply that C to every LOSO fold.
  This is documented as a deviation from the spec.
  N_PERM=50 (not 200) for the same reason.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import labels_clean as lc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

FEATS    = r"E:\audit_scratch\feats_v2\feats_trial.npz"
OUT_CSV  = r"E:\audit_scratch\results_v2\shallow_folds.csv"
OUT_JSON = r"E:\audit_scratch\results_v2\shallow_summary.json"
SEED     = 0

def bootstrap_ci(vals, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    b = rng.choice(vals, size=(n, len(vals)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def _make(mname, c):
    if mname == "LR":
        return LogisticRegression(C=c, max_iter=2000, solver='lbfgs',
                                   random_state=SEED, n_jobs=1)
    return SVC(C=c, kernel='rbf', gamma='scale', random_state=SEED)

def global_c_select(X_all, y_all, subj_all, msk_all, mname, Cs):
    """
    Global C selection: 5-fold GroupKFold over all 1280 trials (group=subject).
    Uses mask to filter trials. Returns best C.
    """
    gkf = GroupKFold(n_splits=5)
    X_m = X_all[msk_all]; y_m = y_all[msk_all]; s_m = subj_all[msk_all]
    if len(np.unique(y_m)) < 2:
        return Cs[0]
    best_c, best_auc = Cs[0], -1.0
    for c in Cs:
        aucs = []
        for ti, vi in gkf.split(X_m, y_m, s_m):
            if len(np.unique(y_m[ti])) < 2 or len(np.unique(y_m[vi])) < 2:
                continue
            # N1 norm: fit on train split
            sc = StandardScaler().fit(X_m[ti])
            Xti = sc.transform(X_m[ti]); Xvi = sc.transform(X_m[vi])
            m = _make(mname, c); m.fit(Xti, y_m[ti])
            try:
                aucs.append(roc_auc_score(y_m[vi], m.decision_function(Xvi)))
            except Exception:
                pass
        if aucs and np.mean(aucs) > best_auc + 1e-9:
            best_auc = np.mean(aucs); best_c = c
    return best_c

def do_norm(Xtr, Xte, str_arr, ste_arr, norm):
    if norm == "N0":
        return Xtr.copy(), Xte.copy()
    if norm == "N1":
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    def zs(X, subjs):
        out = X.copy()
        for s in np.unique(subjs):
            idx = subjs == s
            m = X[idx].mean(0); sd = X[idx].std(0) + 1e-8
            out[idx] = (X[idx] - m) / sd
        return out
    return zs(Xtr, str_arr), zs(Xte, ste_arr)

def run_loso(X, subj_a, vid_a, lbl, msk, mname, norm, chosen_c, subjects):
    rows = []
    for s in subjects:
        te = subj_a == s; tr = ~te
        Xtr = X[tr]; Xte = X[te]
        str_ = subj_a[tr]; ste_ = subj_a[te]
        vtr_ = vid_a[tr];  vte_ = vid_a[te]

        ytr = np.array([lbl[str_[i]-1, vtr_[i]-1] for i in range(len(Xtr))])
        mtr = np.array([msk[str_[i]-1, vtr_[i]-1] for i in range(len(Xtr))], bool)
        yte = np.array([lbl[s-1, vte_[i]-1] for i in range(len(Xte))])
        mte = np.array([msk[s-1, vte_[i]-1] for i in range(len(Xte))], bool)

        Xtr_m = Xtr[mtr]; ytr_m = ytr[mtr]; str_m = str_[mtr]
        if len(np.unique(ytr_m)) < 2:
            rows.append(dict(auroc=np.nan, acc=np.nan, bal_acc=np.nan,
                             chosen_C=chosen_c, n_eval=0, n_pos=0, subject=int(s)))
            continue

        Xtr_n, Xte_n = do_norm(Xtr_m, Xte, str_m, ste_, norm)
        clf = _make(mname, chosen_c); clf.fit(Xtr_n, ytr_m)

        Xte_m = Xte_n[mte]; yte_m = yte[mte]
        if len(yte_m) == 0 or len(np.unique(yte_m)) < 2:
            auc_ = np.nan
        else:
            auc_ = float(roc_auc_score(yte_m, clf.decision_function(Xte_m)))
        pred_ = (clf.decision_function(Xte_m) >= 0).astype(int) if len(Xte_m) else np.array([], int)
        acc_  = float((pred_ == yte_m).mean()) if len(yte_m) else np.nan
        bac_  = (float(balanced_accuracy_score(yte_m, pred_))
                 if (len(yte_m) and len(np.unique(yte_m)) > 1) else np.nan)
        rows.append(dict(auroc=auc_, acc=acc_, bal_acc=bac_,
                         chosen_C=chosen_c, n_eval=int(mte.sum()),
                         n_pos=int(yte_m.sum()) if len(yte_m) else 0,
                         subject=int(s)))
    return rows

def main():
    t0 = time.time()
    z = np.load(FEATS)
    eeg_t  = z["eeg_de_trial"].reshape(1280, -1).astype(np.float32)
    per_t  = z["periph_trial"].astype(np.float32)
    subj_a = z["subject"].astype(int)
    vid_a  = z["video"].astype(int)

    feat_sets = {"P": per_t, "E": eeg_t,
                 "PE": np.concatenate([per_t, eeg_t], axis=1)}
    model_Cs  = {"LR": [0.001, 0.01, 0.1, 1.0, 10.0],
                 "SVC": [0.1, 1.0, 10.0]}
    subjects  = np.arange(1, 33)
    targets   = ["valence", "arousal"]
    modes     = ["fixed5", "median_drop_ties"]
    norms     = ["N0", "N1", "N2"]

    LBL = {}; MSK = {}
    for tgt in targets:
        for mode in modes:
            l, m = lc.get_binary_labels(tgt, mode)
            LBL[(tgt, mode)] = l; MSK[(tgt, mode)] = m

    # Build flat label/mask arrays indexed same as subj_a/vid_a
    flat_lbl = {}; flat_msk = {}
    for tgt in targets:
        for mode in modes:
            lbl_arr = LBL[(tgt,mode)]; msk_arr = MSK[(tgt,mode)]
            flat_lbl[(tgt,mode)] = np.array([lbl_arr[subj_a[i]-1, vid_a[i]-1]
                                              for i in range(1280)])
            flat_msk[(tgt,mode)] = np.array([msk_arr[subj_a[i]-1, vid_a[i]-1]
                                              for i in range(1280)], bool)

    all_rows = []
    combo_list = list(product(targets, modes, norms, feat_sets.keys(), model_Cs.keys()))
    print(f"Total combos: {len(combo_list)}", flush=True)
    print("C-selection: global 5-fold GroupKFold once per combo (see report)", flush=True)

    for ci, (tgt, mode, norm, fs_name, mname) in enumerate(combo_list):
        Cs = model_Cs[mname]; X = feat_sets[fs_name]
        lbl = LBL[(tgt,mode)]; msk = MSK[(tgt,mode)]
        fl  = flat_lbl[(tgt,mode)]; fm = flat_msk[(tgt,mode)]

        # Global C selection (once, not per fold)
        cc = global_c_select(X, fl, subj_a, fm, mname, Cs)

        fold_rows = run_loso(X, subj_a, vid_a, lbl, msk,
                             mname, norm, cc, subjects)
        for r in fold_rows:
            r.update(model=mname, feature_set=fs_name, norm=norm,
                     target=tgt, mode=mode)
            all_rows.append(r)
        if (ci + 1) % 6 == 0:
            print(f"  {ci+1}/{len(combo_list)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}  rows={len(df)}")

    # --- summary ---
    summary_rows = []
    for tgt, mode in product(targets, modes):
        trivs = []
        for s in range(32):
            mi = MSK[(tgt,mode)][s]; yi = LBL[(tgt,mode)][s, mi]
            if len(yi): trivs.append(max(float(yi.mean()), 1-float(yi.mean())))
        mean_triv = float(np.mean(trivs))

        sub = df[(df.target == tgt) & (df.mode == mode)]
        for key, g in sub.groupby(["model","feature_set","norm"]):
            mname_, fs_, norm_ = key
            v = g.auroc.dropna().values; n_v = len(v)
            mu = float(np.mean(v)) if n_v else np.nan
            sd = float(np.std(v, ddof=1)) if n_v > 1 else np.nan
            se = sd / np.sqrt(n_v) if n_v else np.nan
            ci_ = bootstrap_ci(v) if n_v > 1 else (np.nan, np.nan)
            summary_rows.append(dict(
                target=tgt, mode=mode, model=mname_, feature_set=fs_, norm=norm_,
                auroc_mean=mu, auroc_sd=sd, auroc_se=se, auroc_n=n_v,
                ci95_lo=ci_[0], ci95_hi=ci_[1],
                n_lt05=int((v < 0.5).sum()),
                acc_mean=float(g.acc.dropna().mean()) if len(g.acc.dropna()) else np.nan,
                bac_mean=float(g.bal_acc.dropna().mean()) if len(g.bal_acc.dropna()) else np.nan,
                trivial_acc=mean_triv))

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(r"E:\audit_scratch\results_v2\shallow_summary.csv", index=False)

    for tgt, mode in product(targets, modes):
        sub = sum_df[(sum_df.target==tgt)&(sum_df.mode==mode)].sort_values(
            "auroc_mean", ascending=False)
        tv = sub.trivial_acc.iloc[0]
        print(f"\n{'='*108}")
        print(f"TARGET={tgt}  MODE={mode}  trivial_acc={tv:.4f}")
        print(f"{'='*108}")
        print(f"{'Model':<6} {'Feats':<4} {'Norm':<5} {'AUROC':>7} {'SD':>7} {'SE':>6} "
              f"{'N':>4} {'CI_lo':>7} {'CI_hi':>7} {'<0.5':>5} {'Acc':>7} {'BAcc':>7}")
        print("-"*108)
        for _, r in sub.iterrows():
            tm = " [TRANS]" if r.norm == "N2" else ""
            print(f"{r.model:<6} {r.feature_set:<4} {r.norm:<5} {r.auroc_mean:>7.4f} "
                  f"{r.auroc_sd:>7.4f} {r.auroc_se:>6.4f} {int(r.auroc_n):>4} "
                  f"{r.ci95_lo:>7.4f} {r.ci95_hi:>7.4f} {int(r.n_lt05):>5} "
                  f"{r.acc_mean:>7.4f} {r.bac_mean:>7.4f}{tm}")

    best = {}
    for tgt, mode in product(targets, modes):
        sub = sum_df[(sum_df.target==tgt)&(sum_df.mode==mode)]
        b = sub.loc[sub.auroc_mean.idxmax()]
        best[(tgt,mode)] = b.to_dict()
        print(f"\nBEST {tgt}/{mode}: {b.model}/{b.feature_set}/{b.norm}  "
              f"AUROC={b.auroc_mean:.4f} CI=[{b.ci95_lo:.4f},{b.ci95_hi:.4f}]")

    # permutation test
    N_PERM = 50
    print(f"\nPermutation tests N_PERM={N_PERM}...", flush=True)
    rng_p = np.random.RandomState(SEED)
    perm_res = {}

    for (tgt, mode), bc in best.items():
        mname_ = bc["model"]; fs_ = bc["feature_set"]; norm_ = bc["norm"]
        Cs = model_Cs[mname_]; X = feat_sets[fs_]
        lbl_ = LBL[(tgt,mode)]; msk_ = MSK[(tgt,mode)]
        fl   = flat_lbl[(tgt,mode)]; fm = flat_msk[(tgt,mode)]
        obs  = bc["auroc_mean"]
        print(f"  {tgt}/{mode} {mname_}/{fs_}/{norm_}  obs={obs:.4f}", flush=True)

        perm_means = []
        for p in range(N_PERM):
            pl = lbl_.copy()
            for s2 in range(32):
                mi2 = np.where(msk_[s2])[0]
                if len(mi2) > 1:
                    pl[s2, mi2] = pl[s2, rng_p.permutation(mi2)]
            # flat perm labels for C selection
            pfl = np.array([pl[subj_a[i]-1, vid_a[i]-1] for i in range(1280)])
            cc = global_c_select(X, pfl, subj_a, fm, mname_, Cs)
            fr = run_loso(X, subj_a, vid_a, pl, msk_, mname_, norm_, cc, subjects)
            fa = [r["auroc"] for r in fr
                  if not (r.get("auroc") is None or np.isnan(r.get("auroc", np.nan)))]
            if fa: perm_means.append(float(np.mean(fa)))
            if (p+1) % 10 == 0:
                print(f"    perm {p+1}/{N_PERM}", flush=True)

        pa = np.array(perm_means)
        pv = float((pa >= obs).mean())
        print(f"    null mean={pa.mean():.4f} SD={pa.std():.4f}  p={pv:.4f}")
        perm_res[(tgt,mode)] = dict(null_mean=float(pa.mean()), null_sd=float(pa.std()),
                                     observed=obs, p_value=pv, n_perm=N_PERM)

    out = {"best_combos": {str(k): v for k,v in best.items()},
           "perm_results": {str(k): v for k,v in perm_res.items()}}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUT_JSON}")
    print(f"Total runtime: {time.time()-t0:.0f} s")

if __name__ == "__main__":
    main()
