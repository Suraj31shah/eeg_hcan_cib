"""
Domain-shift diagnostics. Answers: is cross-subject failure caused by subject shift,
or is there no learnable signal in these features/labels at all?

D0  Label prior per subject (how skewed are the labels?)
D1  Within-subject AUROC (blocked + shuffled CV)  vs  cross-subject (LOSO) AUROC, same model
D2  Subject-ID decodability from held-out TRIALS of known subjects
    (raw vs per-subject normalized features), against chance = 1 / n_subjects
D3  Variance decomposition: share of feature variance explained by subject vs by emotion label
D4  Effect of per-subject normalization on LOSO AUROC (paired across subjects)

Works on any feature matrix with the npz layout in eeg_common.py, including embeddings
exported from a trained deep model (store them as 'eeg').

Usage
  python domain_shift_diagnostics.py --data feats.npz --features eeg --out results/diag
  python domain_shift_diagnostics.py --demo
"""
import argparse
import json
import os
import warnings
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_common import load_npz, make_demo, subject_normalize, safe_auroc, boot_ci
from shallow_baselines import run_loso

warnings.filterwarnings("ignore")


def lr(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def one_sided(a, ref=0.5):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    try:
        return float(wilcoxon(a - ref, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------------------ D0
def d0_priors(y, subj):
    rates = np.array([y[subj == s].mean() for s in np.unique(subj)])
    maj = np.maximum(rates, 1 - rates)
    return {"pos_rate_mean": float(rates.mean()), "pos_rate_min": float(rates.min()),
            "pos_rate_max": float(rates.max()), "trivial_acc_mean": float(maj.mean()),
            "n_subjects_majority_ge_0.7": int((maj >= 0.7).sum()),
            "n_subjects": int(len(rates))}


# ------------------------------------------------------------------------------ D1
def within_subject_auc(X, y, subj, trial, C, mode, k=5, repeats=3, seed=0):
    out = {}
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        if trial is not None:
            idx = idx[np.argsort(trial[idx])]
        Xs, ys = X[idx], y[idx]
        aucs = []
        for r in range(1 if mode == "blocked" else repeats):
            if mode == "blocked":  # contiguous trial blocks: guards against slow drift leaking
                splits = KFold(k, shuffle=False).split(Xs)
            else:
                splits = StratifiedKFold(k, shuffle=True, random_state=seed + r).split(Xs, ys)
            p = np.full(len(ys), 0.5)
            for tr, te in splits:
                if len(np.unique(ys[tr])) < 2:
                    continue
                p[te] = lr(C).fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
            aucs.append(safe_auroc(ys, p))
        out[int(s)] = float(np.nanmean(aucs))
    return out


def loso_fixed_auc(X, y, subj, C):
    out = {}
    for s in np.unique(subj):
        te = subj == s
        out[int(s)] = safe_auroc(y[te], lr(C).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1])
    return out


# ------------------------------------------------------------------------------ D2
def _norm_from_ref(X, subj, ref_mask, mode):
    """Normalize every row with its subject's statistics computed on reference rows only.
    Using ALL rows would force each subject's mean to exactly 0, which makes held-out
    trials systematically anti-correlated with their own subject and drives accuracy
    BELOW chance (an artifact, not a sign that identity was removed)."""
    if mode == "none":
        return X
    out = X.copy()
    for s in np.unique(subj):
        m = subj == s
        r = m & ref_mask
        mu = X[r].mean(0)
        out[m] = X[m] - mu
        if mode == "zscore":
            out[m] /= X[r].std(0) + 1e-8
    return out


def subject_id_acc(X, subj, mode="none", C=0.1, k=5, seed=0):
    ids = np.unique(subj)
    ymap = np.searchsorted(ids, subj)
    k = min(k, np.bincount(ymap).min())
    if k < 2:
        return float("nan")
    accs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, ymap):
        ref = np.zeros(len(X), bool)
        ref[tr] = True
        Xn = _norm_from_ref(X, subj, ref, mode)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
        accs.append((clf.fit(Xn[tr], ymap[tr]).predict(Xn[te]) == ymap[te]).mean())
    return float(np.mean(accs))


# ------------------------------------------------------------------------------ D3
def eta2(X, groups):
    gm = X.mean(0)
    sst = ((X - gm) ** 2).sum(0) + 1e-12
    ssb = np.zeros(X.shape[1])
    for g in np.unique(groups):
        m = groups == g
        ssb += m.sum() * (X[m].mean(0) - gm) ** 2
    return ssb / sst


def d3_variance(X, y, subj):
    e_subj = eta2(X, subj)
    Xc = subject_normalize(X, subj, "center")  # remove subject means, then look at label
    e_lab = eta2(Xc, y)
    return e_subj, e_lab


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--features", default="eeg", choices=["eeg", "periph", "concat"])
    ap.add_argument("--out", default="results/diag")
    ap.add_argument("--C", type=float, default=0.01, help="LR C for D1 (fixed, no tuning)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        args.data = os.path.join(args.out, "demo.npz")
        make_demo(args.data, signal=0.3, subj_shift=1.5)
    blocks, y, subj, trial = load_npz(args.data)
    if args.features == "concat":
        X = np.hstack([blocks["eeg"], blocks["periph"]])
        blocks_run = {"eeg": X}
    else:
        X = blocks[args.features]
        blocks_run = {"eeg": X}  # run_loso reads variant 'LR_eeg'
    S = len(np.unique(subj))
    R = {"n_trials": int(len(y)), "n_subjects": int(S), "n_features": int(X.shape[1])}
    print(f"[data] {len(y)} trials, {S} subjects, {X.shape[1]} features ({args.features})")

    # D0
    R["D0_label_priors"] = d0_priors(y, subj)
    print("\nD0 label priors:", R["D0_label_priors"])

    # D1
    print("\nD1 within-subject vs cross-subject (LR, fixed C, no tuning)")
    Xz = subject_normalize(X, subj, "zscore")
    w_block = within_subject_auc(Xz, y, subj, trial, args.C, "blocked")
    w_shuf = within_subject_auc(Xz, y, subj, trial, args.C, "shuffled", seed=args.seed)
    l_raw = loso_fixed_auc(X, y, subj, args.C)
    l_z = loso_fixed_auc(Xz, y, subj, args.C)
    ids = sorted(w_block)
    arr = {k: np.array([v[i] for i in ids]) for k, v in
           {"within_blocked": w_block, "within_shuffled": w_shuf,
            "loso_raw": l_raw, "loso_zscore": l_z}.items()}
    R["D1"] = {}
    for k, v in arr.items():
        lo, hi = boot_ci(v)
        R["D1"][k] = {"mean_auroc": float(np.nanmean(v)), "ci95": [lo, hi],
                      "p_greater_than_0.5": one_sided(v)}
        print(f"  {k:16s} AUROC {np.nanmean(v):.3f} [{lo:.3f}, {hi:.3f}]  p(>0.5)={one_sided(v):.3f}")

    # D2
    print("\nD2 subject-ID decodability (held-out trials of known subjects)")
    chance = 1.0 / S
    R["D2"] = {"chance": chance}
    for mode in ("none", "center", "zscore"):
        acc = subject_id_acc(X, subj, mode, seed=args.seed)
        R["D2"][mode] = acc
        print(f"  norm={mode:7s} subject-ID accuracy {acc:.3f}  (chance {chance:.3f}, "
              f"x{acc / chance:.1f})")

    # D3
    e_s, e_l = d3_variance(X, y, subj)
    R["D3"] = {"eta2_subject_mean": float(e_s.mean()), "eta2_subject_median": float(np.median(e_s)),
               "eta2_label_within_subject_mean": float(e_l.mean()),
               "ratio_subject_to_label": float(e_s.mean() / max(e_l.mean(), 1e-12))}
    print(f"\nD3 variance share  subject: {e_s.mean():.3f} (median {np.median(e_s):.3f}) | "
          f"label within subject: {e_l.mean():.4f} | ratio {R['D3']['ratio_subject_to_label']:.0f}x")

    # D4
    print("\nD4 normalization effect with validation-tuned LR (same folds as deep model)")
    aur = {}
    for norm in ("none", "center", "zscore"):
        folds, _ = run_loso(blocks_run, y, subj, "LR_eeg", norm, args.seed, 4, "balanced",
                            verbose=False)
        aur[norm] = {f["test_subject"]: f["auroc"] for f in folds}
        v = np.array(list(aur[norm].values()), float)
        print(f"  norm={norm:7s} LOSO AUROC {np.nanmean(v):.3f}  p(>0.5)={one_sided(v):.3f}")
    R["D4"] = {n: {"mean_auroc": float(np.nanmean(list(d.values())))} for n, d in aur.items()}
    keys = sorted(aur["none"])
    for n in ("center", "zscore"):
        d = np.array([aur[n][k] - aur["none"][k] for k in keys], float)
        try:
            p = float(wilcoxon(d[np.isfinite(d)]).pvalue)
        except ValueError:
            p = float("nan")
        R["D4"][n]["paired_diff_vs_none"] = float(np.nanmean(d))
        R["D4"][n]["paired_p"] = p
        print(f"  {n} - none: {np.nanmean(d):+.3f}  paired Wilcoxon p={p:.3f}")

    # Interpretation (hints, not proof)
    print("\n=== READING THE RESULTS ===")
    wb, ws = R["D1"]["within_blocked"], R["D1"]["within_shuffled"]
    lz = R["D1"]["loso_zscore"]
    if not (wb["p_greater_than_0.5"] < 0.05):
        print("* Within-subject AUROC (blocked) is not above chance: with THIS feature set and a\n"
              "  regularized linear model there is little learnable signal even inside one person.\n"
              "  Domain shift then cannot be the main bottleneck, so DANN is unlikely to help.\n"
              "  Revisit features, labels, preprocessing (and try a richer within-subject model).")
    elif not (lz["p_greater_than_0.5"] < 0.05):
        print("* Signal exists within subjects but does not transfer across them: subject shift or\n"
              "  subject-specific mappings are plausible. Per-subject normalization and domain\n"
              "  adaptation are justified next steps.")
    else:
        print("* Cross-subject transfer is above chance for the shallow model. Compare its AUROC to\n"
              "  the deep models; the deep pipeline may be the weak part.")
    if ws["mean_auroc"] - wb["mean_auroc"] > 0.05:
        print("* Shuffled within-subject CV beats blocked CV by >0.05: slow drift / trial order\n"
              "  leaks into random splits. Report the blocked number.")
    if R["D2"]["none"] / chance > 3:
        print(f"* Subject identity is highly decodable (x{R['D2']['none'] / chance:.0f} chance) from\n"
              "  raw features; normalization reduces it to "
              f"x{R['D2']['zscore'] / chance:.1f} chance.")
    if R["D4"]["zscore"]["paired_p"] < 0.05 and R["D4"]["zscore"]["paired_diff_vs_none"] > 0:
        print("* Per-subject z-scoring significantly improves LOSO AUROC: apply it to the deep\n"
              "  models before trying adversarial methods.")
    print("  These are hints; with ~20-30 subjects, use the CIs above and avoid over-reading.")

    json.dump(R, open(f"{args.out}/diagnostics.json", "w"), indent=2, default=float)
    print(f"\nSaved {args.out}/diagnostics.json")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        names = ["within_blocked", "loso_raw", "loso_zscore"]
        for i in range(len(ids)):
            ax[0].plot(range(3), [arr[n][i] for n in names], color="gray", alpha=0.4, lw=0.8)
        ax[0].boxplot([arr[n] for n in names], positions=range(3), widths=0.4)
        ax[0].axhline(0.5, ls="--", c="r")
        ax[0].set_xticks(range(3)); ax[0].set_xticklabels(names, rotation=15)
        ax[0].set_ylabel("per-subject AUROC"); ax[0].set_title("D1: within vs cross-subject")
        ax[1].bar(["none", "center", "zscore"], [R["D2"][m] for m in ("none", "center", "zscore")])
        ax[1].axhline(chance, ls="--", c="r"); ax[1].set_title("D2: subject-ID accuracy")
        ax[2].hist(e_s, bins=30, alpha=0.6, label="subject")
        ax[2].hist(e_l, bins=30, alpha=0.6, label="label | subject")
        ax[2].set_title("D3: variance share per feature"); ax[2].legend()
        plt.tight_layout(); plt.savefig(f"{args.out}/diagnostics.png", dpi=150)
        print(f"Saved {args.out}/diagnostics.png")


if __name__ == "__main__":
    main()
