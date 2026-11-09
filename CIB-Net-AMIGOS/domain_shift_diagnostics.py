"""
Domain-shift diagnostics (D0-D4) for AMIGOS extracted DE / peripheral features:
  D0: Per-subject label prior skew
  D1: Within-subject blocked & shuffled CV vs cross-subject LOSO AUROC
  D2: Subject-ID decodability (raw vs center vs zscore)
  D3: Variance decomposition (eta^2 subject vs eta^2 label within-subject)
  D4: Paired effect of per-subject normalization under validation-tuned LR (`make_fold`)
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

from eeg_common import boot_ci, load_npz, make_demo, safe_auroc, subject_normalize
from shallow_baselines import run_loso

warnings.filterwarnings("ignore")


def lr(C: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
    )


def one_sided(a, ref: float = 0.5) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    try:
        return float(wilcoxon(a - ref, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def d0_priors(y: np.ndarray, subj: np.ndarray):
    rates = np.array([y[subj == s].mean() for s in np.unique(subj)])
    maj = np.maximum(rates, 1.0 - rates)
    return {
        "pos_rate_mean": float(rates.mean()),
        "pos_rate_min": float(rates.min()),
        "pos_rate_max": float(rates.max()),
        "trivial_acc_mean": float(maj.mean()),
        "n_subjects_majority_ge_0.7": int((maj >= 0.7).sum()),
        "n_subjects": int(len(rates)),
    }


def within_subject_auc(X, y, subj, trial, C: float, mode: str, k: int = 5, repeats: int = 3, seed: int = 0):
    out = {}
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        if trial is not None:
            idx = idx[np.argsort(trial[idx])]
        Xs, ys = X[idx], y[idx]
        if len(np.unique(ys)) < 2 or len(Xs) < 4:
            out[int(s)] = float("nan")
            continue

        min_cls = int(np.bincount(ys).min())
        aucs = []
        n_rep = 1 if mode == "blocked" else repeats
        for r in range(n_rep):
            if mode == "blocked":
                k_eff = min(k, len(Xs))
                splits = KFold(k_eff, shuffle=False).split(Xs)
            else:
                k_eff = min(k, min_cls)
                if k_eff >= 2:
                    splits = StratifiedKFold(k_eff, shuffle=True, random_state=seed + r).split(Xs, ys)
                else:
                    splits = KFold(min(k, len(Xs)), shuffle=True, random_state=seed + r).split(Xs)

            p = np.full(len(ys), 0.5)
            for tr, te in splits:
                if len(np.unique(ys[tr])) < 2:
                    continue
                p[te] = lr(C).fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
            aucs.append(safe_auroc(ys, p))
        out[int(s)] = float(np.nanmean(aucs))
    return out


def loso_fixed_auc(X, y, subj, C: float):
    out = {}
    for s in np.unique(subj):
        te = subj == s
        if len(np.unique(y[~te])) < 2 or len(np.unique(y[te])) < 2:
            out[int(s)] = float("nan")
            continue
        out[int(s)] = safe_auroc(y[te], lr(C).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1])
    return out


def _norm_from_ref(X, subj, ref_mask, mode: str):
    if mode == "none":
        return X
    out = X.copy()
    for s in np.unique(subj):
        m = subj == s
        r = m & ref_mask
        if not r.any():
            continue
        mu = X[r].mean(0)
        out[m] = X[m] - mu
        if mode == "zscore":
            out[m] /= X[r].std(0) + 1e-8
    return out


def subject_id_acc(X, subj, mode: str = "none", C: float = 0.1, k: int = 5, seed: int = 0) -> float:
    ids = np.unique(subj)
    ymap = np.searchsorted(ids, subj)
    k = min(k, int(np.bincount(ymap).min()))
    if k < 2:
        return float("nan")
    accs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, ymap):
        ref = np.zeros(len(X), dtype=bool)
        ref[tr] = True
        Xn = _norm_from_ref(X, subj, ref, mode)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
        accs.append(float((clf.fit(Xn[tr], ymap[tr]).predict(Xn[te]) == ymap[te]).mean()))
    return float(np.mean(accs))


def eta2(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    gm = X.mean(0)
    sst = ((X - gm) ** 2).sum(0) + 1e-12
    ssb = np.zeros(X.shape[1])
    for g in np.unique(groups):
        m = groups == g
        ssb += m.sum() * (X[m].mean(0) - gm) ** 2
    return ssb / sst


def d3_variance(X: np.ndarray, y: np.ndarray, subj: np.ndarray):
    e_subj = eta2(X, subj)
    Xc = subject_normalize(X, subj, "center")
    e_lab = eta2(Xc, y)
    return e_subj, e_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--features", default="eeg", choices=["eeg", "periph", "concat"])
    ap.add_argument("--out", default="results/diag")
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=4)
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
    else:
        X = blocks[args.features]
    blocks_run = {"eeg": X}

    S = len(np.unique(subj))
    R = {"n_trials": int(len(y)), "n_subjects": int(S), "n_features": int(X.shape[1]), "features": args.features}
    print(f"[data] {len(y)} trials, {S} subjects, {X.shape[1]} features ({args.features})")

    R["D0_label_priors"] = d0_priors(y, subj)
    print("\nD0 label priors:", R["D0_label_priors"])

    print("\nD1 within-subject vs cross-subject (LR, fixed C, no tuning)")
    Xz = subject_normalize(X, subj, "zscore")
    w_block = within_subject_auc(Xz, y, subj, trial, args.C, "blocked")
    w_shuf = within_subject_auc(Xz, y, subj, trial, args.C, "shuffled", seed=args.seed)
    l_raw = loso_fixed_auc(X, y, subj, args.C)
    l_z = loso_fixed_auc(Xz, y, subj, args.C)
    ids = sorted(w_block)
    arr = {
        k: np.array([v[i] for i in ids], dtype=float)
        for k, v in {
            "within_blocked": w_block,
            "within_shuffled": w_shuf,
            "loso_raw": l_raw,
            "loso_zscore": l_z,
        }.items()
    }
    R["D1"] = {}
    for k, v in arr.items():
        lo, hi = boot_ci(v)
        p_val = one_sided(v)
        R["D1"][k] = {"mean_auroc": float(np.nanmean(v)), "ci95": [lo, hi], "p_greater_than_0.5": p_val}
        print(f"  {k:16s} AUROC {np.nanmean(v):.3f} [{lo:.3f}, {hi:.3f}]  p(>0.5)={p_val:.3f}")

    print("\nD2 subject-ID decodability (held-out trials of known subjects)")
    chance = 1.0 / S
    R["D2"] = {"chance": chance}
    for mode in ("none", "center", "zscore"):
        acc = subject_id_acc(X, subj, mode, seed=args.seed)
        R["D2"][mode] = acc
        print(f"  norm={mode:7s} subject-ID accuracy {acc:.3f}  (chance {chance:.3f}, x{acc / chance:.1f})")

    e_s, e_l = d3_variance(X, y, subj)
    R["D3"] = {
        "eta2_subject_mean": float(e_s.mean()),
        "eta2_subject_median": float(np.median(e_s)),
        "eta2_label_within_subject_mean": float(e_l.mean()),
        "ratio_subject_to_label": float(e_s.mean() / max(e_l.mean(), 1e-12)),
    }
    print(
        f"\nD3 variance share  subject: {e_s.mean():.3f} (median {np.median(e_s):.3f}) | "
        f"label within subject: {e_l.mean():.4f} | ratio {R['D3']['ratio_subject_to_label']:.0f}x"
    )

    print("\nD4 normalization effect with validation-tuned LR (same folds as deep model)")
    aur = {}
    for norm in ("none", "center", "zscore"):
        folds, _ = run_loso(blocks_run, y, subj, "LR_eeg", norm, args.seed, args.n_val, "balanced", verbose=False)
        aur[norm] = {f["test_subject"]: f["auroc"] for f in folds}
        v = np.array(list(aur[norm].values()), float)
        print(f"  norm={norm:7s} LOSO AUROC {np.nanmean(v):.3f}  p(>0.5)={one_sided(v):.3f}")

    R["D4"] = {n: {"mean_auroc": float(np.nanmean(list(d.values())))} for n, d in aur.items()}
    keys = sorted(aur["none"])
    for n in ("center", "zscore"):
        d = np.array([aur[n][k] - aur["none"][k] for k in keys], float)
        d_fin = d[np.isfinite(d)]
        try:
            p = float(wilcoxon(d_fin).pvalue) if len(d_fin) > 0 else float("nan")
        except ValueError:
            p = float("nan")
        R["D4"][n]["paired_diff_vs_none"] = float(np.nanmean(d))
        R["D4"][n]["paired_p"] = p
        print(f"  {n} - none: {np.nanmean(d):+.3f}  paired Wilcoxon p={p:.3f}")

    with open(f"{args.out}/diagnostics.json", "w") as f:
        json.dump(R, f, indent=2, default=float)
    print(f"\nSaved {args.out}/diagnostics.json")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        names = ["within_blocked", "loso_raw", "loso_zscore"]
        for i in range(len(ids)):
            ax[0].plot(range(3), [arr[n][i] for n in names], color="gray", alpha=0.4, lw=0.8)
        valid_cols = [arr[n][np.isfinite(arr[n])] for n in names]
        ax[0].boxplot([c if len(c) else [0.5] for c in valid_cols], positions=range(3), widths=0.4)
        ax[0].axhline(0.5, ls="--", c="r")
        ax[0].set_xticks(range(3))
        ax[0].set_xticklabels(names, rotation=15)
        ax[0].set_ylabel("per-subject AUROC")
        ax[0].set_title(f"D1 ({args.features}): within vs cross-subject")

        ax[1].bar(["none", "center", "zscore"], [R["D2"][m] for m in ("none", "center", "zscore")])
        ax[1].axhline(chance, ls="--", c="r")
        ax[1].set_title(f"D2 ({args.features}): subject-ID accuracy")

        ax[2].hist(e_s, bins=min(30, max(10, len(e_s) // 2)), alpha=0.6, label="subject")
        ax[2].hist(e_l, bins=min(30, max(10, len(e_l) // 2)), alpha=0.6, label="label | subject")
        ax[2].set_title(f"D3 ({args.features}): variance share")
        ax[2].legend()

        plt.tight_layout()
        plt.savefig(f"{args.out}/diagnostics.png", dpi=150)
        plt.close(fig)
        print(f"Saved {args.out}/diagnostics.png")


if __name__ == "__main__":
    main()

