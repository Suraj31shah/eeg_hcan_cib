"""
Shallow baselines under the IDENTICAL validation-subject LOSO protocol as HCAN-CIB:
  - Held-out test subject never touched
  - Validation subjects drawn with `RandomState(seed*1000 + fold)`
  - Regularization C tuned on validation AUROC, decision threshold tuned on validation BA

Variants:
  T0                      majority/prior baseline
  LR_eeg, SVC_eeg         56-D DE EEG features (14 channels x 4 bands)
  LR_periph, SVC_periph   18-D peripheral features (ECG_R, ECG_L, GSR)
  LR_concat, SVC_concat   74-D early concatenation (EEG + Periph)
  LR_late                 late fusion probability average (LR_eeg + LR_periph)
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eeg_common import (
    boot_ci,
    compute_metrics,
    load_npz,
    make_demo,
    make_fold,
    safe_auroc,
    subject_normalize,
    tune_threshold,
)

warnings.filterwarnings("ignore")
LR_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
SVC_GRID = [0.1, 1.0, 10.0]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_select(Xfit, yfit, Xval, yval, Xte, kind, class_weight, seed):
    if len(np.unique(yfit)) < 2:
        prior = float(yfit.mean()) if len(yfit) > 0 else 0.5
        return np.full(len(Xval), prior), np.full(len(Xte), prior), None

    best = (-np.inf, None, None, None)
    for C in (LR_GRID if kind == "LR" else SVC_GRID):
        if kind == "LR":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=3000, class_weight=class_weight, random_state=seed),
            )
        else:
            clf = make_pipeline(
                StandardScaler(),
                SVC(C=C, kernel="rbf", gamma="scale", class_weight=class_weight, random_state=seed),
            )
        clf.fit(Xfit, yfit)
        if kind == "LR":
            pv, pt = clf.predict_proba(Xval)[:, 1], clf.predict_proba(Xte)[:, 1]
        else:
            pv, pt = _sigmoid(clf.decision_function(Xval)), _sigmoid(clf.decision_function(Xte))
        a = safe_auroc(yval, pv)
        score = a if np.isfinite(a) else -1.0
        if score > best[0]:
            best = (score, pv, pt, C)
    return best[1], best[2], best[3]


def _blocks_for(variant):
    if variant.endswith("_eeg"):
        return ["eeg"]
    if variant.endswith("_periph"):
        return ["periph"]
    if variant.endswith("_concat"):
        return ["eeg", "periph"]
    raise ValueError(variant)


def run_loso(blocks, y, subj, variant, norm="none", seed=0, n_val=4, class_weight="balanced", verbose=True):
    cw = None if class_weight == "none" else class_weight
    nblocks = {k: subject_normalize(v, subj, norm) for k, v in blocks.items()}
    subjects = np.unique(subj)
    folds, preds = [], []

    for fold, ts in enumerate(subjects):
        fit, val, te, val_subj = make_fold(subj, ts, fold, seed, n_val)
        yfit, yval, yte = y[fit], y[val], y[te]

        if variant == "T0":
            prior = float(yfit.mean())
            pv, pt, info = np.full(len(val), prior), np.full(len(te), prior), {}
            pred = np.full(len(te), int(prior >= 0.5))
            thr = 0.5
        else:
            if variant == "LR_late":
                outs = []
                for b in ("eeg", "periph"):
                    outs.append(fit_select(nblocks[b][fit], yfit, nblocks[b][val], yval, nblocks[b][te], "LR", cw, seed))
                pv = np.mean([o[0] for o in outs], axis=0)
                pt = np.mean([o[1] for o in outs], axis=0)
                info = {"C_eeg": outs[0][2], "C_periph": outs[1][2]}
            else:
                kind = variant.split("_")[0]
                X = np.hstack([nblocks[b] for b in _blocks_for(variant)])
                pv, pt, C = fit_select(X[fit], yfit, X[val], yval, X[te], kind, cw, seed)
                info = {"C": C}
            pred = (pt >= 0.5).astype(int)
            thr = tune_threshold(yval, pv)

        m = compute_metrics(yte, pt, pred)
        m["val_thr_metrics"] = compute_metrics(yte, pt, (pt >= thr).astype(int))
        m.update(
            {
                "val_threshold": thr,
                "val_auroc": safe_auroc(yval, pv) if variant != "T0" else 0.5,
                "test_subject": int(ts),
                "val_subjects": val_subj,
                "hparams": info,
            }
        )
        folds.append(m)
        preds.append({"subj": int(ts), "y": yte, "prob1": pt})

    if verbose:
        a = np.array([f["auroc"] for f in folds], float)
        n_fin = max(1, int(np.isfinite(a).sum()))
        se = float(np.nanstd(a, ddof=1) / np.sqrt(n_fin)) if n_fin > 1 else float("nan")
        print(f"  {variant:11s} norm={norm:6s} AUROC {np.nanmean(a):.4f} (SE {se:.4f})")
    return folds, preds


def summarize(name, folds):
    a = np.array([f["auroc"] for f in folds], float)
    a = a[np.isfinite(a)]
    lo, hi = boot_ci(a)
    try:
        p = float(wilcoxon(a - 0.5).pvalue) if len(a) > 0 else float("nan")
    except ValueError:
        p = float("nan")

    def mean(key, thr=False):
        return float(np.nanmean([(f["val_thr_metrics"] if thr else f)[key] for f in folds]))

    return (
        f"| {name} | {np.mean(a):.3f} [{lo:.3f}, {hi:.3f}] | {p:.3f} | {mean('accuracy'):.3f} | "
        f"{mean('delta_trivial'):+.3f} | {mean('balanced_acc', True):.3f} | "
        f"{mean('macro_f1', True):.3f} | {mean('recall_minority', True):.3f} | "
        f"{mean('minority_pred_rate', True):.3f} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="results/shallow")
    ap.add_argument("--norm", default="none", choices=["none", "center", "zscore"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=4)
    ap.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["T0", "LR_eeg", "SVC_eeg", "LR_periph", "SVC_periph", "LR_concat", "SVC_concat", "LR_late"],
    )
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        os.makedirs(args.out, exist_ok=True)
        args.data = os.path.join(args.out, "demo.npz")
        make_demo(args.data, signal=0.4)
        d = np.load(args.data)
        np.savez(
            args.data,
            **{k: d[k] for k in d.files},
            periph=d["eeg"][:, :18] + np.random.RandomState(1).randn(len(d["y"]), 18),
        )

    blocks, y, subj, _ = load_npz(args.data)
    variants = [
        v
        for v in args.variants
        if v == "T0"
        or all(b in blocks for b in (["eeg", "periph"] if v == "LR_late" else _blocks_for(v)))
    ]
    print(
        f"[data] {len(y)} trials, {len(np.unique(subj))} subjects, "
        f"positive rate {y.mean():.3f}, blocks {list(blocks)}; running {variants}"
    )

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for v in variants:
        folds, preds = run_loso(blocks, y, subj, v, args.norm, args.seed, args.n_val, args.class_weight)
        tag = f"{v}_{args.norm}"
        with open(f"{args.out}/fold_results_{tag}.json", "w") as f:
            json.dump(folds, f, indent=2, default=float)
        np.savez(
            f"{args.out}/trial_predictions_{tag}.npz",
            **{f"fold{i + 1}_{k}": val for i, p in enumerate(preds) for k, val in p.items()},
        )
        rows.append(summarize(tag, folds))

    table = (
        "| model | AUROC [95% boot CI] | p vs 0.5 (Wilcoxon) | acc | delta trivial | "
        "bal.acc (thr) | macro-F1 (thr) | minority recall (thr) | minority pred rate (thr) |\n"
        "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    )
    print("\n" + table)
    with open(f"{args.out}/summary_{args.norm}.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()

