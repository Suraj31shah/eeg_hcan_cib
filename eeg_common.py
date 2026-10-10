"""
Shared utilities for the shallow baselines and domain-shift diagnostics.

DATA FORMAT (one .npz file, trial-level or window-level):
    eeg      (N, F_eeg)   float   e.g. DE features, bands x channels flattened
    periph   (N, F_p)     float   OPTIONAL (peripheral / ECG features)
    y        (N,)         int     binary label 0/1 (already binarized: Task P or Task F)
    subject  (N,)         int     subject id
    trial    (N,)         int     OPTIONAL trial id within subject. If rows are windows,
                                  rows sharing (subject, trial) are averaged into one trial.

The fold split reproduces loso_evaluator_fixed.py exactly (same RandomState seed,
same np.unique ordering), so each held-out subject gets the SAME validation subjects
as the deep model. That makes paired comparisons valid.
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, brier_score_loss)


# ----------------------------------------------------------------------------- data
def load_npz(path, aggregate=True):
    z = np.load(path, allow_pickle=False)
    for k in ("y", "subject"):
        if k not in z.files:
            raise KeyError(f"'{k}' missing from {path}; found {z.files}")
    blocks = {k: np.asarray(z[k], dtype=np.float64) for k in ("eeg", "periph") if k in z.files}
    if not blocks:
        raise KeyError("need at least one of 'eeg' / 'periph'")
    y = np.asarray(z["y"]).astype(int)
    subj = np.asarray(z["subject"]).astype(int)
    trial = np.asarray(z["trial"]).astype(int) if "trial" in z.files else None

    if aggregate and trial is not None:
        key = subj.astype(np.int64) * 1_000_000 + trial
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
        if len(uniq) < len(key):  # rows are windows -> average to trial level
            cnt = np.bincount(inv).astype(float)
            for name, X in blocks.items():
                out = np.zeros((len(uniq), X.shape[1]))
                for j in range(X.shape[1]):
                    out[:, j] = np.bincount(inv, weights=X[:, j]) / cnt
                blocks[name] = out
            if not np.all(y[first][inv] == y):
                raise ValueError("label differs inside a trial")
            y, subj, trial = y[first], subj[first], trial[first]
            print(f"[data] aggregated windows -> {len(uniq)} trials")
    for name, X in blocks.items():
        if not np.isfinite(X).all():
            raise ValueError(f"non-finite values in '{name}'")
    return blocks, y, subj, trial


def make_demo(path, n_subj=20, n_trials=40, n_feat=30, signal=0.0, subj_shift=1.5, seed=0):
    """Synthetic data: subject-specific offsets/scales plus an optional weak label signal."""
    rng = np.random.RandomState(seed)
    X, y, s, t = [], [], [], []
    w = rng.randn(n_feat)
    for i in range(n_subj):
        off = rng.randn(n_feat) * subj_shift
        sc = np.exp(rng.randn(n_feat) * 0.3)
        lab = (rng.rand(n_trials) < 0.5).astype(int)
        x = rng.randn(n_trials, n_feat) * sc + off + signal * np.outer(lab - 0.5, w) * sc
        X.append(x); y.append(lab); s.append(np.full(n_trials, i + 1)); t.append(np.arange(n_trials))
    np.savez(path, eeg=np.vstack(X), y=np.concatenate(y), subject=np.concatenate(s),
             trial=np.concatenate(t))


# ------------------------------------------------------------------- normalization
def subject_normalize(X, subj, mode="none"):
    """Label-free per-subject normalization. Uses the held-out subject's own UNLABELED
    features too (transductive); disclose this in the paper."""
    if mode == "none":
        return X
    out = X.copy()
    for s in np.unique(subj):
        m = subj == s
        mu = X[m].mean(0)
        if mode == "center":
            out[m] = X[m] - mu
        elif mode == "zscore":
            out[m] = (X[m] - mu) / (X[m].std(0) + 1e-8)
        else:
            raise ValueError(mode)
    return out


# --------------------------------------------------------------------------- split
def make_fold(subj, test_subject, fold, seed=0, n_val=4):
    """Identical to loso_evaluator_fixed.py."""
    subjects = np.unique(subj)
    test_idx = np.where(subj == test_subject)[0]
    pool = np.where(subj != test_subject)[0]
    rng = np.random.RandomState(seed * 1000 + fold)
    val_subjects = rng.choice(subjects[subjects != test_subject], n_val, replace=False)
    is_val = np.isin(subj[pool], val_subjects)
    return pool[~is_val], pool[is_val], test_idx, [int(v) for v in val_subjects]


# -------------------------------------------------------------------------- metrics
def safe_auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def tune_threshold(y, p):
    cands = np.unique(np.concatenate([np.quantile(p, np.linspace(0.05, 0.95, 37)), [0.5]]))
    best_t, best_ba = 0.5, -1.0
    for t in cands:
        pred = (p >= t).astype(int)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
        if 0.5 * (tpr + tnr) > best_ba + 1e-12:
            best_ba, best_t = 0.5 * (tpr + tnr), float(t)
    return best_t


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def compute_metrics(y, prob, pred):
    """Per-subject metrics. Class roles are explicit: the minority class is the rarer
    class in this test subject (ties -> class 1). Per-class precision/recall/F1 are
    computed separately for each class; they are never aliased to each other."""
    y, prob, pred = np.asarray(y), np.asarray(prob, float), np.asarray(pred)
    prev1 = y.mean()
    mino = 1 if prev1 <= 0.5 else 0
    majo = 1 - mino

    def prf(c):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return float(p), float(r), float(f)

    pm, rm, fm = prf(mino)
    pj, rj, fj = prf(majo)
    auc = safe_auroc(y, prob)
    trivial = max(prev1, 1 - prev1)
    acc = float((pred == y).mean())
    return {
        "accuracy": acc, "balanced_acc": 0.5 * (rm + rj), "macro_f1": 0.5 * (fm + fj),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "g_mean": float(np.sqrt(rm * rj)),
        "precision_minority": pm, "recall_minority": rm, "f1_minority": fm,
        "precision_majority": pj, "recall_majority": rj, "f1_majority": fj,
        "minority_pred_rate": float((pred == mino).mean()), "minority_class": int(mino),
        "trivial_baseline_acc": float(trivial), "delta_trivial": acc - float(trivial),
        "auroc": auc, "gini_auroc": 2 * auc - 1 if np.isfinite(auc) else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y, np.clip(prob, 0, 1))),
        "ece": _ece(y, np.clip(prob, 0, 1)),
    }


# ----------------------------------------------------------------------- statistics
def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = np.random.RandomState(seed).choice(v, (n, len(v))).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
