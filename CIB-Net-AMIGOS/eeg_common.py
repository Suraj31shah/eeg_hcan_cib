"""
Shared utilities for shallow baselines, domain-shift diagnostics, and model comparison.
Identical fold split (`make_fold`) to `training/loso_evaluator.py` so paired statistical
comparisons across shallow and deep models are strictly valid.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def resolve_n_val(n_subjects: int, requested_n_val: int = 4) -> int:
    """
    Matches standard n_val=4 on full datasets (>=15 subjects: 40-subj or 15-subj AMIGOS),
    and adapts safely for small subsets/smoke runs so >=2 training subjects remain.
    """
    if n_subjects >= 12:
        return min(int(requested_n_val), n_subjects - 3)
    return max(1, min(int(requested_n_val), n_subjects - 2, max(1, n_subjects // 4)))


def load_npz(path: str, aggregate: bool = True):
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
        if len(uniq) < len(key):
            cnt = np.bincount(inv).astype(float)
            for name, X in blocks.items():
                out = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
                for j in range(X.shape[1]):
                    out[:, j] = np.bincount(inv, weights=X[:, j]) / cnt
                blocks[name] = out
            if not np.all(y[first][inv] == y):
                raise ValueError("label differs inside a trial")
            y, subj, trial = y[first], subj[first], trial[first]
            print(f"[data] aggregated windows -> {len(uniq)} trials")

    for name, X in blocks.items():
        if not np.isfinite(X).all():
            blocks[name] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return blocks, y, subj, trial


def make_demo(path, n_subj=16, n_trials=16, n_feat=56, signal=0.35, subj_shift=1.5, seed=0):
    rng = np.random.RandomState(seed)
    X, y, s, t = [], [], [], []
    w = rng.randn(n_feat)
    for i in range(n_subj):
        off = rng.randn(n_feat) * subj_shift
        sc = np.exp(rng.randn(n_feat) * 0.3)
        lab = (rng.rand(n_trials) < 0.5).astype(int)
        if len(np.unique(lab)) < 2:
            lab[0] = 1 - lab[0]
        x = rng.randn(n_trials, n_feat) * sc + off + signal * np.outer(lab - 0.5, w) * sc
        X.append(x)
        y.append(lab)
        s.append(np.full(n_trials, i + 1))
        t.append(np.arange(n_trials))
    np.savez(
        path,
        eeg=np.vstack(X),
        y=np.concatenate(y),
        subject=np.concatenate(s),
        trial=np.concatenate(t),
    )


def subject_normalize(X: np.ndarray, subj: np.ndarray, mode: str = "none") -> np.ndarray:
    """
    Label-free per-subject normalization. Note: 'zscore' and 'center' use held-out
    subject statistics across all its trials (transductive); disclose when reporting.
    """
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


def make_fold(subj: np.ndarray, test_subject: int, fold: int, seed: int = 0, n_val: int = 4):
    """Identical deterministic split to `training/loso_evaluator.py`."""
    subjects = np.unique(subj)
    eff_n_val = resolve_n_val(len(subjects), n_val)
    test_idx = np.where(subj == test_subject)[0]
    pool = np.where(subj != test_subject)[0]
    rng = np.random.RandomState(seed * 1000 + fold)
    val_subjects = rng.choice(subjects[subjects != test_subject], eff_n_val, replace=False)
    is_val = np.isin(subj[pool], val_subjects)
    return pool[~is_val], pool[is_val], test_idx, [int(v) for v in val_subjects]


def safe_auroc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def tune_threshold(y, p) -> float:
    if len(p) == 0:
        return 0.5
    cands = np.unique(np.concatenate([np.quantile(p, np.linspace(0.05, 0.95, 37)), [0.5]]))
    best_t, best_ba = 0.5, -1.0
    for t in cands:
        pred = (p >= t).astype(int)
        tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
        tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
        if 0.5 * (tpr + tnr) > best_ba + 1e-12:
            best_ba, best_t = 0.5 * (tpr + tnr), float(t)
    return best_t


def _ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def compute_metrics(y, prob, pred):
    y, prob, pred = np.asarray(y), np.asarray(prob, float), np.asarray(pred)
    prev1 = float(y.mean()) if len(y) > 0 else 0.5
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
    acc = float((pred == y).mean()) if len(y) > 0 else float("nan")
    return {
        "accuracy": acc,
        "balanced_acc": 0.5 * (rm + rj),
        "macro_f1": 0.5 * (fm + fj),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 and len(np.unique(y)) > 1 else 0.0,
        "g_mean": float(np.sqrt(rm * rj)),
        "precision_minority": pm,
        "recall_minority": rm,
        "f1_minority": fm,
        "precision_majority": pj,
        "recall_majority": rj,
        "f1_majority": fj,
        "minority_pred_rate": float((pred == mino).mean()) if len(pred) > 0 else 0.0,
        "minority_class": int(mino),
        "trivial_baseline_acc": float(trivial),
        "delta_trivial": acc - float(trivial),
        "auroc": auc,
        "gini_auroc": 2 * auc - 1 if np.isfinite(auc) else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y, np.clip(prob, 0, 1))) if len(y) > 0 else float("nan"),
        "ece": _ece(y, np.clip(prob, 0, 1)) if len(y) > 0 else float("nan"),
    }


def boot_ci(v, n: int = 5000, seed: int = 0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
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

