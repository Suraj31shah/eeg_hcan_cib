"""
Compare fold-result JSONs from any models (deep or shallow) on a common metric.
Aligns folds by test_subject (not by position), reports mean +/- SE, bootstrap CI,
a test against chance, and paired Wilcoxon vs a reference with Holm correction.

Usage
  python compare_models.py --metric auroc --ref HCAN-CIB \
      HCAN-CIB=results/fold_results_deap.json  V-SMOTE=results/fold_results_deap_smote.json \
      EEGNet=baseline_fold_results_eegnet_deap.txt  LR_eeg=results/shallow/fold_results_LR_eeg_none.json
"""
import argparse
import json
import numpy as np
from scipy.stats import wilcoxon
from eeg_common import boot_ci, holm


def load(path, metric):
    d = json.load(open(path))
    out = {}
    for i, f in enumerate(d):
        out[int(f.get("test_subject", i + 1))] = float(f[metric])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path.json")
    ap.add_argument("--metric", default="auroc")
    ap.add_argument("--ref", default=None, help="reference model name (default: first)")
    ap.add_argument("--chance", type=float, default=0.5, help="chance level of the metric")
    ap.add_argument("--out", default=None, help="write markdown here")
    a = ap.parse_args()

    R = {}
    for m in a.models:
        name, path = m.split("=", 1)
        R[name] = load(path, a.metric)
    ref = a.ref or next(iter(R))
    common = sorted(set.intersection(*[set(v) for v in R.values()]))
    print(f"{len(common)} subjects common to all models")
    M = {k: np.array([v[s] for s in common], float) for k, v in R.items()}

    others = [k for k in M if k != ref]
    raw_p, diffs = [], {}
    for k in others:
        d = M[k] - M[ref]
        diffs[k] = d
        try:
            raw_p.append(float(wilcoxon(d).pvalue))
        except ValueError:
            raw_p.append(float("nan"))
    adj = dict(zip(others, holm(np.nan_to_num(raw_p, nan=1.0)))) if others else {}
    praw = dict(zip(others, raw_p))

    lines = [f"| model | {a.metric} mean +/- SE | 95% boot CI | p vs chance ({a.chance}) | "
             f"Diff vs {ref} | paired p | Holm p |", "|---|---|---|---|---|---|---|"]
    for k, v in M.items():
        v = v[np.isfinite(v)]
        lo, hi = boot_ci(v)
        try:
            pc = wilcoxon(v - a.chance).pvalue
        except ValueError:
            pc = float("nan")
        if k == ref:
            tail = "--- | --- | ---"
        else:
            tail = f"{np.nanmean(diffs[k]):+.3f} | {praw[k]:.3f} | {adj[k]:.3f}"
        lines.append(f"| {k} | {v.mean():.3f} +/- {v.std(ddof=1) / np.sqrt(len(v)):.3f} | "
                     f"[{lo:.3f}, {hi:.3f}] | {pc:.3f} | {tail} |")
    table = "\n".join(lines)
    print(table)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(table + "\n")


if __name__ == "__main__":
    main()
