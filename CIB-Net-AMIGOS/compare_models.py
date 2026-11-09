"""
Compare fold-result JSONs from any models (deep or shallow) on a common metric.
Aligns folds by test_subject, reports mean +/- SE, 95% bootstrap CI, Wilcoxon vs chance,
and paired Wilcoxon vs reference with Holm correction.
"""

import argparse
import json
import numpy as np
from scipy.stats import wilcoxon
from eeg_common import boot_ci, holm


def load(path: str, metric: str):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for i, fold in enumerate(d):
        out[int(fold.get("test_subject", i + 1))] = float(fold[metric])
    return out


def compare(model_specs, metric="auroc", ref=None, chance=0.5, out=None):
    R = {}
    for spec in model_specs:
        name, path = spec.split("=", 1)
        R[name] = load(path, metric)
    ref = ref or next(iter(R))
    common = sorted(set.intersection(*[set(v) for v in R.values()]))
    print(f"{len(common)} subjects common to all models")
    M = {k: np.array([v[s] for s in common], float) for k, v in R.items()}

    others = [k for k in M if k != ref]
    raw_p, diffs = [], {}
    for k in others:
        d = M[k] - M[ref]
        diffs[k] = d
        d_fin = d[np.isfinite(d)]
        try:
            raw_p.append(float(wilcoxon(d_fin).pvalue) if len(d_fin) > 0 else float("nan"))
        except ValueError:
            raw_p.append(float("nan"))

    adj = dict(zip(others, holm(np.nan_to_num(raw_p, nan=1.0)))) if others else {}
    praw = dict(zip(others, raw_p))

    lines = [
        f"| model | {metric} mean +/- SE | 95% boot CI | p vs chance ({chance}) | "
        f"Diff vs {ref} | paired p | Holm p |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in M.items():
        v_fin = v[np.isfinite(v)]
        lo, hi = boot_ci(v_fin)
        se = float(v_fin.std(ddof=1) / np.sqrt(len(v_fin))) if len(v_fin) > 1 else float("nan")
        try:
            pc = float(wilcoxon(v_fin - chance).pvalue) if len(v_fin) > 0 else float("nan")
        except ValueError:
            pc = float("nan")
        if k == ref:
            tail = "--- | --- | ---"
        else:
            tail = f"{np.nanmean(diffs[k]):+.3f} | {praw[k]:.3f} | {adj[k]:.3f}"
        mu = float(np.mean(v_fin)) if len(v_fin) > 0 else float("nan")
        lines.append(
            f"| {k} | {mu:.3f} +/- {se:.3f} | [{lo:.3f}, {hi:.3f}] | {pc:.3f} | {tail} |"
        )
    table = "\n".join(lines)
    print(table)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path.json")
    ap.add_argument("--metric", default="auroc")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--chance", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    compare(args.models, metric=args.metric, ref=args.ref, chance=args.chance, out=args.out)


if __name__ == "__main__":
    main()
