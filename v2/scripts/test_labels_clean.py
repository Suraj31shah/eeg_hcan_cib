"""Tests for labels_clean.py — run and print PASS/FAIL for each assertion."""

import sys
import os
import numpy as np
import pandas as pd

# Make sure we can import labels_clean from the same directory
sys.path.insert(0, os.path.dirname(__file__))
import labels_clean as lc

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    msg = f"{tag}: {name}"
    if detail:
        msg += f"  [{detail}]"
    print(msg)
    results.append((name, cond))
    return cond


# -------------------------------------------------------------------
# T1: shape of load_ratings
# -------------------------------------------------------------------
arr = lc.load_ratings()
check("shape of load_ratings is (32,40,4)", arr.shape == (32, 40, 4),
      f"got {arr.shape}")

# -------------------------------------------------------------------
# T2: no NaN
# -------------------------------------------------------------------
check("no NaN in load_ratings", not np.isnan(arr).any(),
      f"NaN count={np.isnan(arr).sum()}")

# -------------------------------------------------------------------
# T3: fixed5 positive fraction matches CSV
# -------------------------------------------------------------------
labels_f5, mask_f5 = lc.get_binary_labels("valence", "fixed5")
frac_api = float(labels_f5.mean())

csv_path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "deap_labels_from_xls.csv")
df = pd.read_csv(csv_path)
frac_csv = float((df["valence"] > 5.0).mean())

check("fixed5 positive fraction equals CSV fraction",
      abs(frac_api - frac_csv) < 1e-9,
      f"api={frac_api:.6f}  csv={frac_csv:.6f}")

# -------------------------------------------------------------------
# T4: median_drop_ties per-subject class balance within ties+1
# -------------------------------------------------------------------
labels_md, mask_md = lc.get_binary_labels("valence", "median_drop_ties")
ratings_v = lc.get_ratings("valence")

all_ok = True
detail_parts = []
for s in range(32):
    m = float(np.median(ratings_v[s]))
    n_ties = int((ratings_v[s] == m).sum())
    masked_in = mask_md[s]
    n_pos = int(labels_md[s, masked_in].sum())
    n_neg = int((~labels_md[s, masked_in].astype(bool)).sum())
    diff  = abs(n_pos - n_neg)
    if diff > n_ties + 1:
        all_ok = False
        detail_parts.append(f"s{s+1}:diff={diff},ties={n_ties}")

check("median_drop_ties class balance within ties+1 per subject",
      all_ok,
      "; ".join(detail_parts) if detail_parts else "all OK")

# -------------------------------------------------------------------
# T5: dominance and liking match .dat exactly
# -------------------------------------------------------------------
dat_path = r"E:\audit_scratch\labels_from_dat.npy"
dat = np.load(dat_path)   # (32, 40, 4)
dom_match = np.all(np.abs(dat[:,:,2] - arr[:,:,2]) < 1e-9)
lik_match = np.all(np.abs(dat[:,:,3] - arr[:,:,3]) < 1e-9)
check("dominance in load_ratings equals .dat column 2 exactly", dom_match)
check("liking in load_ratings equals .dat column 3 exactly",    lik_match)

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"\nSummary: {n_pass} PASS  {n_fail} FAIL  (total {len(results)})")
sys.exit(0 if n_fail == 0 else 1)
