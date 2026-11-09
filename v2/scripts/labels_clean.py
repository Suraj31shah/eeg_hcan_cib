"""
labels_clean.py  —  Official DEAP label loader for the v2 pipeline.

These ratings come from the official participant_ratings.xls spreadsheet.
The .dat files' valence and arousal columns (columns 0 and 1) must NOT be
used: 439 of 1280 (subject, video) pairs have altered values in those
columns. Dominance and liking (.dat columns 2 and 3) match the spreadsheet
and may be cross-checked, but this module always loads from the .npy file
derived from the spreadsheet.
"""

import os
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_NPY_PATH = os.path.join(_DATA_DIR, "deap_labels_from_xls.npy")


def load_ratings() -> np.ndarray:
    """Return the official spreadsheet ratings.

    Returns
    -------
    np.ndarray  shape (32, 40, 4), float64
        Indexed [subject-1, video_id-1, k] where k = 0 Valence,
        1 Arousal, 2 Dominance, 3 Liking.
        Values are the raw 1-9 ratings from participant_ratings.xls.

    Notes
    -----
    The .dat valence/arousal columns must not be used: 439 rows are altered.
    """
    arr = np.load(os.path.abspath(_NPY_PATH))
    assert arr.shape == (32, 40, 4), f"Unexpected shape {arr.shape}"
    assert arr.dtype == np.float64, f"Unexpected dtype {arr.dtype}"
    return arr


def get_binary_labels(target: str, mode: str):
    """Binarise the official ratings for a given target and mode.

    Parameters
    ----------
    target : {'valence', 'arousal'}
    mode   : {'fixed5', 'median_drop_ties'}
        'fixed5'           : label = (rating > 5.0); mask = all True.
        'median_drop_ties' : per-subject median; label = (rating > median);
                             mask = (rating != median).  Trials where
                             rating == median are excluded (mask False).

    Returns
    -------
    labels : np.ndarray  shape (32, 40), int   (0 or 1)
    mask   : np.ndarray  shape (32, 40), bool
    """
    arr = load_ratings()
    col = {"valence": 0, "arousal": 1}[target]
    ratings = arr[:, :, col]          # (32, 40)

    labels = np.zeros((32, 40), dtype=int)
    mask   = np.ones ((32, 40), dtype=bool)

    if mode == "fixed5":
        labels = (ratings > 5.0).astype(int)
        # mask stays all True

    elif mode == "median_drop_ties":
        for s in range(32):
            m = float(np.median(ratings[s]))
            labels[s] = (ratings[s] > m).astype(int)
            mask[s]   = (ratings[s] != m)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return labels, mask


def get_ratings(target: str) -> np.ndarray:
    """Return raw 1-9 ratings for one target.

    Parameters
    ----------
    target : {'valence', 'arousal'}

    Returns
    -------
    np.ndarray  shape (32, 40), float64
    """
    arr = load_ratings()
    col = {"valence": 0, "arousal": 1}[target]
    return arr[:, :, col]
