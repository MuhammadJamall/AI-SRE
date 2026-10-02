import numpy as np
import pandas as pd


def roc_auc(y, s):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = pd.Series(s).rank().to_numpy()      # average ranks for ties
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    last = np.r_[np.where(np.diff(s))[0], len(s) - 1]     # one point per distinct score
    tp, fp = tp[last], fp[last]
    precision, recall = tp / (tp + fp), tp / tp[-1]
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))
