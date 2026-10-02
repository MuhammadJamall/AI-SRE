import numpy as np


def confusion(y_true, y_pred) -> dict:
    yt, yp = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {"tn": int((~yt & ~yp).sum()), "fp": int((~yt & yp).sum()),
            "fn": int((yt & ~yp).sum()), "tp": int((yt & yp).sum())}


def prf(c: dict) -> dict:
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def point_adjust(y_true, y_pred) -> np.ndarray:
    """If any point of a true anomaly segment is detected, count the whole segment as detected.
    Common in SMD papers but MORE LENIENT than point-wise; always report both."""
    yt = np.asarray(y_true).astype(bool)
    adj = np.asarray(y_pred).astype(bool).copy()
    d = np.diff(np.concatenate([[0], yt.astype(int), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if adj[s:e].any():
            adj[s:e] = True
    return adj


def best_f1(y_true, scores, n: int = 200):
    """ORACLE reference: best F1 over thresholds, chosen using the test labels.
    Not an operating result, only shows how good the scores could be with a perfect threshold."""
    yt = np.asarray(y_true).astype(bool)
    if not yt.any():
        return None
    best = {"threshold": None, "f1": -1.0}
    for t in np.unique(np.quantile(scores, np.linspace(0.5, 0.999, n))):
        f1 = prf(confusion(yt, scores > t))["f1"]
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": float(f1)}
    return best


def evaluate(y_true, y_pred, scores=None) -> dict:
    c = confusion(y_true, y_pred)
    out = {**c, **prf(c)}
    ca = confusion(y_true, point_adjust(y_true, y_pred))
    out["point_adjusted"] = {**ca, **prf(ca)}
    if scores is not None:
        out["oracle_best_f1"] = best_f1(y_true, scores)
    return out
