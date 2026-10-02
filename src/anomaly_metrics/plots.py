import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_score_distribution(val_err, test_err, labels, threshold, path):
    eps = 1e-12
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(np.log10(min(val_err.min(), test_err.min()) + eps),
                       np.log10(max(val_err.max(), test_err.max()) + eps), 60)
    ax.hist(np.log10(val_err + eps), bins, alpha=0.5, label="validation (normal)", density=True)
    ax.hist(np.log10(test_err[labels == 0] + eps), bins, alpha=0.5, label="test normal", density=True)
    if (labels == 1).any():
        ax.hist(np.log10(test_err[labels == 1] + eps), bins, alpha=0.5, label="test anomaly", density=True)
    ax.axvline(np.log10(threshold + eps), color="k", ls="--", label="threshold")
    ax.set_xlabel("log10 reconstruction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_timeline(raw_err, labels, preds, threshold, path):
    x = np.arange(len(raw_err))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(x, raw_err, lw=0.6)
    a1.axhline(threshold, color="k", ls="--")
    a1.set_yscale("log")
    a1.set_ylabel("error")
    a2.fill_between(x, 0, 1, where=labels == 1, step="mid", color="red", alpha=0.5, label="ground truth")
    a2.fill_between(x, -1, 0, where=preds == 1, step="mid", color="blue", alpha=0.5, label="predicted")
    a2.set_yticks([])
    a2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_example_window(test_scaled, test_err, labels, feature_names, path, context=150, top_k=4):
    if not (labels == 1).any():
        return None
    s = int(np.argmax(labels == 1))
    e = s
    while e < len(labels) and labels[e] == 1:
        e += 1
    lo, hi = max(0, s - context), min(len(labels), e + context)
    top = np.argsort(test_err[s:e].mean(axis=0))[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 4))
    for f in top:
        ax.plot(np.arange(lo, hi), test_scaled[lo:hi, f], label=feature_names[f], lw=0.8)
    ax.axvspan(s, e, color="red", alpha=0.15, label="true anomaly")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
