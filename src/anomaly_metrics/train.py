import random

import numpy as np
import torch
import torch.nn.functional as nnf

from src.anomaly_metrics.data import windows_view
from src.anomaly_metrics.model import LSTMAutoencoder
from src.common.logging_utils import get_logger

log = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "cpu") -> torch.device:
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("MPS requested but not available; using CPU")
    return torch.device("cpu")  # 'auto' and 'cpu' -> cpu (most reliable for LSTMs)


def _batches(view, idx, batch_size, device):
    for i in range(0, len(idx), batch_size):
        b = view[idx[i:i + batch_size]]
        yield torch.from_numpy(np.ascontiguousarray(b)).to(device)


@torch.no_grad()
def _eval_loss(model, view, idx, batch_size, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb in _batches(view, idx, batch_size, device):
        total += nnf.mse_loss(model(xb), xb).item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


def train_model(fit_x, val_x, cfg, device, seed):
    """fit_x / val_x: scaled (T, F) float32 arrays. Returns (model, history)."""
    set_seed(seed)
    window, n_features = cfg["window"], fit_x.shape[1]
    model = LSTMAutoencoder(n_features, cfg["hidden_size"], cfg["latent_size"], cfg["num_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    fit_view, val_view = windows_view(fit_x, window), windows_view(val_x, window)
    fit_idx = np.arange(0, len(fit_view), cfg["train_stride"])
    val_idx = np.arange(0, len(val_view), cfg["train_stride"])
    rng = np.random.default_rng(seed)

    history = {"train_loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total, n = 0.0, 0
        for xb in _batches(fit_view, rng.permutation(fit_idx), cfg["batch_size"], device):
            opt.zero_grad()
            loss = nnf.mse_loss(model(xb), xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        tr, va = total / n, _eval_loss(model, val_view, val_idx, cfg["batch_size"], device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        log.info("epoch %d  train=%.5f  val=%.5f", epoch + 1, tr, va)

        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log.info("early stopping at epoch %d", epoch + 1)
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def reconstruction_errors(model, x, window, device, batch_size=512) -> np.ndarray:
    """Per-point, per-feature squared error (T, F).
    Point t uses the last step of the window ending at t; the first window-1 points
    use the first window's own positions."""
    model.eval()
    view = windows_view(x, window)
    out = np.zeros(x.shape, dtype="float32")
    for i in range(0, len(view), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(view[i:i + batch_size])).to(device)
        err = ((model(xb) - xb) ** 2).cpu().numpy()
        out[i + window - 1: i + window - 1 + len(err)] = err[:, -1, :]
        if i == 0:
            out[:window - 1] = err[0, :window - 1, :]
    return out
