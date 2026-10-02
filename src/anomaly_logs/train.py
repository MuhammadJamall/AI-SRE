import numpy as np
import torch
import torch.nn.functional as nnf

from src.anomaly_logs.data import make_batches, pad_batch
from src.anomaly_logs.model import NextEventModel
from src.anomaly_logs.scoring import score_sequences
from src.anomaly_metrics.train import get_device, set_seed  # noqa: F401  (shared helpers)
from src.common.logging_utils import get_logger

log = get_logger(__name__)


def _val_loss(model, val_enc, val_counts, device) -> float:
    s = score_sequences(model, val_enc, device, topg=1)
    return float((val_counts * s["nll_sum"]).sum() / (val_counts * s["n_tokens"]).sum())


def train_model(fit_enc, fit_counts, val_enc, val_counts, vocab_size, cfg, device, seed):
    """Trains on unique normal sequences, sampled in proportion to their frequency
    (equivalent to training on all traces, but far cheaper)."""
    set_seed(seed)
    model = NextEventModel(vocab_size, cfg["embedding_size"], cfg["hidden_size"],
                           cfg["num_layers"], cfg.get("rnn_type", "lstm")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    p = fit_counts / fit_counts.sum()

    history = {"train_loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        sample = rng.choice(len(fit_enc), size=cfg["samples_per_epoch"], p=p)
        total, ntok = 0.0, 0
        for bidx in make_batches(sample, fit_enc, cfg["batch_size"], rng):
            inp, tgt, mask = (torch.from_numpy(a).to(device) for a in pad_batch([fit_enc[i] for i in bidx]))
            opt.zero_grad()
            tok = nnf.cross_entropy(model(inp).transpose(1, 2), tgt, reduction="none")
            loss = (tok * mask).sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float((tok * mask).sum().item())
            ntok += int(mask.sum().item())
        tr, va = total / ntok, _val_loss(model, val_enc, val_counts, device)
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
