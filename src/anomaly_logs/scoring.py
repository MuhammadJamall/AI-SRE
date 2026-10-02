import numpy as np
import torch
import torch.nn.functional as nnf

from src.anomaly_logs.data import pad_batch


@torch.no_grad()
def score_sequences(model, encoded, device, topg=9, batch_size=1024) -> dict:
    """Per-sequence stats: nll_sum, n_tokens, max_nll, max_pos, topg_miss (steps whose
    true next event was not among the model's top-g predictions)."""
    model.eval()
    n = len(encoded)
    out = {"nll_sum": np.zeros(n), "n_tokens": np.zeros(n), "max_nll": np.zeros(n),
           "max_pos": np.zeros(n, dtype=np.int64), "topg_miss": np.zeros(n, dtype=np.int64)}
    order = np.argsort([len(e) for e in encoded], kind="stable")
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        inp, tgt, mask = pad_batch([encoded[j] for j in idx])
        inp, tgt, mask = (torch.from_numpy(a).to(device) for a in (inp, tgt, mask))
        logits = model(inp)
        nll = nnf.cross_entropy(logits.transpose(1, 2), tgt, reduction="none") * mask
        g = min(topg, logits.size(-1))
        hit = (logits.topk(g, dim=-1).indices == tgt.unsqueeze(-1)).any(-1)
        mx, pos = nll.max(dim=1)
        out["nll_sum"][idx] = nll.sum(1).cpu().numpy()
        out["n_tokens"][idx] = mask.sum(1).cpu().numpy()
        out["max_nll"][idx] = mx.cpu().numpy()
        out["max_pos"][idx] = pos.cpu().numpy()
        out["topg_miss"][idx] = ((~hit) & mask).sum(1).cpu().numpy()
    return out
