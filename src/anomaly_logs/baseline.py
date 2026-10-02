import math

import numpy as np

from src.anomaly_logs.data import END, PAD, START


class NGramModel:
    """Smoothed n-gram next-event model trained on normal sequences (sequence-likelihood baseline)."""

    def __init__(self, order: int = 3, vocab_size: int = 33, smoothing: float = 0.01):
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order, self.k = order, smoothing
        self.n_outputs = vocab_size - 2          # PAD and START can never be a next token
        self.ctx_next, self.ctx_total = {}, {}

    def _steps(self, enc):
        e, n = list(map(int, enc)), self.order - 1
        for t in range(1, len(e)):
            yield tuple(e[max(0, t - n):t]), e[t]

    def fit(self, encoded, counts):
        for enc, c in zip(encoded, counts):
            for ctx, nxt in self._steps(enc):
                d = self.ctx_next.setdefault(ctx, {})
                d[nxt] = d.get(nxt, 0) + int(c)
                self.ctx_total[ctx] = self.ctx_total.get(ctx, 0) + int(c)
        return self

    def _nll(self, ctx, nxt) -> float:
        d = self.ctx_next.get(ctx)
        if d is None:
            num, den = self.k, self.k * self.n_outputs
        else:
            num, den = d.get(nxt, 0) + self.k, self.ctx_total[ctx] + self.k * self.n_outputs
        return -math.log(num / den)

    def score(self, encoded) -> dict:
        n = len(encoded)
        out = {"nll_sum": np.zeros(n), "n_tokens": np.zeros(n), "max_nll": np.zeros(n),
               "max_pos": np.zeros(n, dtype=np.int64)}
        for i, enc in enumerate(encoded):
            nll = [self._nll(ctx, nxt) for ctx, nxt in self._steps(enc)]
            out["nll_sum"][i], out["n_tokens"][i] = sum(nll), len(nll)
            out["max_nll"][i], out["max_pos"][i] = max(nll), int(np.argmax(nll))
        return out

    def to_dict(self) -> dict:
        return {"order": self.order, "smoothing": self.k, "n_outputs": self.n_outputs,
                "counts": {",".join(map(str, c)): {str(k): v for k, v in d.items()}
                           for c, d in self.ctx_next.items()}}
