import torch.nn as nn

from src.anomaly_logs.data import PAD


class NextEventModel(nn.Module):
    """DeepLog-style next-event predictor over event ids (LSTM or GRU)."""

    def __init__(self, vocab_size, embedding_size=16, hidden_size=64, num_layers=1, rnn_type="lstm"):
        super().__init__()
        cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(rnn_type)
        if cls is None:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.rnn = cls(embedding_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):                      # (B, T) -> (B, T, V)
        h, _ = self.rnn(self.embed(x))
        return self.out(h)
