import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM -> latent bottleneck -> decoder LSTM -> reconstruction of the window."""

    def __init__(self, n_features: int, hidden_size: int = 64, latent_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, W, F)
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.decoder(d)
        return self.out(y)
