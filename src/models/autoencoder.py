from __future__ import annotations

from torch import nn


class ConvAE1D(nn.Module):
    """Conv1D autoencoder architecture preserved from the comparison notebook."""

    def __init__(self, n_features: int, enc_channels: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        c1, c2 = enc_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c2, c2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(c2, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, n_features, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))
