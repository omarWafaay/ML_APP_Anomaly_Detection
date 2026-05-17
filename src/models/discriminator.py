from __future__ import annotations

from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class Discriminator1D(nn.Module):
    """Real-vs-fake window discriminator preserved from the AAE notebook cells."""

    def __init__(
        self,
        n_features: int,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
        input_len: int = 128,
        neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = n_features
        for i, out_channels in enumerate(hidden_channels):
            layers.append(
                spectral_norm(
                    nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
                )
            )
            if i > 0:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.LeakyReLU(neg_slope, inplace=True))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)
        flat_len = (input_len // (2 ** len(hidden_channels))) * hidden_channels[-1]
        self.fc = spectral_norm(nn.Linear(flat_len, 1))

    def forward(self, x):
        z = self.conv(x).flatten(1)
        return self.fc(z).squeeze(-1)
