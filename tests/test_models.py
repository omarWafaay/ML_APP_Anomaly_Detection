import torch

from src.models import ConvAE1D, Discriminator1D


def test_conv_ae_reconstruction_shape_matches_input():
    model = ConvAE1D(n_features=5, enc_channels=(8, 4))
    x = torch.randn(2, 5, 128)

    y = model(x)

    assert y.shape == x.shape


def test_conv_ae_forecasting_length_matches_target_half():
    model = ConvAE1D(n_features=5, enc_channels=(8, 4))
    x = torch.randn(2, 5, 64)

    y = model(x)

    assert y.shape == (2, 5, 64)


def test_discriminator_outputs_one_score_per_window():
    model = Discriminator1D(n_features=5, hidden_channels=(8, 16, 32), input_len=128)
    x = torch.randn(3, 5, 128)

    scores = model(x)

    assert scores.shape == (3,)
