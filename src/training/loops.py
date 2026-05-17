from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.models import ConvAE1D, Discriminator1D


@dataclass(frozen=True)
class AETrainingResult:
    model: ConvAE1D
    train_losses: list[float]
    val_losses: list[float]
    best_val: float


@dataclass(frozen=True)
class AAETrainingResult:
    generator: ConvAE1D
    discriminator: Discriminator1D
    history: dict[str, list[float]]
    best_val_mse: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> float:
    is_train = optimizer is not None
    device = torch.device(device)
    model.train(is_train)
    total, n_seen = 0.0, 0
    for x_input, x_target in loader:
        x_input = x_input.to(device, non_blocking=True)
        x_target = x_target.to(device, non_blocking=True)
        loss = loss_fn(model(x_input), x_target)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += loss.item() * x_input.size(0)
        n_seen += x_input.size(0)
    return total / max(n_seen, 1)


def train_autoencoder(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_features: int,
    enc_channels: tuple[int, int] = (64, 32),
    epochs: int = 60,
    learning_rate: float = 1e-3,
    patience: int = 10,
    grad_clip: float | None = 1.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> AETrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    model = ConvAE1D(n_features=n_features, enc_channels=enc_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val, best_state, remaining = float("inf"), None, patience

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer, grad_clip)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining = patience
            marker = " *"
        else:
            remaining -= 1
            marker = ""
        if verbose:
            print(f"  epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}{marker}", flush=True)
        if not improved:
            if remaining <= 0:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val {best_val:.6f})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid validation state")
    model.load_state_dict(best_state)
    return AETrainingResult(model=model, train_losses=train_losses, val_losses=val_losses, best_val=best_val)


def hinge_loss_discriminator(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def hinge_loss_generator(d_fake: torch.Tensor) -> torch.Tensor:
    return -d_fake.mean()


def generator_adversarial_loss(d_fake: torch.Tensor, mode: str = "linear") -> torch.Tensor:
    if mode == "linear":
        return -d_fake.mean()
    if mode == "hinge":
        return F.relu(1.0 - d_fake).mean()
    raise ValueError(f"Unsupported generator adversarial loss: {mode}")


def train_aae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_features: int,
    enc_channels: tuple[int, int] = (64, 32),
    input_len: int = 128,
    epochs: int = 60,
    lr_generator: float = 1e-3,
    lr_discriminator: float = 1e-4,
    lambda_adv: float = 0.01,
    betas: tuple[float, float] = (0.5, 0.999),
    generator_loss_mode: str = "linear",
    patience: int = 10,
    grad_clip: float | None = 1.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> AAETrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    generator = ConvAE1D(n_features=n_features, enc_channels=enc_channels).to(device)
    discriminator = Discriminator1D(n_features=n_features, input_len=input_len).to(device)
    opt_generator = torch.optim.Adam(generator.parameters(), lr=lr_generator, betas=betas)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=betas)
    mse = nn.MSELoss()

    history = {
        "train_mse": [],
        "val_mse": [],
        "loss_D": [],
        "loss_G_adv": [],
        "d_real": [],
        "d_fake": [],
    }
    best_val, best_state, remaining = float("inf"), None, patience

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        sums = {key: 0.0 for key in ("mse", "loss_D", "loss_G_adv", "d_real", "d_fake")}
        n_seen = 0

        for x_input, x_target in train_loader:
            x_input = x_input.to(device, non_blocking=True)
            x_target = x_target.to(device, non_blocking=True)

            opt_discriminator.zero_grad(set_to_none=True)
            with torch.no_grad():
                x_fake_for_discriminator = generator(x_input)
            d_real = discriminator(x_target)
            d_fake = discriminator(x_fake_for_discriminator)
            loss_discriminator = hinge_loss_discriminator(d_real, d_fake)
            loss_discriminator.backward()
            opt_discriminator.step()

            opt_generator.zero_grad(set_to_none=True)
            x_fake = generator(x_input)
            mse_loss = mse(x_fake, x_target)
            adv_loss = generator_adversarial_loss(discriminator(x_fake), mode=generator_loss_mode)
            total_generator_loss = mse_loss + lambda_adv * adv_loss
            total_generator_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_generator.step()

            batch_size = x_input.size(0)
            n_seen += batch_size
            sums["mse"] += mse_loss.item() * batch_size
            sums["loss_D"] += loss_discriminator.item() * batch_size
            sums["loss_G_adv"] += adv_loss.item() * batch_size
            sums["d_real"] += d_real.mean().item() * batch_size
            sums["d_fake"] += d_fake.mean().item() * batch_size

        if n_seen == 0:
            raise RuntimeError("AAE training loader produced no batches")

        val_mse = _validation_mse(generator, val_loader, mse, device)
        history["train_mse"].append(sums["mse"] / n_seen)
        history["val_mse"].append(val_mse)
        history["loss_D"].append(sums["loss_D"] / n_seen)
        history["loss_G_adv"].append(sums["loss_G_adv"] / n_seen)
        history["d_real"].append(sums["d_real"] / n_seen)
        history["d_fake"].append(sums["d_fake"] / n_seen)

        improved = val_mse < best_val - 1e-6
        if improved:
            best_val = val_mse
            # Saving both networks at the same epoch avoids inconsistent AAE checkpoints.
            best_state = {
                "generator": copy.deepcopy(generator.state_dict()),
                "discriminator": copy.deepcopy(discriminator.state_dict()),
            }
            remaining = patience
            marker = " *"
        else:
            remaining -= 1
            marker = ""
        if verbose:
            print(
                f"  epoch {epoch:03d} | MSE train {history['train_mse'][-1]:.6f} | "
                f"val {val_mse:.6f} | D {history['loss_D'][-1]:.4f} | "
                f"G_adv {history['loss_G_adv'][-1]:+.4f}{marker}",
                flush=True,
            )
        if not improved:
            if remaining <= 0:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val {best_val:.6f})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("AAE training finished without a valid validation state")
    generator.load_state_dict(best_state["generator"])
    discriminator.load_state_dict(best_state["discriminator"])
    return AAETrainingResult(
        generator=generator,
        discriminator=discriminator,
        history=history,
        best_val_mse=best_val,
    )


def _validation_mse(
    model: nn.Module,
    loader: DataLoader,
    mse: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n_seen = 0.0, 0
    with torch.no_grad():
        for x_input, x_target in loader:
            x_input = x_input.to(device, non_blocking=True)
            x_target = x_target.to(device, non_blocking=True)
            total += mse(model(x_input), x_target).item() * x_input.size(0)
            n_seen += x_input.size(0)
    return total / max(n_seen, 1)
