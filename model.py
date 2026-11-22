"""
model.py
PyTorch LSTM Autoencoder architecture, training routine, and reconstruction error computation.
"""

from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMEncoder(nn.Module):
    """Encoder network: compresses (batch_size, seq_len, input_dim)

    into a compact latent bottleneck vector (batch_size, latent_dim).
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 24,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Use last step hidden representation
        last_hidden = out[:, -1, :]  # (batch_size, hidden_dim)
        latent = self.relu(self.linear(last_hidden))  # (batch_size, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """Decoder network: expands latent bottleneck vector back into

    the original sequence shape (batch_size, seq_len, input_dim).
    """

    def __init__(
        self,
        latent_dim: int = 24,
        hidden_dim: int = 64,
        output_dim: int = 2,
        seq_len: int = 20,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_project = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (batch_size, latent_dim)
        h = self.relu(self.latent_project(latent))  # (batch_size, hidden_dim)
        # Repeat vector along sequence dimension
        repeated = h.unsqueeze(1).repeat(
            1, self.seq_len, 1
        )  # (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(repeated)
        reconstruction = self.output_layer(
            out
        )  # (batch_size, seq_len, output_dim)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """Complete LSTM Autoencoder for time-series anomaly detection."""

    def __init__(
        self,
        input_dim: int = 2,
        seq_len: int = 20,
        hidden_dim: int = 64,
        latent_dim: int = 24,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def train_autoencoder(
    model: LSTMAutoencoder,
    X_train: np.ndarray,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.002,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[float]:
    """Train the LSTM Autoencoder using Adam and Mean Squared Error (MSE) loss.

    Optionally invokes progress_callback(epoch, total_epochs, current_loss).
    """
    model.to(device)
    model.train()

    tensor_x = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        epoch_losses: List[float] = []
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        loss_history.append(mean_loss)

        if progress_callback is not None:
            progress_callback(epoch, epochs, mean_loss)

    return loss_history


def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    X: np.ndarray,
    mode: str = "last_day",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Computes sample-wise reconstruction error (MSE).

    Modes:
    - 'last_day': Mean squared error on the final time step of the window (t_w).
                  Captures sudden, instantaneous market shocks.
    - 'window_avg': Mean squared error averaged over all time steps in the window.
                    Captures persistent regime changes and multi-day structural shifts.
    """
    model.to(device)
    model.eval()

    tensor_x = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_diffs: List[np.ndarray] = []

    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)
            # Element-wise squared difference: (B, seq_len, features)
            diff = (batch_x - reconstructed).pow(2).cpu().numpy()
            all_diffs.append(diff)

    # Concatenate all batches: shape (N, seq_len, features)
    diffs = np.concatenate(all_diffs, axis=0)

    if mode == "last_day":
        # Average across features on the last time step
        errors = diffs[:, -1, :].mean(axis=-1)
    elif mode == "window_avg":
        # Average across both time steps and features
        errors = diffs.mean(axis=(1, 2))
    else:
        raise ValueError(
            f"Chế độ phát hiện '{mode}' không hợp lệ. Chọn 'last_day' hoặc 'window_avg'."
        )

    return errors
