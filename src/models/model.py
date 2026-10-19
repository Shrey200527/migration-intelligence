from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    input_size: int = 2  # lat, long
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    forecast_horizon: int = 5


class TrajectoryLSTM(nn.Module):
    """
    Simple encoder LSTM with linear head that predicts a sequence of future
    coordinates from a window of past coordinates.
    """

    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.forecast_horizon * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 2) past coordinates
        returns: (B, H, 2) predicted future coordinates
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        preds = self.head(last_hidden)
        preds = preds.view(-1, self.cfg.forecast_horizon, 2)
        return preds


def build_model(
    input_size: int = 2,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    forecast_horizon: int = 5,
) -> Tuple[TrajectoryLSTM, LSTMConfig]:
    cfg = LSTMConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        forecast_horizon=forecast_horizon,
    )
    model = TrajectoryLSTM(cfg)
    return model, cfg

