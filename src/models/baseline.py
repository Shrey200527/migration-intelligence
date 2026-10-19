"""Simple persistence trajectory forecasting baseline."""
import torch


def persistence_forecast(past: torch.Tensor, forecast_horizon: int) -> torch.Tensor:
    """Repeat each sequence's last [latitude, longitude] for the forecast horizon."""
    if not isinstance(past, torch.Tensor) or past.ndim != 3 or past.shape[1] < 1 or past.shape[2] != 2:
        raise ValueError("past must have shape [batch, time>=1, 2]")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be positive")
    if not torch.isfinite(past).all():
        raise ValueError("past coordinates must be finite")
    return past[:, -1:, :].expand(-1, forecast_horizon, -1).clone()
