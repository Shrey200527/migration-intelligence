"""Safe loading and prediction for the trained trajectory LSTM checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.models.model import LSTMConfig, TrajectoryLSTM, build_model


class InferenceError(ValueError):
    """The requested trajectory or the loaded model cannot be used for inference."""


class InvalidForecastError(RuntimeError):
    """The model produced a malformed, non-finite, or out-of-bounds forecast."""


def default_checkpoint_path() -> Path:
    """Return the repository's canonical trained-checkpoint location."""
    configured = os.environ.get("MODEL_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().absolute()
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "artifacts" / "pipeline" / "evaluation" / "trajectory_lstm.pt"


class TrajectoryPredictor:
    """A single loaded model reused for synchronous CPU or CUDA predictions."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None):
        self.checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("The configured model checkpoint does not exist.")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.model, self.input_length, self.forecast_horizon = self._construct_model(checkpoint)
        except Exception as exc:
            raise ValueError("The model checkpoint is malformed or incompatible with this serving version.") from exc
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _construct_model(checkpoint: object) -> tuple[TrajectoryLSTM, int, int]:
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary.")
        model_cfg = checkpoint.get("model_cfg")
        traj_cfg = checkpoint.get("traj_cfg")
        state = checkpoint.get("model_state")
        if not isinstance(model_cfg, dict) or not isinstance(traj_cfg, dict) or not isinstance(state, dict):
            raise ValueError("Checkpoint metadata or weights are missing.")

        model_fields = set(LSTMConfig.__dataclass_fields__)
        if set(model_cfg) != model_fields:
            raise ValueError("Model configuration fields do not match the supported architecture.")
        try:
            parsed_model_cfg = LSTMConfig(**model_cfg)
            input_length = traj_cfg["input_length"]
            horizon = traj_cfg["forecast_horizon"]
        except (TypeError, KeyError) as exc:
            raise ValueError("Trajectory configuration is incomplete.") from exc
        if (type(input_length) is not int or input_length < 1
                or type(horizon) is not int or horizon < 1
                or parsed_model_cfg.input_size != 2
                or parsed_model_cfg.forecast_horizon != horizon
                or type(parsed_model_cfg.hidden_size) is not int or parsed_model_cfg.hidden_size < 1
                or type(parsed_model_cfg.num_layers) is not int or parsed_model_cfg.num_layers < 1
                or isinstance(parsed_model_cfg.dropout, bool)
                or not isinstance(parsed_model_cfg.dropout, (int, float))
                or not math.isfinite(parsed_model_cfg.dropout)
                or not 0 <= parsed_model_cfg.dropout < 1):
            raise ValueError("Checkpoint model and trajectory settings are invalid or inconsistent.")

        model, _ = build_model(**model_cfg)
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("Checkpoint weights do not match the configured model architecture.") from exc
        return model, input_length, horizon

    def _validated_array(self, sequence: Iterable[Iterable[float]]) -> np.ndarray:
        try:
            array = np.asarray(sequence, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InferenceError("Trajectory must contain numeric latitude/longitude pairs.") from exc
        if array.shape != (self.input_length, 2):
            raise InferenceError(
                f"Trajectory must contain exactly {self.input_length} [latitude, longitude] coordinates."
            )
        if not np.isfinite(array).all():
            raise InferenceError("Trajectory coordinates must be finite numbers.")
        if np.any((array[:, 0] < -90) | (array[:, 0] > 90)):
            raise InferenceError("Trajectory latitude must be between -90 and 90 degrees.")
        if np.any((array[:, 1] < -180) | (array[:, 1] > 180)):
            raise InferenceError("Trajectory longitude must be between -180 and 180 degrees.")
        return array

    def predict_batch(self, sequences: list[Iterable[Iterable[float]]]) -> list[list[list[float]]]:
        if not sequences:
            raise InferenceError("At least one trajectory is required.")
        arrays = [self._validated_array(sequence) for sequence in sequences]
        inputs = torch.from_numpy(np.stack(arrays)).to(self.device)
        with torch.inference_mode():
            predictions = self.model(inputs)
        expected_shape = (len(sequences), self.forecast_horizon, 2)
        if tuple(predictions.shape) != expected_shape:
            raise InvalidForecastError("Loaded model returned an incompatible forecast shape.")
        output = predictions.detach().cpu().numpy()
        if not np.isfinite(output).all():
            raise InvalidForecastError("Loaded model returned non-finite forecast coordinates.")
        if (np.any((output[:, :, 0] < -90) | (output[:, :, 0] > 90))
                or np.any((output[:, :, 1] < -180) | (output[:, :, 1] > 180))):
            raise InvalidForecastError("Loaded model returned coordinates outside valid latitude/longitude bounds.")
        return output.astype(float).tolist()

    def predict(self, sequence: Iterable[Iterable[float]]) -> list[list[float]]:
        return self.predict_batch([sequence])[0]
