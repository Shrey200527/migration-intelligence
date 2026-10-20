import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.model import build_model
from src.serving.api import create_app
from src.serving.inference import InferenceError, TrajectoryPredictor, default_checkpoint_path


def write_checkpoint(path: Path, input_length: int = 4, horizon: int = 3) -> None:
    torch.manual_seed(19)
    model, model_cfg = build_model(hidden_size=4, num_layers=1, dropout=0, forecast_horizon=horizon)
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": model_cfg.__dict__,
        "traj_cfg": {"input_length": input_length, "forecast_horizon": horizon,
                     "min_points_per_track": input_length + horizon, "max_animals": 2},
        "seed": 19,
    }, path)


def asgi_request(app, method: str, path: str, body: object | None = None):
    """Small standard-library ASGI client so tests do not require httpx."""
    request_body = b"" if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    messages = [{"type": "http.request", "body": request_body, "more_body": False}]
    response = {"status": None, "body": bytearray(), "headers": []}

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"].extend(message.get("body", b""))

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "server": ("testserver", 80), "client": ("127.0.0.1", 12345), "state": {},
    }
    asyncio.run(app(scope, receive, send))
    payload = json.loads(response["body"]) if response["body"] else None
    return response["status"], payload


class InferenceTests(unittest.TestCase):
    def test_checkpoint_default_and_environment_override_are_local_and_predictable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_CHECKPOINT", None)
            self.assertTrue(default_checkpoint_path().as_posix().endswith(
                "/artifacts/pipeline/evaluation/trajectory_lstm.pt"))
        with tempfile.TemporaryDirectory() as temp:
            configured = Path(temp) / "alternate.pt"
            with patch.dict(os.environ, {"MODEL_CHECKPOINT": str(configured)}):
                self.assertEqual(default_checkpoint_path(), configured.absolute())

    def test_valid_checkpoint_loads_once_and_predicts_configured_horizon(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            write_checkpoint(checkpoint)
            predictor = TrajectoryPredictor(checkpoint, device="cpu")
            self.assertFalse(predictor.model.training)
            forecast = predictor.predict([[20, 30], [20.1, 30.1], [20.2, 30.2], [20.3, 30.3]])
            self.assertEqual(len(forecast), 3)
            self.assertTrue(all(len(coordinate) == 2 for coordinate in forecast))
            with patch.object(predictor.model, "forward", wraps=predictor.model.forward) as forward:
                batch = predictor.predict_batch([
                    [[20, 30], [20.1, 30.1], [20.2, 30.2], [20.3, 30.3]],
                    [[-20, -30], [-20.1, -30.1], [-20.2, -30.2], [-20.3, -30.3]],
                ])
            self.assertEqual(tuple(forward.call_args.args[0].shape), (2, 4, 2))
            self.assertEqual(len(batch), 2)
            self.assertTrue(all(len(forecast) == 3 for forecast in batch))

    def test_missing_and_incompatible_checkpoints_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint does not exist"):
                TrajectoryPredictor(root / "missing.pt", device="cpu")
            malformed = root / "malformed.pt"
            torch.save({"model_state": {}}, malformed)
            with self.assertRaisesRegex(ValueError, "malformed or incompatible"):
                TrajectoryPredictor(malformed, device="cpu")
            incompatible = root / "incompatible.pt"
            write_checkpoint(incompatible, input_length=4, horizon=3)
            payload = torch.load(incompatible, weights_only=True)
            payload["model_cfg"]["forecast_horizon"] = 2
            torch.save(payload, incompatible)
            with self.assertRaisesRegex(ValueError, "malformed or incompatible"):
                TrajectoryPredictor(incompatible, device="cpu")

    def test_invalid_sequence_shapes_nonfinite_and_coordinate_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            write_checkpoint(checkpoint)
            predictor = TrajectoryPredictor(checkpoint, device="cpu")
            cases = [
                ([[20, 30], [20, 30]], "exactly 4"),
                ([[20, 30], [20, 30], [20, 30], [20]], "numeric latitude/longitude pairs"),
                ([[float("nan"), 30]] * 4, "finite numbers"),
                ([[91, 30]] * 4, "latitude must be between"),
                ([[20, 181]] * 4, "longitude must be between"),
            ]
            for sequence, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(InferenceError, message):
                    predictor.predict(sequence)


class ServingApiTests(unittest.TestCase):
    def test_health_readiness_and_single_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            app = create_app(str(root / "missing.pt"))
            self.assertEqual(asgi_request(app, "GET", "/health"), (200, {"status": "ok"}))
            status, payload = asgi_request(app, "GET", "/ready")
            self.assertEqual(status, 503)
            self.assertNotIn(str(root), json.dumps(payload))

            checkpoint = root / "model.pt"
            write_checkpoint(checkpoint)
            ready_app = create_app(str(checkpoint))
            self.assertEqual(asgi_request(ready_app, "GET", "/ready"), (200, {"status": "ready"}))
            status, payload = asgi_request(ready_app, "POST", "/predict", {
                "trajectory": [[20, 30], [20.1, 30.1], [20.2, 30.2], [20.3, 30.3]],
            })
            self.assertEqual(status, 200)
            self.assertEqual(len(payload["forecast"]), 3)
            self.assertEqual(payload["coordinate_order"], "latitude,longitude")
            self.assertEqual(payload["coordinate_units"], "degrees")

    def test_batch_and_invalid_requests_return_safe_responses(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            write_checkpoint(checkpoint)
            app = create_app(str(checkpoint))
            sequence = [[20, 30], [20.1, 30.1], [20.2, 30.2], [20.3, 30.3]]

            status, payload = asgi_request(app, "POST", "/predict/batch", {
                "trajectories": [sequence, sequence],
            })
            self.assertEqual(status, 200)
            self.assertEqual(len(payload["forecasts"]), 2)
            self.assertTrue(all(len(forecast) == 3 for forecast in payload["forecasts"]))

            status, _ = asgi_request(app, "POST", "/predict/batch", {
                "trajectories": [sequence, [[91, 30]] * 4],
            })
            self.assertEqual(status, 422)

            for trajectories, expected in (([], 422), ([[[20, 30]] * 4] * 65, 422)):
                with self.subTest(expected=expected, count=len(trajectories)):
                    status, _ = asgi_request(app, "POST", "/predict/batch", {"trajectories": trajectories})
                    self.assertEqual(status, expected)
            invalid_requests = [
                ({"trajectory": [[20, 30], [20, 30]]}, "exactly 4"),
                ({"trajectory": [[91, 30]] * 4}, "latitude"),
                ({"trajectory": [[20, 181]] * 4}, "longitude"),
                ({"trajectory": [[20, 30, 40]] * 4}, "422"),
            ]
            for request, expected_text in invalid_requests:
                with self.subTest(expected=expected_text):
                    status, payload = asgi_request(app, "POST", "/predict", request)
                    self.assertEqual(status, 422)
                    self.assertNotIn(str(temp), json.dumps(payload))
                    self.assertNotIn("Traceback", json.dumps(payload))

            unavailable = create_app(str(Path(temp) / "not-here.pt"))
            status, payload = asgi_request(unavailable, "POST", "/predict", {"trajectory": sequence})
            self.assertEqual(status, 503)
            self.assertNotIn(str(temp), json.dumps(payload))

    def test_docker_readiness_healthcheck_uses_valid_dockerfile_cmd_form(self):
        dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
        contents = dockerfile.read_text(encoding="utf-8")
        lines = contents.splitlines()
        healthcheck_index = next(index for index, line in enumerate(lines)
                                 if line.lstrip().startswith("HEALTHCHECK"))
        healthcheck = lines[healthcheck_index].strip()
        following = lines[healthcheck_index + 1].strip()
        self.assertIn("HEALTHCHECK", healthcheck)
        self.assertTrue(following.startswith("CMD ["), following)
        self.assertIn("/ready", following)
        self.assertNotIn("CMD-SHELL", contents)


if __name__ == "__main__":
    unittest.main()
