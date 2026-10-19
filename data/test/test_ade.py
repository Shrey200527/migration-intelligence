
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.train import evaluate_ade, evaluate_fde


class FixedPredictionModel(nn.Module):
    """Predict one degree north at every forecast step."""

    def forward(self, past):
        batch_size = past.shape[0]
        return torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
            dtype=torch.float32,
            device=past.device,
        ).expand(batch_size, -1, -1)


past = torch.zeros((2, 1, 2), dtype=torch.float32)

# Three actual future positions, all at (0, 0).
future = torch.zeros((2, 3, 2), dtype=torch.float32)

loader = DataLoader(TensorDataset(past, future), batch_size=2)
model = FixedPredictionModel()
device = torch.device("cpu")

ade_km = evaluate_ade(model, loader, device)
fde_km = evaluate_fde(model, loader, device)

print(f"ADE: {ade_km:.3f} km")
print(f"FDE: {fde_km:.3f} km")

assert abs(ade_km - 111.195) < 0.1
assert abs(fde_km - 111.195) < 0.1

print("PASS: ADE and FDE calculations are correct.")
