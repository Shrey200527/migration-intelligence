"""Geographic trajectory metrics. Coordinates are always [latitude, longitude]."""
import torch

EARTH_RADIUS_KM = 6371.0088


def validate_coordinate_tensor(points: torch.Tensor, name: str = "coordinates") -> None:
    """Reject malformed, non-finite, or out-of-range [latitude, longitude] tensors."""
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch tensor")
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2) ordered [latitude, longitude]")
    if not torch.isfinite(points).all():
        raise ValueError(f"{name} contain non-finite latitude/longitude values")
    if (points[..., 0].abs() > 90).any():
        raise ValueError(f"{name} contain latitude values outside [-90, 90] degrees")
    if (points[..., 1].abs() > 180).any():
        raise ValueError(f"{name} contain longitude values outside [-180, 180] degrees")


def _validate_points(points_a: torch.Tensor, points_b: torch.Tensor) -> None:
    if not isinstance(points_a, torch.Tensor) or not isinstance(points_b, torch.Tensor):
        raise TypeError("Coordinates must be torch tensors")
    if points_a.shape != points_b.shape or points_a.ndim < 1 or points_a.shape[-1] != 2:
        raise ValueError("Coordinate tensors must have matching shapes ending in [latitude, longitude]")
    if points_a.device != points_b.device:
        raise ValueError("Coordinate tensors must be on the same device")
    validate_coordinate_tensor(points_a, "First geographic coordinate input")
    validate_coordinate_tensor(points_b, "Second geographic coordinate input")


def haversine_distance_km(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
    """Great-circle distances in kilometres for matching (..., 2) tensors."""
    _validate_points(points_a, points_b)
    lat_a, lon_a = torch.deg2rad(points_a).unbind(dim=-1)
    lat_b, lon_b = torch.deg2rad(points_b).unbind(dim=-1)
    delta_lat, delta_lon = lat_b - lat_a, lon_b - lon_a
    a = torch.sin(delta_lat / 2).square() + torch.cos(lat_a) * torch.cos(lat_b) * torch.sin(delta_lon / 2).square()
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(torch.clamp(a, 0.0, 1.0)))
