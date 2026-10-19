
import torch


def haversine_distance_km(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate great-circle distances between latitude/longitude points.

    Args:
        points_a: Tensor shaped (..., 2), in degrees (latitude, longitude).
        points_b: Tensor shaped (..., 2), in degrees (latitude, longitude).

    Returns:
        Distances in kilometres, with shape matching the broadcasted
        leading dimensions.
    """
    if points_a.shape[-1] != 2 or points_b.shape[-1] != 2:
        raise ValueError("Both inputs must end with latitude/longitude pairs.")

    if points_a.device != points_b.device:
        raise ValueError("Both inputs must be on the same device.")

    if not torch.isfinite(points_a).all() or not torch.isfinite(points_b).all():
        raise ValueError("Coordinates must contain only finite values.")

    lat_a, lon_a = points_a.unbind(dim=-1)
    lat_b, lon_b = points_b.unbind(dim=-1)

    lat_a = torch.deg2rad(lat_a)
    lon_a = torch.deg2rad(lon_a)
    lat_b = torch.deg2rad(lat_b)
    lon_b = torch.deg2rad(lon_b)

    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a

    a = (
        torch.sin(delta_lat / 2).square()
        + torch.cos(lat_a)
        * torch.cos(lat_b)
        * torch.sin(delta_lon / 2).square()
    )

    # Guard against floating-point rounding outside the valid range.
    a = torch.clamp(a, min=0.0, max=1.0)

    earth_radius_km = 6371.0088
    return 2 * earth_radius_km * torch.asin(torch.sqrt(a))
