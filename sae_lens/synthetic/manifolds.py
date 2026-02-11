"""
Manifold feature generation for synthetic SAE experiments.

This module provides tools for creating feature manifolds (circular, spherical, toroidal)
that can be embedded in high-dimensional activation spaces. These manifolds represent
concepts that are inherently multi-dimensional (e.g., days of the week, spatial directions).

Key concepts:
- Circular manifolds (S¹): Periodic features like days, months, angles
- Spherical manifolds (S²): Directional features like spatial orientations
- Toroidal manifolds (S¹ × S¹): Compositional periodic features like hour+day

References:
    Engels et al. (2025): "Not All Language Model Features Are One-Dimensionally Linear"
    Li et al. (2025): "The Geometry of Concepts: Sparse Autoencoder Feature Structure"
    Olah & Batson (2023): "Feature Manifold Toy Model"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
import torch.nn as nn


class ManifoldType(str, Enum):
    """Types of manifolds supported."""

    CIRCULAR = "circular"
    SPHERICAL = "spherical"
    TOROIDAL = "toroidal"


@dataclass
class ManifoldConfig:
    """
    Configuration for a single manifold feature.

    Attributes:
        manifold_type: Type of manifold (circular, spherical, toroidal)
        num_points: Number of discrete points to sample on the manifold
        embedding_dim: Number of dimensions to use for embedding in activation space
        intrinsic_dim: Intrinsic dimensionality of the manifold
        noise_level: Amount of noise to add (0.0 = none, 0.1 = 10% perturbation)
        name: Optional name for this manifold (e.g., "day_of_week", "direction")
    """

    manifold_type: ManifoldType
    num_points: int
    embedding_dim: int
    intrinsic_dim: int
    noise_level: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_points < 2:
            raise ValueError(f"num_points must be >= 2, got {self.num_points}")
        if self.embedding_dim < self.intrinsic_dim:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be >= intrinsic_dim ({self.intrinsic_dim})"
            )
        if not 0.0 <= self.noise_level <= 1.0:
            raise ValueError(f"noise_level must be in [0, 1], got {self.noise_level}")


def generate_circular_manifold(
    num_points: int,
    noise_level: float = 0.0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate points uniformly distributed on a circle (S¹).

    The circle is represented in 2D as (cos θ, sin θ) where θ ∈ [0, 2π).
    This is useful for periodic features like days of the week, months, angles.

    Args:
        num_points: Number of points to sample on the circle
        noise_level: Amount of tangent noise to add (0.0 = none, 0.1 = 10% angular noise)
        device: Device to create tensors on

    Returns:
        coords_2d: Tensor of shape (num_points, 2) with 2D circular coordinates
        angles: Tensor of shape (num_points,) with the angles in [0, 2π)
    """
    # Generate evenly spaced angles
    angles = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]

    if noise_level > 0:
        # Add tangent noise (angular perturbation)
        angle_noise = torch.randn(num_points, device=device) * noise_level
        angles_noisy = angles + angle_noise
        # Keep angles in [0, 2π) range
        angles_noisy = angles_noisy % (2 * torch.pi)
    else:
        angles_noisy = angles

    # Convert to 2D Cartesian coordinates
    coords_2d = torch.stack([torch.cos(angles_noisy), torch.sin(angles_noisy)], dim=1)

    # Ensure unit norm (may drift slightly due to noise)
    coords_2d = coords_2d / coords_2d.norm(dim=1, keepdim=True).clamp(min=1e-8)

    return coords_2d, angles


def generate_spherical_manifold(
    num_points: int,
    noise_level: float = 0.0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate points uniformly distributed on a sphere (S²) using Fibonacci sphere algorithm.

    The sphere is represented in 3D as unit vectors (x, y, z) with x² + y² + z² = 1.
    This is useful for directional features like spatial orientations, gaze directions.

    Args:
        num_points: Number of points to sample on the sphere
        noise_level: Amount of tangent noise to add (perpendicular to radius)
        device: Device to create tensors on

    Returns:
        coords_3d: Tensor of shape (num_points, 3) with 3D spherical coordinates
        angles: Tuple of (phi, theta) tensors, each of shape (num_points,)
            phi: Polar angle in [0, π]
            theta: Azimuthal angle in [0, 2π)
    """
    # Fibonacci sphere algorithm for uniform distribution
    indices = torch.arange(num_points, device=device, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / num_points)  # Polar angle
    golden_ratio = (1 + torch.sqrt(torch.tensor(5.0, device=device))) / 2
    theta = 2 * torch.pi * indices / golden_ratio  # Azimuthal angle
    theta = theta % (2 * torch.pi)

    # Convert to Cartesian coordinates
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    coords_3d = torch.stack([x, y, z], dim=1)

    if noise_level > 0:
        # Add tangent noise (perpendicular to radial direction)
        tangent_noise = torch.randn_like(coords_3d) * noise_level

        # Project out radial component to keep noise tangent to sphere
        radial_component = (tangent_noise * coords_3d).sum(dim=1, keepdim=True)
        tangent_noise = tangent_noise - radial_component * coords_3d

        # Add noise and renormalize
        coords_3d = coords_3d + tangent_noise
        coords_3d = coords_3d / coords_3d.norm(dim=1, keepdim=True).clamp(min=1e-8)

    return coords_3d, (phi, theta)


def generate_toroidal_manifold(
    num_points_major: int,
    num_points_minor: int,
    noise_level: float = 0.0,
    device: str | torch.device = "cpu",
    representation: Literal["4d", "3d"] = "4d",
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate points on a torus (S¹ × S¹).

    A torus is the product of two circles, representing compositional periodic features
    like (hour of day, day of week) or (latitude, longitude) on Earth.

    Args:
        num_points_major: Number of points on the major circle
        num_points_minor: Number of points on the minor circle
        noise_level: Amount of angular noise to add
        device: Device to create tensors on
        representation: '4d' for (cos θ₁, sin θ₁, cos θ₂, sin θ₂) or
                       '3d' for embedded torus in 3D space

    Returns:
        coords: Tensor of shape (num_points_total, dim) with toroidal coordinates
            dim = 4 for '4d', dim = 3 for '3d'
        angles: Tuple of (theta_major, theta_minor) tensors
    """
    num_points_total = num_points_major * num_points_minor

    # Generate grid of angles
    theta_major = torch.linspace(
        0, 2 * torch.pi, num_points_major + 1, device=device
    )[:-1]
    theta_minor = torch.linspace(
        0, 2 * torch.pi, num_points_minor + 1, device=device
    )[:-1]

    # Create meshgrid
    theta_major_grid, theta_minor_grid = torch.meshgrid(
        theta_major, theta_minor, indexing="ij"
    )
    theta_major_flat = theta_major_grid.reshape(-1)
    theta_minor_flat = theta_minor_grid.reshape(-1)

    if noise_level > 0:
        # Add angular noise
        major_noise = torch.randn(num_points_total, device=device) * noise_level
        minor_noise = torch.randn(num_points_total, device=device) * noise_level
        theta_major_flat = (theta_major_flat + major_noise) % (2 * torch.pi)
        theta_minor_flat = (theta_minor_flat + minor_noise) % (2 * torch.pi)

    if representation == "4d":
        # 4D representation: direct product of circles
        coords = torch.stack(
            [
                torch.cos(theta_major_flat),
                torch.sin(theta_major_flat),
                torch.cos(theta_minor_flat),
                torch.sin(theta_minor_flat),
            ],
            dim=1,
        )
    else:  # '3d'
        # 3D embedded torus: (R + r cos θ₂) cos θ₁, (R + r cos θ₂) sin θ₁, r sin θ₂
        # where R = major radius, r = minor radius
        R = 2.0  # Major radius
        r = 1.0  # Minor radius
        x = (R + r * torch.cos(theta_minor_flat)) * torch.cos(theta_major_flat)
        y = (R + r * torch.cos(theta_minor_flat)) * torch.sin(theta_major_flat)
        z = r * torch.sin(theta_minor_flat)
        coords = torch.stack([x, y, z], dim=1)

        # Normalize to unit scale (optional, for consistency)
        coords = coords / coords.abs().max()

    return coords, (theta_major_flat, theta_minor_flat)


def embed_manifold_in_high_dimensional_space(
    manifold_coords: torch.Tensor,
    embedding_dim: int,
    total_dim: int,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Embed low-dimensional manifold coordinates into high-dimensional activation space.

    This creates a random smooth embedding from the manifold's intrinsic space
    into the full activation space (e.g., 768 dimensions for GPT-2 scale models).

    Args:
        manifold_coords: Tensor of shape (num_points, intrinsic_dim) with manifold coordinates
        embedding_dim: Number of dimensions to use for embedding (e.g., 16-32)
        total_dim: Total dimensionality of activation space (e.g., 768)
        seed: Random seed for reproducible embedding
        device: Device to create tensors on

    Returns:
        embedded: Tensor of shape (num_points, total_dim) with high-dimensional embeddings
    """
    num_points, intrinsic_dim = manifold_coords.shape

    if embedding_dim < intrinsic_dim:
        raise ValueError(
            f"embedding_dim ({embedding_dim}) must be >= intrinsic_dim ({intrinsic_dim})"
        )
    if total_dim < embedding_dim:
        raise ValueError(
            f"total_dim ({total_dim}) must be >= embedding_dim ({embedding_dim})"
        )

    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Create random projection matrix: intrinsic_dim -> embedding_dim
    W_embed = torch.randn(
        intrinsic_dim, embedding_dim, device=device, generator=generator
    )

    # Orthogonalize to preserve distances (use QR decomposition)
    W_embed, _ = torch.linalg.qr(W_embed)

    # Project manifold to embedding_dim subspace
    embedded_low = manifold_coords @ W_embed  # (num_points, embedding_dim)

    # Place in full dimensional space at random location
    embedded_full = torch.zeros(num_points, total_dim, device=device)

    # Random starting position for the embedding
    start_idx = torch.randint(
        0, total_dim - embedding_dim + 1, (1,), generator=generator
    ).item()

    embedded_full[:, start_idx : start_idx + embedding_dim] = embedded_low

    # Normalize to unit norm (consistent with FeatureDictionary convention)
    embedded_full = embedded_full / embedded_full.norm(
        dim=1, keepdim=True
    ).clamp(min=1e-8)

    return embedded_full


class ManifoldFeatureDictionary(nn.Module):
    """
    Feature dictionary containing both independent features and manifold features.

    This extends the standard feature dictionary to support manifolds alongside
    independent 1D features, enabling realistic synthetic benchmarks that test
    SAE performance on geometric structure.

    Attributes:
        independent_features: Tensor of shape (num_independent, hidden_dim)
            Standard independent feature vectors
        manifold_features: List of tensors, one per manifold
            Each tensor has shape (num_points, hidden_dim)
        manifold_configs: List of ManifoldConfig objects describing each manifold
        manifold_ranges: List of (start_idx, end_idx) tuples in the full feature space
    """

    def __init__(
        self,
        num_independent: int,
        manifold_configs: list[ManifoldConfig],
        hidden_dim: int,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ):
        """
        Create a hybrid feature dictionary with both independent and manifold features.

        Args:
            num_independent: Number of independent 1D features
            manifold_configs: List of configurations for manifold features
            hidden_dim: Dimensionality of the activation space
            device: Device to create tensors on
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.num_independent = num_independent
        self.hidden_dim = hidden_dim
        self.manifold_configs = manifold_configs
        self.device = device

        # Generate independent features (random unit vectors)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        independent = torch.randn(
            num_independent, hidden_dim, device=device, generator=generator
        )
        independent = independent / independent.norm(dim=1, keepdim=True).clamp(
            min=1e-8
        )
        self.register_buffer("independent_features", independent)

        # Generate manifold features
        manifold_features_list = []
        manifold_ranges = []
        current_idx = num_independent

        for i, config in enumerate(manifold_configs):
            # Generate manifold points in intrinsic space
            if config.manifold_type == ManifoldType.CIRCULAR:
                coords, _ = generate_circular_manifold(
                    config.num_points, config.noise_level, device
                )
            elif config.manifold_type == ManifoldType.SPHERICAL:
                coords, _ = generate_spherical_manifold(
                    config.num_points, config.noise_level, device
                )
            elif config.manifold_type == ManifoldType.TOROIDAL:
                # For toroidal, assume square grid
                side = int(config.num_points**0.5)
                coords, _ = generate_toroidal_manifold(
                    side, side, config.noise_level, device, representation="4d"
                )
            else:
                raise ValueError(f"Unknown manifold type: {config.manifold_type}")

            # Embed in high-dimensional space
            embedded = embed_manifold_in_high_dimensional_space(
                coords,
                config.embedding_dim,
                hidden_dim,
                seed=seed + i if seed is not None else None,
                device=device,
            )

            manifold_features_list.append(embedded)
            manifold_ranges.append((current_idx, current_idx + config.num_points))
            current_idx += config.num_points

        # Store manifold features as buffers
        for i, manifold_features in enumerate(manifold_features_list):
            self.register_buffer(f"manifold_features_{i}", manifold_features)

        self.manifold_ranges = manifold_ranges
        self.num_manifolds = len(manifold_configs)
        self.total_features = current_idx

    @property
    def num_features(self) -> int:
        """Total number of features (independent + all manifold points)."""
        return self.total_features

    def get_all_feature_vectors(self) -> torch.Tensor:
        """
        Get all feature vectors concatenated into a single matrix.

        Returns:
            Tensor of shape (total_features, hidden_dim)
        """
        manifold_features = [
            getattr(self, f"manifold_features_{i}") for i in range(self.num_manifolds)
        ]
        return torch.cat([self.independent_features] + manifold_features, dim=0)

    def forward(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """
        Convert feature activations to hidden activations.

        Args:
            feature_activations: Tensor of shape (batch, total_features)

        Returns:
            Tensor of shape (batch, hidden_dim)
        """
        all_features = self.get_all_feature_vectors()
        return feature_activations @ all_features

    def get_manifold_metadata(self) -> list[dict]:
        """
        Get metadata about each manifold for evaluation purposes.

        Returns:
            List of dicts with keys: 'manifold_id', 'type', 'start_idx', 'end_idx',
                'num_points', 'intrinsic_dim', 'name'
        """
        metadata = []
        for i, config in enumerate(self.manifold_configs):
            start_idx, end_idx = self.manifold_ranges[i]
            metadata.append(
                {
                    "manifold_id": i,
                    "type": config.manifold_type.value,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_points": config.num_points,
                    "intrinsic_dim": config.intrinsic_dim,
                    "embedding_dim": config.embedding_dim,
                    "name": config.name or f"manifold_{i}",
                }
            )
        return metadata


def compute_geodesic_distance_circular(angles: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise geodesic distances on a circle.

    For points at angles θᵢ and θⱼ on a circle, the geodesic distance is
    the shorter arc length: min(|θᵢ - θⱼ|, 2π - |θᵢ - θⱼ|).

    Args:
        angles: Tensor of shape (N,) with angles in [0, 2π)

    Returns:
        Tensor of shape (N, N) with pairwise geodesic distances
    """
    N = angles.shape[0]
    angles_i = angles.unsqueeze(1).expand(N, N)
    angles_j = angles.unsqueeze(0).expand(N, N)

    diff = torch.abs(angles_i - angles_j)
    geodesic = torch.minimum(diff, 2 * torch.pi - diff)

    return geodesic


def compute_geodesic_distance_spherical(coords_3d: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise geodesic distances on a sphere.

    For unit vectors p and q on a sphere, the geodesic distance (great circle distance)
    is arccos(p · q).

    Args:
        coords_3d: Tensor of shape (N, 3) with unit vectors

    Returns:
        Tensor of shape (N, N) with pairwise geodesic distances
    """
    # Compute dot products
    dot_products = coords_3d @ coords_3d.T

    # Clamp to [-1, 1] to avoid numerical issues with arccos
    dot_products = torch.clamp(dot_products, -1.0, 1.0)

    # Geodesic distance is arccos(dot product)
    geodesic = torch.acos(dot_products)

    return geodesic
