"""
Unit tests for manifold generation and manipulation.

Tests cover:
- Circular manifold generation and properties
- Spherical manifold generation and properties
- Toroidal manifold generation
- High-dimensional embedding
- ManifoldFeatureDictionary
- Geodesic distance computation
"""

import pytest
import torch

from sae_lens.synthetic.manifolds import (
    ManifoldConfig,
    ManifoldFeatureDictionary,
    ManifoldType,
    compute_geodesic_distance_circular,
    compute_geodesic_distance_spherical,
    embed_manifold_in_high_dimensional_space,
    generate_circular_manifold,
    generate_spherical_manifold,
    generate_toroidal_manifold,
)


class TestCircularManifold:
    """Tests for circular manifold generation."""

    def test_circular_manifold_shape(self):
        """Test that circular manifold has correct shape."""
        num_points = 32
        coords, angles = generate_circular_manifold(num_points, device="cpu")

        assert coords.shape == (num_points, 2)
        assert angles.shape == (num_points,)

    def test_circular_manifold_unit_norm(self):
        """Test that points lie on unit circle."""
        num_points = 100
        coords, _ = generate_circular_manifold(num_points, device="cpu")

        norms = torch.norm(coords, dim=1)
        assert torch.allclose(norms, torch.ones(num_points), atol=1e-6)

    def test_circular_manifold_uniform_spacing(self):
        """Test that points are approximately uniformly spaced (with no noise)."""
        num_points = 16
        coords, angles = generate_circular_manifold(num_points, noise_level=0.0, device="cpu")

        # Check angles are evenly spaced
        expected_spacing = 2 * torch.pi / num_points
        angle_diffs = torch.diff(angles)

        assert torch.allclose(angle_diffs, torch.full_like(angle_diffs, expected_spacing), atol=1e-5)

    def test_circular_manifold_with_noise(self):
        """Test that noise perturbs but maintains unit norm."""
        num_points = 50
        coords, angles = generate_circular_manifold(num_points, noise_level=0.1, device="cpu")

        # Still on unit circle
        norms = torch.norm(coords, dim=1)
        assert torch.allclose(norms, torch.ones(num_points), atol=1e-6)

        # Angles should be perturbed (not perfectly uniform)
        angle_diffs = torch.diff(angles)
        expected_spacing = 2 * torch.pi / num_points
        # Should have some variance
        assert angle_diffs.std() > 0.01


class TestSphericalManifold:
    """Tests for spherical manifold generation."""

    def test_spherical_manifold_shape(self):
        """Test that spherical manifold has correct shape."""
        num_points = 64
        coords, (phi, theta) = generate_spherical_manifold(num_points, device="cpu")

        assert coords.shape == (num_points, 3)
        assert phi.shape == (num_points,)
        assert theta.shape == (num_points,)

    def test_spherical_manifold_unit_norm(self):
        """Test that points lie on unit sphere."""
        num_points = 100
        coords, _ = generate_spherical_manifold(num_points, device="cpu")

        norms = torch.norm(coords, dim=1)
        assert torch.allclose(norms, torch.ones(num_points), atol=1e-6)

    def test_spherical_manifold_coverage(self):
        """Test that points cover the sphere (not all on one hemisphere)."""
        num_points = 200
        coords, _ = generate_spherical_manifold(num_points, device="cpu")

        # Check x, y, z coordinates span positive and negative
        for dim in range(3):
            assert coords[:, dim].min() < -0.5
            assert coords[:, dim].max() > 0.5

    def test_spherical_manifold_with_noise(self):
        """Test that noise maintains unit norm."""
        num_points = 50
        coords, _ = generate_spherical_manifold(num_points, noise_level=0.1, device="cpu")

        norms = torch.norm(coords, dim=1)
        assert torch.allclose(norms, torch.ones(num_points), atol=1e-6)


class TestToroidalManifold:
    """Tests for toroidal manifold generation."""

    def test_toroidal_manifold_shape_4d(self):
        """Test 4D toroidal manifold shape."""
        num_major, num_minor = 8, 8
        coords, (theta_major, theta_minor) = generate_toroidal_manifold(
            num_major, num_minor, representation="4d", device="cpu"
        )

        total_points = num_major * num_minor
        assert coords.shape == (total_points, 4)
        assert theta_major.shape == (total_points,)
        assert theta_minor.shape == (total_points,)

    def test_toroidal_manifold_shape_3d(self):
        """Test 3D embedded toroidal manifold shape."""
        num_major, num_minor = 8, 8
        coords, _ = generate_toroidal_manifold(
            num_major, num_minor, representation="3d", device="cpu"
        )

        total_points = num_major * num_minor
        assert coords.shape == (total_points, 3)

    def test_toroidal_4d_structure(self):
        """Test that 4D torus is product of two circles."""
        num_major, num_minor = 4, 4
        coords, _ = generate_toroidal_manifold(
            num_major, num_minor, noise_level=0.0, representation="4d", device="cpu"
        )

        # First two dims should be a circle, last two dims should be a circle
        circle1 = coords[:, :2]
        circle2 = coords[:, 2:]

        # Check each unique point on first circle
        for i in range(num_major):
            points = circle1[i * num_minor : (i + 1) * num_minor]
            # All should have same first circle coordinates
            assert torch.allclose(points[:, 0], points[0, 0].expand(num_minor), atol=1e-5)
            assert torch.allclose(points[:, 1], points[0, 1].expand(num_minor), atol=1e-5)


class TestHighDimensionalEmbedding:
    """Tests for embedding manifolds in high-dimensional space."""

    def test_embedding_shape(self):
        """Test that embedding produces correct shape."""
        num_points = 32
        intrinsic_dim = 2
        embedding_dim = 16
        total_dim = 128

        manifold_coords = torch.randn(num_points, intrinsic_dim)
        embedded = embed_manifold_in_high_dimensional_space(
            manifold_coords, embedding_dim, total_dim, device="cpu"
        )

        assert embedded.shape == (num_points, total_dim)

    def test_embedding_unit_norm(self):
        """Test that embedded points have unit norm."""
        num_points = 50
        manifold_coords = torch.randn(num_points, 2)

        embedded = embed_manifold_in_high_dimensional_space(
            manifold_coords, 16, 128, device="cpu"
        )

        norms = torch.norm(embedded, dim=1)
        assert torch.allclose(norms, torch.ones(num_points), atol=1e-6)

    def test_embedding_preserves_distances_approximately(self):
        """Test that embedding approximately preserves relative distances."""
        num_points = 20
        manifold_coords = torch.randn(num_points, 2)

        # Compute distances in original space
        original_dists = torch.cdist(manifold_coords, manifold_coords)

        # Embed
        embedded = embed_manifold_in_high_dimensional_space(
            manifold_coords, 16, 128, seed=42, device="cpu"
        )

        # Compute distances in embedded space
        embedded_dists = torch.cdist(embedded, embedded)

        # Check rank correlation (distances should be monotonically related)
        from scipy.stats import spearmanr

        original_flat = original_dists[torch.triu_indices(num_points, num_points, offset=1)]
        embedded_flat = embedded_dists[torch.triu_indices(num_points, num_points, offset=1)]

        corr, _ = spearmanr(original_flat.numpy(), embedded_flat.numpy())
        assert corr > 0.8  # Should have strong positive correlation

    def test_embedding_reproducibility(self):
        """Test that same seed produces same embedding."""
        manifold_coords = torch.randn(10, 2)

        emb1 = embed_manifold_in_high_dimensional_space(
            manifold_coords, 8, 32, seed=123, device="cpu"
        )
        emb2 = embed_manifold_in_high_dimensional_space(
            manifold_coords, 8, 32, seed=123, device="cpu"
        )

        assert torch.allclose(emb1, emb2)


class TestGeodesicDistances:
    """Tests for geodesic distance computation."""

    def test_circular_geodesic_diagonal_zero(self):
        """Test that geodesic distance from point to itself is zero."""
        angles = torch.linspace(0, 2 * torch.pi, 10)
        dists = compute_geodesic_distance_circular(angles)

        assert torch.allclose(torch.diag(dists), torch.zeros(10))

    def test_circular_geodesic_symmetry(self):
        """Test that geodesic distance matrix is symmetric."""
        angles = torch.rand(20) * 2 * torch.pi
        dists = compute_geodesic_distance_circular(angles)

        assert torch.allclose(dists, dists.T)

    def test_circular_geodesic_opposite_points(self):
        """Test geodesic distance between opposite points is π."""
        # Two points at 0 and π
        angles = torch.tensor([0.0, torch.pi])
        dists = compute_geodesic_distance_circular(angles)

        assert torch.isclose(dists[0, 1], torch.tensor(torch.pi))

    def test_circular_geodesic_max_pi(self):
        """Test that max geodesic distance on circle is π."""
        angles = torch.rand(100) * 2 * torch.pi
        dists = compute_geodesic_distance_circular(angles)

        # Max possible distance is π
        assert dists.max() <= torch.pi + 1e-5

    def test_spherical_geodesic_diagonal_zero(self):
        """Test that geodesic distance from point to itself is zero."""
        coords = torch.randn(10, 3)
        coords = coords / coords.norm(dim=1, keepdim=True)

        dists = compute_geodesic_distance_spherical(coords)

        assert torch.allclose(torch.diag(dists), torch.zeros(10), atol=1e-6)

    def test_spherical_geodesic_symmetry(self):
        """Test that geodesic distance matrix is symmetric."""
        coords = torch.randn(20, 3)
        coords = coords / coords.norm(dim=1, keepdim=True)

        dists = compute_geodesic_distance_spherical(coords)

        assert torch.allclose(dists, dists.T)

    def test_spherical_geodesic_opposite_points(self):
        """Test geodesic distance between antipodal points is π."""
        # Two opposite points
        coords = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

        dists = compute_geodesic_distance_spherical(coords)

        assert torch.isclose(dists[0, 1], torch.tensor(torch.pi))

    def test_spherical_geodesic_orthogonal_points(self):
        """Test geodesic distance between orthogonal points is π/2."""
        coords = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        dists = compute_geodesic_distance_spherical(coords)

        assert torch.isclose(dists[0, 1], torch.tensor(torch.pi / 2), atol=1e-5)


class TestManifoldFeatureDictionary:
    """Tests for ManifoldFeatureDictionary."""

    def test_initialization(self):
        """Test basic initialization."""
        configs = [
            ManifoldConfig(ManifoldType.CIRCULAR, 16, 8, 2, name="day_of_week"),
            ManifoldConfig(ManifoldType.SPHERICAL, 32, 12, 3, name="direction"),
        ]

        feature_dict = ManifoldFeatureDictionary(
            num_independent=100, manifold_configs=configs, hidden_dim=64, device="cpu"
        )

        assert feature_dict.num_independent == 100
        assert feature_dict.num_manifolds == 2
        assert feature_dict.total_features == 100 + 16 + 32

    def test_feature_vectors_shape(self):
        """Test that all feature vectors have correct shape."""
        configs = [ManifoldConfig(ManifoldType.CIRCULAR, 8, 4, 2)]

        feature_dict = ManifoldFeatureDictionary(
            num_independent=50, manifold_configs=configs, hidden_dim=32, device="cpu"
        )

        all_features = feature_dict.get_all_feature_vectors()
        assert all_features.shape == (58, 32)  # 50 + 8 = 58 features

    def test_feature_vectors_unit_norm(self):
        """Test that all feature vectors have unit norm."""
        configs = [
            ManifoldConfig(ManifoldType.CIRCULAR, 10, 4, 2),
            ManifoldConfig(ManifoldType.SPHERICAL, 20, 6, 3),
        ]

        feature_dict = ManifoldFeatureDictionary(
            num_independent=30, manifold_configs=configs, hidden_dim=48, device="cpu"
        )

        all_features = feature_dict.get_all_feature_vectors()
        norms = torch.norm(all_features, dim=1)

        assert torch.allclose(norms, torch.ones(60), atol=1e-5)

    def test_forward_pass(self):
        """Test forward pass through feature dictionary."""
        configs = [ManifoldConfig(ManifoldType.CIRCULAR, 8, 4, 2)]

        feature_dict = ManifoldFeatureDictionary(
            num_independent=20, manifold_configs=configs, hidden_dim=16, device="cpu"
        )

        # Create random feature activations
        batch_size = 5
        feature_acts = torch.randn(batch_size, feature_dict.num_features)

        # Forward pass
        hidden = feature_dict(feature_acts)

        assert hidden.shape == (batch_size, 16)

    def test_manifold_metadata(self):
        """Test that manifold metadata is correctly generated."""
        configs = [
            ManifoldConfig(ManifoldType.CIRCULAR, 10, 5, 2, name="circle1"),
            ManifoldConfig(ManifoldType.SPHERICAL, 20, 8, 3, name="sphere1"),
        ]

        feature_dict = ManifoldFeatureDictionary(
            num_independent=15, manifold_configs=configs, hidden_dim=24, device="cpu"
        )

        metadata = feature_dict.get_manifold_metadata()

        assert len(metadata) == 2
        assert metadata[0]["name"] == "circle1"
        assert metadata[0]["start_idx"] == 15
        assert metadata[0]["end_idx"] == 25
        assert metadata[0]["intrinsic_dim"] == 2

        assert metadata[1]["name"] == "sphere1"
        assert metadata[1]["start_idx"] == 25
        assert metadata[1]["end_idx"] == 45
        assert metadata[1]["intrinsic_dim"] == 3

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same feature dictionary."""
        configs = [ManifoldConfig(ManifoldType.CIRCULAR, 8, 4, 2)]

        dict1 = ManifoldFeatureDictionary(
            num_independent=10, manifold_configs=configs, hidden_dim=16, seed=42, device="cpu"
        )

        dict2 = ManifoldFeatureDictionary(
            num_independent=10, manifold_configs=configs, hidden_dim=16, seed=42, device="cpu"
        )

        features1 = dict1.get_all_feature_vectors()
        features2 = dict2.get_all_feature_vectors()

        assert torch.allclose(features1, features2)


class TestManifoldConfig:
    """Tests for ManifoldConfig validation."""

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = ManifoldConfig(
            manifold_type=ManifoldType.CIRCULAR,
            num_points=16,
            embedding_dim=8,
            intrinsic_dim=2,
            noise_level=0.1,
        )

        assert config.manifold_type == ManifoldType.CIRCULAR
        assert config.num_points == 16

    def test_invalid_num_points(self):
        """Test that invalid num_points raises error."""
        with pytest.raises(ValueError, match="num_points must be >= 2"):
            ManifoldConfig(
                manifold_type=ManifoldType.CIRCULAR,
                num_points=1,
                embedding_dim=8,
                intrinsic_dim=2,
            )

    def test_invalid_embedding_dim(self):
        """Test that embedding_dim < intrinsic_dim raises error."""
        with pytest.raises(ValueError, match="embedding_dim.*must be >= intrinsic_dim"):
            ManifoldConfig(
                manifold_type=ManifoldType.CIRCULAR,
                num_points=16,
                embedding_dim=1,
                intrinsic_dim=2,
            )

    def test_invalid_noise_level(self):
        """Test that invalid noise_level raises error."""
        with pytest.raises(ValueError, match="noise_level must be in"):
            ManifoldConfig(
                manifold_type=ManifoldType.CIRCULAR,
                num_points=16,
                embedding_dim=8,
                intrinsic_dim=2,
                noise_level=1.5,
            )
