"""
Evaluation metrics for manifold-aware sparse autoencoders.

This module provides specialized metrics for assessing how well SAEs recover
and represent geometric manifold structure, going beyond standard point-wise
feature recovery metrics (MCC, F1).

Key metrics:
- Manifold detection: Identify groups of latents representing manifolds
- Geodesic preservation: Measure if SAE preserves manifold distances
- Topology preservation: Use persistent homology to verify topological structure
- Curvature accuracy: Assess if intrinsic curvature is captured

References:
    Michaud et al. (2024): "Understanding SAE Scaling with Feature Manifolds"
    Kriegeskorte et al. (2008): "Representational Similarity Analysis"
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.stats import spearmanr

from sae_lens.synthetic.manifolds import (
    ManifoldType,
    compute_geodesic_distance_circular,
    compute_geodesic_distance_spherical,
)


@dataclass
class ManifoldAlignmentResult:
    """
    Results from manifold alignment evaluation.

    Attributes:
        manifold_id: Which manifold this result is for
        manifold_type: Type of manifold (circular, spherical, toroidal)
        geodesic_correlation: Spearman correlation between geodesic distances
        euclidean_correlation: Correlation with Euclidean distances (for comparison)
        intrinsic_dim_estimate: Estimated intrinsic dimensionality from SAE
        detected_latent_group: Indices of SAE latents representing this manifold
        num_latents_used: How many latents the SAE uses for this manifold
    """

    manifold_id: int
    manifold_type: str
    geodesic_correlation: float
    euclidean_correlation: float
    intrinsic_dim_estimate: int
    detected_latent_group: list[int]
    num_latents_used: int


@dataclass
class TopologyPreservationResult:
    """
    Results from topological analysis using persistent homology.

    Attributes:
        manifold_id: Which manifold this result is for
        h0_score: H0 (connected components) preservation score
        h1_score: H1 (circular loops) preservation score
        h2_score: H2 (spherical shells) preservation score
        bottleneck_distance: Bottleneck distance between persistence diagrams
    """

    manifold_id: int
    h0_score: float
    h1_score: float
    h2_score: float
    bottleneck_distance: float


def detect_manifold_clusters_via_coactivation(
    sae_activations: torch.Tensor,
    threshold_correlation: float = 0.3,
    min_cluster_size: int = 2,
) -> list[list[int]]:
    """
    Detect groups of SAE latents that consistently co-activate (potential manifolds).

    When an SAE learns a manifold, multiple latents often fire together to tile
    the manifold surface. This function identifies such groups via co-activation patterns.

    Args:
        sae_activations: Tensor of shape (num_samples, num_latents) with binary activations
        threshold_correlation: Minimum co-activation frequency to consider as neighbors
        min_cluster_size: Minimum size for a cluster to be considered a manifold

    Returns:
        List of clusters, where each cluster is a list of latent indices
    """
    num_latents = sae_activations.shape[1]

    # Binarize activations
    binary_acts = (sae_activations > 0).float()

    # Compute co-activation frequency matrix
    # co_activation[i,j] = fraction of samples where both latent i and j are active
    co_activation = (binary_acts.T @ binary_acts) / binary_acts.shape[0]

    # Threshold to adjacency matrix
    adjacency = (co_activation > threshold_correlation).float()
    adjacency.fill_diagonal_(0)  # Remove self-loops

    # Find connected components using depth-first search
    visited = torch.zeros(num_latents, dtype=torch.bool)
    clusters = []

    def dfs(node: int, cluster: list[int]) -> None:
        visited[node] = True
        cluster.append(node)
        neighbors = torch.where(adjacency[node] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                dfs(neighbor.item(), cluster)

    for node in range(num_latents):
        if not visited[node]:
            cluster: list[int] = []
            dfs(node, cluster)
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

    return clusters


def analyze_decoder_subspace_geometry(
    decoder_weights: torch.Tensor,
    latent_group: list[int],
) -> dict:
    """
    Analyze geometric structure of decoder columns for a group of latents.

    If latents represent a manifold, their decoder columns should span a
    low-dimensional subspace with manifold structure (e.g., circular, spherical).

    Args:
        decoder_weights: Tensor of shape (hidden_dim, num_latents)
        latent_group: Indices of latents to analyze

    Returns:
        Dict with keys:
            'intrinsic_dim': Estimated intrinsic dimensionality
            'singular_values': Top singular values
            'circularity_score': How circular the structure is (if 2D)
            'sphericity_score': How spherical the structure is (if 3D)
    """
    # Extract decoder columns for this group
    group_weights = decoder_weights[:, latent_group]  # (hidden_dim, num_in_group)

    # Perform SVD to find intrinsic dimensionality
    U, S, Vt = torch.linalg.svd(group_weights, full_matrices=False)

    # Estimate intrinsic dimension (number of significant singular values)
    threshold = 0.1 * S[0].item()
    intrinsic_dim = (S > threshold).sum().item()

    # Project to top principal components for geometry analysis
    top_k = min(3, len(S))
    projected = Vt[:top_k].T  # (num_in_group, top_k)

    result = {
        "intrinsic_dim": intrinsic_dim,
        "singular_values": S.cpu().tolist(),
        "circularity_score": 0.0,
        "sphericity_score": 0.0,
    }

    # Check for circular structure (2D)
    if intrinsic_dim == 2 and projected.shape[1] >= 2:
        result["circularity_score"] = measure_circularity(projected[:, :2])

    # Check for spherical structure (3D)
    if intrinsic_dim == 3 and projected.shape[1] >= 3:
        result["sphericity_score"] = measure_sphericity(projected[:, :3])

    return result


def measure_circularity(points_2d: torch.Tensor) -> float:
    """
    Measure how circular a set of 2D points is.

    Fits a circle to the points and computes how well they lie on that circle.
    Score of 1.0 means perfect circle, 0.0 means very non-circular.

    Args:
        points_2d: Tensor of shape (N, 2) with 2D point coordinates

    Returns:
        Circularity score in [0, 1]
    """
    points_np = points_2d.detach().cpu().numpy()

    def circle_residuals(params, points):
        cx, cy, r = params
        distances = ((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2) ** 0.5
        return distances - r

    # Initial guess: center at mean, radius from variance
    cx_init = points_np[:, 0].mean()
    cy_init = points_np[:, 1].mean()
    r_init = ((points_np[:, 0] - cx_init) ** 2 + (points_np[:, 1] - cy_init) ** 2).mean() ** 0.5

    try:
        result = least_squares(
            circle_residuals, [cx_init, cy_init, r_init], args=(points_np,)
        )
        cx, cy, r = result.x
        residuals = circle_residuals(result.x, points_np)

        # Circularity = 1 - (std of residuals / radius)
        circularity = max(0.0, 1.0 - (residuals.std() / (r + 1e-8)))
        return float(circularity)
    except Exception:
        return 0.0


def measure_sphericity(points_3d: torch.Tensor) -> float:
    """
    Measure how spherical a set of 3D points is.

    Fits a sphere to the points and computes how well they lie on that sphere.
    Score of 1.0 means perfect sphere, 0.0 means very non-spherical.

    Args:
        points_3d: Tensor of shape (N, 3) with 3D point coordinates

    Returns:
        Sphericity score in [0, 1]
    """
    points_np = points_3d.detach().cpu().numpy()

    def sphere_residuals(params, points):
        cx, cy, cz, r = params
        distances = (
            (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 + (points[:, 2] - cz) ** 2
        ) ** 0.5
        return distances - r

    # Initial guess: center at mean, radius from variance
    center_init = points_np.mean(axis=0)
    r_init = ((points_np - center_init) ** 2).sum(axis=1).mean() ** 0.5

    try:
        result = least_squares(
            sphere_residuals,
            [center_init[0], center_init[1], center_init[2], r_init],
            args=(points_np,),
        )
        cx, cy, cz, r = result.x
        residuals = sphere_residuals(result.x, points_np)

        # Sphericity = 1 - (std of residuals / radius)
        sphericity = max(0.0, 1.0 - (residuals.std() / (r + 1e-8)))
        return float(sphericity)
    except Exception:
        return 0.0


def compute_manifold_alignment_score(
    gt_manifold_coords: torch.Tensor,
    gt_manifold_type: ManifoldType,
    sae_latent_activations: torch.Tensor,
    latent_group: list[int],
) -> ManifoldAlignmentResult:
    """
    Compute alignment between ground-truth manifold and SAE-learned representation.

    Measures how well the SAE preserves geodesic distances on the manifold.
    Good alignment means geodesic distances in ground truth correlate with
    Euclidean distances in SAE latent space.

    Args:
        gt_manifold_coords: Ground-truth manifold coordinates (N, intrinsic_dim)
        gt_manifold_type: Type of manifold
        sae_latent_activations: SAE activations (N_samples, num_latents)
        latent_group: Indices of SAE latents representing this manifold

    Returns:
        ManifoldAlignmentResult with alignment metrics
    """
    # Extract activations for the detected latent group
    if len(latent_group) == 0:
        return ManifoldAlignmentResult(
            manifold_id=-1,
            manifold_type=gt_manifold_type.value,
            geodesic_correlation=0.0,
            euclidean_correlation=0.0,
            intrinsic_dim_estimate=0,
            detected_latent_group=[],
            num_latents_used=0,
        )

    group_acts = sae_latent_activations[:, latent_group]

    # Compute ground-truth geodesic distances
    if gt_manifold_type == ManifoldType.CIRCULAR:
        # For circular manifolds, gt_manifold_coords should contain angles
        if gt_manifold_coords.shape[1] == 2:
            # If given as (cos, sin), convert to angles
            angles = torch.atan2(gt_manifold_coords[:, 1], gt_manifold_coords[:, 0])
        else:
            angles = gt_manifold_coords.squeeze()
        gt_distances = compute_geodesic_distance_circular(angles)

    elif gt_manifold_type == ManifoldType.SPHERICAL:
        # For spherical manifolds, coords should be 3D unit vectors
        gt_distances = compute_geodesic_distance_spherical(gt_manifold_coords)

    else:  # Toroidal or unknown
        # Fall back to Euclidean distance in intrinsic space
        gt_distances = torch.cdist(gt_manifold_coords, gt_manifold_coords)

    # Compute SAE latent space distances
    sae_distances = torch.cdist(group_acts, group_acts)

    # Also compute Euclidean distances in ground-truth space for comparison
    gt_euclidean = torch.cdist(gt_manifold_coords, gt_manifold_coords)

    # Flatten upper triangular parts (exclude diagonal)
    n = gt_distances.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1)

    gt_dist_flat = gt_distances[triu_indices[0], triu_indices[1]].cpu().numpy()
    sae_dist_flat = sae_distances[triu_indices[0], triu_indices[1]].cpu().numpy()
    gt_euclidean_flat = gt_euclidean[triu_indices[0], triu_indices[1]].cpu().numpy()

    # Compute Spearman correlation (rank correlation, robust to monotonic transforms)
    geodesic_corr, _ = spearmanr(gt_dist_flat, sae_dist_flat)
    euclidean_corr, _ = spearmanr(gt_euclidean_flat, sae_dist_flat)

    # Estimate intrinsic dimensionality from SAE (via PCA)
    _, S, _ = torch.linalg.svd(group_acts.T, full_matrices=False)
    threshold = 0.1 * S[0].item()
    intrinsic_dim = (S > threshold).sum().item()

    return ManifoldAlignmentResult(
        manifold_id=-1,  # Will be set by caller
        manifold_type=gt_manifold_type.value,
        geodesic_correlation=float(geodesic_corr) if not torch.isnan(torch.tensor(geodesic_corr)) else 0.0,
        euclidean_correlation=float(euclidean_corr) if not torch.isnan(torch.tensor(euclidean_corr)) else 0.0,
        intrinsic_dim_estimate=intrinsic_dim,
        detected_latent_group=latent_group,
        num_latents_used=len(latent_group),
    )


def compute_topology_preservation_score(
    gt_manifold_coords: torch.Tensor,
    sae_latent_activations: torch.Tensor,
    latent_group: list[int],
    max_dim: int = 2,
) -> TopologyPreservationResult:
    """
    Measure topological preservation using persistent homology.

    Compares persistence diagrams of the ground-truth manifold and the
    SAE-learned representation to verify that topological features
    (connected components, loops, voids) are preserved.

    Args:
        gt_manifold_coords: Ground-truth manifold coordinates
        sae_latent_activations: SAE activations
        latent_group: Indices of SAE latents for this manifold
        max_dim: Maximum homology dimension to compute (0=components, 1=loops, 2=voids)

    Returns:
        TopologyPreservationResult with preservation scores
    """
    try:
        from ripser import ripser
        from persim import bottleneck
    except ImportError:
        # If ripser not installed, return default scores
        return TopologyPreservationResult(
            manifold_id=-1,
            h0_score=0.0,
            h1_score=0.0,
            h2_score=0.0,
            bottleneck_distance=float("inf"),
        )

    if len(latent_group) == 0:
        return TopologyPreservationResult(
            manifold_id=-1, h0_score=0.0, h1_score=0.0, h2_score=0.0, bottleneck_distance=float("inf")
        )

    group_acts = sae_latent_activations[:, latent_group]

    # Compute persistence diagrams
    gt_result = ripser(gt_manifold_coords.cpu().numpy(), maxdim=max_dim)
    sae_result = ripser(group_acts.cpu().numpy(), maxdim=max_dim)

    gt_diagrams = gt_result["dgms"]
    sae_diagrams = sae_result["dgms"]

    # Compute preservation scores for each homology dimension
    h0_score = 0.0
    h1_score = 0.0
    h2_score = 0.0

    # H0: Connected components
    if len(gt_diagrams) > 0 and len(sae_diagrams) > 0:
        h0_dist = bottleneck(gt_diagrams[0], sae_diagrams[0])
        h0_score = float(max(0.0, 1.0 - h0_dist))

    # H1: Loops (e.g., for circular manifolds)
    if len(gt_diagrams) > 1 and len(sae_diagrams) > 1:
        h1_dist = bottleneck(gt_diagrams[1], sae_diagrams[1])
        h1_score = float(max(0.0, 1.0 - h1_dist))

    # H2: Voids (e.g., for spherical manifolds)
    if len(gt_diagrams) > 2 and len(sae_diagrams) > 2:
        h2_dist = bottleneck(gt_diagrams[2], sae_diagrams[2])
        h2_score = float(max(0.0, 1.0 - h2_dist))

    # Overall bottleneck distance (average across dimensions)
    total_dist = 0.0
    count = 0
    for i in range(min(len(gt_diagrams), len(sae_diagrams))):
        if len(gt_diagrams[i]) > 0 and len(sae_diagrams[i]) > 0:
            total_dist += bottleneck(gt_diagrams[i], sae_diagrams[i])
            count += 1

    avg_bottleneck = total_dist / count if count > 0 else float("inf")

    return TopologyPreservationResult(
        manifold_id=-1,
        h0_score=h0_score,
        h1_score=h1_score,
        h2_score=h2_score,
        bottleneck_distance=float(avg_bottleneck),
    )


def compute_manifold_aware_mcc(
    gt_feature_activations: torch.Tensor,
    sae_latent_activations: torch.Tensor,
    manifold_ranges: list[tuple[int, int]],
    device: str | torch.device = "cpu",
) -> tuple[float, list[float]]:
    """
    Compute Matthews Correlation Coefficient for manifold features.

    For each ground-truth manifold point, finds the best-matching SAE latent
    using optimal bipartite matching (Hungarian algorithm), then computes MCC.

    Args:
        gt_feature_activations: Ground-truth activations (num_samples, num_features)
        sae_latent_activations: SAE activations (num_samples, num_latents)
        manifold_ranges: List of (start_idx, end_idx) for each manifold
        device: Device for computations

    Returns:
        average_mcc: Average MCC across all manifolds
        per_manifold_mcc: List of MCC scores for each manifold
    """
    from sklearn.metrics import matthews_corrcoef

    per_manifold_mcc = []

    for start_idx, end_idx in manifold_ranges:
        # Ground-truth activations for this manifold
        gt_manifold = gt_feature_activations[:, start_idx:end_idx]
        gt_binary = (gt_manifold > 0).float().cpu().numpy()

        # Binarize SAE activations
        sae_binary = (sae_latent_activations > 0).float().cpu().numpy()

        num_gt_points = gt_binary.shape[1]
        num_sae_latents = sae_binary.shape[1]

        # Compute MCC matrix: (num_gt_points, num_sae_latents)
        mcc_matrix = torch.zeros(num_gt_points, num_sae_latents, device=device)

        for i in range(num_gt_points):
            for j in range(num_sae_latents):
                try:
                    mcc = matthews_corrcoef(gt_binary[:, i], sae_binary[:, j])
                    mcc_matrix[i, j] = mcc if not torch.isnan(torch.tensor(mcc)) else 0.0
                except Exception:
                    mcc_matrix[i, j] = 0.0

        # Optimal bipartite matching (Hungarian algorithm)
        # Maximize MCC, so negate for linear_sum_assignment (which minimizes)
        row_ind, col_ind = linear_sum_assignment(-mcc_matrix.cpu().numpy())

        # Average MCC for matched pairs
        matched_mccs = mcc_matrix[row_ind, col_ind]
        manifold_mcc = matched_mccs.mean().item()
        per_manifold_mcc.append(float(manifold_mcc))

    average_mcc = sum(per_manifold_mcc) / len(per_manifold_mcc) if per_manifold_mcc else 0.0

    return float(average_mcc), per_manifold_mcc


@dataclass
class ComprehensiveManifoldEvaluation:
    """
    Complete evaluation results for manifold-aware SAE.

    Attributes:
        manifold_detection_rate: Fraction of manifolds successfully detected
        alignment_results: List of ManifoldAlignmentResult per manifold
        topology_results: List of TopologyPreservationResult per manifold
        average_geodesic_correlation: Mean geodesic preservation across manifolds
        average_topology_score: Mean topology preservation across manifolds
        manifold_mcc: Average MCC for manifold features
        per_manifold_mcc: MCC scores per manifold
        average_latents_per_manifold: Mean number of latents used per manifold
    """

    manifold_detection_rate: float
    alignment_results: list[ManifoldAlignmentResult]
    topology_results: list[TopologyPreservationResult]
    average_geodesic_correlation: float
    average_topology_score: float
    manifold_mcc: float
    per_manifold_mcc: list[float]
    average_latents_per_manifold: float


def evaluate_manifold_sae(
    gt_feature_activations: torch.Tensor,
    sae_latent_activations: torch.Tensor,
    manifold_metadata: list[dict],
    gt_manifold_coords_list: list[torch.Tensor],
    device: str | torch.device = "cpu",
) -> ComprehensiveManifoldEvaluation:
    """
    Comprehensive evaluation of SAE performance on manifold features.

    Args:
        gt_feature_activations: Ground-truth feature activations
        sae_latent_activations: SAE latent activations
        manifold_metadata: List of dicts with manifold information
        gt_manifold_coords_list: List of ground-truth manifold coordinates
        device: Device for computations

    Returns:
        ComprehensiveManifoldEvaluation with all metrics
    """
    # Detect manifold clusters in SAE
    detected_clusters = detect_manifold_clusters_via_coactivation(sae_latent_activations)

    # Match detected clusters to ground-truth manifolds
    # (Simple heuristic: match by size similarity)
    num_manifolds = len(manifold_metadata)
    manifold_sizes = [meta["num_points"] for meta in manifold_metadata]

    # Assign clusters to manifolds (greedy matching)
    assigned_clusters = [[] for _ in range(num_manifolds)]
    used_clusters = set()

    for i, expected_size in enumerate(manifold_sizes):
        best_cluster_idx = None
        best_score = float("inf")

        for j, cluster in enumerate(detected_clusters):
            if j in used_clusters:
                continue
            size_diff = abs(len(cluster) - expected_size)
            if size_diff < best_score:
                best_score = size_diff
                best_cluster_idx = j

        if best_cluster_idx is not None:
            assigned_clusters[i] = detected_clusters[best_cluster_idx]
            used_clusters.add(best_cluster_idx)

    # Compute alignment for each manifold
    alignment_results = []
    for i, (meta, gt_coords) in enumerate(zip(manifold_metadata, gt_manifold_coords_list)):
        manifold_type = ManifoldType(meta["type"])
        latent_group = assigned_clusters[i]

        result = compute_manifold_alignment_score(
            gt_coords, manifold_type, sae_latent_activations, latent_group
        )
        result.manifold_id = i
        alignment_results.append(result)

    # Compute topology preservation
    topology_results = []
    for i, (meta, gt_coords) in enumerate(zip(manifold_metadata, gt_manifold_coords_list)):
        latent_group = assigned_clusters[i]

        result = compute_topology_preservation_score(
            gt_coords, sae_latent_activations, latent_group
        )
        result.manifold_id = i
        topology_results.append(result)

    # Compute manifold-aware MCC
    manifold_ranges = [(meta["start_idx"], meta["end_idx"]) for meta in manifold_metadata]
    manifold_mcc, per_manifold_mcc = compute_manifold_aware_mcc(
        gt_feature_activations, sae_latent_activations, manifold_ranges, device
    )

    # Aggregate metrics
    detection_rate = sum(1 for cluster in assigned_clusters if len(cluster) > 0) / num_manifolds

    avg_geodesic_corr = sum(r.geodesic_correlation for r in alignment_results) / len(
        alignment_results
    )

    avg_topology_score = sum(
        (r.h0_score + r.h1_score + r.h2_score) / 3 for r in topology_results
    ) / len(topology_results)

    avg_latents_per_manifold = sum(r.num_latents_used for r in alignment_results) / len(
        alignment_results
    )

    return ComprehensiveManifoldEvaluation(
        manifold_detection_rate=detection_rate,
        alignment_results=alignment_results,
        topology_results=topology_results,
        average_geodesic_correlation=avg_geodesic_corr,
        average_topology_score=avg_topology_score,
        manifold_mcc=manifold_mcc,
        per_manifold_mcc=per_manifold_mcc,
        average_latents_per_manifold=avg_latents_per_manifold,
    )
