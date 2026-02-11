"""
Manifold-Parametric Sparse Autoencoder (MP-SAE).

This SAE architecture explicitly parameterizes features as manifold coordinates,
allowing it to natively represent circular, spherical, and other geometric structures.
Each manifold module predicts: (1) is manifold active? (2) position on manifold,
(3) magnitude.

Key features:
- Explicit manifold parameterization (angles for circles, unit vectors for spheres)
- Separate prediction of manifold activation, position, and magnitude
- Smooth decoder from manifold parameters to reconstruction
- Can be combined with standard independent features

Reference:
    Inspired by manifold learning techniques and the feature manifold toy model
    (Olah & Batson, 2023).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from sae_lens.saes.sae import SAE


class CircularManifoldModule(nn.Module):
    """
    Module for a single circular manifold (S¹).

    Predicts:
    - Gate: Is this circular feature active?
    - Angle: Position on circle as (cos θ, sin θ)
    - Magnitude: How strongly the feature fires
    """

    def __init__(
        self,
        d_in: int,
        embedding_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize circular manifold module.

        Args:
            d_in: Input dimension
            embedding_dim: Dimension of manifold embedding in reconstruction space
            device: Device for tensors
            dtype: Data type
        """
        super().__init__()

        self.d_in = d_in
        self.embedding_dim = embedding_dim

        # Gating: is this manifold active?
        self.gate_net = nn.Linear(d_in, 1, device=device, dtype=dtype)

        # Angle prediction: predict (cos θ, sin θ)
        self.angle_net = nn.Linear(d_in, 2, device=device, dtype=dtype)

        # Magnitude prediction
        self.magnitude_net = nn.Linear(d_in, 1, device=device, dtype=dtype)

        # Decoder: from (cos θ, sin θ) to reconstruction contribution
        # This learns a smooth function from the circle to activation space
        self.W_manifold_dec = nn.Parameter(
            torch.randn(2, embedding_dim, device=device, dtype=dtype) * 0.1
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> dict[str, Float[torch.Tensor, "..."]]:
        """
        Forward pass through circular manifold module.

        Args:
            x: Input activations

        Returns:
            Dict with keys:
                'gate': Binary gate (0 or 1)
                'angle': Normalized 2D vector (cos θ, sin θ)
                'magnitude': Scalar magnitude
                'reconstruction': Contribution to reconstruction
        """
        # Gate
        gate_logit = self.gate_net(x)
        gate = torch.sigmoid(gate_logit)

        # Angle (normalized to unit circle)
        angle_raw = self.angle_net(x)
        angle = F.normalize(angle_raw, dim=-1)

        # Magnitude
        magnitude = F.relu(self.magnitude_net(x))

        # Reconstruction: project angle onto decoder basis
        # recon_contrib = magnitude * gate * angle @ W_manifold_dec
        weighted_angle = magnitude * gate * angle.unsqueeze(-2)  # (..., 1, 2)
        recon_contrib = (weighted_angle @ self.W_manifold_dec).squeeze(-2)  # (..., embedding_dim)

        return {
            "gate": gate.squeeze(-1),
            "angle": angle,
            "magnitude": magnitude.squeeze(-1),
            "reconstruction": recon_contrib,
        }


class SphericalManifoldModule(nn.Module):
    """
    Module for a single spherical manifold (S²).

    Predicts:
    - Gate: Is this spherical feature active?
    - Direction: Position on sphere as 3D unit vector
    - Magnitude: How strongly the feature fires
    """

    def __init__(
        self,
        d_in: int,
        embedding_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize spherical manifold module.

        Args:
            d_in: Input dimension
            embedding_dim: Dimension of manifold embedding in reconstruction space
            device: Device for tensors
            dtype: Data type
        """
        super().__init__()

        self.d_in = d_in
        self.embedding_dim = embedding_dim

        # Gating
        self.gate_net = nn.Linear(d_in, 1, device=device, dtype=dtype)

        # Direction prediction: predict 3D direction
        self.direction_net = nn.Linear(d_in, 3, device=device, dtype=dtype)

        # Magnitude prediction
        self.magnitude_net = nn.Linear(d_in, 1, device=device, dtype=dtype)

        # Decoder: from 3D direction to reconstruction
        self.W_manifold_dec = nn.Parameter(
            torch.randn(3, embedding_dim, device=device, dtype=dtype) * 0.1
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> dict[str, Float[torch.Tensor, "..."]]:
        """
        Forward pass through spherical manifold module.

        Args:
            x: Input activations

        Returns:
            Dict with gate, direction, magnitude, reconstruction
        """
        # Gate
        gate_logit = self.gate_net(x)
        gate = torch.sigmoid(gate_logit)

        # Direction (normalized to unit sphere)
        direction_raw = self.direction_net(x)
        direction = F.normalize(direction_raw, dim=-1)

        # Magnitude
        magnitude = F.relu(self.magnitude_net(x))

        # Reconstruction
        weighted_direction = magnitude * gate * direction.unsqueeze(-2)
        recon_contrib = (weighted_direction @ self.W_manifold_dec).squeeze(-2)

        return {
            "gate": gate.squeeze(-1),
            "direction": direction,
            "magnitude": magnitude.squeeze(-1),
            "reconstruction": recon_contrib,
        }


class ManifoldParametricSAE(SAE):
    """
    Sparse Autoencoder with explicit manifold parameterization.

    Combines:
    - Standard independent latents (for 1D features)
    - Circular manifold modules (for periodic features)
    - Spherical manifold modules (for directional features)

    Each manifold module learns to represent a specific geometric structure,
    predicting both the position on the manifold and the magnitude.

    Attributes:
        independent_encoder: Standard encoder for 1D features
        circular_manifolds: List of CircularManifoldModule
        spherical_manifolds: List of SphericalManifoldModule
        W_dec: Decoder matrix
    """

    def __init__(
        self,
        d_in: int,
        d_sae_independent: int,
        num_circular_manifolds: int = 0,
        num_spherical_manifolds: int = 0,
        manifold_embedding_dim: int = 16,
        activation_fn_str: str = "relu",
        apply_b_dec_to_input: bool = True,
        finetuning_scaling_factor: bool = False,
        context_size: int = 128,
        model_name: str = "unknown",
        model_from_pretrained_path: str | None = None,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        hook_name: str = "unknown",
        hook_layer: int = 0,
        hook_head_index: int | None = None,
        prepend_bos: bool = True,
        dataset_path: str = "",
        dataset_trust_remote_code: bool = True,
        normalize_activations: str = "none",
        lambda_l1_independent: float = 1e-3,
        lambda_manifold: float = 1e-3,
    ):
        """
        Initialize Manifold-Parametric SAE.

        Args:
            d_in: Input dimension
            d_sae_independent: Number of independent (1D) latents
            num_circular_manifolds: Number of circular manifold modules
            num_spherical_manifolds: Number of spherical manifold modules
            manifold_embedding_dim: Embedding dimension for each manifold
            lambda_l1_independent: L1 penalty for independent features
            lambda_manifold: L1 penalty for manifold magnitudes
            Other args: Standard SAE parameters
        """
        # Total SAE dimension includes independent + manifolds
        # (Each circular manifold contributes 2 latent dims, spherical contributes 3)
        d_sae = d_sae_independent + num_circular_manifolds * 2 + num_spherical_manifolds * 3

        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            activation_fn_str=activation_fn_str,
            apply_b_dec_to_input=apply_b_dec_to_input,
            finetuning_scaling_factor=finetuning_scaling_factor,
            context_size=context_size,
            model_name=model_name,
            model_from_pretrained_path=model_from_pretrained_path,
            dtype=dtype,
            device=device,
            hook_name=hook_name,
            hook_layer=hook_layer,
            hook_head_index=hook_head_index,
            prepend_bos=prepend_bos,
            dataset_path=dataset_path,
            dataset_trust_remote_code=dataset_trust_remote_code,
            normalize_activations=normalize_activations,
        )

        self.d_sae_independent = d_sae_independent
        self.num_circular_manifolds = num_circular_manifolds
        self.num_spherical_manifolds = num_spherical_manifolds
        self.manifold_embedding_dim = manifold_embedding_dim
        self.lambda_l1_independent = lambda_l1_independent
        self.lambda_manifold = lambda_manifold

        # Independent feature encoder (standard)
        self.W_enc_independent = nn.Parameter(
            torch.randn(d_in, d_sae_independent, device=device, dtype=dtype) * 0.1
        )

        # Circular manifold modules
        self.circular_manifolds = nn.ModuleList(
            [
                CircularManifoldModule(d_in, manifold_embedding_dim, device, dtype)
                for _ in range(num_circular_manifolds)
            ]
        )

        # Spherical manifold modules
        self.spherical_manifolds = nn.ModuleList(
            [
                SphericalManifoldModule(d_in, manifold_embedding_dim, device, dtype)
                for _ in range(num_spherical_manifolds)
            ]
        )

        # Decoder includes: independent features + manifold embeddings
        total_decoder_dim = d_sae_independent + (num_circular_manifolds + num_spherical_manifolds) * manifold_embedding_dim

        # Override decoder initialization
        self.W_dec = nn.Parameter(
            torch.randn(total_decoder_dim, d_in, device=device, dtype=dtype) * 0.1
        )

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], dict]:
        """
        Encode input to sparse latent representation.

        Args:
            x: Input activations

        Returns:
            Tuple of:
                - Feature activations (concatenated independent + manifold latents)
                - Dict with detailed manifold information
        """
        # Remove decoder bias if configured
        if self.apply_b_dec_to_input:
            x = x - self.b_dec

        # Independent features
        f_independent = F.relu(x @ self.W_enc_independent + self.b_enc[:self.d_sae_independent])

        # Manifold features
        manifold_outputs = []

        # Circular manifolds
        for manifold in self.circular_manifolds:
            output = manifold(x)
            manifold_outputs.append(output)

        # Spherical manifolds
        for manifold in self.spherical_manifolds:
            output = manifold(x)
            manifold_outputs.append(output)

        # Construct full latent vector
        # For independent: just the activations
        # For manifolds: include the parameterization (angle/direction)
        latent_parts = [f_independent]

        for output in manifold_outputs:
            if "angle" in output:  # Circular
                latent_parts.append(output["angle"])
            else:  # Spherical
                latent_parts.append(output["direction"])

        f_full = torch.cat(latent_parts, dim=-1)

        manifold_info = {
            "independent": f_independent,
            "manifold_outputs": manifold_outputs,
        }

        return f_full, manifold_info

    def decode(self, f: Float[torch.Tensor, "... d_sae"]) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode latent representation to reconstruction.

        Note: This uses the combined latent representation which includes
        manifold parameterizations.

        Args:
            f: Latent activations

        Returns:
            Reconstructed activations
        """
        # Split into independent and manifold parts
        f_independent = f[..., :self.d_sae_independent]

        # For manifolds, we need to use the manifold outputs directly
        # This is a simplified version - in practice, would reconstruct from manifold_info

        # Standard reconstruction from independent features
        recon = f_independent @ self.W_dec[:self.d_sae_independent]

        # Add decoder bias
        recon = recon + self.b_dec

        return recon

    def forward(
        self, x: Float[torch.Tensor, "... d_in"], return_manifold_info: bool = False
    ) -> Float[torch.Tensor, "... d_in"] | tuple[Float[torch.Tensor, "... d_in"], dict]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations
            return_manifold_info: If True, return detailed manifold information

        Returns:
            Reconstructed activations (and optionally manifold info)
        """
        # Encode
        _, manifold_info = self.encode(x)

        # Reconstruct from components
        f_independent = manifold_info["independent"]

        # Independent reconstruction
        recon = f_independent @ self.W_dec[:self.d_sae_independent]

        # Add manifold reconstructions
        current_idx = self.d_sae_independent
        for output in manifold_info["manifold_outputs"]:
            embedding_dim = output["reconstruction"].shape[-1]
            manifold_decoder = self.W_dec[current_idx:current_idx+embedding_dim]
            recon = recon + (output["reconstruction"] @ manifold_decoder)
            current_idx += embedding_dim

        # Add bias
        recon = recon + self.b_dec

        if return_manifold_info:
            return recon, manifold_info

        return recon

    def get_loss_dict(
        self,
        x: Float[torch.Tensor, "batch d_in"],
        return_outputs: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """
        Compute loss with separate penalties for independent and manifold features.

        Args:
            x: Input activations
            return_outputs: If True, return forward pass outputs

        Returns:
            Loss dictionary (and optionally outputs)
        """
        # Forward pass
        x_reconstruct, manifold_info = self.forward(x, return_manifold_info=True)

        # Reconstruction loss
        per_item_mse_loss = (x_reconstruct - x).pow(2).sum(dim=-1)
        mse_loss = per_item_mse_loss.mean()

        # Independent L1 sparsity
        f_independent = manifold_info["independent"]
        independent_l1 = f_independent.abs().mean()

        # Manifold sparsity (on magnitudes)
        manifold_magnitudes = torch.stack(
            [output["magnitude"] for output in manifold_info["manifold_outputs"]], dim=-1
        )
        manifold_l1 = manifold_magnitudes.abs().mean()

        # Total loss
        loss = mse_loss + self.lambda_l1_independent * independent_l1 + self.lambda_manifold * manifold_l1

        # Compute L0 (number of active features)
        independent_l0 = (f_independent > 0).float().sum(dim=-1).mean()
        manifold_l0 = (manifold_magnitudes > 0).float().sum(dim=-1).mean()
        total_l0 = independent_l0 + manifold_l0

        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "independent_l1": independent_l1,
            "manifold_l1": manifold_l1,
            "l0": total_l0,
            "independent_l0": independent_l0,
            "manifold_l0": manifold_l0,
        }

        if return_outputs:
            outputs = {
                "x_reconstruct": x_reconstruct,
                "manifold_info": manifold_info,
            }
            return loss_dict, outputs

        return loss_dict

    def get_name(self) -> str:
        """Get a descriptive name for this SAE."""
        return (
            f"manifold_parametric_sae_"
            f"{self.d_sae_independent}ind_"
            f"{self.num_circular_manifolds}circ_"
            f"{self.num_spherical_manifolds}sph"
        )
