"""
Grouped Latent Sparse Autoencoder (GL-SAE).

This SAE architecture organizes latents into predefined groups, where each group
can specialize to represent a potential manifold structure. Group-level gating
provides sparsity at the manifold level, while within-group features represent
points on the manifold.

Key features:
- Latents organized into groups
- Group-level gating (which groups are active?)
- Within-group encoding for manifold points
- Can efficiently represent circular/spherical manifolds

Reference:
    Extends the gated SAE architecture (Rajamanoharan et al., 2024) with
    explicit group structure for manifold representation.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from sae_lens.saes.sae import SAE


class GroupedLatentSAE(SAE):
    """
    Sparse Autoencoder with grouped latent structure for manifold representation.

    Attributes:
        num_groups: Number of latent groups
        latents_per_group: Number of latents in each group
        W_enc_shared: Shared encoder projection
        group_encoders: List of group-specific encoders
        group_gate: Gating network for group selection
        W_dec: Decoder matrix (standard)
        b_enc: Encoder bias
        b_dec: Decoder bias
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        num_groups: int,
        latents_per_group: int | None = None,
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
        lambda_group: float = 1e-3,
        lambda_feature: float = 1e-3,
    ):
        """
        Initialize Grouped Latent SAE.

        Args:
            d_in: Input dimension
            d_sae: Total number of latents across all groups
            num_groups: Number of groups
            latents_per_group: Latents per group (if None, computed as d_sae // num_groups)
            lambda_group: Sparsity penalty for group activation
            lambda_feature: Sparsity penalty for within-group features
            Other args: Standard SAE parameters
        """
        if latents_per_group is None:
            latents_per_group = d_sae // num_groups

        if d_sae != num_groups * latents_per_group:
            raise ValueError(
                f"d_sae ({d_sae}) must equal num_groups ({num_groups}) * "
                f"latents_per_group ({latents_per_group})"
            )

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

        self.num_groups = num_groups
        self.latents_per_group = latents_per_group
        self.lambda_group = lambda_group
        self.lambda_feature = lambda_feature

        # Shared encoder
        self.W_enc_shared = nn.Parameter(
            torch.randn(d_in, d_in, device=device, dtype=dtype) * 0.1
        )

        # Group-specific encoders
        self.group_encoders = nn.ModuleList(
            [
                nn.Linear(d_in, latents_per_group, device=device, dtype=dtype, bias=False)
                for _ in range(num_groups)
            ]
        )

        # Group gating network
        self.group_gate = nn.Linear(d_in, num_groups, device=device, dtype=dtype)

        # Initialize decoder (will use standard SAE decoder)
        # Decoder columns correspond to groups: [group0_latents, group1_latents, ...]

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode input to sparse latent representation.

        Args:
            x: Input activations of shape (..., d_in)

        Returns:
            Sparse latent activations of shape (..., d_sae)
        """
        # Remove decoder bias if configured
        if self.apply_b_dec_to_input:
            x = x - self.b_dec

        # Shared representation
        h = F.relu(x @ self.W_enc_shared + self.b_enc[:self.d_in])

        # Group gating: which groups should be active?
        group_logits = self.group_gate(x)
        group_probs = torch.sigmoid(group_logits)

        # Encode within each group
        group_features = []
        for i, encoder in enumerate(self.group_encoders):
            # Encode this group
            f_group = F.relu(encoder(h))

            # Gate by group probability
            f_group = f_group * group_probs[..., i : i + 1]

            group_features.append(f_group)

        # Concatenate all groups
        f = torch.cat(group_features, dim=-1)

        return f

    def forward(
        self, x: Float[torch.Tensor, "... d_in"], return_group_info: bool = False
    ) -> Float[torch.Tensor, "... d_in"] | tuple[Float[torch.Tensor, "... d_in"], dict]:
        """
        Forward pass through the SAE.

        Args:
            x: Input activations
            return_group_info: If True, return dict with group activation info

        Returns:
            Reconstructed activations (and optionally group info dict)
        """
        # Encode
        feature_acts = self.encode(x)

        # Decode
        x_reconstruct = self.decode(feature_acts)

        if return_group_info:
            # Reshape features to (batch, num_groups, latents_per_group)
            batch_shape = feature_acts.shape[:-1]
            f_grouped = feature_acts.reshape(*batch_shape, self.num_groups, self.latents_per_group)

            # Compute group-level statistics
            group_l0 = (f_grouped.abs().sum(dim=-1) > 0).float()
            group_magnitude = f_grouped.abs().sum(dim=-1)

            info = {
                "group_activations": group_l0,
                "group_magnitudes": group_magnitude,
                "features_grouped": f_grouped,
            }
            return x_reconstruct, info

        return x_reconstruct

    def get_loss_dict(
        self,
        x: Float[torch.Tensor, "batch d_in"],
        return_outputs: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """
        Compute loss with group-level and feature-level sparsity penalties.

        Args:
            x: Input activations
            return_outputs: If True, return forward pass outputs

        Returns:
            Loss dictionary (and optionally outputs dict)
        """
        # Forward pass
        feature_acts = self.encode(x)
        x_reconstruct, group_info = self.forward(x, return_group_info=True)

        # Reconstruction loss (MSE)
        per_item_mse_loss = (x_reconstruct - x).pow(2).sum(dim=-1)
        mse_loss = per_item_mse_loss.mean()

        # Group sparsity: L0 on groups (number of active groups)
        group_acts = group_info["group_activations"]
        group_l0 = group_acts.sum(dim=-1).mean()

        # Feature sparsity: L1 within groups
        feature_l1 = feature_acts.abs().mean()

        # Total loss
        loss = mse_loss + self.lambda_group * group_l0 + self.lambda_feature * feature_l1

        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "group_l0": group_l0,
            "feature_l1": feature_l1,
            "l0": (feature_acts > 0).float().sum(dim=-1).mean(),
        }

        if return_outputs:
            outputs = {
                "feature_acts": feature_acts,
                "x_reconstruct": x_reconstruct,
                "group_info": group_info,
            }
            return loss_dict, outputs

        return loss_dict

    def get_name(self) -> str:
        """Get a descriptive name for this SAE."""
        return (
            f"grouped_latent_sae_"
            f"{self.num_groups}groups_"
            f"{self.latents_per_group}per_"
            f"{self.d_sae}total"
        )
