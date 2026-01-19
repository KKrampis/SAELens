import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, cast, overload

import torch
import wandb
from safetensors.torch import save_file
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from sae_lens import __version__
from sae_lens.config import SAETrainerConfig
from sae_lens.constants import (
    ACTIVATION_SCALER_CFG_FILENAME,
    SPARSITY_FILENAME,
    TRAINER_STATE_FILENAME,
)
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.optim import CoefficientScheduler, get_lr_scheduler
from sae_lens.training.types import DataProvider
from sae_lens.util import path_or_tmp_dir


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


def _update_sae_lens_training_version(sae: TrainingSAE[Any]) -> None:
    """
    Make sure we record the version of SAELens used for the training run
    """
    sae.cfg.sae_lens_training_version = str(__version__)


class SaveCheckpointFn(Protocol):
    def __call__(
        self,
        checkpoint_path: Path | None,
    ) -> None: ...


Evaluator = Callable[[T_TRAINING_SAE, DataProvider, ActivationScaler], dict[str, Any]]


@dataclass
class PerSAETrainerState(Generic[T_TRAINING_SAE]):
    """
    State maintained per-SAE during training.

    This encapsulates all per-SAE state including the model, optimizer,
    schedulers, activation scaler, and sparsity tracking metrics.
    """

    sae: T_TRAINING_SAE
    optimizer: Adam
    lr_scheduler: LRScheduler
    coefficient_schedulers: dict[str, CoefficientScheduler]
    grad_scaler: torch.amp.GradScaler
    activation_scaler: ActivationScaler

    # Sparsity tracking
    act_freq_scores: torch.Tensor
    n_forward_passes_since_fired: torch.Tensor
    n_frac_active_samples: int = 0

    # Training state
    started_fine_tuning: bool = False

    def feature_sparsity(self) -> torch.Tensor:
        if self.n_frac_active_samples == 0:
            return torch.zeros_like(self.act_freq_scores)
        return self.act_freq_scores / self.n_frac_active_samples

    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity())

    def dead_neurons(self, dead_feature_window: int) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > dead_feature_window).bool()

    def get_coefficients(self) -> dict[str, float]:
        return {
            name: scheduler.value
            for name, scheduler in self.coefficient_schedulers.items()
        }


class SAETrainer(Generic[T_TRAINING_SAE, T_TRAINING_SAE_CONFIG]):
    """
    Trainer for Sparse Autoencoder (SAE) models.

    Supports training either a single SAE or multiple SAEs simultaneously
    on the same training data. When multiple SAEs are provided as a dict,
    each SAE is trained on the same batch of activations each step, which
    is useful for hyperparameter sweeps.

    Args:
        cfg: Training configuration.
        sae: Either a single SAE or a dict mapping names to SAEs.
        data_provider: Iterator yielding training batches.
        evaluator: Optional evaluation function.
        save_checkpoint_fn: Optional custom checkpoint saving function.
    """

    data_provider: DataProvider
    evaluator: Evaluator[T_TRAINING_SAE] | None

    # Internal state
    _sae_states: dict[str, PerSAETrainerState[T_TRAINING_SAE]]
    _input_was_single_sae: bool
    _wandb_runs: dict[str, Any] | None  # wandb.Run objects for multi-SAE mode

    @overload
    def __init__(
        self,
        cfg: SAETrainerConfig,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        evaluator: Evaluator[T_TRAINING_SAE] | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        cfg: SAETrainerConfig,
        sae: dict[str, T_TRAINING_SAE],
        data_provider: DataProvider,
        evaluator: Evaluator[T_TRAINING_SAE] | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
    ) -> None: ...

    def __init__(
        self,
        cfg: SAETrainerConfig,
        sae: T_TRAINING_SAE | dict[str, T_TRAINING_SAE],
        data_provider: DataProvider,
        evaluator: Evaluator[T_TRAINING_SAE] | None = None,
        save_checkpoint_fn: SaveCheckpointFn | None = None,
    ) -> None:
        self.data_provider = data_provider
        self.evaluator = evaluator
        self.save_checkpoint_fn = save_checkpoint_fn
        self.cfg = cfg

        self.n_training_steps: int = 0
        self.n_training_samples: int = 0

        # Normalize input to dict
        if isinstance(sae, dict):
            if not sae:
                raise ValueError("sae dict cannot be empty")
            self._input_was_single_sae = False
            sae_dict = sae
        else:
            self._input_was_single_sae = True
            sae_dict = {"default": sae}

        # Initialize per-SAE state for each SAE
        self._sae_states = {}
        for key, sae_instance in sae_dict.items():
            self._sae_states[key] = self._create_sae_state(sae_instance)

        # Setup checkpoint thresholds
        self.checkpoint_thresholds: list[int] = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_samples,
                    math.ceil(
                        cfg.total_training_samples / (self.cfg.n_checkpoints + 1)
                    ),
                )
            )[1:]

        # Setup autocast
        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type=self.cfg.device,
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()

        # Initialize wandb runs for multi-SAE mode
        self._wandb_runs = None
        if cfg.logger.log_to_wandb and not self._input_was_single_sae:
            self._init_wandb_runs()

    def _create_sae_state(
        self, sae: T_TRAINING_SAE
    ) -> PerSAETrainerState[T_TRAINING_SAE]:
        """Create per-SAE trainer state."""
        _update_sae_lens_training_version(sae)

        optimizer = Adam(
            sae.parameters(),
            lr=self.cfg.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        assert self.cfg.lr_end is not None  # set in config post-init
        lr_scheduler = get_lr_scheduler(
            self.cfg.lr_scheduler_name,
            lr=self.cfg.lr,
            optimizer=optimizer,
            warm_up_steps=self.cfg.lr_warm_up_steps,
            decay_steps=self.cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=self.cfg.lr_end,
            num_cycles=self.cfg.n_restart_cycles,
        )

        coefficient_schedulers: dict[str, CoefficientScheduler] = {}
        for name, coeff_cfg in sae.get_coefficients().items():
            if not isinstance(coeff_cfg, TrainCoefficientConfig):
                coeff_cfg = TrainCoefficientConfig(value=coeff_cfg, warm_up_steps=0)
            coefficient_schedulers[name] = CoefficientScheduler(
                warm_up_steps=coeff_cfg.warm_up_steps,
                final_value=coeff_cfg.value,
            )

        grad_scaler = torch.amp.GradScaler(
            device=self.cfg.device, enabled=self.cfg.autocast
        )

        return PerSAETrainerState(
            sae=sae,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            coefficient_schedulers=coefficient_schedulers,
            grad_scaler=grad_scaler,
            activation_scaler=ActivationScaler(),
            act_freq_scores=torch.zeros(sae.cfg.d_sae, device=self.cfg.device),
            n_forward_passes_since_fired=torch.zeros(
                sae.cfg.d_sae, device=self.cfg.device
            ),
            n_frac_active_samples=0,
            started_fine_tuning=False,
        )

    def _init_wandb_runs(self) -> None:
        """Initialize separate wandb runs for each SAE in multi-SAE mode."""
        self._wandb_runs = {}
        group_id = self.cfg.logger.wandb_id or cast(Any, wandb).util.generate_id()

        for sae_key in self._sae_states:
            run_name = self.cfg.logger.run_name or "sae_training"
            run = wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                name=f"{sae_key}/{run_name}",
                group=group_id,
                config={"sae_key": sae_key},
                reinit=True,
            )
            self._wandb_runs[sae_key] = run

    # SAE accessors
    @property
    def sae_keys(self) -> list[str]:
        """Get list of SAE keys."""
        return list(self._sae_states.keys())

    def get_sae(self, key: str) -> T_TRAINING_SAE:
        """Get SAE by key."""
        return self._sae_states[key].sae

    def get_state(self, key: str) -> PerSAETrainerState[T_TRAINING_SAE]:
        """Get full per-SAE state by key."""
        return self._sae_states[key]

    def fit(self) -> T_TRAINING_SAE | dict[str, T_TRAINING_SAE]:
        """
        Train the SAE(s).

        Returns the trained SAE(s) in the same format as provided to the constructor:
        a single SAE if a single SAE was provided, or a dict if a dict was provided.
        """
        # Move all SAEs to device
        for state in self._sae_states.values():
            state.sae.to(self.cfg.device)

        pbar = tqdm(
            total=self.cfg.total_training_samples,
            desc="Training SAE" if self._input_was_single_sae else "Training SAEs",
        )

        # Handle activation normalization per SAE
        for state in self._sae_states.values():
            if state.sae.cfg.normalize_activations == "expected_average_only_in":
                state.activation_scaler.estimate_scaling_factor(
                    d_in=state.sae.cfg.d_in,
                    data_provider=self.data_provider,
                    n_batches_for_norm_estimate=int(1e3),
                )

        # Train loop
        while self.n_training_samples < self.cfg.total_training_samples:
            # Get batch ONCE - shared across all SAEs
            first_sae = next(iter(self._sae_states.values())).sae
            batch = next(self.data_provider).to(first_sae.device)
            self.n_training_samples += batch.shape[0]

            # Train each SAE on the same batch (each SAE applies its own scaling)
            step_outputs: dict[str, TrainStepOutput] = {}
            for sae_key, state in self._sae_states.items():
                scaled_batch = state.activation_scaler(batch)
                step_output = self._train_step_for_sae(
                    sae_key=sae_key,
                    state=state,
                    sae_in=scaled_batch,
                )
                step_outputs[sae_key] = step_output

            # Logging
            if self.cfg.logger.log_to_wandb:
                self._log_train_steps(step_outputs)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_outputs, pbar)

        # Fold activation norm scaling factor into each SAE
        for state in self._sae_states.values():
            if state.activation_scaler.scaling_factor is not None:
                state.sae.fold_activation_norm_scaling_factor(
                    state.activation_scaler.scaling_factor
                )
                state.activation_scaler.scaling_factor = None

        if self.cfg.save_final_checkpoint:
            self.save_checkpoint(checkpoint_name=f"final_{self.n_training_samples}")

        # Finish wandb runs for multi-SAE mode
        if self._wandb_runs is not None:
            for run in self._wandb_runs.values():
                run.finish()

        pbar.close()
        return self._get_return_value()

    def _get_return_value(self) -> T_TRAINING_SAE | dict[str, T_TRAINING_SAE]:
        """Return SAE(s) in same format as input."""
        if self._input_was_single_sae:
            return self._sae_states["default"].sae
        return {key: state.sae for key, state in self._sae_states.items()}

    def _train_step_for_sae(
        self,
        sae_key: str,
        state: PerSAETrainerState[T_TRAINING_SAE],
        sae_in: torch.Tensor,
    ) -> TrainStepOutput:
        """Execute a single training step for one SAE."""
        state.sae.train()

        # Log and reset sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.logger.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict_for_sae(
                    sae_key, state
                )
                self._log_to_wandb(sae_key, sparsity_log_dict)
            self._reset_running_sparsity_stats_for_sae(state)

        # Forward pass with autocast
        with self.autocast_if_enabled:
            train_step_output = state.sae.training_forward_pass(
                step_input=TrainStepInput(
                    sae_in=sae_in,
                    dead_neuron_mask=state.dead_neurons(self.cfg.dead_feature_window),
                    coefficients=state.get_coefficients(),
                    n_training_steps=self.n_training_steps,
                ),
            )

            # Update sparsity tracking
            with torch.no_grad():
                firing_feats = train_step_output.feature_acts.bool().float()
                did_fire = firing_feats.sum(-2).bool()
                if did_fire.is_sparse:
                    did_fire = did_fire.to_dense()
                state.n_forward_passes_since_fired += 1
                state.n_forward_passes_since_fired[did_fire] = 0
                state.act_freq_scores += firing_feats.sum(0)
                state.n_frac_active_samples += self.cfg.train_batch_size_samples

        # Backward pass
        state.grad_scaler.scale(train_step_output.loss).backward()
        state.grad_scaler.unscale_(state.optimizer)
        torch.nn.utils.clip_grad_norm_(state.sae.parameters(), 1.0)
        state.grad_scaler.step(state.optimizer)
        state.grad_scaler.update()

        state.optimizer.zero_grad()
        state.lr_scheduler.step()
        for scheduler in state.coefficient_schedulers.values():
            scheduler.step()

        return train_step_output

    def _log_to_wandb(self, sae_key: str, log_dict: dict[str, Any]) -> None:
        """Log to wandb, handling both single and multi-SAE modes."""
        if self._input_was_single_sae:
            wandb.log(log_dict, step=self.n_training_steps)
        else:
            assert self._wandb_runs is not None
            # Use the specific run for this SAE
            run = self._wandb_runs[sae_key]
            run.log(log_dict, step=self.n_training_steps)

    @torch.no_grad()
    def _log_train_steps(self, step_outputs: dict[str, TrainStepOutput]) -> None:
        """Log training metrics for all SAEs."""
        if (self.n_training_steps + 1) % self.cfg.logger.wandb_log_frequency != 0:
            return

        for sae_key, step_output in step_outputs.items():
            state = self._sae_states[sae_key]
            log_dict = self._build_train_step_log_dict_for_sae(
                state=state,
                output=step_output,
                n_training_samples=self.n_training_samples,
            )
            self._log_to_wandb(sae_key, log_dict)

    @torch.no_grad()
    def _run_and_log_evals(self) -> None:
        """Run and log evaluations for all SAEs."""
        if (self.n_training_steps + 1) % (
            self.cfg.logger.wandb_log_frequency
            * self.cfg.logger.eval_every_n_wandb_logs
        ) != 0:
            return

        for sae_key, state in self._sae_states.items():
            state.sae.eval()
            eval_metrics = (
                self.evaluator(state.sae, self.data_provider, state.activation_scaler)
                if self.evaluator is not None
                else {}
            )
            for key, value in state.sae.log_histograms().items():
                eval_metrics[key] = wandb.Histogram(value)  # type: ignore

            self._log_to_wandb(sae_key, eval_metrics)
            state.sae.train()

    @torch.no_grad()
    def get_coefficients(self) -> dict[str, float]:
        """Get coefficients (single-SAE mode only, for backward compatibility)."""
        if not self._input_was_single_sae:
            raise ValueError("Cannot access .get_coefficients() in multi-SAE mode.")
        return self._sae_states["default"].get_coefficients()

    @torch.no_grad()
    def _build_train_step_log_dict_for_sae(
        self,
        state: PerSAETrainerState[T_TRAINING_SAE],
        output: TrainStepOutput,
        n_training_samples: int,
    ) -> dict[str, Any]:
        """Build log dict for a single SAE's training step."""
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        l0 = feature_acts.bool().float().sum(-1).to_dense().mean()
        current_learning_rate = state.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance_legacy = 1 - per_token_l2_loss / total_variance
        explained_variance = 1 - per_token_l2_loss.mean() / total_variance.mean()

        log_dict = {
            "losses/overall_loss": loss,
            "metrics/explained_variance_legacy": explained_variance_legacy.mean().item(),
            "metrics/explained_variance_legacy_std": explained_variance_legacy.std().item(),
            "metrics/explained_variance": explained_variance.item(),
            "metrics/l0": l0.item(),
            "sparsity/mean_passes_since_fired": state.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": state.dead_neurons(self.cfg.dead_feature_window)
            .sum()
            .item(),
            "details/current_learning_rate": current_learning_rate,
            "details/n_training_samples": n_training_samples,
            **{
                f"details/{name}_coefficient": scheduler.value
                for name, scheduler in state.coefficient_schedulers.items()
            },
        }
        for loss_name, loss_value in output.losses.items():
            log_dict[f"losses/{loss_name}"] = _unwrap_item(loss_value)

        for metric_name, metric_value in output.metrics.items():
            log_dict[f"metrics/{metric_name}"] = _unwrap_item(metric_value)

        return log_dict

    @torch.no_grad()
    def _build_sparsity_log_dict_for_sae(
        self,
        sae_key: str,  # noqa: ARG002
        state: PerSAETrainerState[T_TRAINING_SAE],
    ) -> dict[str, Any]:
        """Build sparsity log dict for a single SAE."""
        log_feature_sparsity = state.log_feature_sparsity()
        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())  # type: ignore
        feature_sparsity = state.feature_sparsity()
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats_for_sae(
        self, state: PerSAETrainerState[T_TRAINING_SAE]
    ) -> None:
        """Reset sparsity stats for a single SAE."""
        state.act_freq_scores = torch.zeros(
            state.sae.cfg.d_sae,
            device=self.cfg.device,
        )
        state.n_frac_active_samples = 0

    # Legacy methods for single-SAE mode backward compatibility
    def _train_step(
        self,
        sae: T_TRAINING_SAE,  # noqa: ARG002
        sae_in: torch.Tensor,
    ) -> TrainStepOutput:
        """Legacy train step method for backward compatibility."""
        if not self._input_was_single_sae:
            raise ValueError("_train_step is only available in single-SAE mode.")
        return self._train_step_for_sae("default", self._sae_states["default"], sae_in)

    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput) -> None:
        """Legacy log method for backward compatibility."""
        if not self._input_was_single_sae:
            raise ValueError("_log_train_step is only available in single-SAE mode.")
        self._log_train_steps({"default": step_output})

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_samples: int,
    ) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        if not self._input_was_single_sae:
            raise ValueError(
                "_build_train_step_log_dict is only available in single-SAE mode."
            )
        return self._build_train_step_log_dict_for_sae(
            state=self._sae_states["default"],
            output=output,
            n_training_samples=n_training_samples,
        )

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        """Legacy method for backward compatibility."""
        if not self._input_was_single_sae:
            raise ValueError(
                "_build_sparsity_log_dict is only available in single-SAE mode."
            )
        return self._build_sparsity_log_dict_for_sae(
            "default", self._sae_states["default"]
        )

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        """Legacy method for backward compatibility."""
        if not self._input_was_single_sae:
            raise ValueError(
                "_reset_running_sparsity_stats is only available in single-SAE mode."
            )
        self._reset_running_sparsity_stats_for_sae(self._sae_states["default"])

    def save_checkpoint(
        self,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        """
        Save checkpoint(s) for trained SAE(s).

        For single SAE mode, saves in flat structure (backward compatible).
        For multi-SAE mode, creates subdirectories for each SAE.
        """
        checkpoint_path = None
        if self.cfg.checkpoint_path is not None or self.cfg.logger.log_to_wandb:
            with path_or_tmp_dir(self.cfg.checkpoint_path) as base_checkpoint_path:
                checkpoint_path = base_checkpoint_path / checkpoint_name
                checkpoint_path.mkdir(exist_ok=True, parents=True)

                if self._input_was_single_sae:
                    self._save_single_sae_checkpoint(checkpoint_path, wandb_aliases)
                else:
                    self._save_multi_sae_checkpoint(checkpoint_path, wandb_aliases)

        if self.save_checkpoint_fn is not None:
            self.save_checkpoint_fn(checkpoint_path=checkpoint_path)

    def _save_single_sae_checkpoint(
        self,
        checkpoint_path: Path,
        wandb_aliases: list[str] | None,
    ) -> None:
        """Save checkpoint in original flat format for backward compatibility."""
        state = self._sae_states["default"]

        weights_path, cfg_path = state.sae.save_model(str(checkpoint_path))
        sparsity_path = checkpoint_path / SPARSITY_FILENAME
        save_file({"sparsity": state.log_feature_sparsity()}, sparsity_path)

        # Include global state in trainer state for backward compatibility
        self._save_sae_trainer_state(state, checkpoint_path, include_global_state=True)
        self._save_shared_state(checkpoint_path)

        if self.cfg.logger.log_to_wandb:
            self.cfg.logger.log(
                self,
                weights_path,
                cfg_path,
                sparsity_path=sparsity_path,
                wandb_aliases=wandb_aliases,
            )

    def _save_multi_sae_checkpoint(
        self,
        checkpoint_path: Path,
        wandb_aliases: list[str] | None,
    ) -> None:
        """Save checkpoint with per-SAE subdirectories."""
        # Save each SAE in its own subdirectory
        for sae_key, state in self._sae_states.items():
            sae_dir = checkpoint_path / sae_key
            sae_dir.mkdir(exist_ok=True, parents=True)

            weights_path, cfg_path = state.sae.save_model(str(sae_dir))
            sparsity_path = sae_dir / SPARSITY_FILENAME
            save_file({"sparsity": state.log_feature_sparsity()}, sparsity_path)

            self._save_sae_trainer_state(state, sae_dir)

            # Log to wandb for this specific SAE's run
            if self.cfg.logger.log_to_wandb and self._wandb_runs is not None:
                run = self._wandb_runs[sae_key]
                sae_name = state.sae.get_name().replace("/", "__")
                model_artifact = wandb.Artifact(
                    sae_name,
                    type="model",
                    metadata=dict(self.cfg.__dict__),
                )
                model_artifact.add_file(str(weights_path))
                model_artifact.add_file(str(cfg_path))
                run.log_artifact(model_artifact, aliases=wandb_aliases)

                sparsity_artifact = wandb.Artifact(
                    f"{sae_name}_log_feature_sparsity",
                    type="log_feature_sparsity",
                    metadata=dict(self.cfg.__dict__),
                )
                sparsity_artifact.add_file(str(sparsity_path))
                run.log_artifact(sparsity_artifact)

        # Save shared state
        shared_dir = checkpoint_path / "shared"
        shared_dir.mkdir(exist_ok=True, parents=True)
        self._save_shared_state(shared_dir)

    def _save_sae_trainer_state(
        self,
        state: PerSAETrainerState[T_TRAINING_SAE],
        path: Path,
        include_global_state: bool = False,
    ) -> None:
        """Save per-SAE trainer state."""
        scheduler_state_dicts = {
            name: scheduler.state_dict()
            for name, scheduler in state.coefficient_schedulers.items()
        }
        state_dict: dict[str, Any] = {
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": state.lr_scheduler.state_dict(),
            "act_freq_scores": state.act_freq_scores,
            "n_forward_passes_since_fired": state.n_forward_passes_since_fired,
            "n_frac_active_samples": state.n_frac_active_samples,
            "started_fine_tuning": state.started_fine_tuning,
            "coefficient_schedulers": scheduler_state_dicts,
        }
        # Include global state for single-SAE mode backward compatibility
        if include_global_state:
            state_dict["n_training_samples"] = self.n_training_samples
            state_dict["n_training_steps"] = self.n_training_steps
        torch.save(state_dict, str(path / TRAINER_STATE_FILENAME))

        # Save per-SAE activation scaler
        state.activation_scaler.save(str(path / ACTIVATION_SCALER_CFG_FILENAME))

    def _save_shared_state(self, path: Path) -> None:
        """Save state shared across all SAEs (global training progress)."""
        path.mkdir(exist_ok=True, parents=True)

        # For multi-SAE mode, save global training progress
        if not self._input_was_single_sae:
            torch.save(
                {
                    "n_training_steps": self.n_training_steps,
                    "n_training_samples": self.n_training_samples,
                },
                str(path / "global_trainer_state.pt"),
            )

    def save_trainer_state(self, checkpoint_path: Path) -> None:
        """Legacy method for backward compatibility."""
        if not self._input_was_single_sae:
            raise ValueError(
                "save_trainer_state is only available in single-SAE mode. "
                "Use save_checkpoint for multi-SAE mode."
            )
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        state = self._sae_states["default"]
        # Include global state for backward compatibility
        self._save_sae_trainer_state(state, checkpoint_path, include_global_state=True)
        self._save_shared_state(checkpoint_path)

    def load_trainer_state(self, checkpoint_path: Path | str) -> None:
        """
        Load trainer state from a checkpoint.

        Only single-SAE checkpoint loading is currently supported.
        """
        checkpoint_path = Path(checkpoint_path)

        # Check if this is a multi-SAE checkpoint
        shared_dir = checkpoint_path / "shared"

        if shared_dir.exists():
            raise NotImplementedError(
                "Resuming from multi-SAE checkpoints is not yet supported. "
                "Multi-SAE checkpoints can be saved but not loaded."
            )

        if not self._input_was_single_sae:
            raise ValueError("Cannot load single-SAE checkpoint into multi-SAE trainer")

        self._load_single_sae_state(checkpoint_path)

    def _load_single_sae_state(self, checkpoint_path: Path) -> None:
        """Load single SAE checkpoint (backward compatible)."""
        state = self._sae_states["default"]
        state.activation_scaler.load(checkpoint_path / ACTIVATION_SCALER_CFG_FILENAME)

        state_dict = torch.load(checkpoint_path / TRAINER_STATE_FILENAME)
        state.optimizer.load_state_dict(state_dict["optimizer"])
        state.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.n_training_samples = state_dict["n_training_samples"]
        self.n_training_steps = state_dict["n_training_steps"]
        state.act_freq_scores = state_dict["act_freq_scores"]
        state.n_forward_passes_since_fired = state_dict["n_forward_passes_since_fired"]
        state.n_frac_active_samples = state_dict["n_frac_active_samples"]
        state.started_fine_tuning = state_dict["started_fine_tuning"]

        for name, scheduler_state in state_dict["coefficient_schedulers"].items():
            state.coefficient_schedulers[name].load_state_dict(scheduler_state)

    @torch.no_grad()
    def _checkpoint_if_needed(self) -> None:
        if (
            self.checkpoint_thresholds
            and self.n_training_samples > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(checkpoint_name=str(self.n_training_samples))
            self.checkpoint_thresholds.pop(0)

    @torch.no_grad()
    def _update_pbar(
        self,
        step_outputs: dict[str, TrainStepOutput],
        pbar: tqdm,  # type: ignore
        update_interval: int = 100,
    ) -> None:
        if self.n_training_steps % update_interval == 0:
            # Use first SAE's output for progress bar display
            first_output = next(iter(step_outputs.values()))
            loss_strs = " | ".join(
                f"{loss_name}: {_unwrap_item(loss_value):.5f}"
                for loss_name, loss_value in first_output.losses.items()
            )
            desc = f"{self.n_training_steps}| {loss_strs}"
            if not self._input_was_single_sae:
                desc = f"[{len(self._sae_states)} SAEs] " + desc
            pbar.set_description(desc)
            pbar.update(update_interval * self.cfg.train_batch_size_samples)


def _unwrap_item(item: float | torch.Tensor) -> float:
    return item.item() if isinstance(item, torch.Tensor) else item
