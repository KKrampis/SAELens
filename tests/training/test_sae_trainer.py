import json
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.llm_sae_training_runner import (
    LanguageModelSAETrainingRunner,
    LLMSaeEvaluator,
)
from sae_lens.saes.sae import TrainingSAE
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.config import LoggingConfig, SAETrainerConfig
from sae_lens.training.sae_trainer import (
    PerSAETrainerState,
    SAETrainer,
    TrainStepOutput,
    _log_feature_sparsity,
    _update_sae_lens_training_version,
)
from tests.helpers import (
    TINYSTORIES_MODEL,
    assert_close,
    build_runner_cfg,
    load_model_cached,
)


@pytest.fixture
def cfg():
    return build_runner_cfg(d_in=64, d_sae=128)


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(
    model: HookedTransformer,
    cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig],
):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig]):
    return StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


@pytest.fixture
def trainer(  # type: ignore
    cfg: LanguageModelSAERunnerConfig[StandardTrainingSAEConfig],
    model: HookedTransformer,
    training_sae: StandardTrainingSAE,
    activation_store: ActivationsStore,
):
    evaluator = LLMSaeEvaluator(
        model,
        activation_store,
        eval_batch_size_prompts=cfg.eval_batch_size_prompts,
        n_eval_batches=cfg.n_eval_batches,
        model_kwargs=cfg.model_kwargs,
    )
    return SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=training_sae,
        data_provider=activation_store,
        evaluator=evaluator,
    )


def modify_sae_output(
    sae: StandardTrainingSAE, modifier: Callable[[torch.Tensor], Any]
):
    """
    Helper to modify the output of the SAE forward pass for use in patching, for use in patch side_effect.
    We need real grads during training, so we can't just mock the whole forward pass directly.
    """

    def modified_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = TrainingSAE.forward(sae, *args, **kwargs)
        return modifier(output)

    return modified_forward


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts(
    trainer: SAETrainer[StandardTrainingSAE, StandardTrainingSAEConfig],
) -> None:
    layer_acts = next(trainer.data_provider)

    # intentionally train on the same activations 5 times to ensure loss decreases
    state = trainer.get_state("default")
    train_outputs = [
        trainer._train_step(
            sae=state.sae,
            sae_in=layer_acts,
        )
        for _ in range(5)
    ]

    # ensure loss decreases with each training step
    for output, next_output in zip(train_outputs[:-1], train_outputs[1:]):
        assert output.loss > next_output.loss
    assert (
        state.n_frac_active_samples == 20
    )  # should increment each step by batch_size (5*4)


def test_train_step__output_looks_reasonable(trainer: SAETrainer[Any, Any]) -> None:
    layer_acts = next(trainer.data_provider)
    state = trainer.get_state("default")

    output = trainer._train_step(
        sae=state.sae,
        sae_in=layer_acts,
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert_close(output.sae_in, layer_acts)
    assert output.sae_out.shape == output.sae_in.shape
    assert output.feature_acts.shape == (4, 128)  # batch_size, d_sae
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert output.losses.get("ghost_grad_loss", 0) == 0
    assert state.n_frac_active_samples == 4
    assert state.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert_close(
        state.act_freq_scores,
        (output.feature_acts.abs() > 0).float().sum(0),
    )


def test_train_step__sparsity_updates_based_on_feature_act_sparsity(
    trainer: SAETrainer[StandardTrainingSAE, StandardTrainingSAEConfig],
) -> None:
    trainer._reset_running_sparsity_stats()
    layer_acts = next(trainer.data_provider)
    state = trainer.get_state("default")

    train_output = trainer._train_step(
        sae=state.sae,
        sae_in=layer_acts,
    )
    feature_acts = train_output.feature_acts

    # should increase by batch_size
    assert state.n_frac_active_samples == 4
    # add freq scores for all non-zero feature acts
    assert_close(
        state.act_freq_scores,
        (feature_acts > 0).float().sum(dim=0),
    )

    # check that features that just fired have n_forward_passes_since_fired = 0
    assert (
        state.n_forward_passes_since_fired[((feature_acts > 0).float()[-1] == 1)].max()
        == 0
    )
    assert train_output.feature_acts is feature_acts


def test_log_feature_sparsity__handles_zeroes_by_default_fp32() -> None:
    fp32_zeroes = torch.tensor([0], dtype=torch.float32)
    assert _log_feature_sparsity(fp32_zeroes).item() != float("-inf")


# TODO: currently doesn't work for fp16, we should address this
@pytest.mark.skip(reason="Currently doesn't work for fp16")
def test_log_feature_sparsity__handles_zeroes_by_default_fp16() -> None:
    fp16_zeroes = torch.tensor([0], dtype=torch.float16)
    assert _log_feature_sparsity(fp16_zeroes).item() != float("-inf")


@pytest.mark.parametrize("sparse_feature_acts", [True, False])
def test_build_train_step_log_dict(
    trainer: SAETrainer[StandardTrainingSAE, StandardTrainingSAEConfig],
    sparse_feature_acts: bool,
) -> None:
    train_output = TrainStepOutput(
        sae_in=torch.tensor([[-1, 0], [0, 2], [1, 1]]).float(),
        sae_out=torch.tensor([[0, 0], [0, 2], [0.5, 1]]).float(),
        feature_acts=torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]]).float(),
        hidden_pre=torch.tensor([[-1, 0, 0, 1], [1, -1, 0, 1], [1, -1, 1, 1]]).float(),
        loss=torch.tensor(0.5),
        losses={
            "mse_loss": torch.tensor(0.25),
            "l1_loss": torch.tensor(0.1),
        },
        metrics={
            "topk_threshold": torch.tensor(0.5),
        },
    )
    if sparse_feature_acts:
        train_output.feature_acts = train_output.feature_acts.to_sparse_coo()

    # we're relying on the trainer only for some of the metrics here
    # we should more / less try to break this and push
    # everything through the train step output if we can.
    state = trainer.get_state("default")
    log_dict = trainer._build_train_step_log_dict(
        output=train_output, n_training_samples=123
    )
    expected = {
        "losses/mse_loss": 0.25,
        "losses/l1_loss": train_output.losses["l1_loss"].item(),
        "losses/overall_loss": 0.5,
        "metrics/explained_variance": 0.6875,
        "metrics/explained_variance_legacy": 0.75,
        "metrics/explained_variance_legacy_std": 0.25,
        "metrics/l0": 2.0,
        "sparsity/mean_passes_since_fired": state.n_forward_passes_since_fired.mean().item(),
        "sparsity/dead_features": state.dead_neurons(trainer.cfg.dead_feature_window)
        .sum()
        .item(),
        "details/current_learning_rate": 2e-4,
        "details/l1_coefficient": state.sae.cfg.l1_coefficient,
        "details/n_training_samples": 123,
        "metrics/topk_threshold": 0.5,
    }
    assert log_dict.keys() == expected.keys()
    assert log_dict == pytest.approx(expected)


def test_train_sae_group_on_language_model__runs(
    ts_model: HookedTransformer,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = build_runner_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=20,
        context_size=8,
    )
    # just a tiny datast which will run quickly
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    sae = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=activation_store,
    ).fit()

    assert isinstance(sae, TrainingSAE)


def test_update_sae_lens_training_version_sets_the_current_version():
    cfg = build_runner_cfg(sae_lens_training_version="0.1.0")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    _update_sae_lens_training_version(sae)
    assert sae.cfg.sae_lens_training_version == str(__version__)


def test_checkpoints_save_runner_cfg(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cfg = build_runner_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=100,  # Increased to ensure we hit checkpoints
        context_size=8,
        n_checkpoints=2,  # Explicitly request 2 checkpoints during training
        save_final_checkpoint=True,  # Enable final checkpoint
    )

    # Create a small dataset
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    runner = LanguageModelSAETrainingRunner(
        cfg, override_model=ts_model, override_sae=sae
    )
    runner.activations_store = activation_store

    trainer = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=activation_store,
        save_checkpoint_fn=runner.save_checkpoint,
    )

    # Train the model - this should create checkpoints
    trainer.fit()
    checkpoint_cfg_paths = list(checkpoint_dir.glob("**/cfg.json"))
    checkpoint_runner_cfg_paths = list(checkpoint_dir.glob("**/runner_cfg.json"))
    # We should have exactly 3 checkpoints, including the final checkpoint:
    assert (
        len(checkpoint_cfg_paths) == 3
    ), f"Expected 3 sae cfg but got {len(checkpoint_cfg_paths)}"
    assert (
        len(checkpoint_runner_cfg_paths) == 3
    ), f"Expected 3 runner cfg but got {len(checkpoint_runner_cfg_paths)}"

    for checkpoint_cfg_path in checkpoint_cfg_paths:
        with open(checkpoint_cfg_path) as f:
            checkpoint_cfg = json.load(f)
        assert checkpoint_cfg == cfg.sae.to_dict()

    for checkpoint_runner_cfg_path in checkpoint_runner_cfg_paths:
        with open(checkpoint_runner_cfg_path) as f:
            runner_cfg = json.load(f)

        expected_cfg = cfg.to_dict()
        # seqpos_slice is a tuple when saved, but list when loaded. Just ignore it.
        del runner_cfg["seqpos_slice"]
        del expected_cfg["seqpos_slice"]
        assert runner_cfg == expected_cfg


def test_skips_saving_checkpoint_when_checkpoint_path_is_none(
    ts_model: HookedTransformer,
):
    cfg = build_runner_cfg(
        checkpoint_path=None,
        training_tokens=100,  # Increased to ensure we hit checkpoints
        context_size=8,
        n_checkpoints=2,  # Explicitly request 2 checkpoints during training
        save_final_checkpoint=True,  # Enable final checkpoint
    )
    trainer_cfg = cfg.to_sae_trainer_config()

    assert trainer_cfg.checkpoint_path is None

    # Create a small dataset
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    runner = LanguageModelSAETrainingRunner(
        cfg, override_model=ts_model, override_sae=sae
    )
    runner.activations_store = activation_store

    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=activation_store,
        save_checkpoint_fn=runner.save_checkpoint,
    )

    # Train the model - this should create checkpoints
    trainer.fit()


def test_estimated_norm_scaling_factor_persistence(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    """Test that estimated_norm_scaling_factor is correctly persisted in intermediate checkpoints
    but not in the final checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cfg = build_runner_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=100,  # Increased to ensure we hit checkpoints
        context_size=8,
        normalize_activations="expected_average_only_in",
        n_checkpoints=2,  # Explicitly request 2 checkpoints during training
        save_final_checkpoint=True,  # Enable final checkpoint
    )

    # Create a small dataset
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    runner = LanguageModelSAETrainingRunner(
        cfg, override_model=ts_model, override_sae=sae
    )
    runner.activations_store = activation_store

    trainer = SAETrainer(
        sae=sae,
        data_provider=activation_store,
        cfg=cfg.to_sae_trainer_config(),
        save_checkpoint_fn=runner.save_checkpoint,
    )

    # Train the model - this should create checkpoints
    trainer.fit()
    checkpoint_paths = list(checkpoint_dir.glob("**/activation_scaler.json"))
    # We should have exactly 3 checkpoints including the final checkpoint:
    assert (
        len(checkpoint_paths) == 3
    ), f"Expected 3 checkpoints but got {len(checkpoint_paths)}"
    during_checkpoints = []
    final_checkpoints = []
    for path in checkpoint_paths:
        with open(path) as f:
            data = json.load(f)
        if "final" in path.parent.name:
            final_checkpoints.append(data)
        else:
            during_checkpoints.append(data)

    assert (
        len(during_checkpoints) == 2
    ), f"Expected 2 other checkpoints but got {len(during_checkpoints)}"
    assert (
        len(final_checkpoints) == 1
    ), f"Expected 1 final checkpoint but got {len(final_checkpoints)}"
    during_checkpoint = during_checkpoints[0]
    final_checkpoint = final_checkpoints[0]

    # Check intermediate checkpoints have the scaling factor
    assert "scaling_factor" in during_checkpoint
    assert during_checkpoint["scaling_factor"] is not None

    # Final checkpoint should NOT have the scaling factor as it's been folded into the weights
    assert final_checkpoint.get("scaling_factor") is None


def test_sae_trainer_saves_final_checkpoint_when_enabled(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    """Test that SAETrainer saves final checkpoint when save_final_checkpoint is True."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cfg = build_runner_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=20,
        context_size=8,
        save_final_checkpoint=True,  # Enable final checkpoint
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    trainer = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=activation_store,
    )

    trainer.fit()

    # Check that final checkpoint was saved
    final_checkpoint_dir = checkpoint_dir / "final_20"
    assert final_checkpoint_dir.exists()
    assert (final_checkpoint_dir / "sae_weights.safetensors").exists()
    assert (final_checkpoint_dir / "cfg.json").exists()
    assert (final_checkpoint_dir / "sparsity.safetensors").exists()


def test_sae_trainer_skips_final_checkpoint_when_disabled(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    """Test that SAETrainer skips final checkpoint when save_final_checkpoint is False."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cfg = build_runner_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=20,
        context_size=8,
        save_final_checkpoint=False,  # Disable final checkpoint
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    trainer = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae,
        data_provider=activation_store,
    )

    trainer.fit()

    # Check that final checkpoint was NOT saved
    final_checkpoint_dir = checkpoint_dir / "final_20"
    assert not final_checkpoint_dir.exists()


def test_SAETrainer_save_and_load_from_checkpoint(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    cfg = build_runner_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=20,
        context_size=8,
        save_final_checkpoint=False,  # Disable final checkpoint
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 1000)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae1 = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    sae2 = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    trainer = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae1,
        data_provider=activation_store,
    )
    state = trainer.get_state("default")

    for param in sae1.parameters():
        param.grad = torch.randn_like(param)
    state.optimizer.step()

    trainer.n_training_steps = 17
    trainer.n_training_samples = 170
    state.activation_scaler.scaling_factor = 1.0
    state.act_freq_scores = torch.tensor([1.0, 2.0, 3.0])
    state.n_forward_passes_since_fired = torch.tensor([1.0, 2.0, 3.0])
    state.n_frac_active_samples = 170

    for _ in range(17):
        state.coefficient_schedulers["l1"].step()

    trainer.save_trainer_state(checkpoint_dir)

    new_trainer = SAETrainer(
        cfg=cfg.to_sae_trainer_config(),
        sae=sae2,
        data_provider=activation_store,
    )
    new_trainer.load_trainer_state(checkpoint_dir)
    new_state_obj = new_trainer.get_state("default")

    assert new_trainer.n_training_steps == trainer.n_training_steps
    assert new_trainer.n_training_samples == trainer.n_training_samples
    assert (
        new_state_obj.activation_scaler.scaling_factor
        == state.activation_scaler.scaling_factor
    )
    assert torch.allclose(new_state_obj.act_freq_scores, state.act_freq_scores)
    assert torch.allclose(
        new_state_obj.n_forward_passes_since_fired, state.n_forward_passes_since_fired
    )
    assert new_state_obj.n_frac_active_samples == state.n_frac_active_samples
    assert new_state_obj.started_fine_tuning == state.started_fine_tuning
    assert (
        new_state_obj.coefficient_schedulers["l1"].current_step
        == state.coefficient_schedulers["l1"].current_step
    )

    # compare optimizer state dicts
    old_state = state.optimizer.state_dict()
    new_state = new_state_obj.optimizer.state_dict()
    assert old_state.keys() == new_state.keys()
    for key in old_state:
        if isinstance(old_state[key], dict):
            assert old_state[key].keys() == new_state[key].keys()
            for param_key in old_state[key]:
                old_val = old_state[key][param_key]
                new_val = new_state[key][param_key]
                if isinstance(old_val, dict):
                    assert old_val.keys() == new_val.keys()
                    for nested_key in old_val:
                        if torch.is_tensor(old_val[nested_key]):
                            assert torch.allclose(
                                old_val[nested_key], new_val[nested_key]
                            )
                        else:
                            assert old_val[nested_key] == new_val[nested_key]
                elif torch.is_tensor(old_val):
                    assert torch.allclose(old_val, new_val)
                else:
                    assert old_val == new_val
        elif torch.is_tensor(old_state[key]):
            assert torch.allclose(old_state[key], new_state[key])
        else:
            assert old_state[key] == new_state[key]


def test_multi_sae_training_matches_serial_training():
    """
    Test that training SAEs in multi-SAE mode produces identical results to training
    them serially, given the same initial weights and training data.
    """
    d_in = 32
    d_sae = 16
    batch_size = 64
    num_batches = 20
    total_samples = batch_size * num_batches

    # Create fixed training data
    fixed_data = [torch.randn(batch_size, d_in) for _ in range(num_batches)]

    # Create config
    cfg = SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=batch_size,
        lr=1e-3,
        feature_sampling_window=num_batches + 10,  # Don't reset during training
        dead_feature_window=num_batches + 10,
        logger=LoggingConfig(log_to_wandb=False),
    )

    # Create base SAE config
    sae_cfg = StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae)

    # Create first SAE and save its initial state
    sae_base = StandardTrainingSAE.from_dict(sae_cfg.to_dict())
    initial_state_dict = {k: v.clone() for k, v in sae_base.state_dict().items()}

    # Create 4 SAEs with identical initial weights
    sae1 = StandardTrainingSAE.from_dict(sae_cfg.to_dict())
    sae1.load_state_dict({k: v.clone() for k, v in initial_state_dict.items()})

    sae2 = StandardTrainingSAE.from_dict(sae_cfg.to_dict())
    sae2.load_state_dict({k: v.clone() for k, v in initial_state_dict.items()})

    sae3 = StandardTrainingSAE.from_dict(sae_cfg.to_dict())
    sae3.load_state_dict({k: v.clone() for k, v in initial_state_dict.items()})

    sae4 = StandardTrainingSAE.from_dict(sae_cfg.to_dict())
    sae4.load_state_dict({k: v.clone() for k, v in initial_state_dict.items()})

    # Verify initial weights are identical
    for key in initial_state_dict:
        assert torch.allclose(sae1.state_dict()[key], sae2.state_dict()[key])
        assert torch.allclose(sae1.state_dict()[key], sae3.state_dict()[key])
        assert torch.allclose(sae1.state_dict()[key], sae4.state_dict()[key])

    # Train sae1 serially
    trainer1 = SAETrainer(
        cfg=cfg,
        sae=sae1,
        data_provider=iter([d.clone() for d in fixed_data]),
    )
    trained_sae1 = trainer1.fit()

    # Train sae2 serially
    trainer2 = SAETrainer(
        cfg=cfg,
        sae=sae2,
        data_provider=iter([d.clone() for d in fixed_data]),
    )
    trained_sae2 = trainer2.fit()

    # Train sae3 and sae4 together in multi-SAE mode
    trainer_multi = SAETrainer(
        cfg=cfg,
        sae={"a": sae3, "b": sae4},
        data_provider=iter([d.clone() for d in fixed_data]),
    )
    trained_saes = trainer_multi.fit()

    assert isinstance(trained_saes, dict)
    assert set(trained_saes.keys()) == {"a", "b"}

    # Assert weights match between serial and multi-SAE training
    for key in trained_sae1.state_dict():
        # Serial SAE 1 should match multi-SAE "a"
        assert torch.allclose(
            trained_sae1.state_dict()[key],
            trained_saes["a"].state_dict()[key],
            rtol=1e-5,
            atol=1e-5,
        ), f"Mismatch for key {key} between serial sae1 and multi-SAE 'a'"

        # Serial SAE 2 should match multi-SAE "b"
        assert torch.allclose(
            trained_sae2.state_dict()[key],
            trained_saes["b"].state_dict()[key],
            rtol=1e-5,
            atol=1e-5,
        ), f"Mismatch for key {key} between serial sae2 and multi-SAE 'b'"


def test_multi_sae_trainer_basic_functionality():
    """Test basic multi-SAE trainer functionality."""
    d_in = 32
    d_sae_1 = 16
    d_sae_2 = 24  # Different d_sae to test heterogeneous SAEs
    batch_size = 32
    num_batches = 5
    total_samples = batch_size * num_batches

    fixed_data = [torch.randn(batch_size, d_in) for _ in range(num_batches)]

    cfg = SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=batch_size,
        lr=1e-3,
        feature_sampling_window=num_batches + 10,
        dead_feature_window=num_batches + 10,
        logger=LoggingConfig(log_to_wandb=False),
    )

    # Create SAEs with different d_sae values
    sae1 = StandardTrainingSAE.from_dict(
        StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae_1).to_dict()
    )
    sae2 = StandardTrainingSAE.from_dict(
        StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae_2).to_dict()
    )

    # Create multi-SAE trainer
    trainer = SAETrainer(
        cfg=cfg,
        sae={"small": sae1, "large": sae2},
        data_provider=iter(fixed_data),
    )

    # Test sae_keys property
    assert set(trainer.sae_keys) == {"small", "large"}

    # Test get_sae method
    assert trainer.get_sae("small") is sae1
    assert trainer.get_sae("large") is sae2

    # Test get_state method
    state_small = trainer.get_state("small")
    assert isinstance(state_small, PerSAETrainerState)
    assert state_small.sae is sae1

    # Test training
    result = trainer.fit()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"small", "large"}
    assert result["small"].cfg.d_sae == d_sae_1
    assert result["large"].cfg.d_sae == d_sae_2


def test_multi_sae_checkpoint_saves_per_sae_subdirs(tmp_path: Path):
    """Test that multi-SAE checkpoints are saved in per-SAE subdirectories."""
    d_in = 32
    d_sae = 16
    batch_size = 32
    num_batches = 5
    total_samples = batch_size * num_batches

    fixed_data = [torch.randn(batch_size, d_in) for _ in range(num_batches)]

    cfg = SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=batch_size,
        checkpoint_path=str(tmp_path / "checkpoints"),
        save_final_checkpoint=True,
        logger=LoggingConfig(log_to_wandb=False),
    )

    sae1 = StandardTrainingSAE.from_dict(
        StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae).to_dict()
    )
    sae2 = StandardTrainingSAE.from_dict(
        StandardTrainingSAEConfig(d_in=d_in, d_sae=d_sae).to_dict()
    )

    trainer = SAETrainer(
        cfg=cfg,
        sae={"sae_a": sae1, "sae_b": sae2},
        data_provider=iter(fixed_data),
    )
    trainer.fit()

    # Check checkpoint structure
    checkpoint_dir = tmp_path / "checkpoints" / f"final_{total_samples}"
    assert checkpoint_dir.exists()

    # Each SAE should have its own subdirectory
    assert (checkpoint_dir / "sae_a").exists()
    assert (checkpoint_dir / "sae_b").exists()
    assert (checkpoint_dir / "shared").exists()

    # Check that each SAE dir has the expected files
    for sae_key in ["sae_a", "sae_b"]:
        sae_dir = checkpoint_dir / sae_key
        assert (sae_dir / "sae_weights.safetensors").exists()
        assert (sae_dir / "cfg.json").exists()
        assert (sae_dir / "sparsity.safetensors").exists()
        assert (sae_dir / "trainer_state.pt").exists()
        assert (sae_dir / "activation_scaler.json").exists()

    # Check shared directory (only has global trainer state for multi-SAE mode)
    assert (checkpoint_dir / "shared" / "global_trainer_state.pt").exists()
