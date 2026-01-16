from typing import Type

import pytest

from sae_lens.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    SAETrainerConfig,
    _default_cached_activations_path,
)
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

test_cases_for_seqpos = [
    ((None, 10, -1), ValueError),
    ((None, 10, 0), ValueError),
    ((5, 5, None), ValueError),
    ((6, 3, None), ValueError),
]


@pytest.mark.parametrize("seqpos_slice, expected_error", test_cases_for_seqpos)
def test_sae_training_runner_config_seqpos(
    seqpos_slice: tuple[int, int], expected_error: Type[BaseException]
):
    context_size = 10
    with pytest.raises(expected_error):
        LanguageModelSAERunnerConfig(
            sae=StandardTrainingSAEConfig(d_in=10, d_sae=10),
            seqpos_slice=seqpos_slice,
            context_size=context_size,
        )


def test_LanguageModelSAERunnerConfig_hook_eval_deprecated_usage():
    with pytest.warns(
        DeprecationWarning,
        match="The 'hook_eval' field is deprecated and will be removed in v7.0.0. ",
    ):
        LanguageModelSAERunnerConfig(
            sae=StandardTrainingSAEConfig(d_in=10, d_sae=10),
            hook_eval="blocks.0.hook_output",
        )


@pytest.mark.parametrize("seqpos_slice, expected_error", test_cases_for_seqpos)
def test_cache_activations_runner_config_seqpos(
    seqpos_slice: tuple[int, int],
    expected_error: Type[BaseException],
):
    with pytest.raises(expected_error):
        CacheActivationsRunnerConfig(
            dataset_path="",
            model_name="",
            model_batch_size=1,
            hook_name="",
            d_in=1,
            training_tokens=100,
            context_size=10,
            seqpos_slice=seqpos_slice,
        )


def test_default_cached_activations_path():
    assert (
        _default_cached_activations_path(
            dataset_path="ds_path",
            model_name="model_name",
            hook_name="hook_name",
            hook_head_index=None,
        )
        == "activations/ds_path/model_name/hook_name"
    )


def test_LanguageModelSAERunnerConfig_to_dict_and_from_dict():
    cfg = LanguageModelSAERunnerConfig(
        sae=JumpReLUTrainingSAEConfig(
            d_in=5,
            d_sae=10,
            jumprelu_init_threshold=0.1,
            jumprelu_bandwidth=0.1,
            jumprelu_sparsity_loss_mode="tanh",
        ),
        seqpos_slice=(0, 10),
        context_size=10,
    )
    cfg_dict = cfg.to_dict()
    assert cfg_dict == cfg.to_dict()
    assert cfg == LanguageModelSAERunnerConfig.from_dict(cfg_dict)


def test_LanguageModelSAERunnerConfig_errors_when_loading_from_dict_with_missing_fields():
    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_in=5, d_sae=10),
        seqpos_slice=(0, 10),
        context_size=10,
    )
    with pytest.raises(
        ValueError, match="sae field is required in the config dictionary"
    ):
        test_dict = cfg.to_dict()
        del test_dict["sae"]
        LanguageModelSAERunnerConfig.from_dict(test_dict)
    with pytest.raises(
        ValueError, match="architecture field is required in the sae dictionary"
    ):
        test_dict = cfg.to_dict()
        del test_dict["sae"]["architecture"]
        LanguageModelSAERunnerConfig.from_dict(test_dict)
    with pytest.raises(
        ValueError, match="logger field is required in the config dictionary"
    ):
        test_dict = cfg.to_dict()
        del test_dict["logger"]
        LanguageModelSAERunnerConfig.from_dict(test_dict)


def test_sae_trainer_config_defaults_match_runner_config():
    """
    Test that SAETrainerConfig defaults match LanguageModelSAERunnerConfig defaults.

    This ensures that when both configs use their default values, the resulting
    SAETrainerConfig from to_sae_trainer_config() matches a directly constructed
    SAETrainerConfig with defaults.
    """
    training_tokens = 100_000

    # Create runner config with minimal required fields, relying on defaults
    runner_cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_in=64, d_sae=128),
        training_tokens=training_tokens,
        verbose=False,  # Suppress logging during test
    )
    trainer_cfg_from_runner = runner_cfg.to_sae_trainer_config()

    # Create trainer config directly using defaults
    trainer_cfg_direct = SAETrainerConfig(
        total_training_samples=training_tokens,
    )

    # The checkpoint_path from runner has a unique ID appended in __post_init__,
    # so we skip that field in the comparison
    # Compare all fields that should have matching defaults
    assert trainer_cfg_from_runner.lr == trainer_cfg_direct.lr
    assert trainer_cfg_from_runner.lr_end == trainer_cfg_direct.lr_end
    assert (
        trainer_cfg_from_runner.lr_scheduler_name
        == trainer_cfg_direct.lr_scheduler_name
    )
    assert (
        trainer_cfg_from_runner.lr_warm_up_steps == trainer_cfg_direct.lr_warm_up_steps
    )
    assert trainer_cfg_from_runner.lr_decay_steps == trainer_cfg_direct.lr_decay_steps
    assert (
        trainer_cfg_from_runner.n_restart_cycles == trainer_cfg_direct.n_restart_cycles
    )
    assert trainer_cfg_from_runner.adam_beta1 == trainer_cfg_direct.adam_beta1
    assert trainer_cfg_from_runner.adam_beta2 == trainer_cfg_direct.adam_beta2
    assert (
        trainer_cfg_from_runner.train_batch_size_samples
        == trainer_cfg_direct.train_batch_size_samples
    )
    assert (
        trainer_cfg_from_runner.dead_feature_window
        == trainer_cfg_direct.dead_feature_window
    )
    assert (
        trainer_cfg_from_runner.feature_sampling_window
        == trainer_cfg_direct.feature_sampling_window
    )
    assert trainer_cfg_from_runner.n_checkpoints == trainer_cfg_direct.n_checkpoints
    assert (
        trainer_cfg_from_runner.save_final_checkpoint
        == trainer_cfg_direct.save_final_checkpoint
    )
    assert trainer_cfg_from_runner.device == trainer_cfg_direct.device
    assert trainer_cfg_from_runner.autocast == trainer_cfg_direct.autocast
    assert (
        trainer_cfg_from_runner.total_training_samples
        == trainer_cfg_direct.total_training_samples
    )

    # Logger should have the same default structure (log_to_wandb=True, etc.)
    assert type(trainer_cfg_from_runner.logger) == type(trainer_cfg_direct.logger)
    assert (
        trainer_cfg_from_runner.logger.log_to_wandb
        == trainer_cfg_direct.logger.log_to_wandb
    )
    assert (
        trainer_cfg_from_runner.logger.wandb_log_frequency
        == trainer_cfg_direct.logger.wandb_log_frequency
    )
    assert (
        trainer_cfg_from_runner.logger.eval_every_n_wandb_logs
        == trainer_cfg_direct.logger.eval_every_n_wandb_logs
    )
