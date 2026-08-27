import random
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from mattergen.generator import CrystalGenerator, seed_all


def test_seed_all_resets_random_generators() -> None:
    seed_all(42)
    values = (random.random(), np.random.random(), torch.rand(1))

    seed_all(42)

    assert random.random() == values[0]
    assert np.random.random() == values[1]
    assert torch.equal(torch.rand(1), values[2])


def test_crystal_generator_applies_seed_before_sampling(tmp_path) -> None:
    checkpoint_info = Mock(config=OmegaConf.create({}))
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=1,
        num_batches=1,
        seed=42,
    )
    generator._model = Mock()
    generator.load_sampling_config = Mock(
        return_value=OmegaConf.create(
            {"sampler_partial": {"_target_": "builtins.dict", "_partial_": True}}
        )
    )
    generator.get_condition_loader = Mock(return_value=[])

    with (
        patch("mattergen.generator.seed_all") as apply_seed,
        patch("mattergen.generator.instantiate", return_value=Mock(return_value=Mock())),
        patch("mattergen.generator.draw_samples_from_sampler", return_value=[]),
    ):
        generator.generate(output_dir=tmp_path)

    apply_seed.assert_called_once_with(42)
