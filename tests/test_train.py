import argparse
import pickle

import numpy as np
import pytest
import torch

from train import MODALITY_SPECS, SeedVigDataset, SeedVigModel, split_indices, train


def make_batch(batch_size=2):
    batch = {}
    for spec in MODALITY_SPECS.values():
        for key in spec["keys"]:
            batch[key] = torch.rand(batch_size, *spec["shape"])
    return batch


def test_full_model_produces_bounded_regression_output():
    model = SeedVigModel(list(MODALITY_SPECS))
    output = model(make_batch())

    assert output.shape == (2,)
    assert torch.all((0 <= output) & (output <= 1))
    output.mean().backward()


def test_model_supports_modality_ablation():
    model = SeedVigModel(["eeg_2hz", "eog"])

    assert model(make_batch()).shape == (2,)


def test_dataset_rejects_misaligned_features():
    with pytest.raises(ValueError, match="labels have 3"):
        SeedVigDataset({"feature": np.ones((2, 4))}, np.ones(3), {"feature"})


def test_group_split_keeps_records_separate():
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    train_indices, val_indices = split_indices(9, 0.34, 42, groups)

    train_groups = {groups[index] for index in train_indices}
    val_groups = {groups[index] for index in val_indices}
    assert train_groups.isdisjoint(val_groups)
    assert train_indices
    assert val_indices


def test_training_pipeline_saves_checkpoint(tmp_path):
    sample_count = 6
    inputs = {
        key: np.random.default_rng(42).random((sample_count, 36), dtype=np.float32)
        for key in MODALITY_SPECS["eog"]["keys"]
    }
    for filename, value in (
        ("inputs.pickle", inputs),
        ("outputs.pickle", np.linspace(0, 1, sample_count, dtype=np.float32)),
        ("groups.pickle", ["record-a"] * 3 + ["record-b"] * 3),
    ):
        with (tmp_path / filename).open("wb") as handle:
            pickle.dump(value, handle)

    output_path = tmp_path / "model.pth"
    args = argparse.Namespace(
        data_dir=tmp_path,
        output=output_path,
        cuda_device=0,
        batch_size=2,
        epochs=1,
        learning_rate=0.001,
        momentum=0.9,
        val_ratio=0.5,
        seed=42,
        workers=0,
        modalities=["eog"],
    )

    train(args)

    checkpoint = torch.load(output_path, weights_only=True)
    assert checkpoint["modalities"] == ["eog"]
    assert checkpoint["model_state_dict"]
