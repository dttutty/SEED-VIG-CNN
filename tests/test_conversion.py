import pickle

import numpy as np
import scipy.io as scio

from mat_to_pickle import (
    EEG_FEATURES,
    EEG_SUB_FEATURES,
    EOG_SUB_FEATURES,
    convert_dataset,
)


def write_record(root, filename, sample_count=3, missing_feature=None):
    label_dir = root / "perclos_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    scio.savemat(label_dir / filename, {"perclos": np.linspace(0, 1, sample_count)})

    for relative_folder, shape in EEG_FEATURES.values():
        folder = root / relative_folder
        folder.mkdir(parents=True, exist_ok=True)
        features = {
            name: np.ones((shape[0], sample_count, shape[1]), dtype=np.float32)
            for name in EEG_SUB_FEATURES
            if name != missing_feature
        }
        scio.savemat(folder / filename, features)

    eog_dir = root / "EOG_Feature"
    eog_dir.mkdir(parents=True, exist_ok=True)
    scio.savemat(
        eog_dir / filename,
        {
            name: np.ones((sample_count, 36), dtype=np.float32)
            for name in EOG_SUB_FEATURES
        },
    )


def test_convert_dataset_keeps_only_complete_records(tmp_path):
    data_root = tmp_path / "SEED-VIG"
    output_dir = tmp_path / "output"
    write_record(data_root, "complete.mat")
    write_record(data_root, "incomplete.mat", missing_feature="de_LDS")

    inputs, outputs, groups = convert_dataset(data_root, output_dir)

    assert len(outputs) == 3
    assert groups == ["complete.mat"] * 3
    assert inputs["EEG_Feature_2Hz_psd_LDS"].shape == (3, 17, 25)
    assert inputs["Forehead_EEG_Feature_5Bands_de_LDS"].shape == (3, 4, 5)
    assert inputs["EOG_Feature_features_table_ica"].shape == (3, 36)
    assert "incomplete.mat" in (output_dir / "error_logs.txt").read_text()

    with (output_dir / "outputs.pickle").open("rb") as handle:
        saved_outputs = pickle.load(handle)
    np.testing.assert_array_equal(saved_outputs, outputs)
