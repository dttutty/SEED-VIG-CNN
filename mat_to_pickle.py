import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.io as scio
from tqdm import tqdm

EEG_FEATURES = {
    "EEG_Feature_2Hz": ("EEG_Feature_2Hz", (17, 25)),
    "EEG_Feature_5Bands": ("EEG_Feature_5Bands", (17, 5)),
    "Forehead_EEG_Feature_2Hz": (
        "Forehead_EEG/EEG_Feature_2Hz",
        (4, 25),
    ),
    "Forehead_EEG_Feature_5Bands": (
        "Forehead_EEG/EEG_Feature_5Bands",
        (4, 5),
    ),
}
EEG_SUB_FEATURES = ("psd_movingAve", "psd_LDS", "de_movingAve", "de_LDS")
EOG_SUB_FEATURES = (
    "features_table_ica",
    "features_table_minus",
    "features_table_icav_minh",
)


def load_mat(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return scio.loadmat(path)
    except Exception as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc


def require_feature(mat: dict, key: str, path: Path) -> np.ndarray:
    if key not in mat:
        raise KeyError(f"Feature '{key}' not found in {path}")
    return np.asarray(mat[key])


def load_record(
    data_root: Path, filename: str
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    label_path = data_root / "perclos_labels" / filename
    label_mat = load_mat(label_path)
    labels = require_feature(label_mat, "perclos", label_path).reshape(-1)
    sample_count = len(labels)
    if sample_count == 0:
        raise ValueError(f"No PERCLOS labels found in {label_path}")

    record = {}
    for prefix, (relative_folder, expected_shape) in EEG_FEATURES.items():
        path = data_root / relative_folder / filename
        mat = load_mat(path)
        for sub_feature in EEG_SUB_FEATURES:
            values = require_feature(mat, sub_feature, path)
            if values.ndim != 3:
                raise ValueError(
                    f"Expected a 3D array for {sub_feature} in {path}, got {values.shape}"
                )
            values = values.transpose(1, 0, 2)
            if values.shape != (sample_count, *expected_shape):
                raise ValueError(
                    f"Unexpected shape for {sub_feature} in {path}: {values.shape}; "
                    f"expected {(sample_count, *expected_shape)}"
                )
            record[f"{prefix}_{sub_feature}"] = values

    eog_path = data_root / "EOG_Feature" / filename
    eog_mat = load_mat(eog_path)
    for sub_feature in EOG_SUB_FEATURES:
        values = require_feature(eog_mat, sub_feature, eog_path)
        if values.shape != (sample_count, 36):
            raise ValueError(
                f"Unexpected shape for {sub_feature} in {eog_path}: {values.shape}; "
                f"expected {(sample_count, 36)}"
            )
        record[f"EOG_Feature_{sub_feature}"] = values

    return record, labels


def dump_pickle(value, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_dataset(
    data_root: Path, output_dir: Path
) -> tuple[dict, np.ndarray, list[str]]:
    label_dir = data_root / "perclos_labels"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"PERCLOS label directory not found: {label_dir}")

    filenames = sorted(path.name for path in label_dir.glob("*.mat"))
    if not filenames:
        raise FileNotFoundError(f"No .mat label files found in {label_dir}")

    feature_chunks: dict[str, list[np.ndarray]] = {}
    label_chunks = []
    groups = []
    error_logs = []

    for filename in tqdm(filenames, desc="Processing records"):
        try:
            record, labels = load_record(data_root, filename)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            error_logs.append(f"{filename}: {exc}")
            continue

        for key, values in record.items():
            feature_chunks.setdefault(key, []).append(values)
        label_chunks.append(labels)
        groups.extend([filename] * len(labels))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "error_logs.txt").write_text(
        "\n".join(error_logs) + ("\n" if error_logs else ""), encoding="utf-8"
    )
    if not label_chunks:
        raise RuntimeError("No complete records were converted; see error_logs.txt")

    inputs = {
        key: np.concatenate(chunks, axis=0) for key, chunks in feature_chunks.items()
    }
    outputs = np.concatenate(label_chunks).astype(np.float32, copy=False)
    dump_pickle(inputs, output_dir / "inputs.pickle")
    dump_pickle(outputs, output_dir / "outputs.pickle")
    dump_pickle(groups, output_dir / "groups.pickle")

    print(
        f"Converted {len(label_chunks)}/{len(filenames)} records "
        f"({len(outputs)} samples) to {output_dir}"
    )
    if error_logs:
        print(f"Skipped {len(error_logs)} incomplete records; see error_logs.txt")
    return inputs, outputs, groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SEED-VIG MATLAB features")
    parser.add_argument(
        "--data-root", type=Path, default=Path("SEED-VIG"), help="SEED-VIG root"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Pickle output directory"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert_dataset(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
