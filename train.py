import argparse
import pickle
import random
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset

EEG_SUB_FEATURES = ("psd_movingAve", "psd_LDS", "de_movingAve", "de_LDS")
EOG_SUB_FEATURES = (
    "features_table_ica",
    "features_table_minus",
    "features_table_icav_minh",
)
MODALITY_SPECS = {
    "eeg_2hz": {
        "keys": tuple(f"EEG_Feature_2Hz_{name}" for name in EEG_SUB_FEATURES),
        "shape": (17, 25),
        "hidden_sizes": (340, 80),
    },
    "eeg_5bands": {
        "keys": tuple(f"EEG_Feature_5Bands_{name}" for name in EEG_SUB_FEATURES),
        "shape": (17, 5),
        "hidden_sizes": (80,),
    },
    "forehead_eeg_2hz": {
        "keys": tuple(f"Forehead_EEG_Feature_2Hz_{name}" for name in EEG_SUB_FEATURES),
        "shape": (4, 25),
        "hidden_sizes": (80,),
    },
    "forehead_eeg_5bands": {
        "keys": tuple(
            f"Forehead_EEG_Feature_5Bands_{name}" for name in EEG_SUB_FEATURES
        ),
        "shape": (4, 5),
        "hidden_sizes": (),
    },
    "eog": {
        "keys": tuple(f"EOG_Feature_{name}" for name in EOG_SUB_FEATURES),
        "shape": (36,),
    },
}


class SeedVigDataset(Dataset):
    def __init__(self, inputs: dict, outputs: np.ndarray, feature_keys: set[str]):
        self.outputs = np.asarray(outputs, dtype=np.float32).reshape(-1)
        self.inputs = {}
        for key in sorted(feature_keys):
            if key not in inputs:
                raise KeyError(
                    f"Required feature '{key}' is missing from inputs.pickle"
                )
            values = np.asarray(inputs[key], dtype=np.float32)
            if len(values) != len(self.outputs):
                raise ValueError(
                    f"Feature '{key}' has {len(values)} samples, but labels have "
                    f"{len(self.outputs)}"
                )
            self.inputs[key] = values

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features = {
            key: torch.from_numpy(values[index]) for key, values in self.inputs.items()
        }
        return features, torch.tensor(self.outputs[index], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.outputs)


class EegBranch(nn.Module):
    def __init__(self, keys: tuple[str, ...], shape: tuple[int, int], hidden_sizes):
        super().__init__()
        self.keys = keys
        self.extractor = nn.Sequential(
            nn.Conv2d(len(keys), 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
        )

        sizes = (4 * shape[0] * shape[1], *hidden_sizes, 108)
        layers = []
        for input_size, output_size in pairwise(sizes):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        self.projection = nn.Sequential(*layers)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = torch.stack([inputs[key] for key in self.keys], dim=1)
        return self.projection(self.extractor(features))


class EogBranch(nn.Module):
    def __init__(self, keys: tuple[str, ...]):
        super().__init__()
        self.keys = keys

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([inputs[key] for key in self.keys], dim=1).flatten(1)


class SeedVigModel(nn.Module):
    def __init__(self, modalities: list[str]):
        super().__init__()
        self.modalities = tuple(modalities)
        branches = {}
        for modality in modalities:
            spec = MODALITY_SPECS[modality]
            if modality == "eog":
                branches[modality] = EogBranch(spec["keys"])
            else:
                branches[modality] = EegBranch(
                    spec["keys"], spec["shape"], spec["hidden_sizes"]
                )
        self.branches = nn.ModuleDict(branches)
        self.regressor = nn.Sequential(
            nn.Linear(108 * len(modalities), 108),
            nn.ReLU(),
            nn.Linear(108, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.branches[name](inputs) for name in self.modalities]
        return self.regressor(torch.cat(features, dim=1)).squeeze(1)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def split_indices(
    sample_count: int,
    val_ratio: float,
    seed: int,
    groups: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    if groups is not None:
        if len(groups) != sample_count:
            raise ValueError("groups.pickle must contain one group for every sample")
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            shuffled_groups = rng.permutation(unique_groups)
            val_group_count = min(
                max(1, round(len(unique_groups) * val_ratio)), len(unique_groups) - 1
            )
            val_groups = set(shuffled_groups[:val_group_count])
            val_indices = [i for i, group in enumerate(groups) if group in val_groups]
            train_indices = [
                i for i, group in enumerate(groups) if group not in val_groups
            ]
            return train_indices, val_indices

    indices = rng.permutation(sample_count)
    val_count = min(max(1, round(sample_count * val_ratio)), sample_count - 1)
    return indices[val_count:].tolist(), indices[:val_count].tolist()


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_absolute_error = 0.0
    sample_count = 0

    for features, targets in loader:
        features = {key: value.to(device) for key, value in features.items()}
        targets = targets.to(device)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            predictions = model(features)
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = len(targets)
        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(predictions - targets).sum().item()
        sample_count += batch_size

    return {
        "loss": total_loss / sample_count,
        "mae": total_absolute_error / sample_count,
    }


def train(
    args: argparse.Namespace,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> SeedVigModel:
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    inputs = load_pickle(args.data_dir / "inputs.pickle")
    outputs = load_pickle(args.data_dir / "outputs.pickle")
    groups_path = args.data_dir / "groups.pickle"
    groups = load_pickle(groups_path) if groups_path.is_file() else None

    feature_keys = {
        key for modality in args.modalities for key in MODALITY_SPECS[modality]["keys"]
    }
    dataset = SeedVigDataset(inputs, outputs, feature_keys)
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for training")
    train_indices, val_indices = split_indices(
        len(dataset), args.val_ratio, args.seed, groups
    )

    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, **loader_options
    )

    model = SeedVigModel(args.modalities).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum
    )

    split_type = "record" if groups is not None and len(set(groups)) >= 2 else "sample"
    print(
        f"Training on {device} with {len(train_indices)} training and "
        f"{len(val_indices)} validation samples ({split_type}-level split)"
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        metrics = {
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"val_mae={metrics['val_mae']:.4f}"
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "modalities": args.modalities,
        },
        args.output,
    )
    print(f"Training complete. Model saved to {args.output}")
    return model


def build_parser(
    description: str = "Train the SEED-VIG fusion model",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("model.pth"))
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=tuple(MODALITY_SPECS),
        default=list(MODALITY_SPECS),
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
