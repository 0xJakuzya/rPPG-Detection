import argparse
import random
from pathlib import Path

import numpy as np
import torch

from models.loss import CNNLoss
from models.baseline import PatchCNN
from src import config
from src.dataset import (
    build_dataloaders,
    describe_dataset,
    discover_window_files,
    split_by_patient,
)

def run(args=None) -> None:
    args = parse_args(args)
    fix_seed(args.seed)

    train_config = {
        "DATA_DIR": Path(args.data_dir),
        "OUTPUT": Path(args.output),
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "LR": args.lr,
        "VAL_SPLIT": args.val_split,
        "SPECTRAL_ALPHA": args.spectral_alpha,
        "SEED": args.seed,
        "NUM_WORKERS": args.num_workers,
        "DEVICE": torch.device(args.device),
        "MODEL_NAME": "Multi-ROI Patch CNN",
    }

    print_step(1, "Load prepared windows")
    files = discover_window_files(train_config["DATA_DIR"])
    describe_dataset(files)

    print_step(2, "Split by patient")
    split = split_by_patient(files, train_config["VAL_SPLIT"], train_config["SEED"])
    print(f"train windows: {len(split.train_indices)} from {len(split.train_patients)} patients")
    print(f"val windows:   {len(split.val_indices)} from {len(split.val_patients)} patients")

    print_step(3, "Build dataloaders")
    train_loader, val_loader = build_dataloaders(files, split, train_config)
    print(f"train batches: {len(train_loader)}")
    print(f"val batches:   {len(val_loader)}")

    print_step(4, "Build training objects")
    model = PatchCNN().to(train_config["DEVICE"])
    criterion = CNNLoss(train_config["SPECTRAL_ALPHA"])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["LR"])
    print(f"device: {train_config['DEVICE']}")
    print(f"model: {train_config['MODEL_NAME']}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"output: {train_config['OUTPUT']}")

    print_step(5, "Train")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, train_config)
    save_results(model, history, train_config)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: PatchCNN,
    train_loader,
    val_loader,
    criterion: CNNLoss,
    optimizer: torch.optim.Optimizer,
    train_config: dict,
) -> dict[str, list[float]]:
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, train_config["EPOCHS"] + 1):
        train_loss = run_epoch(model, train_loader, criterion, train_config["DEVICE"], optimizer)
        val_loss = run_epoch(model, val_loader, criterion, train_config["DEVICE"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            train_config["BEST_STATE"] = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

        marker = "*" if improved else " "
        print(
            f"{marker} epoch {epoch:03d}/{train_config['EPOCHS']} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} best={best_val_loss:.5f}"
        )

    return history


def run_epoch(
    model: PatchCNN,
    loader,
    criterion: CNNLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    for patches, target_ppg in loader:
        patches = patches.to(device)
        target_ppg = target_ppg.to(device)

        with torch.set_grad_enabled(is_training):
            predicted_ppg = model(patches)
            loss = criterion(predicted_ppg, target_ppg)

        if is_training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = patches.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def save_results(model: PatchCNN, history: dict[str, list[float]], train_config: dict) -> None:
    output_path = train_config["OUTPUT"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "BEST_STATE" in train_config:
        model.load_state_dict(train_config["BEST_STATE"])
    torch.save(model.to("cpu").state_dict(), output_path)
    for metric, values in history.items():
        metric_path = output_path.with_name(metric)
        np.save(metric_path, np.array(values, dtype=np.float32))
    print()
    print(f"saved best model: {output_path}")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

def print_step(index: int, title: str) -> None:
    print()
    print(f"[{index}/5] {title}")

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-ROI patch CNN.")
    parser.add_argument("--data-dir", default="data/mcd_rppg_windows")
    parser.add_argument("--output", default=config.PHYSNET_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=config.PHYSNET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.PHYSNET_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.PHYSNET_LR)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--spectral-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(args)

if __name__ == "__main__":
    run()
