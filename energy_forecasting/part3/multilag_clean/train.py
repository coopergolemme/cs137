"""
Training script for the multi-lag HybridCNNTransformer.

Same loop and hyperparameters as the lookback-model trainer. The only
difference: each batch carries one extra tensor (`future_demand_lag_24`)
that is threaded into model.forward().
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import EnergyDataset
from model import get_model


def compute_demand_stats(dataset):
    indices = set()
    for start in dataset.valid_starts:
        indices.update(range(start, start + dataset.S))
    demand = dataset.demand_values[sorted(indices)]
    mean = torch.tensor(demand.mean(axis=0), dtype=torch.float32)
    std  = torch.tensor(demand.std(axis=0),  dtype=torch.float32)
    std  = std.clamp(min=1.0)
    return mean, std


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        hist_w, hist_d, hist_cal, fut_w, fut_cal, lag168, lag24, target = [
            x.to(device) for x in batch
        ]
        pred = model(hist_w, hist_d, hist_cal, fut_w, fut_cal, lag168, lag24)
        total += loss_fn(pred, target).item() * hist_w.size(0)
        n += hist_w.size(0)
    return total / max(n, 1)


def train(config: dict = None):
    if config is None:
        config = CONFIG

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(config["checkpoint_dir"], config["checkpoint_name"])

    print("Building datasets …")
    train_ds = EnergyDataset(
        weather_base_path=config["weather_base_path"],
        demand_dir=config["demand_dir"],
        S=config["S"],
        grid_size=config["grid_size"],
        split_years=config["train_years"],
    )
    val_ds = EnergyDataset(
        weather_base_path=config["weather_base_path"],
        demand_dir=config["demand_dir"],
        S=config["S"],
        grid_size=config["grid_size"],
        split_years=config["val_years"],
    )
    print(f"  Train samples: {len(train_ds):,}   Val samples: {len(val_ds):,}")

    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config["num_workers"] > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    model = get_model(config).to(device)
    demand_mean, demand_std = compute_demand_stats(train_ds)
    model.set_demand_stats(demand_mean, demand_std)
    print(f"  Demand stats — mean: {demand_mean.mean():.1f} MW, std: {demand_std.mean():.1f} MW")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    seq_len = (config["S"] + 24) * (config["P"] + 1)
    print(f"  Transformer sequence length: {seq_len:,} tokens")

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config["num_epochs"], eta_min=config["lr_min"]
    )
    loss_fn = nn.L1Loss() if config["loss"] == "mae" else nn.MSELoss()

    best_val_loss = float("inf")
    log_interval = max(1, len(train_loader) // 5)

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            hist_w, hist_d, hist_cal, fut_w, fut_cal, lag168, lag24, target = [
                x.to(device) for x in batch
            ]
            optimiser.zero_grad(set_to_none=True)
            pred = model(hist_w, hist_d, hist_cal, fut_w, fut_cal, lag168, lag24)
            loss = loss_fn(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimiser.step()

            epoch_loss += loss.item() * hist_w.size(0)
            n += hist_w.size(0)

            if step % log_interval == 0:
                print(f"  Epoch {epoch} [{step}/{len(train_loader)}]  loss: {loss.item():.2f} MW")

        train_loss = epoch_loss / max(n, 1)
        val_loss = validate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{config['num_epochs']}  "
              f"train: {train_loss:.2f}  val: {val_loss:.2f}  ({elapsed:.0f}s)")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                    "demand_mean": model.demand_mean.cpu(),
                    "demand_std":  model.demand_std.cpu(),
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model (val_loss={val_loss:.2f})")

    print(f"\nTraining complete.  Best val loss: {best_val_loss:.2f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-lag HybridCNNTransformer")
    for key, val in CONFIG.items():
        if isinstance(val, (int, float, str)):
            t = type(val)
            parser.add_argument(f"--{key}", type=t, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(CONFIG)
    for key, val in vars(args).items():
        if val is not None:
            cfg[key] = val
    train(cfg)
