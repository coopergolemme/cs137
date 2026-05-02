"""
Mechanistic case studies: find the worst-MAPE val 2023 windows for the
lookback model and plot prediction vs actual vs naive-lag (the 168 h prior).

This shows *why* the model fails on holidays (lag = regular weekday) and
extreme weather (lag = mild week).

Outputs PNGs of the 6 worst windows by aggregate MAPE.

Usage:
    python analysis_cases.py preds_lookback.npz
"""

import argparse
import os

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local_data import Part3Data

_US_HOLIDAYS = holidays.US()


def per_window_mape(preds, tgts):
    """(N,) MAPE per window, averaged over horizon and zones."""
    return np.mean(np.abs((tgts - preds) / (tgts + 1e-6)) * 100, axis=(1, 2))


def label_window(start_hour, S, fut_hours):
    """Build a one-line description of the future window."""
    fut_dti = pd.to_datetime(fut_hours.astype(np.int64), unit="h", utc=True)
    first = fut_dti[0]
    is_weekend = first.dayofweek >= 5
    is_holiday = first.normalize().date() in _US_HOLIDAYS
    label = first.strftime("%Y-%m-%d %H:%MZ")
    tags = []
    if is_holiday:
        tags.append(f"HOLIDAY ({_US_HOLIDAYS.get(first.normalize().date())})")
    if is_weekend:
        tags.append("weekend")
    return label + ((" — " + ", ".join(tags)) if tags else "")


def plot_window(idx, preds, tgts, lag, fut_hours, zones, mape_val, label, out_path):
    """Plot prediction / actual / naive-lag for all 8 zones in one figure."""
    fut_dti = pd.to_datetime(fut_hours.astype(np.int64), unit="h", utc=True)
    n_zones = len(zones)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    axes = axes.flatten()
    for j, z in enumerate(zones):
        ax = axes[j]
        ax.plot(fut_dti, tgts[:, j], "k-", label="actual")
        ax.plot(fut_dti, preds[:, j], "C0--", label="lookback pred")
        ax.plot(fut_dti, lag[:, j], "C3:", label="naive lag (168 h)")
        ax.set_title(z)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Window {idx} — {label}\nWindow MAPE: {mape_val:.2f}%", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(pred_path, S=72, out_dir="cases", top_k=6):
    os.makedirs(out_dir, exist_ok=True)

    npz = np.load(pred_path, allow_pickle=True)
    preds = npz["predictions"].astype(np.float32)
    tgts = npz["targets"].astype(np.float32)
    start_hours = npz["start_hours"].astype(np.int64)
    fut_hours = npz["fut_hours"].astype(np.int64)
    zones = [str(z) for z in npz["zones"]]

    print(f"Loaded {len(preds):,} val windows from {pred_path}")

    # Per-window MAPE → rank
    win_mape = per_window_mape(preds, tgts)
    print(f"Window MAPE: min={win_mape.min():.2f}, p50={np.median(win_mape):.2f}, p95={np.percentile(win_mape, 95):.2f}, max={win_mape.max():.2f}")

    # Need naive lag = demand from 168 h before each future hour
    # We'll fetch from Part3Data
    data = Part3Data()
    print("Fetching naive-lag demand for each window …")
    lag_for_window = np.zeros_like(preds)
    for i, fh in enumerate(fut_hours):
        lag_hours = fh - 168
        idxs = [data.h2d_idx[int(h)] for h in lag_hours]
        lag_for_window[i] = data.demand[idxs]

    # Top-K worst
    order = np.argsort(-win_mape)
    print(f"\nTop {top_k} worst windows:")
    for rank in range(top_k):
        idx = int(order[rank])
        label = label_window(start_hours[idx], S, fut_hours[idx])
        m = float(win_mape[idx])
        print(f"  rank {rank+1}: idx {idx:5d}   MAPE {m:6.2f}%   {label}")
        out_path = os.path.join(out_dir, f"worst_{rank+1:02d}_idx{idx}.png")
        plot_window(
            idx,
            preds[idx], tgts[idx], lag_for_window[idx],
            fut_hours[idx], zones, m, label, out_path,
        )

    # Also plot 2 representative best-case windows for contrast
    print(f"\nTop 2 best windows (for contrast):")
    for rank in range(2):
        idx = int(order[-(rank + 1)])
        label = label_window(start_hours[idx], S, fut_hours[idx])
        m = float(win_mape[idx])
        print(f"  rank {rank+1}: idx {idx:5d}   MAPE {m:6.2f}%   {label}")
        out_path = os.path.join(out_dir, f"best_{rank+1:02d}_idx{idx}.png")
        plot_window(
            idx,
            preds[idx], tgts[idx], lag_for_window[idx],
            fut_hours[idx], zones, m, label, out_path,
        )

    print(f"\nSaved case plots to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_file")
    p.add_argument("--S", type=int, default=72)
    p.add_argument("--out_dir", default="cases")
    p.add_argument("--top_k", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.pred_file, S=args.S, out_dir=args.out_dir, top_k=args.top_k)
