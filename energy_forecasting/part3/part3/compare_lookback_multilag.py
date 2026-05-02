"""
Final comparison: lookback vs multi-lag.

Bootstrap 90% CIs on the *paired* per-window MAPE difference, plus
per-regime contrasts.
"""

import json
import os

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_US_HOLIDAYS = holidays.US()


def load(path):
    npz = np.load(path, allow_pickle=True)
    return {
        "preds":       npz["predictions"].astype(np.float32),
        "tgts":        npz["targets"].astype(np.float32),
        "start_hours": npz["start_hours"].astype(np.int64),
        "fut_hours":   npz["fut_hours"].astype(np.int64),
        "zones":       [str(z) for z in npz["zones"]],
    }


def per_window_mape(preds, tgts):
    return np.mean(np.abs((tgts - preds) / (tgts + 1e-6)) * 100, axis=(1, 2))


def bootstrap_ci(values, n_boot=1000, seed=137):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def calendar_masks(start_hours, S=72):
    fut0 = start_hours + S
    dti = pd.to_datetime(fut0.astype(np.int64), unit="h", utc=True)
    is_weekend = dti.dayofweek >= 5
    dates = dti.normalize().date
    is_holiday = np.array([d in _US_HOLIDAYS for d in dates])
    is_weekday = (~is_weekend) & (~is_holiday)
    return {
        "holiday": is_holiday & ~is_weekend,
        "weekend": is_weekend & ~is_holiday,
        "weekday": is_weekday,
    }


def main():
    lb = load("preds_lookback.npz")
    ml = load("preds_multilag.npz")
    assert np.array_equal(lb["start_hours"], ml["start_hours"])

    lb_w = per_window_mape(lb["preds"], lb["tgts"])
    ml_w = per_window_mape(ml["preds"], ml["tgts"])
    diff = ml_w - lb_w   # negative = multilag improves

    overall_lb = float(lb_w.mean())
    overall_ml = float(ml_w.mean())
    overall_diff = float(diff.mean())
    diff_lo, diff_hi = bootstrap_ci(diff)

    print("=== Overall comparison (90% bootstrap CI) ===")
    print(f"Lookback : {overall_lb:.3f}%")
    print(f"Multilag : {overall_ml:.3f}%")
    print(f"Δ (multilag − lookback) = {overall_diff:+.3f} pp  CI [{diff_lo:+.3f}, {diff_hi:+.3f}]")

    masks = calendar_masks(lb["start_hours"])
    print("\n=== Per-calendar-regime Δ (multilag − lookback) ===")
    for label, mask in masks.items():
        if mask.sum() == 0:
            continue
        d = diff[mask]
        m = float(d.mean())
        lo, hi = bootstrap_ci(d)
        print(f"  {label:10s} n={int(mask.sum()):5d}  Δ = {m:+.3f} pp  CI [{lo:+.3f}, {hi:+.3f}]")

    # Per-zone
    print("\n=== Per-zone Δ (multilag − lookback) ===")
    zones = lb["zones"]
    Z = len(zones)
    for j, z in enumerate(zones):
        lb_z = np.mean(np.abs((lb["tgts"][:, :, j] - lb["preds"][:, :, j]) /
                              (lb["tgts"][:, :, j] + 1e-6)) * 100, axis=1)
        ml_z = np.mean(np.abs((ml["tgts"][:, :, j] - ml["preds"][:, :, j]) /
                              (ml["tgts"][:, :, j] + 1e-6)) * 100, axis=1)
        d = ml_z - lb_z
        m = float(d.mean())
        lo, hi = bootstrap_ci(d)
        print(f"  {z:10s}  Δ = {m:+.3f} pp  CI [{lo:+.3f}, {hi:+.3f}]")

    # Worst-window comparison
    print("\n=== Sept 5 (worst-windows) comparison ===")
    fut0 = lb["start_hours"] + 72
    fut0_dti = pd.to_datetime(fut0.astype(np.int64), unit="h", utc=True)
    sept5 = fut0_dti.date == pd.Timestamp("2023-09-05").date()
    print(f"  windows on 2023-09-05: {int(sept5.sum())}")
    print(f"  lookback MAPE  on Sept 5: {lb_w[sept5].mean():.2f}%  (max {lb_w[sept5].max():.2f}%)")
    print(f"  multilag MAPE  on Sept 5: {ml_w[sept5].mean():.2f}%  (max {ml_w[sept5].max():.2f}%)")

    out = {
        "overall": {
            "lookback_mape": overall_lb,
            "multilag_mape": overall_ml,
            "delta_pp": overall_diff,
            "delta_ci90": [diff_lo, diff_hi],
        },
    }
    with open("analysis_out_multilag/comparison_summary.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
