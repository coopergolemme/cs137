"""
Per-regime MAPE breakdown for one or more prediction .npz files.

For each model, computes MAPE sliced along:
  - forecast horizon hour (1..24)
  - load zone
  - calendar regime: holiday / weekend / weekday
  - weather regime: top-5% / bottom-5% / median temperature day

Outputs:
  results.csv            — long-format table of all regime MAPE values
  fig_horizon.png        — per-horizon MAPE bar chart (one bar per model)
  fig_zone.png           — per-zone MAPE
  fig_calendar.png       — holiday / weekend / weekday MAPE
  fig_weather.png        — extreme-cold / median / extreme-heat MAPE

Usage:
    python analysis_regimes.py preds_lookback.npz preds_lookback_clim.npz
"""

import argparse
import os

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local_data import Part3Data

_US_HOLIDAYS = holidays.US()


def mape(y, yhat):
    """Element-wise mean absolute percentage error."""
    return float(np.mean(np.abs((y - yhat) / (y + 1e-6))) * 100)


def per_horizon_mape(preds, tgts):
    """(24,) MAPE per forecast hour, averaged over windows and zones."""
    out = np.zeros(24)
    for h in range(24):
        out[h] = mape(tgts[:, h, :], preds[:, h, :])
    return out


def per_zone_mape(preds, tgts, zones):
    """(Z,) MAPE per zone, averaged over windows and horizon."""
    return {z: mape(tgts[:, :, j], preds[:, :, j]) for j, z in enumerate(zones)}


def calendar_buckets(start_hours):
    """
    For each window, label its first future hour as 'holiday' / 'weekend' /
    'weekday'. Returns (N,) string array.
    """
    # Future starts at start_hour + S; we don't know S here, but the typical
    # pattern: start_hour is the first historical hour. Caller passes S so we
    # can offset.
    raise NotImplementedError("use bucket_windows() instead")


def bucket_windows(start_hours, S):
    """
    Label each window by its first-future-hour timestamp.
    Returns dict of np.ndarray[bool] masks over the window axis.
    """
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


def weather_buckets(data: Part3Data, start_hours, S):
    """
    Bucket each window by the mean temperature (channel 0) of its 24 future
    hours: extreme_cold (bottom 5%), median (45-55%), extreme_heat (top 5%).
    """
    # Map start_hour -> mean temp over the future window
    fut_temps = []
    for s in start_hours:
        wi_fut = [data.h2w_idx[int(h)] for h in range(int(s) + S, int(s) + S + 24)]
        # Channel 0 of pooled weather, spatial mean
        fut_temps.append(float(data.weather[wi_fut, :, :, 0].mean()))
    fut_temps = np.array(fut_temps)
    lo = np.percentile(fut_temps, 5)
    hi = np.percentile(fut_temps, 95)
    p45, p55 = np.percentile(fut_temps, [45, 55])
    return {
        "extreme_cold": fut_temps <= lo,
        "median":       (fut_temps >= p45) & (fut_temps <= p55),
        "extreme_heat": fut_temps >= hi,
    }


def load_pred(path):
    npz = np.load(path, allow_pickle=True)
    return {
        "preds":       npz["predictions"].astype(np.float32),
        "tgts":        npz["targets"].astype(np.float32),
        "start_hours": npz["start_hours"].astype(np.int64),
        "fut_hours":   npz["fut_hours"].astype(np.int64),
        "zones":       [str(z) for z in npz["zones"]],
    }


def shorten_label(path):
    return os.path.splitext(os.path.basename(path))[0].replace("preds_", "")


def main(pred_paths, S=72, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading prediction files …")
    preds_by_model = {shorten_label(p): load_pred(p) for p in pred_paths}

    # Use the first model's zones as canonical
    canonical = next(iter(preds_by_model.values()))
    zones = canonical["zones"]

    print("Loading Part3Data for weather buckets …")
    data = Part3Data()

    # Bucket masks share the same start_hours across all model files (same val set)
    start_hours = canonical["start_hours"]
    cal_buckets = bucket_windows(start_hours, S)
    print("  Bucketing windows by future-window weather …")
    wx_buckets = weather_buckets(data, start_hours, S)

    rows = []   # for results.csv

    # ── Overall + horizon ────────────────────────────────────────────────────
    print("\n=== Overall MAPE ===")
    overall = {}
    for name, p in preds_by_model.items():
        m = mape(p["tgts"], p["preds"])
        overall[name] = m
        rows.append({"model": name, "regime": "overall", "bucket": "all", "mape": m, "n": len(p["preds"])})
        print(f"  {name:40s}  {m:6.3f}%")

    print("\n=== Per-horizon MAPE (hour 1..24) ===")
    horizon = {name: per_horizon_mape(p["tgts"], p["preds"]) for name, p in preds_by_model.items()}
    for name, arr in horizon.items():
        for h, v in enumerate(arr):
            rows.append({"model": name, "regime": "horizon", "bucket": f"h{h+1:02d}", "mape": v, "n": len(preds_by_model[name]["preds"])})
        print(f"  {name}")
        print(f"    h01–h06: {arr[:6].round(3)}")
        print(f"    h07–h12: {arr[6:12].round(3)}")
        print(f"    h13–h18: {arr[12:18].round(3)}")
        print(f"    h19–h24: {arr[18:24].round(3)}")

    print("\n=== Per-zone MAPE ===")
    zone_mapes = {name: per_zone_mape(p["tgts"], p["preds"], p["zones"]) for name, p in preds_by_model.items()}
    for name, zm in zone_mapes.items():
        for z, v in zm.items():
            rows.append({"model": name, "regime": "zone", "bucket": z, "mape": v, "n": len(preds_by_model[name]["preds"])})
        print(f"  {name}: " + "  ".join(f"{z}={v:.3f}" for z, v in zm.items()))

    print("\n=== Calendar regime MAPE ===")
    cal_mapes = {}
    for name, p in preds_by_model.items():
        cal_mapes[name] = {}
        for label, mask in cal_buckets.items():
            n = int(mask.sum())
            if n == 0:
                cal_mapes[name][label] = (np.nan, n)
                continue
            v = mape(p["tgts"][mask], p["preds"][mask])
            cal_mapes[name][label] = (v, n)
            rows.append({"model": name, "regime": "calendar", "bucket": label, "mape": v, "n": n})
        print(f"  {name}: " + "  ".join(f"{lab}={v[0]:.3f}(n={v[1]})" for lab, v in cal_mapes[name].items()))

    print("\n=== Weather regime MAPE ===")
    wx_mapes = {}
    for name, p in preds_by_model.items():
        wx_mapes[name] = {}
        for label, mask in wx_buckets.items():
            n = int(mask.sum())
            if n == 0:
                wx_mapes[name][label] = (np.nan, n)
                continue
            v = mape(p["tgts"][mask], p["preds"][mask])
            wx_mapes[name][label] = (v, n)
            rows.append({"model": name, "regime": "weather", "bucket": label, "mape": v, "n": n})
        print(f"  {name}: " + "  ".join(f"{lab}={v[0]:.3f}(n={v[1]})" for lab, v in wx_mapes[name].items()))

    # ── Save CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    model_names = list(preds_by_model.keys())

    # Horizon
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, arr in horizon.items():
        ax.plot(np.arange(1, 25), arr, marker="o", label=name)
    ax.set_xlabel("Forecast horizon (hour ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Per-horizon MAPE on val 2023")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_horizon.png"), dpi=150)
    plt.close(fig)

    # Zone
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(zones))
    width = 0.8 / max(len(model_names), 1)
    for i, name in enumerate(model_names):
        vals = [zone_mapes[name][z] for z in zones]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(zones, rotation=20)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Per-zone MAPE on val 2023")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_zone.png"), dpi=150)
    plt.close(fig)

    # Calendar
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["holiday", "weekend", "weekday"]
    x = np.arange(len(cats))
    for i, name in enumerate(model_names):
        vals = [cal_mapes[name][c][0] for c in cats]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(cats)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by calendar regime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_calendar.png"), dpi=150)
    plt.close(fig)

    # Weather
    fig, ax = plt.subplots(figsize=(7, 4))
    cats = ["extreme_cold", "median", "extreme_heat"]
    for i, name in enumerate(model_names):
        vals = [wx_mapes[name][c][0] for c in cats]
        ax.bar(np.arange(len(cats)) + i * width, vals, width, label=name)
    ax.set_xticks(np.arange(len(cats)) + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(cats)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by future-window temperature regime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_weather.png"), dpi=150)
    plt.close(fig)

    print(f"Saved fig_horizon.png, fig_zone.png, fig_calendar.png, fig_weather.png to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pred_files", nargs="+",
                   help="One or more preds_*.npz files to compare")
    p.add_argument("--S", type=int, default=72)
    p.add_argument("--out_dir", default=".")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.pred_files, S=args.S, out_dir=args.out_dir)
