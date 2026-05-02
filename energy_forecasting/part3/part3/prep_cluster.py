"""
Cluster-side preprocessing for Part 3 diagnostic study.

Loads every hourly weather tensor for 2019..2023, spatial-pools each to 5x5
(matching the lookback model's grid_size), and saves a single compact tensor
plus the int64 hours-since-epoch index. Also bundles demand CSVs into one
.npz so the laptop side can do everything else without the cluster.

Output (~few hundred MB total — small enough to scp):
    weather_pooled_2019_2023.pt   (T, 5, 5, 7) float32 + meta
    demand_all.npz                demand values + timestamps + zone names

Run via SLURM (see prep_cluster.slurm), not on the login node.
"""

import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DATA_BASE = "/cluster/tufts/c26sp1cs0137/data/assignment3_data"
WEATHER_DIR = os.path.join(DATA_BASE, "weather_data")
DEMAND_DIR = os.path.join(DATA_BASE, "energy_demand_data")

OUT_DIR = "/cluster/tufts/c26sp1cs0137/kkapli01/part3_prep_out"
GRID = 5
YEARS = [2019, 2020, 2021, 2022, 2023]


def pool_weather_file(path: str, grid: int) -> torch.Tensor:
    """Load one (450,449,C) weather .pt file and pool to (grid,grid,C)."""
    t = torch.load(path, weights_only=True).float()      # (H, W, C)
    t = t.permute(2, 0, 1).unsqueeze(0)                  # (1, C, H, W)
    t = F.adaptive_avg_pool2d(t, (grid, grid))           # (1, C, G, G)
    return t.squeeze(0).permute(1, 2, 0).contiguous()    # (G, G, C)


def load_all_weather():
    """
    Walk weather_data/{year}/X_YYYYMMDDHH.pt for every year in YEARS,
    pool each to GRID, return a single tensor + matched int64 hour index.
    """
    rows = []                                 # list of (hour_int64, Tensor)
    for year in YEARS:
        files = sorted(glob.glob(os.path.join(WEATHER_DIR, str(year), "X_*.pt")))
        print(f"  {year}: {len(files):,} files")
        sys.stdout.flush()
        t0 = time.time()
        for i, path in enumerate(files):
            stem = os.path.basename(path)[2:-3]   # YYYYMMDDHH
            dt = pd.to_datetime(stem, format="%Y%m%d%H", utc=True)
            hour = int(dt.value // (10 ** 9 * 3600))
            rows.append((hour, pool_weather_file(path, GRID)))
            if (i + 1) % 1000 == 0:
                print(f"    {i+1:,}/{len(files):,}  ({time.time()-t0:.0f}s)")
                sys.stdout.flush()
        print(f"  {year} done in {time.time()-t0:.0f}s")
        sys.stdout.flush()

    rows.sort(key=lambda r: r[0])
    hours = torch.tensor([r[0] for r in rows], dtype=torch.int64)
    weather = torch.stack([r[1] for r in rows]).float()    # (T, G, G, C)
    return hours, weather


def load_all_demand():
    """Concat every per-year demand CSV into one DataFrame, save to .npz."""
    csvs = sorted(glob.glob(os.path.join(DEMAND_DIR, "target_energy_zonal_*.csv")))
    dfs = [pd.read_csv(p, parse_dates=["timestamp_utc"]) for p in csvs]
    df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)
    zone_cols = [c for c in df.columns if c != "timestamp_utc"]
    demand = df[zone_cols].values.astype(np.float32)            # (T, Z)

    dti = pd.DatetimeIndex(df["timestamp_utc"])
    if dti.tz is None:
        dti = dti.tz_localize("UTC")
    hours = (dti.astype(np.int64) // (10 ** 9 * 3600)).values   # (T,)
    return hours, demand, zone_cols


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    print("Loading + pooling weather …")
    t0 = time.time()
    hours, weather = load_all_weather()
    print(f"  weather: {weather.shape}  ({time.time()-t0:.0f}s)")

    out_w = os.path.join(OUT_DIR, "weather_pooled_2019_2023.pt")
    torch.save({"hours": hours, "weather": weather, "grid": GRID}, out_w)
    print(f"  saved {out_w}  ({os.path.getsize(out_w)/1e6:.1f} MB)")

    print("Loading demand …")
    d_hours, demand, zone_cols = load_all_demand()
    out_d = os.path.join(OUT_DIR, "demand_all.npz")
    np.savez(out_d, hours=d_hours, demand=demand, zones=np.array(zone_cols))
    print(f"  saved {out_d}  ({os.path.getsize(out_d)/1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
