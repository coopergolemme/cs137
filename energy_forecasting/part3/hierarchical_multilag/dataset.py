"""
EnergyDataset — loads ISO-NE demand CSVs and per-hour weather tensors.

Each sample covers a window of S historical hours followed by 24 future hours.
Weather maps are spatially pooled from (450, 449) to (grid_size, grid_size)
on load to keep memory usage tractable during training.

Directory layout expected:
    weather_base_path/
        {year}/
            X_{YYYYMMDDHH}.pt          # shape (450, 449, 7) float32

    demand_dir/
        target_energy_zonal_{year}.csv  # columns: timestamp_utc, ME, NH, …
"""

import os
import glob

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_US_HOLIDAYS = holidays.US()


# ─── Calendar helper ────────────────────────────────────────────────────────

def compute_calendar_features(hours_int64: np.ndarray) -> np.ndarray:
    """
    Convert an array of int64 hours-since-epoch values into cyclical calendar
    features.

    Args:
        hours_int64: (N,) int64 — hours since Unix epoch (UTC)

    Returns:
        (N, 8) float32:
            [hour_sin, hour_cos, dow_sin, dow_cos,
             month_sin, month_cos, is_weekend, is_holiday]
    """
    dti = pd.to_datetime(hours_int64.astype(np.int64), unit="h", utc=True)
    N = len(hours_int64)
    out = np.empty((N, 8), dtype=np.float32)
    out[:, 0] = np.sin(2 * np.pi * dti.hour / 24)
    out[:, 1] = np.cos(2 * np.pi * dti.hour / 24)
    out[:, 2] = np.sin(2 * np.pi * dti.dayofweek / 7)
    out[:, 3] = np.cos(2 * np.pi * dti.dayofweek / 7)
    out[:, 4] = np.sin(2 * np.pi * (dti.month - 1) / 12)
    out[:, 5] = np.cos(2 * np.pi * (dti.month - 1) / 12)
    out[:, 6] = (dti.dayofweek >= 5).astype(np.float32)  # is_weekend
    dates = dti.normalize().date
    out[:, 7] = np.array([float(d in _US_HOLIDAYS) for d in dates], dtype=np.float32)
    return out


# ─── Dataset ─────────────────────────────────────────────────────────────────

class EnergyDataset(Dataset):
    """
    Dataset for day-ahead energy demand forecasting.

    A window belongs to split_years if the *first future hour* falls in one
    of those years.  This lets val windows use the last S hours of the
    preceding year as historical context.

    Args:
        weather_base_path: root directory containing year subdirectories of
                           weather .pt files
        demand_dir:        directory containing per-year demand CSVs
        S:                 historical lookback in hours (≤ 168)
        grid_size:         spatial downsampling target for weather maps
        split_years:       which calendar years count as this split

    Returns per __getitem__:
        hist_weather:       (S, grid_size, grid_size, C)  float32
        hist_demand:        (S, Z)                        float32
        hist_calendar:      (S, 8)                        float32
        future_weather:     (24, grid_size, grid_size, C) float32
        future_calendar:    (24, 8)                       float32
        future_demand_lag:  (24, Z)  demand from exactly 168 h prior  float32
        target_demand:      (24, Z)                       float32
    """

    def __init__(self, weather_base_path: str, demand_dir: str,
                 S: int, grid_size: int, split_years: list):
        self.weather_base_path = weather_base_path
        self.S = S
        self.H = 24
        self.grid_size = grid_size

        # ── Load full multi-year demand CSV ──────────────────────────────────
        csv_paths = sorted(glob.glob(os.path.join(demand_dir, "target_energy_zonal_*.csv")))
        if not csv_paths:
            raise FileNotFoundError(f"No demand CSVs found in {demand_dir}")

        dfs = [pd.read_csv(p, parse_dates=["timestamp_utc"]) for p in csv_paths]
        demand_df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)

        self.zone_cols = [c for c in demand_df.columns if c != "timestamp_utc"]
        self.demand_values = demand_df[self.zone_cols].values.astype(np.float32)  # (T, Z)

        # Convert timestamps to hours-since-UTC-epoch (int64) — same units as
        # evaluate.py's future_time.
        dti = pd.DatetimeIndex(demand_df["timestamp_utc"])
        if dti.tz is None:
            dti = dti.tz_localize("UTC")
        self.hours_since_epoch = (dti.astype(np.int64) // (10 ** 9 * 3600)).values  # (T,)

        # Precompute calendar features for every row in the full demand table.
        self.calendar = compute_calendar_features(self.hours_since_epoch)  # (T, 7)

        # ── Identify valid window starts ─────────────────────────────────────
        # A window of total length S+24 starting at index i is valid when:
        #   1. There are enough rows after i (i + S + 24 <= T)
        #   2. The first future hour (at index i+S) is in split_years
        #   3. The 168-hour lag is available (i + S >= 168)
        T = len(demand_df)
        window_size = S + self.H
        future_years = pd.DatetimeIndex(demand_df["timestamp_utc"]).year.values

        self.valid_starts = [
            i for i in range(T - window_size + 1)
            if future_years[i + S] in split_years and i + S >= 168
        ]

        if len(self.valid_starts) == 0:
            raise ValueError(
                f"No valid windows for split_years={split_years} with S={S}. "
                "Check that the demand CSVs cover the requested years."
            )

        # Preload all needed weather hours into memory as pooled tensors.
        # Raw files are 450×449×7 (~5.6 MB each) but pool to grid_size²×7
        # (~700 B each), so the entire cache fits easily in RAM.
        needed_hours = set()
        for start in self.valid_starts:
            needed_hours.update(self.hours_since_epoch[start:start + S + self.H].tolist())
        print(f"  Preloading {len(needed_hours):,} weather files …")
        self._weather_cache = {}
        for hour in sorted(needed_hours):
            self._weather_cache[hour] = self._load_weather_from_disk(hour)
        print("  Weather cache ready.")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def _load_weather_from_disk(self, hour: int) -> torch.Tensor:
        dt = pd.Timestamp(int(hour), unit="h", tz="UTC")
        path = os.path.join(
            self.weather_base_path,
            str(dt.year),
            f"X_{dt.strftime('%Y%m%d%H')}.pt",
        )
        t = torch.load(path, weights_only=True).float()   # (450, 449, C)
        t = t.permute(2, 0, 1).unsqueeze(0)               # (1, C, 450, 449)
        t = F.adaptive_avg_pool2d(t, (self.grid_size, self.grid_size))
        return t.squeeze(0).permute(1, 2, 0)              # (G, G, C)

    def __getitem__(self, idx: int):
        start = self.valid_starts[idx]
        hi = slice(start, start + self.S)            # history indices
        fi = slice(start + self.S, start + self.S + self.H)  # future indices

        # ── Demand ───────────────────────────────────────────────────────────
        hist_demand   = torch.from_numpy(self.demand_values[hi])   # (S, Z)
        target_demand = torch.from_numpy(self.demand_values[fi])   # (24, Z)

        # 168-h weekly lag — one-week-prior demand.
        lag_168_start = start + self.S - 168
        future_demand_lag_168 = torch.from_numpy(
            self.demand_values[lag_168_start : lag_168_start + self.H]     # (24, Z)
        )
        # 24-h lag — last 24 hours of history (yesterday-same-hour demand).
        lag_24_start = start + self.S - 24
        future_demand_lag_24 = torch.from_numpy(
            self.demand_values[lag_24_start : lag_24_start + self.H]       # (24, Z)
        )

        # ── Calendar ─────────────────────────────────────────────────────────
        hist_calendar   = torch.from_numpy(self.calendar[hi])      # (S, 7)
        future_calendar = torch.from_numpy(self.calendar[fi])      # (24, 7)

        # ── Weather (load and pool each hour) ────────────────────────────────
        hist_hours   = self.hours_since_epoch[hi]
        future_hours = self.hours_since_epoch[fi]

        hist_weather = torch.stack(
            [self._weather_cache[int(h)] for h in hist_hours]
        )   # (S, G, G, C)
        future_weather = torch.stack(
            [self._weather_cache[int(h)] for h in future_hours]
        )   # (24, G, G, C)

        return (
            hist_weather,
            hist_demand,
            hist_calendar,
            future_weather,
            future_calendar,
            future_demand_lag_168,
            future_demand_lag_24,
            target_demand,
        )
