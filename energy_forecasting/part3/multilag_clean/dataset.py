"""
Multi-lag EnergyDataset.

Same as the lookback dataset, but each window now also returns a 24-hour
lag tensor (yesterday-same-hour demand) alongside the existing 168-hour
weekly lag.

Window definition (length S+24):
    history hours: t - S    .. t - 1
    future  hours: t        .. t + 23     (length 24)

Lag tensors (each shape (24, Z), aligned to future hours):
    future_demand_lag_168:  hours t-168 .. t-145
    future_demand_lag_24:   hours t-24  .. t-1     (last 24 h of history)

Both are inside the 168-h history window, so the model is evaluator-safe.
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


def compute_calendar_features(hours_int64: np.ndarray) -> np.ndarray:
    dti = pd.to_datetime(hours_int64.astype(np.int64), unit="h", utc=True)
    N = len(hours_int64)
    out = np.empty((N, 8), dtype=np.float32)
    out[:, 0] = np.sin(2 * np.pi * dti.hour / 24)
    out[:, 1] = np.cos(2 * np.pi * dti.hour / 24)
    out[:, 2] = np.sin(2 * np.pi * dti.dayofweek / 7)
    out[:, 3] = np.cos(2 * np.pi * dti.dayofweek / 7)
    out[:, 4] = np.sin(2 * np.pi * (dti.month - 1) / 12)
    out[:, 5] = np.cos(2 * np.pi * (dti.month - 1) / 12)
    out[:, 6] = (dti.dayofweek >= 5).astype(np.float32)
    dates = dti.normalize().date
    out[:, 7] = np.array([float(d in _US_HOLIDAYS) for d in dates], dtype=np.float32)
    return out


class EnergyDataset(Dataset):
    def __init__(self, weather_base_path: str, demand_dir: str,
                 S: int, grid_size: int, split_years: list):
        self.weather_base_path = weather_base_path
        self.S = S
        self.H = 24
        self.grid_size = grid_size

        csv_paths = sorted(glob.glob(os.path.join(demand_dir, "target_energy_zonal_*.csv")))
        if not csv_paths:
            raise FileNotFoundError(f"No demand CSVs found in {demand_dir}")
        dfs = [pd.read_csv(p, parse_dates=["timestamp_utc"]) for p in csv_paths]
        demand_df = pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)

        self.zone_cols = [c for c in demand_df.columns if c != "timestamp_utc"]
        self.demand_values = demand_df[self.zone_cols].values.astype(np.float32)

        dti = pd.DatetimeIndex(demand_df["timestamp_utc"])
        if dti.tz is None:
            dti = dti.tz_localize("UTC")
        self.hours_since_epoch = (dti.astype(np.int64) // (10 ** 9 * 3600)).values
        self.calendar = compute_calendar_features(self.hours_since_epoch)

        T = len(demand_df)
        window_size = S + self.H
        future_years = pd.DatetimeIndex(demand_df["timestamp_utc"]).year.values
        self.valid_starts = [
            i for i in range(T - window_size + 1)
            if future_years[i + S] in split_years and i + S >= 168
        ]
        if not self.valid_starts:
            raise ValueError(
                f"No valid windows for split_years={split_years} with S={S}."
            )

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
        t = torch.load(path, weights_only=True).float()
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = F.adaptive_avg_pool2d(t, (self.grid_size, self.grid_size))
        return t.squeeze(0).permute(1, 2, 0)

    def __getitem__(self, idx: int):
        start = self.valid_starts[idx]
        hi = slice(start, start + self.S)
        fi = slice(start + self.S, start + self.S + self.H)

        hist_demand   = torch.from_numpy(self.demand_values[hi])
        target_demand = torch.from_numpy(self.demand_values[fi])

        lag_168_start = start + self.S - 168
        future_demand_lag_168 = torch.from_numpy(
            self.demand_values[lag_168_start : lag_168_start + self.H]
        )
        # 24-hour lag: hours t-24..t-1 = last 24 of the historical window
        lag_24_start = start + self.S - 24
        future_demand_lag_24 = torch.from_numpy(
            self.demand_values[lag_24_start : lag_24_start + self.H]
        )

        hist_calendar   = torch.from_numpy(self.calendar[hi])
        future_calendar = torch.from_numpy(self.calendar[fi])

        hist_hours   = self.hours_since_epoch[hi]
        future_hours = self.hours_since_epoch[fi]
        hist_weather = torch.stack([self._weather_cache[int(h)] for h in hist_hours])
        future_weather = torch.stack([self._weather_cache[int(h)] for h in future_hours])

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
