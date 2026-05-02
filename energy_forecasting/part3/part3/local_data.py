"""
Local data loader for Part 3 analysis.

Reads the cluster-prepped bundle and exposes one method to build a window
(hist + future) for any start hour. Used by run_inference.py and
run_perturbation.py.
"""

import os

import holidays
import numpy as np
import pandas as pd
import torch

_US_HOLIDAYS = holidays.US()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
S_DEFAULT = 72
H = 24
LAG_HOURS = 168


def compute_calendar_features(hours_int64: np.ndarray) -> np.ndarray:
    """Same 8-D cyclical encoding as energy_forecasting/dataset.py."""
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


class Part3Data:
    """
    Indexed view of pooled weather + demand for any year.

    Attributes:
        weather:    (T_w, G, G, C) float32, hours-aligned with weather_hours
        weather_hours: (T_w,) int64
        demand:     (T_d, Z) float32
        demand_hours:  (T_d,) int64
        zones:      list[str] of length Z
        h2w_idx:    dict hour_int64 -> index into weather (None if missing)
        h2d_idx:    dict hour_int64 -> index into demand   (None if missing)
    """

    def __init__(self, data_dir: str = DATA_DIR):
        wpath = os.path.join(data_dir, "weather_pooled_2019_2023.pt")
        dpath = os.path.join(data_dir, "demand_all.npz")

        wbundle = torch.load(wpath, weights_only=True)
        self.weather = wbundle["weather"].float()           # (T_w, G, G, C)
        self.weather_hours = wbundle["hours"].numpy().astype(np.int64)  # (T_w,)
        self.grid = int(wbundle["grid"])

        d = np.load(dpath, allow_pickle=True)
        self.demand = d["demand"].astype(np.float32)        # (T_d, Z)
        self.demand_hours = d["hours"].astype(np.int64)     # (T_d,)
        self.zones = list(d["zones"])

        # hour -> index lookups
        self.h2w_idx = {int(h): i for i, h in enumerate(self.weather_hours)}
        self.h2d_idx = {int(h): i for i, h in enumerate(self.demand_hours)}

    # ── Window generation ───────────────────────────────────────────────────

    def valid_starts_for_year(self, year: int, S: int = S_DEFAULT) -> np.ndarray:
        """
        Return all start_hour values such that:
          - first future hour falls in `year`
          - 168-h weekly lag is available
          - all S+24 weather hours are present in our pooled cache
        """
        # demand-side constraint
        d_year = pd.to_datetime(self.demand_hours, unit="h", utc=True).year
        starts = []
        for i in range(len(self.demand_hours) - S - H + 1):
            if d_year[i + S] != year:
                continue
            if i + S < LAG_HOURS:
                continue
            window_hours = self.demand_hours[i : i + S + H]
            lag_hour = self.demand_hours[i + S - LAG_HOURS]
            need = list(window_hours) + [self.demand_hours[i + S - LAG_HOURS + j] for j in range(H)]
            if any(int(h) not in self.h2w_idx for h in window_hours):
                continue
            starts.append(int(self.demand_hours[i]))
        return np.array(starts, dtype=np.int64)

    def get_window(self, start_hour: int, S: int = S_DEFAULT,
                   weather_override: torch.Tensor = None):
        """
        Build the 7 tensors the lookback model expects, for the window
        starting at `start_hour`.

        weather_override: if given, must be (S + H, G, G, C). Replaces both
                          hist and future weather (used for climatology test).

        Returns dict of CPU tensors (no batch dim).
        """
        # Locate the window in the demand table
        di = self.h2d_idx[int(start_hour)]
        hist_d_hours = self.demand_hours[di : di + S]
        fut_d_hours = self.demand_hours[di + S : di + S + H]
        lag_d_hours = self.demand_hours[di + S - LAG_HOURS : di + S - LAG_HOURS + H]

        hist_demand   = torch.from_numpy(self.demand[di : di + S])
        target_demand = torch.from_numpy(self.demand[di + S : di + S + H])
        future_demand_lag = torch.from_numpy(self.demand[di + S - LAG_HOURS : di + S - LAG_HOURS + H])
        future_demand_lag_24 = torch.from_numpy(self.demand[di + S - 24 : di + S - 24 + H])

        hist_calendar   = torch.from_numpy(compute_calendar_features(hist_d_hours))
        future_calendar = torch.from_numpy(compute_calendar_features(fut_d_hours))

        # Weather
        if weather_override is not None:
            assert weather_override.shape[0] == S + H, \
                f"weather_override must be (S+H, G, G, C), got {weather_override.shape}"
            hist_weather   = weather_override[:S]
            future_weather = weather_override[S:]
        else:
            wi_hist = [self.h2w_idx[int(h)] for h in hist_d_hours]
            wi_fut  = [self.h2w_idx[int(h)] for h in fut_d_hours]
            hist_weather   = self.weather[wi_hist]
            future_weather = self.weather[wi_fut]

        return {
            "hist_weather":         hist_weather,
            "hist_demand":          hist_demand,
            "hist_calendar":        hist_calendar,
            "future_weather":       future_weather,
            "future_calendar":      future_calendar,
            "future_demand_lag":    future_demand_lag,
            "future_demand_lag_24": future_demand_lag_24,
            "target_demand":        target_demand,
            "fut_hours":            fut_d_hours,
            "start_hour":           int(start_hour),
        }

    # ── Climatology helper ──────────────────────────────────────────────────

    def compute_climatology(self, train_years=(2019, 2020, 2021, 2022)) -> dict:
        """
        Per (month, day, hour) mean weather over train_years.
        Returns dict (month, day, hour) -> Tensor (G, G, C).
        Uses MM-DD-HH key so leap-day Feb 29 is handled if present.
        """
        dti = pd.to_datetime(self.weather_hours, unit="h", utc=True)
        years = dti.year.values
        m = dti.month.values
        d = dti.day.values
        h = dti.hour.values

        mask = np.isin(years, list(train_years))
        keys = list(zip(m[mask], d[mask], h[mask]))
        idxs = np.where(mask)[0]

        bucket = {}
        for k, idx in zip(keys, idxs):
            bucket.setdefault(k, []).append(idx)

        clim = {k: self.weather[v].mean(dim=0) for k, v in bucket.items()}
        return clim


if __name__ == "__main__":
    # sanity check
    pd_obj = Part3Data()
    print(f"weather: {tuple(pd_obj.weather.shape)}  hours: {len(pd_obj.weather_hours):,}")
    print(f"demand:  {tuple(pd_obj.demand.shape)}    hours: {len(pd_obj.demand_hours):,}  zones: {pd_obj.zones}")
    starts = pd_obj.valid_starts_for_year(2023)
    print(f"val 2023 valid starts: {len(starts):,}")
    w = pd_obj.get_window(int(starts[0]))
    for k, v in w.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k}: {v}")
