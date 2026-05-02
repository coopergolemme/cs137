"""
Run a HybridCNNTransformer checkpoint over all val 2023 windows.

Output:
    predictions:  (N, 24, Z) float32 — denormalised MWh
    targets:      (N, 24, Z) float32
    start_hours:  (N,) int64
    fut_hours:    (N, 24) int64
    zones:        (Z,) str

Optional:
    --perturbation climatology   replace weather with per-(m,d,h) climatology
                                 from train years 2019-2022.

Usage:
    python run_inference.py --ckpt ../checkpoints/lookback/best_model_lookback.pt \
                            --out preds_lookback.npz
    python run_inference.py --ckpt ../checkpoints/lookback/best_model_lookback.pt \
                            --out preds_lookback_clim.npz --perturbation climatology
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# Allow importing the existing model.py without changing sys.path globally
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # adds energy_forecasting/

from model import HybridCNNTransformer   # noqa: E402

from local_data import Part3Data   # noqa: E402


def build_model_from_ckpt(ckpt_path: str) -> HybridCNNTransformer:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = HybridCNNTransformer(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.set_demand_stats(ck["demand_mean"], ck["demand_std"])
    model.eval()
    return model, cfg


def climatology_lookup(data: Part3Data, hours: np.ndarray, clim_dict: dict) -> torch.Tensor:
    """
    For each int64 hour in `hours`, return the climatology weather as a
    stacked (len(hours), G, G, C) tensor.
    """
    dti = pd.to_datetime(hours.astype(np.int64), unit="h", utc=True)
    keys = list(zip(dti.month, dti.day, dti.hour))
    out = []
    for k in keys:
        if k in clim_dict:
            out.append(clim_dict[k])
        elif (k[0], 28, k[2]) in clim_dict and k[0] == 2 and k[1] == 29:
            # Feb 29 fallback to Feb 28
            out.append(clim_dict[(2, 28, k[2])])
        else:
            raise KeyError(f"No climatology for {k}")
    return torch.stack(out)


def collate_window(d: dict, weather_override: torch.Tensor = None) -> dict:
    """Add a batch dim of 1 to each tensor, optionally swapping weather."""
    w = d.copy()
    if weather_override is not None:
        S = w["hist_weather"].shape[0]
        w["hist_weather"] = weather_override[:S]
        w["future_weather"] = weather_override[S:]
    out = {}
    for k in ["hist_weather", "hist_demand", "hist_calendar",
              "future_weather", "future_calendar", "future_demand_lag",
              "target_demand"]:
        out[k] = w[k].unsqueeze(0)
    out["fut_hours"] = w["fut_hours"]
    out["start_hour"] = w["start_hour"]
    return out


def stack_batch(items: list) -> dict:
    """Stack a list of single-window dicts into a batch."""
    out = {}
    for k in ["hist_weather", "hist_demand", "hist_calendar",
              "future_weather", "future_calendar", "future_demand_lag",
              "target_demand"]:
        out[k] = torch.stack([it[k].squeeze(0) for it in items])
    out["fut_hours"] = np.stack([it["fut_hours"] for it in items])
    out["start_hours"] = np.array([it["start_hour"] for it in items], dtype=np.int64)
    return out


def run(ckpt_path: str, out_path: str, perturbation: str = None,
        batch_size: int = 16, year: int = 2023):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    model.to(device)
    S = cfg["S"]
    print(f"  S={S}  grid={cfg['grid_size']}  D={cfg['D']}  layers={cfg['num_transformer_layers']}")

    print("Loading local data …")
    data = Part3Data()
    starts = data.valid_starts_for_year(year, S=S)
    print(f"  val {year} windows: {len(starts):,}")

    clim = None
    if perturbation == "climatology":
        print("Computing climatology from train years 2019–2022 …")
        t0 = time.time()
        clim = data.compute_climatology()
        print(f"  ({time.time()-t0:.1f}s, {len(clim):,} (m,d,h) keys)")

    all_preds, all_tgts, all_starts, all_fut = [], [], [], []

    print(f"Running inference (batch_size={batch_size}) …")
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            batch_starts = starts[bi : bi + batch_size]
            items = []
            for s in batch_starts:
                w = data.get_window(int(s), S=S)
                if clim is not None:
                    all_hours = np.concatenate([
                        np.arange(int(s), int(s) + S, dtype=np.int64),
                        w["fut_hours"]
                    ])
                    override = climatology_lookup(data, all_hours, clim)
                    w["hist_weather"] = override[:S]
                    w["future_weather"] = override[S:]
                items.append(collate_window(w))

            b = stack_batch(items)
            pred = model(
                b["hist_weather"].to(device),
                b["hist_demand"].to(device),
                b["hist_calendar"].to(device),
                b["future_weather"].to(device),
                b["future_calendar"].to(device),
                b["future_demand_lag"].to(device),
            )

            all_preds.append(pred.cpu().numpy())
            all_tgts.append(b["target_demand"].numpy())
            all_starts.append(b["start_hours"])
            all_fut.append(b["fut_hours"])

            if (bi // batch_size) % 25 == 0:
                done = bi + len(batch_starts)
                rate = done / max(time.time() - t0, 1e-3)
                eta = (len(starts) - done) / max(rate, 1e-3)
                print(f"  {done:,}/{len(starts):,}  ({rate:.1f}/s, ETA {eta:.0f}s)")

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)
    start_hours = np.concatenate(all_starts)
    fut_hours = np.concatenate(all_fut)

    # Sanity: overall MAPE
    mape = float(np.mean(np.abs((targets - predictions) / (targets + 1e-6))) * 100)
    print(f"\nOverall MAPE: {mape:.4f}%")

    np.savez(out_path,
             predictions=predictions,
             targets=targets,
             start_hours=start_hours,
             fut_hours=fut_hours,
             zones=np.array(data.zones))
    print(f"Saved {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perturbation", choices=["climatology"], default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--year", type=int, default=2023)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.ckpt, args.out, args.perturbation, args.batch_size, args.year)
