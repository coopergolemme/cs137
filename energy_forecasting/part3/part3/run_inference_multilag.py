"""
Inference for the multi-lag HybridCNNTransformerMultilag checkpoint.

Mirrors run_inference.py but threads the second (24-h) lag tensor and
imports the multi-lag model class.

Usage:
    python run_inference_multilag.py \
        --ckpt /path/to/best_model_multilag.pt \
        --out preds_multilag.npz
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Add the multi-lag dir for its custom model class
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "multilag"))

from model import HybridCNNTransformerMultilag   # noqa: E402

from local_data import Part3Data   # noqa: E402


def build_model_from_ckpt(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = HybridCNNTransformerMultilag(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.set_demand_stats(ck["demand_mean"], ck["demand_std"])
    model.eval()
    return model, cfg


def get_window_with_24h_lag(data: Part3Data, start_hour: int, S: int):
    """Same as data.get_window but also returns lag_24 (last 24 h of history)."""
    di = data.h2d_idx[int(start_hour)]
    fut_d_hours = data.demand_hours[di + S : di + S + 24]
    hist_d_hours = data.demand_hours[di : di + S]

    hist_demand   = torch.from_numpy(data.demand[di : di + S])
    target_demand = torch.from_numpy(data.demand[di + S : di + S + 24])

    # 168-h lag = first 24 h of the (most recent) 168-h history
    # which is rows di + S - 168 .. di + S - 145
    future_demand_lag_168 = torch.from_numpy(
        data.demand[di + S - 168 : di + S - 144]
    )
    # 24-h lag = last 24 h of history
    future_demand_lag_24 = torch.from_numpy(
        data.demand[di + S - 24 : di + S]
    )

    from local_data import compute_calendar_features
    hist_calendar   = torch.from_numpy(compute_calendar_features(hist_d_hours))
    future_calendar = torch.from_numpy(compute_calendar_features(fut_d_hours))

    wi_hist = [data.h2w_idx[int(h)] for h in hist_d_hours]
    wi_fut  = [data.h2w_idx[int(h)] for h in fut_d_hours]
    hist_weather   = data.weather[wi_hist]
    future_weather = data.weather[wi_fut]

    return {
        "hist_weather": hist_weather,
        "hist_demand":  hist_demand,
        "hist_calendar": hist_calendar,
        "future_weather": future_weather,
        "future_calendar": future_calendar,
        "future_demand_lag_168": future_demand_lag_168,
        "future_demand_lag_24":  future_demand_lag_24,
        "target_demand": target_demand,
        "fut_hours": fut_d_hours,
        "start_hour": int(start_hour),
    }


def stack_batch(items):
    out = {}
    for k in ["hist_weather", "hist_demand", "hist_calendar",
              "future_weather", "future_calendar",
              "future_demand_lag_168", "future_demand_lag_24", "target_demand"]:
        out[k] = torch.stack([it[k] for it in items])
    out["fut_hours"]   = np.stack([it["fut_hours"] for it in items])
    out["start_hours"] = np.array([it["start_hour"] for it in items], dtype=np.int64)
    return out


def run(ckpt_path, out_path, batch_size=16, year=2023):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    model.to(device)
    S = cfg["S"]
    print(f"  S={S}  grid={cfg['grid_size']}  D={cfg['D']}  layers={cfg['num_transformer_layers']}")
    print(f"  num_lags={cfg.get('num_lags', 2)}")

    print("Loading local data …")
    data = Part3Data()
    starts = data.valid_starts_for_year(year, S=S)
    print(f"  val {year} windows: {len(starts):,}")

    all_preds, all_tgts, all_starts, all_fut = [], [], [], []
    t0 = time.time()
    print(f"Running inference (batch_size={batch_size}) …")
    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            batch_starts = starts[bi : bi + batch_size]
            items = [get_window_with_24h_lag(data, int(s), S) for s in batch_starts]
            b = stack_batch(items)
            pred = model(
                b["hist_weather"].to(device),
                b["hist_demand"].to(device),
                b["hist_calendar"].to(device),
                b["future_weather"].to(device),
                b["future_calendar"].to(device),
                b["future_demand_lag_168"].to(device),
                b["future_demand_lag_24"].to(device),
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
    targets     = np.concatenate(all_tgts)
    start_hours = np.concatenate(all_starts)
    fut_hours   = np.concatenate(all_fut)

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
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--year", type=int, default=2023)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.ckpt, args.out, args.batch_size, args.year)
