"""
Inference for Cooper's HierarchicalCNNTransformer (Part 2 model).

Mirrors run_inference.py but imports the hierarchical class. Same
forward signature (6 inputs), so we can reuse the data adapter.

Usage:
    python run_inference_hierarchical.py \
        --ckpt ../hierarchical/best_model_hierarchical.pt \
        --out preds_hierarchical_2022.npz --year 2022

    # with climatology perturbation (uses train_years=[2019,2020,2021,2023])
    python run_inference_hierarchical.py \
        --ckpt ../hierarchical/best_model_hierarchical.pt \
        --out preds_hierarchical_clim_2022.npz --year 2022 \
        --perturbation climatology
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hierarchical"))

from model import HierarchicalCNNTransformer   # noqa: E402

from local_data import Part3Data   # noqa: E402


def build_model_from_ckpt(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = HierarchicalCNNTransformer(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.set_demand_stats(ck["demand_mean"], ck["demand_std"])
    model.eval()
    return model, cfg


def climatology_lookup(hours: np.ndarray, clim_dict: dict) -> torch.Tensor:
    dti = pd.to_datetime(hours.astype(np.int64), unit="h", utc=True)
    keys = list(zip(dti.month, dti.day, dti.hour))
    out = []
    for k in keys:
        if k in clim_dict:
            out.append(clim_dict[k])
        elif k[0] == 2 and k[1] == 29 and (2, 28, k[2]) in clim_dict:
            out.append(clim_dict[(2, 28, k[2])])
        else:
            raise KeyError(f"No climatology for {k}")
    return torch.stack(out)


def run(ckpt_path, out_path, perturbation=None, batch_size=8, year=2022):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    model.to(device)
    S = cfg["S"]
    train_years = tuple(cfg.get("train_years", [2019, 2020, 2021, 2023]))
    print(f"  S={S}  grid={cfg['grid_size']}  D={cfg['D']}")
    print(f"  train_years={train_years}  val_years={cfg.get('val_years')}")

    print("Loading local data …")
    data = Part3Data()
    starts = data.valid_starts_for_year(year, S=S)
    print(f"  test {year} windows: {len(starts):,}")

    clim = None
    if perturbation == "climatology":
        print(f"Computing climatology from train years {train_years} …")
        t0 = time.time()
        clim = data.compute_climatology(train_years=train_years)
        print(f"  ({time.time()-t0:.1f}s, {len(clim):,} (m,d,h) keys)")

    all_preds, all_tgts, all_starts, all_fut = [], [], [], []
    print(f"Running inference (batch_size={batch_size}) …")
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            batch_starts = starts[bi : bi + batch_size]
            items = [data.get_window(int(s), S=S) for s in batch_starts]
            if clim is not None:
                for w in items:
                    all_hours = np.concatenate([
                        np.arange(w["start_hour"], w["start_hour"] + S, dtype=np.int64),
                        w["fut_hours"],
                    ])
                    override = climatology_lookup(all_hours, clim)
                    w["hist_weather"] = override[:S]
                    w["future_weather"] = override[S:]

            hist_w = torch.stack([w["hist_weather"]   for w in items]).to(device)
            hist_d = torch.stack([w["hist_demand"]    for w in items]).to(device)
            hist_c = torch.stack([w["hist_calendar"]  for w in items]).to(device)
            fut_w  = torch.stack([w["future_weather"] for w in items]).to(device)
            fut_c  = torch.stack([w["future_calendar"] for w in items]).to(device)
            lag    = torch.stack([w["future_demand_lag"] for w in items]).to(device)
            tgt    = torch.stack([w["target_demand"]   for w in items])

            pred = model(hist_w, hist_d, hist_c, fut_w, fut_c, lag)

            all_preds.append(pred.cpu().numpy())
            all_tgts.append(tgt.numpy())
            all_starts.append(np.array([w["start_hour"] for w in items], dtype=np.int64))
            all_fut.append(np.stack([w["fut_hours"] for w in items]))

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
             predictions=predictions, targets=targets,
             start_hours=start_hours, fut_hours=fut_hours,
             zones=np.array(data.zones))
    print(f"Saved {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--perturbation", choices=["climatology"], default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--year", type=int, default=2022)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.ckpt, args.out, args.perturbation, args.batch_size, args.year)
