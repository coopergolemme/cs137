"""
Final Part 3 analysis pass.

  • Bootstrap 90% CIs for overall MAPE and per-zone Δ-MAPE
    (real-weather − climatology). 1,000 resamples over windows.

  • One headline figure: per-zone Δ-MAPE with CIs, ranked.

  • One figure summarising horizon × weather-condition contribution.

  • A summary JSON (numbers ready to paste into the writeup).

Usage (from energy_forecasting/part3/):
    python final_analysis.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


REAL_PATH = "preds_lookback.npz"
CLIM_PATH = "preds_lookback_clim.npz"
OUT_DIR = "analysis_out"
N_BOOT = 1000
SEED = 137


def load(path):
    npz = np.load(path, allow_pickle=True)
    return {
        "preds":       npz["predictions"].astype(np.float32),
        "tgts":        npz["targets"].astype(np.float32),
        "start_hours": npz["start_hours"].astype(np.int64),
        "fut_hours":   npz["fut_hours"].astype(np.int64),
        "zones":       [str(z) for z in npz["zones"]],
    }


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / (y + 1e-6))) * 100)


def per_window_mape(preds, tgts):
    """(N,) MAPE per window."""
    return np.mean(np.abs((tgts - preds) / (tgts + 1e-6)) * 100, axis=(1, 2))


def per_window_zone_mape(preds, tgts, j):
    """(N,) MAPE per window for zone j."""
    return np.mean(np.abs((tgts[:, :, j] - preds[:, :, j]) / (tgts[:, :, j] + 1e-6)) * 100, axis=1)


def bootstrap_ci(values_per_window, n_boot=N_BOOT, seed=SEED):
    """
    Bootstrap 90% CI of the mean of per-window MAPEs.
    Resampling at the WINDOW level (not per-element) is the correct
    treatment because adjacent windows share weather/demand and aren't
    independent samples.
    """
    rng = np.random.default_rng(seed)
    n = len(values_per_window)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values_per_window[idx].mean()
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading predictions …")
    real = load(REAL_PATH)
    clim = load(CLIM_PATH)
    zones = real["zones"]
    Z = len(zones)
    n = len(real["preds"])

    # Sanity: same start_hours
    assert np.array_equal(real["start_hours"], clim["start_hours"])

    # ── Per-window per-overall MAPE ──────────────────────────────────────────
    real_w = per_window_mape(real["preds"], real["tgts"])
    clim_w = per_window_mape(clim["preds"], clim["tgts"])
    delta_w = clim_w - real_w   # cost of climatology

    overall_real = float(real_w.mean())
    overall_clim = float(clim_w.mean())
    overall_delta = overall_clim - overall_real

    real_lo, real_hi = bootstrap_ci(real_w)
    clim_lo, clim_hi = bootstrap_ci(clim_w)
    delta_lo, delta_hi = bootstrap_ci(delta_w)

    print("\n=== Bootstrap 90% CIs (1,000 resamples) ===")
    print(f"Lookback (real weather):  {overall_real:.3f}%  CI [{real_lo:.3f}, {real_hi:.3f}]")
    print(f"Lookback (climatology):   {overall_clim:.3f}%  CI [{clim_lo:.3f}, {clim_hi:.3f}]")
    print(f"Δ (weather contribution): {overall_delta:.3f} pp CI [{delta_lo:.3f}, {delta_hi:.3f}]")

    # ── Per-zone Δ-MAPE with CI ──────────────────────────────────────────────
    print("\n=== Per-zone Δ-MAPE with bootstrap CIs ===")
    zone_results = []
    for j, z in enumerate(zones):
        rw = per_window_zone_mape(real["preds"], real["tgts"], j)
        cw = per_window_zone_mape(clim["preds"], clim["tgts"], j)
        d = cw - rw
        m_real = float(rw.mean())
        m_clim = float(cw.mean())
        m_delta = float(d.mean())
        d_lo, d_hi = bootstrap_ci(d)
        zone_results.append({
            "zone": z, "real": m_real, "clim": m_clim,
            "delta": m_delta, "delta_lo": d_lo, "delta_hi": d_hi,
        })
        print(f"  {z:10s}  real={m_real:5.2f}  clim={m_clim:5.2f}  "
              f"Δ={m_delta:5.2f}pp  CI [{d_lo:5.2f}, {d_hi:5.2f}]")

    # Sort by delta descending for the figure
    zone_results_sorted = sorted(zone_results, key=lambda r: -r["delta"])

    # ── Headline figure: per-zone Δ-MAPE with CI ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    names = [r["zone"] for r in zone_results_sorted]
    deltas = np.array([r["delta"] for r in zone_results_sorted])
    los = np.array([r["delta_lo"] for r in zone_results_sorted])
    his = np.array([r["delta_hi"] for r in zone_results_sorted])
    err = np.stack([deltas - los, his - deltas])
    bars = ax.bar(names, deltas, yerr=err, capsize=4, color="#6794a7", edgecolor="#33586a")
    ax.set_ylabel("Δ MAPE (climatology − real weather)  [pp]")
    ax.set_title("Per-zone weather contribution\n"
                 "Higher bar = more dependent on real weather signal")
    ax.axhline(overall_delta, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Overall Δ = {overall_delta:.2f} pp")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_delta_per_zone.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/fig_delta_per_zone.png")

    # ── Per-horizon real vs clim figure (refined) ────────────────────────────
    h_real = np.array([
        mape(real["tgts"][:, h, :], real["preds"][:, h, :]) for h in range(24)
    ])
    h_clim = np.array([
        mape(clim["tgts"][:, h, :], clim["preds"][:, h, :]) for h in range(24)
    ])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hours = np.arange(1, 25)
    ax.plot(hours, h_real, "C0-o", label="real weather (lookback)")
    ax.plot(hours, h_clim, "C3-s", label="climatology weather")
    ax.fill_between(hours, h_real, h_clim, alpha=0.2, color="C2",
                    label="weather contribution")
    ax.set_xlabel("Forecast horizon (hour ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Weather contribution grows with horizon")
    ax.set_xticks(hours[::2])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_horizon_decomp.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/fig_horizon_decomp.png")

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {
        "n_val_windows": n,
        "n_zones": Z,
        "overall": {
            "lookback_real_mape": overall_real,
            "lookback_real_ci90": [real_lo, real_hi],
            "lookback_clim_mape": overall_clim,
            "lookback_clim_ci90": [clim_lo, clim_hi],
            "delta_pp": overall_delta,
            "delta_ci90": [delta_lo, delta_hi],
        },
        "per_zone": zone_results,
        "per_horizon_real": h_real.tolist(),
        "per_horizon_clim": h_clim.tolist(),
    }
    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
