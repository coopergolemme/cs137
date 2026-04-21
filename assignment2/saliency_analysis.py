"""
Part 2: Saliency Maps / Sensitivity Analysis
Computes gradients of model outputs w.r.t. input to reveal which spatial
regions and input variables drive the 24-hour forecast.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader, Subset

from dataset import WeatherDataset
from model import WeatherCNN

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = '/cluster/tufts/c26sp1cs0137/cgolem01/best_model_0.7339.pth'
DATA_DIR    = '/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset'
OUT_DIR     = '/cluster/tufts/c26sp1cs0137/cgolem01'
N_SAMPLES   = 200          # samples to average over
BATCH_SIZE  = 8
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target output index → name mapping
OUTPUT_NAMES = [
    'TMP@2m (Temperature)',
    'RH@2m (Rel. Humidity)',
    'UGRD@10m (U-wind)',
    'VGRD@10m (V-wind)',
    'GUST (Wind Gust)',
    'APCP (Precipitation)',
    'Binary (Extreme Event)',
]

# Input channel names (42 channels)
CHANNEL_NAMES = [
    'TMP@2m', 'RH@2m', 'UGRD@10m', 'VGRD@10m', 'GUST', 'DSWRF',
    'APCP_1hr', 'CAPE', 'DPT@1000', 'DPT@500', 'DPT@700', 'DPT@850', 'DPT@925',
    'HGT@1000', 'HGT@500', 'HGT@700', 'HGT@850', 'HGT@sfc',
    'TMP@1000', 'TMP@500', 'TMP@700', 'TMP@850', 'TMP@925',
    'UGRD@1000', 'UGRD@250', 'UGRD@500', 'UGRD@700', 'UGRD@850', 'UGRD@925',
    'VGRD@1000', 'VGRD@250', 'VGRD@500', 'VGRD@700', 'VGRD@850', 'VGRD@925',
    'TCDC', 'HCDC', 'MCDC', 'LCDC', 'PWAT', 'RHPW', 'VIL',
]

# Lambert Conformal grid extent in km (approx, for axis labels)
GRID_X_KM = np.linspace(0, 1344, 449)   # ~1344 km west→east
GRID_Y_KM = np.linspace(0, 1347, 450)   # ~1347 km south→north

# The Jumbo station pixel location (from metadata)
JUMBO_Y, JUMBO_X = 177, 263


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(path, in_channels=42):
    model = WeatherCNN(in_channels=in_channels, out_channels=7, use_residual=True)
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def compute_saliency(model, x_batch, output_idx):
    """
    Vanilla gradient saliency: |d output[output_idx] / d x|
    Returns tensor of shape (B, C, H, W).
    """
    x = x_batch.to(DEVICE).float()
    x = x.clone().detach()
    x.requires_grad_(True)          # x is now a leaf tensor on the correct device
    out = model(x)                  # (B, 7)
    scalar = out[:, output_idx].sum()
    model.zero_grad()
    scalar.backward()
    return x.grad.detach().abs()    # (B, C, H, W)


def accumulate_saliency(model, loader, n_samples):
    """
    Returns dict: output_idx → averaged saliency (C, H, W) and (H, W)
    Also returns a sample input (C, H, W) for context.
    """
    saliency_sum  = {i: None for i in range(7)}
    saliency_sq   = {i: None for i in range(7)}
    count = 0
    sample_input = None

    for x, _ in loader:
        if count >= n_samples:
            break
        # drop NaN samples
        valid = ~torch.isnan(x).view(x.size(0), -1).any(dim=1)
        x = x[valid]
        if x.size(0) == 0:
            continue

        if sample_input is None:
            sample_input = x[0].cpu()

        for out_idx in range(7):
            sal = compute_saliency(model, x, out_idx)   # (B, C, H, W)
            sal_cpu = sal.cpu()
            if saliency_sum[out_idx] is None:
                saliency_sum[out_idx] = sal_cpu.sum(0)
                saliency_sq[out_idx]  = (sal_cpu ** 2).sum(0)
            else:
                saliency_sum[out_idx] += sal_cpu.sum(0)
                saliency_sq[out_idx]  += (sal_cpu ** 2).sum(0)

        count += x.size(0)
        print(f'  processed {count}/{n_samples} samples', end='\r', flush=True)

    print()
    avg_sal = {}
    for i in range(7):
        if saliency_sum[i] is not None:
            avg_sal[i] = saliency_sum[i] / count          # (C, H, W)
        else:
            avg_sal[i] = torch.zeros(42, 450, 449)

    return avg_sal, sample_input


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_spatial_saliency(avg_sal):
    """
    Fig 1: 7-panel spatial saliency maps (sum over channels, then normalize).
    Shows which grid regions most influence each output.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    fig.suptitle('Saliency Maps: Mean |∂output / ∂input| summed over channels\n'
                 '(24-hr forecast, averaged over 200 validation samples)', fontsize=13)

    for i in range(7):
        sal_spatial = avg_sal[i].sum(0).numpy()   # (H, W)  — sum over channels
        # percentile clip for visibility
        vmax = np.percentile(sal_spatial, 99)
        im = axes[i].imshow(sal_spatial, origin='lower', cmap='hot',
                            vmin=0, vmax=vmax, aspect='auto')
        axes[i].set_title(OUTPUT_NAMES[i], fontsize=10)
        axes[i].set_xlabel('West → East (grid col)')
        axes[i].set_ylabel('South → North (grid row)')
        # mark Jumbo station
        axes[i].plot(JUMBO_X, JUMBO_Y, 'b^', markersize=8, label='Jumbo station')
        axes[i].legend(fontsize=7)
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    axes[7].axis('off')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'saliency_spatial.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved {path}')


def plot_channel_importance(avg_sal):
    """
    Fig 2: Per-channel (variable) sensitivity bar chart for each output.
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.ravel()
    fig.suptitle('Input Channel Sensitivity\n'
                 'Mean |∂output / ∂input| summed over spatial dims', fontsize=13)

    for i in range(7):
        # sum over H, W → per-channel importance
        ch_imp = avg_sal[i].sum(dim=(1, 2)).numpy()   # (C,)
        # sort by importance
        order = np.argsort(ch_imp)[::-1]
        top_n = 20
        top_idx = order[:top_n]
        top_vals = ch_imp[top_idx]
        top_names = [CHANNEL_NAMES[j] for j in top_idx]

        axes[i].barh(range(top_n), top_vals[::-1], color='steelblue')
        axes[i].set_yticks(range(top_n))
        axes[i].set_yticklabels(top_names[::-1], fontsize=8)
        axes[i].set_title(f'Output: {OUTPUT_NAMES[i]}', fontsize=9)
        axes[i].set_xlabel('Mean |grad| (spatial sum)')

    axes[7].axis('off')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'saliency_channels.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved {path}')


def plot_key_channels_spatial(avg_sal):
    """
    Fig 3: For the two most important outputs (Temperature + Precipitation),
    show the spatial saliency of the top-3 most sensitive input channels.
    This reveals *where* each variable matters most.
    """
    focus_outputs = [0, 5]   # Temperature, Precipitation
    focus_names   = ['Temperature (TMP@2m)', 'Precipitation (APCP)']

    fig, axes = plt.subplots(len(focus_outputs), 3, figsize=(18, 10))
    fig.suptitle('Spatial Saliency by Top Input Channels\n'
                 '(rows = output target, cols = most sensitive input channel)', fontsize=12)

    for row, (out_idx, out_name) in enumerate(zip(focus_outputs, focus_names)):
        ch_imp = avg_sal[out_idx].sum(dim=(1, 2)).numpy()
        top3 = np.argsort(ch_imp)[::-1][:3]

        for col, ch_idx in enumerate(top3):
            sal = avg_sal[out_idx][ch_idx].numpy()   # (H, W)
            vmax = np.percentile(sal, 99)
            im = axes[row, col].imshow(sal, origin='lower', cmap='YlOrRd',
                                       vmin=0, vmax=vmax, aspect='auto')
            axes[row, col].set_title(
                f'{out_name}\n← Input: {CHANNEL_NAMES[ch_idx]}', fontsize=9)
            axes[row, col].set_xlabel('West → East')
            axes[row, col].set_ylabel('South → North')
            axes[row, col].plot(JUMBO_X, JUMBO_Y, 'b^', markersize=8)
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'saliency_top_channels.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved {path}')


def plot_upstream_analysis(avg_sal):
    """
    Fig 4: Wind-field saliency overlay — shows whether the model's sensitivity
    pattern aligns with prevailing westerly/southwesterly upstream flow.
    Focuses on U-wind and V-wind inputs to understand directionality.
    """
    # Channels: UGRD@10m=2, VGRD@10m=3, UGRD@500mb=25, VGRD@500mb=31
    wind_channels = {
        'UGRD@10m (surface U)': 2,
        'VGRD@10m (surface V)': 3,
        'UGRD@500mb (upper U)': 25,
        'VGRD@500mb (upper V)': 31,
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle('Wind-Channel Spatial Saliency per Output\n'
                 'Reveals upstream sensitivity pattern (prevailing westerlies)', fontsize=12)

    for col, (wname, wch) in enumerate(wind_channels.items()):
        for row, (out_idx, out_label) in enumerate([(0, 'Temp'), (5, 'Precip')]):
            sal = avg_sal[out_idx][wch].numpy()   # (H, W)
            vmax = np.percentile(sal, 99)
            im = axes[row, col].imshow(sal, origin='lower', cmap='Blues',
                                        vmin=0, vmax=vmax, aspect='auto')
            axes[row, col].set_title(f'{out_label}\n{wname}', fontsize=9)
            axes[row, col].set_xlabel('W → E')
            axes[row, col].set_ylabel('S → N')
            # Jumbo station
            axes[row, col].plot(JUMBO_X, JUMBO_Y, 'r^', markersize=9,
                                label='Jumbo stn')
            # Draw rough "upstream" rectangle (west & southwest of station)
            from matplotlib.patches import Rectangle
            rect = Rectangle((max(0, JUMBO_X-140), max(0, JUMBO_Y-60)),
                              140, 120, linewidth=1.5, edgecolor='green',
                              facecolor='none', linestyle='--', label='Upstream box')
            axes[row, col].add_patch(rect)
            axes[row, col].legend(fontsize=7)
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'saliency_upstream.png')
    plt.savefig(path, dpi=120)
    plt.close()
    print(f'Saved {path}')


def print_channel_ranking(avg_sal):
    """Print top-10 channel rankings for each output."""
    print('\n' + '='*60)
    print('CHANNEL SENSITIVITY RANKINGS (top 10 per output)')
    print('='*60)
    for i in range(7):
        ch_imp = avg_sal[i].sum(dim=(1, 2)).numpy()
        order = np.argsort(ch_imp)[::-1][:10]
        print(f'\n[{OUTPUT_NAMES[i]}]')
        for rank, ch in enumerate(order, 1):
            print(f'  {rank:2d}. {CHANNEL_NAMES[ch]:25s}  score={ch_imp[ch]:.4f}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f'Device: {DEVICE}')
    print('Loading model...')
    model = load_model(MODEL_PATH)

    print('Loading validation dataset (2020)...')
    val_ds = WeatherDataset(
        data_dir=DATA_DIR,
        years_to_include=['2020']
    )
    # Use a fixed subset for reproducibility
    indices = list(range(min(N_SAMPLES * 4, len(val_ds))))
    subset  = Subset(val_ds, indices)
    loader  = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f'Computing saliency over up to {N_SAMPLES} valid samples...')
    avg_sal, sample_input = accumulate_saliency(model, loader, N_SAMPLES)

    print('Plotting...')
    plot_spatial_saliency(avg_sal)
    plot_channel_importance(avg_sal)
    plot_key_channels_spatial(avg_sal)
    plot_upstream_analysis(avg_sal)
    print_channel_ranking(avg_sal)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
