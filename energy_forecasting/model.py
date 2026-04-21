"""
Hybrid CNN-Transformer for ISO-NE day-ahead energy demand forecasting.

Architecture overview
─────────────────────
For each of the S+24 hours in a window:
  • P = grid_size² spatial tokens  ← shared-weight CNN feature extractor
  • 1  tabular token               ← linear projection of demand + calendar

Total sequence length: (S+24) × (P+1)
  e.g. S=72, P=100  →  96 × 101 = 9 696 tokens  (see OOM warning below)

Positional embeddings (both learnable):
  • spatial_pos  (P, D)       — which grid cell am I?
  • timestep_pos (S+24, D)    — which hour am I?

After a standard TransformerEncoder the tabular tokens of the 24 future
hours are extracted and passed through a 2-layer MLP → (24, Z) predictions.

Evaluation interface
────────────────────
The evaluate.py harness calls:
    adapted = model.adapt_inputs(hist_weather, hist_energy,
                                  future_weather, future_time)
    pred = model(*adapted)

adapt_inputs:
  • spatially pools raw (450 × 449) weather maps to (grid_size × grid_size)
  • subsamples history from 168 h → S h (last S hours)
  • derives calendar features from future_time (int64 hours-since-epoch)

forward:
    hist_weather:    (B, S,  G, G, C)
    hist_demand:     (B, S,  Z)
    hist_calendar:   (B, S,  F)
    future_weather:  (B, 24, G, G, C)
    future_calendar: (B, 24, F)
  → predictions:     (B, 24, Z)   raw / denormalised MWh

OOM warning
───────────
With S=72, P=100: seq_len = 9 696.  PyTorch ≥ 2.0 uses a memory-efficient
scaled_dot_product_attention kernel so the O(seq²) attention matrix is
never fully materialised, but activation memory still scales with seq².
Keep batch_size ≤ 2 on a 24 GB GPU, or reduce grid_size / S.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import compute_calendar_features


# ─── CNN feature extractor ────────────────────────────────────────────────────

class CNNFeatureExtractor(nn.Module):
    """
    Lightweight CNN applied identically to every timestep's (G × G × C)
    weather map.  It acts as a spatial feature extractor; the spatial
    downsampling from the raw (450 × 449) map is handled upstream by
    adaptive average pooling (in dataset.__getitem__ and adapt_inputs).

    Two conv layers maintain the (G × G) spatial resolution; each grid cell
    becomes a D-dimensional token.

    Args:
        in_channels:  C  — number of weather variables
        cnn_channels: list of two ints, intermediate channel widths
        D:            output embedding dimension per spatial token
    """

    def __init__(self, in_channels: int, cnn_channels: list, D: int):
        super().__init__()
        ch = cnn_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, ch[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.GELU(),
            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.GELU(),
        )
        self.proj = nn.Linear(ch[1], D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, G, G)  — N = B*T after folding batch and time dims
        Returns:
            (N, P, D)  — P = G*G spatial tokens per timestep
        """
        feat = self.encoder(x)                    # (N, ch[-1], G, G)
        feat = feat.flatten(2).transpose(1, 2)    # (N, P, ch[-1])
        return self.proj(feat)                    # (N, P, D)


# ─── Full model ───────────────────────────────────────────────────────────────

class HybridCNNTransformer(nn.Module):

    def __init__(self, config: dict):
        super().__init__()

        S   = config["S"]
        Z   = config["Z"]
        C   = config["C"]
        D   = config["D"]
        G   = config["grid_size"]
        F_cal = config["num_calendar_features"]
        P   = G * G

        self.S     = S
        self.Z     = Z
        self.C     = C
        self.D     = D
        self.G     = G
        self.P     = P
        self.F_cal = F_cal

        # Demand normalisation — call set_demand_stats() before training.
        self.register_buffer("demand_mean", torch.zeros(Z))
        self.register_buffer("demand_std",  torch.ones(Z))

        # ── Shared CNN backbone (same weights for every timestep) ────────────
        self.cnn = CNNFeatureExtractor(C, config["cnn_channels"], D)

        # ── Tabular embedders ────────────────────────────────────────────────
        tab_dim = Z + F_cal
        self.hist_embedder   = nn.Linear(tab_dim, D)
        self.future_embedder = nn.Linear(tab_dim, D)

        # ── Positional embeddings ────────────────────────────────────────────
        self.spatial_pos  = nn.Parameter(torch.empty(P, D))
        self.timestep_pos = nn.Parameter(torch.empty(S + 24, D))
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)
        nn.init.trunc_normal_(self.timestep_pos, std=0.02)

        # ── Transformer encoder (batch_first to avoid seq-first reshape) ─────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config["num_heads"],
            dim_feedforward=config["mlp_hidden_dim"] * 2,
            dropout=config["transformer_dropout"],
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=config["num_transformer_layers"]
        )

        # ── Prediction head ──────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(D, config["mlp_hidden_dim"]),
            nn.GELU(),
            nn.Linear(config["mlp_hidden_dim"], Z),
        )

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Demand normalisation helpers ─────────────────────────────────────────

    def set_demand_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Store z-score parameters computed from the training set."""
        self.demand_mean.copy_(mean.to(self.demand_mean.device))
        self.demand_std.copy_(std.to(self.demand_std.device))

    def _norm(self, x):
        return (x - self.demand_mean) / (self.demand_std + 1e-8)

    def _denorm(self, x):
        return x * (self.demand_std + 1e-8) + self.demand_mean

    # ── Evaluation adapter ────────────────────────────────────────────────────

    def adapt_inputs(
        self,
        history_weather:  torch.Tensor,   # (B, 168, 450, 449, C)
        history_energy:   torch.Tensor,   # (B, 168, Z)
        future_weather:   torch.Tensor,   # (B, 24, 450, 449, C)
        future_time:      torch.Tensor,   # (B, 24) int64 hours-since-epoch
    ) -> tuple:
        """
        Prepares raw evaluation inputs for forward().  Mirrors the feature
        extraction that EnergyDataset.__getitem__ performs during training.

        Steps:
          1. Subsample history to the last S hours.
          2. Spatially pool weather from (450 × 449) to (G × G).
          3. Derive calendar features from future_time (and inferred hist_time).

        Returns a tuple that is unpacked directly into forward().
        """
        B = history_weather.size(0)
        device = history_weather.device

        # 1. Subsample history to last S hours
        hist_weather = history_weather[:, -self.S:]   # (B, S, 450, 449, C)
        hist_energy  = history_energy[:, -self.S:]    # (B, S, Z)

        # 2. Spatial pooling for history weather
        hist_weather   = self._pool_weather(hist_weather,   self.G)  # (B, S,  G, G, C)
        future_weather = self._pool_weather(future_weather, self.G)  # (B, 24, G, G, C)

        # 3. Calendar features — compute from hours-since-epoch
        #    History starts S hours before the first future hour.
        fut_hours = future_time.cpu().numpy().astype(np.int64)   # (B, 24)
        hist_start_hours = fut_hours[:, 0:1] - self.S            # (B, 1)
        hist_hours = hist_start_hours + np.arange(self.S)        # (B, S)

        hist_cal   = np.stack([compute_calendar_features(hist_hours[b])
                               for b in range(B)])               # (B, S, 7)
        future_cal = np.stack([compute_calendar_features(fut_hours[b])
                               for b in range(B)])               # (B, 24, 7)

        hist_cal   = torch.from_numpy(hist_cal).to(device)    # (B, S, 7)
        future_cal = torch.from_numpy(future_cal).to(device)  # (B, 24, 7)

        # 4. Weekly lag: first 24 hours of the 168-h history = same wall-clock
        #    hours as the forecast window, one week prior.
        future_demand_lag = history_energy[:, :24, :].to(device)   # (B, 24, Z)

        return hist_weather, hist_energy, hist_cal, future_weather, future_cal, future_demand_lag

    @staticmethod
    def _pool_weather(weather: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Adaptive-average-pool a batch of weather maps from (H, W) to
        (grid_size, grid_size).

        Args:
            weather:   (B, T, H, W, C)
            grid_size: target spatial size

        Returns:
            (B, T, G, G, C)
        """
        B, T, H, W, C = weather.shape
        x = weather.view(B * T, H, W, C).permute(0, 3, 1, 2)        # (B*T, C, H, W)
        x = F.adaptive_avg_pool2d(x, (grid_size, grid_size))         # (B*T, C, G, G)
        return x.permute(0, 2, 3, 1).view(B, T, grid_size, grid_size, C)  # (B,T,G,G,C)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        hist_weather:      torch.Tensor,   # (B, S,  G, G, C)
        hist_demand:       torch.Tensor,   # (B, S,  Z)   raw MWh
        hist_calendar:     torch.Tensor,   # (B, S,  F)
        future_weather:    torch.Tensor,   # (B, 24, G, G, C)
        future_calendar:   torch.Tensor,   # (B, 24, F)
        future_demand_lag: torch.Tensor,   # (B, 24, Z)   raw MWh, 168 h prior
    ) -> torch.Tensor:                     # (B, 24, Z)   raw MWh (denormalised)
        B = hist_weather.size(0)
        T = self.S + 24   # total timesteps in the window

        # ── CNN backbone (shared weights, applied to all T timesteps) ────────
        weather = torch.cat([hist_weather, future_weather], dim=1)  # (B, T, G, G, C)
        weather_flat = weather.view(B * T, self.G, self.G, self.C)
        weather_flat = weather_flat.permute(0, 3, 1, 2)             # (B*T, C, G, G)
        spatial_tokens = self.cnn(weather_flat)                     # (B*T, P, D)
        spatial_tokens = spatial_tokens.view(B, T, self.P, self.D)  # (B, T, P, D)

        # ── Tabular embeddings ───────────────────────────────────────────────
        hist_demand_norm = self._norm(hist_demand)                        # (B, S,  Z)
        future_lag_norm  = self._norm(future_demand_lag)                  # (B, 24, Z)

        hist_tab   = torch.cat([hist_demand_norm, hist_calendar],  dim=-1)  # (B, S,  Z+F)
        future_tab = torch.cat([future_lag_norm,  future_calendar], dim=-1)  # (B, 24, Z+F)

        hist_tab_emb   = self.hist_embedder(hist_tab)     # (B, S,  D)
        future_tab_emb = self.future_embedder(future_tab) # (B, 24, D)
        tab_emb = torch.cat([hist_tab_emb, future_tab_emb], dim=1)  # (B, T, D)

        # ── Positional embeddings ────────────────────────────────────────────
        # spatial_pos[None, None, :, :]: (1, 1, P, D) — same for every (b, t)
        # timestep_pos[None, :, None, :]: (1, T, 1, D) — same for every (b, p)
        spatial_tokens = (
            spatial_tokens
            + self.spatial_pos[None, None, :, :]        # which grid cell
            + self.timestep_pos[None, :, None, :]       # which hour
        )
        tab_emb = tab_emb + self.timestep_pos[None, :, :]   # (B, T, D)

        # ── Assemble sequence: [P spatial | 1 tabular] per timestep ─────────
        tab_emb = tab_emb.unsqueeze(2)                       # (B, T, 1, D)
        groups  = torch.cat([spatial_tokens, tab_emb], dim=2)  # (B, T, P+1, D)
        seq     = groups.view(B, T * (self.P + 1), self.D)   # (B, seq_len, D)

        # ── Transformer ──────────────────────────────────────────────────────
        seq_out = self.transformer(seq)                          # (B, seq_len, D)
        seq_out = seq_out.view(B, T, self.P + 1, self.D)        # (B, T, P+1, D)

        # ── Prediction head ──────────────────────────────────────────────────
        # Tabular token for each timestep is at index self.P (last in its group).
        # Slice only the 24 future timesteps (indices S … S+23).
        future_tab_out = seq_out[:, self.S:, self.P, :]  # (B, 24, D)
        pred_norm = self.head(future_tab_out)             # (B, 24, Z)

        return self._denorm(pred_norm)                    # (B, 24, Z)  raw MWh


# ─── Factory ─────────────────────────────────────────────────────────────────

def get_model(metadata: dict = None) -> HybridCNNTransformer:
    """
    Build and return a HybridCNNTransformer.

    When called by evaluate.py, `metadata` contains:
        {"n_zones": 8, "n_weather_vars": 7, "history_len": 168,
         "future_len": 24, "zone_names": [...]}
    When called from train.py, pass the CONFIG dict directly, or pass None
    to use CONFIG defaults.

    The function merges metadata keys into CONFIG so both call sites work.
    """
    from config import CONFIG
    cfg = dict(CONFIG)   # start from defaults

    if metadata is not None:
        # Map evaluate.py metadata keys → CONFIG keys
        mapping = {
            "n_zones":        "Z",
            "n_weather_vars": "C",
        }
        for src, dst in mapping.items():
            if src in metadata:
                cfg[dst] = metadata[src]

    return HybridCNNTransformer(cfg)
