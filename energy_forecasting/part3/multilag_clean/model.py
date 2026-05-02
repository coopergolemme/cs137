"""
Multi-lag HybridCNNTransformer.

Identical to the lookback model except the future tabular embedder takes
TWO lag tensors (168 h + 24 h prior) rather than one. Architecturally the
only change is `future_embedder.in_features = num_lags*Z + F_cal`.

Forward signature gains one extra arg: `future_demand_lag_24`.

adapt_inputs() pulls both lags from the 168 h history slot, so the
evaluator contract is preserved.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import compute_calendar_features


class CNNFeatureExtractor(nn.Module):
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

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.flatten(2).transpose(1, 2)
        return self.proj(feat)


class HybridCNNTransformerMultilag(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        S   = config["S"]
        Z   = config["Z"]
        C   = config["C"]
        D   = config["D"]
        G   = config["grid_size"]
        F_cal = config["num_calendar_features"]
        num_lags = config.get("num_lags", 2)
        P = G * G

        self.S = S
        self.Z = Z
        self.C = C
        self.D = D
        self.G = G
        self.P = P
        self.F_cal = F_cal
        self.num_lags = num_lags

        self.register_buffer("demand_mean", torch.zeros(Z))
        self.register_buffer("demand_std",  torch.ones(Z))

        self.cnn = CNNFeatureExtractor(C, config["cnn_channels"], D)

        # Historical tabular: [demand, calendar] = (Z + F_cal) dim
        self.hist_embedder   = nn.Linear(Z + F_cal, D)
        # Future tabular:    [lag_1, lag_2, ..., lag_K, calendar] = (K*Z + F_cal)
        self.future_embedder = nn.Linear(num_lags * Z + F_cal, D)

        self.spatial_pos  = nn.Parameter(torch.empty(P, D))
        self.timestep_pos = nn.Parameter(torch.empty(S + 24, D))
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)
        nn.init.trunc_normal_(self.timestep_pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config["num_heads"],
            dim_feedforward=config["mlp_hidden_dim"] * 2,
            dropout=config["transformer_dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=config["num_transformer_layers"]
        )

        self.head = nn.Sequential(
            nn.Linear(D, config["mlp_hidden_dim"]),
            nn.GELU(),
            nn.Linear(config["mlp_hidden_dim"], Z),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_demand_stats(self, mean: torch.Tensor, std: torch.Tensor):
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
        B = history_weather.size(0)
        device = history_weather.device

        # Subsample history to last S hours
        hist_weather = history_weather[:, -self.S:]
        hist_energy  = history_energy[:, -self.S:]

        # Spatial pooling
        hist_weather   = self._pool_weather(hist_weather,   self.G)
        future_weather = self._pool_weather(future_weather, self.G)

        # Calendar features
        fut_hours = future_time.cpu().numpy().astype(np.int64)
        hist_start_hours = fut_hours[:, 0:1] - self.S
        hist_hours = hist_start_hours + np.arange(self.S)
        hist_cal   = np.stack([compute_calendar_features(hist_hours[b]) for b in range(B)])
        future_cal = np.stack([compute_calendar_features(fut_hours[b])  for b in range(B)])
        hist_cal   = torch.from_numpy(hist_cal).to(device)
        future_cal = torch.from_numpy(future_cal).to(device)

        # Both lags from the 168 h history slot
        # 168-h lag: first 24 h of history
        future_demand_lag_168 = history_energy[:, :24, :].to(device)
        # 24-h lag: last 24 h of history
        future_demand_lag_24  = history_energy[:, -24:, :].to(device)

        return (hist_weather, hist_energy, hist_cal,
                future_weather, future_cal,
                future_demand_lag_168, future_demand_lag_24)

    @staticmethod
    def _pool_weather(weather, grid_size):
        B, T, H, W, C = weather.shape
        x = weather.view(B * T, H, W, C).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (grid_size, grid_size))
        return x.permute(0, 2, 3, 1).view(B, T, grid_size, grid_size, C)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        hist_weather,
        hist_demand,
        hist_calendar,
        future_weather,
        future_calendar,
        future_demand_lag_168,
        future_demand_lag_24,
    ):
        B = hist_weather.size(0)
        T = self.S + 24

        # CNN
        weather = torch.cat([hist_weather, future_weather], dim=1)
        weather_flat = weather.view(B * T, self.G, self.G, self.C).permute(0, 3, 1, 2)
        spatial_tokens = self.cnn(weather_flat).view(B, T, self.P, self.D)

        # Tabular embeddings — historical: [demand, calendar]
        hist_demand_norm = self._norm(hist_demand)
        hist_tab = torch.cat([hist_demand_norm, hist_calendar], dim=-1)
        hist_tab_emb = self.hist_embedder(hist_tab)

        # Tabular embeddings — future: [lag_168, lag_24, calendar]
        lag168_norm = self._norm(future_demand_lag_168)
        lag24_norm  = self._norm(future_demand_lag_24)
        future_tab = torch.cat([lag168_norm, lag24_norm, future_calendar], dim=-1)
        future_tab_emb = self.future_embedder(future_tab)

        tab_emb = torch.cat([hist_tab_emb, future_tab_emb], dim=1)

        # Positional embeddings
        spatial_tokens = (
            spatial_tokens
            + self.spatial_pos[None, None, :, :]
            + self.timestep_pos[None, :, None, :]
        )
        tab_emb = tab_emb + self.timestep_pos[None, :, :]

        # Assemble + transformer
        tab_emb = tab_emb.unsqueeze(2)
        groups  = torch.cat([spatial_tokens, tab_emb], dim=2)
        seq     = groups.view(B, T * (self.P + 1), self.D)
        seq_out = self.transformer(seq).view(B, T, self.P + 1, self.D)

        future_tab_out = seq_out[:, self.S:, self.P, :]
        pred_norm = self.head(future_tab_out)
        return self._denorm(pred_norm)


def get_model(metadata: dict = None):
    from config import CONFIG
    cfg = dict(CONFIG)
    if metadata is not None:
        mapping = {"n_zones": "Z", "n_weather_vars": "C"}
        for src, dst in mapping.items():
            if src in metadata:
                cfg[dst] = metadata[src]
    return HybridCNNTransformerMultilag(cfg)
