import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = "/cluster/tufts/c26sp1cs0137/data/assignment3_data"
WEATHER_PATH = os.path.join(DATA_PATH, "weather_data")
DEMAND_PATH = os.path.join(DATA_PATH, "energy_demand_data")

CHECKPOINT_PATH = "./best_model.pt"
MAPE_PLOT_PATH = "./mape_per_epoch.png"

EPS = 1e-6


# ================= DATASET =================
class EnergyDataset(Dataset):
    def __init__(self, S=12):
        self.S = S

        # ---- WEATHER ----
        self.weather_files = sorted(glob.glob(
            os.path.join(WEATHER_PATH, "**/*.pt"), recursive=True
        ))

        # ---- DEMAND ----
        dfs = []
        for f in sorted(glob.glob(os.path.join(DEMAND_PATH, "*.csv"))):
            df = pd.read_csv(f)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # assume first column is timestamp
        timestamps = pd.to_datetime(df.iloc[:, 0])

        # raw demand values, one column per load zone
        self.raw_demand = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        self.demand_mean = self.raw_demand.mean(axis=0).astype(np.float32)
        self.demand_std = (self.raw_demand.std(axis=0) + EPS).astype(np.float32)

        # dataset returns normalized demand for training targets and historical inputs
        self.demand = ((self.raw_demand - self.demand_mean) / self.demand_std).astype(np.float32)

        # ---- CREATE NORMALIZED CALENDAR FEATURES ----
        # hour: 0..23, dayofweek: 0..6, month: 1..12
        self.calendar = np.stack([
            timestamps.dt.hour.values / 23.0,
            timestamps.dt.dayofweek.values / 6.0,
            (timestamps.dt.month.values - 1.0) / 11.0,
        ], axis=1).astype(np.float32)

        # ---- ALIGN LENGTHS ----
        N = min(len(self.weather_files), len(self.demand))
        self.weather_files = self.weather_files[:N]
        self.raw_demand = self.raw_demand[:N]
        self.demand = self.demand[:N]
        self.calendar = self.calendar[:N]

    def __len__(self):
        return len(self.demand) - self.S - 24

    def __getitem__(self, idx):
        max_idx = len(self.demand) - self.S - 24 - 1
        if idx > max_idx:
            idx = max_idx

        # ---- WEATHER ----
        hist_w = torch.stack([
            torch.load(self.weather_files[idx + i])
            for i in range(self.S)
        ]).permute(0, 3, 1, 2)

        fut_w = torch.stack([
            torch.load(self.weather_files[idx + self.S + i])
            for i in range(24)
        ]).permute(0, 3, 1, 2)

        # ---- NORMALIZED DEMAND ----
        hist_y = torch.tensor(self.demand[idx:idx + self.S], dtype=torch.float32)
        fut_y = torch.tensor(self.demand[idx + self.S:idx + self.S + 24], dtype=torch.float32)

        # ---- NORMALIZED CALENDAR ----
        hist_c = torch.tensor(self.calendar[idx:idx + self.S], dtype=torch.float32)
        fut_c = torch.tensor(self.calendar[idx + self.S:idx + self.S + 24], dtype=torch.float32)

        # ---- TABULAR ----
        hist_tab = torch.cat([hist_y, hist_c], dim=-1)
        zeros = torch.zeros((24, hist_y.shape[1]), dtype=torch.float32)
        fut_tab = torch.cat([zeros, fut_c], dim=-1)

        return hist_w.float(), fut_w.float(), hist_tab.float(), fut_tab.float(), fut_y.float()


# ================= NORMALIZATION =================
def compute_weather_norm(loader, max_batches=100):
    """Compute scalar weather mean/std from training-style weather tensors."""
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch_idx, (hist_w, fut_w, _, _, _) in enumerate(loader):
        w = torch.cat([hist_w, fut_w], dim=1).float()
        total_sum += w.sum().item()
        total_sq_sum += (w ** 2).sum().item()
        total_count += w.numel()

        if batch_idx + 1 >= max_batches:
            break

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = max(var, EPS) ** 0.5

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


# ================= MODEL =================
class CNNPatchEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, 8, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = self.conv(x)
        _, D, Hp, Wp = x.shape
        x = x.reshape(B, T, Hp * Wp, D)
        return x


class Model(nn.Module):
    def __init__(self, weather_channels, tab_dim, num_zones):
        super().__init__()

        self.embed_dim = 32
        self.num_zones = num_zones
        self.calendar_dim = tab_dim - num_zones

        self.max_steps = 40
        self.max_patches = 256
        self.time_pos = nn.Parameter(torch.randn(1, self.max_steps, 1, self.embed_dim))
        self.token_pos = nn.Parameter(torch.randn(1, 1, self.max_patches + 1, self.embed_dim))

        self.cnn = CNNPatchEncoder(weather_channels, self.embed_dim)
        self.tab = nn.Linear(tab_dim, self.embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, num_zones),
        )

        # These buffers make input normalization part of the model, so it happens
        # during both training and evaluation/inference.
        self.register_buffer("weather_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("weather_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("demand_mean", torch.zeros(num_zones, dtype=torch.float32))
        self.register_buffer("demand_std", torch.ones(num_zones, dtype=torch.float32))

        # If True, forward returns predictions in original MWh scale.
        # If False, forward returns normalized predictions.
        self.return_raw = False

    def set_normalization(self, weather_mean, weather_std, demand_mean, demand_std):
        self.weather_mean.copy_(torch.as_tensor(weather_mean, dtype=torch.float32).reshape(()))
        self.weather_std.copy_(torch.as_tensor(weather_std, dtype=torch.float32).reshape(()))
        self.demand_mean.copy_(torch.as_tensor(demand_mean, dtype=torch.float32))
        self.demand_std.copy_(torch.as_tensor(demand_std, dtype=torch.float32))

    def _standardize_weather_shape(self, w, name):
        w = torch.as_tensor(w, dtype=torch.float32, device=self.weather_mean.device)

        if w.dim() == 4:
            # either (T, C, H, W) or (T, H, W, C)
            w = w.unsqueeze(0)
        elif w.dim() != 5:
            raise ValueError(f"Unexpected {name} shape: {tuple(w.shape)}")

        if w.shape[2] == 7:
            return w
        if w.shape[-1] == 7:
            return w.permute(0, 1, 4, 2, 3)

        raise ValueError(f"Unexpected {name} shape: {tuple(w.shape)}")

    def _normalize_weather(self, w):
        return (w - self.weather_mean) / (self.weather_std + EPS)

    def _normalize_calendar_if_needed(self, c):
        """Normalize calendar features if they look raw; leave them alone if already 0..1."""
        c = c.float()
        if c.numel() == 0:
            return c

        # Expected columns: hour, dayofweek, month. Supports already-normalized values too.
        if c.shape[-1] >= 3 and c.max() > 1.5:
            c = c.clone()
            c[..., 0] = c[..., 0] / 23.0
            c[..., 1] = c[..., 1] / 6.0
            c[..., 2] = (c[..., 2] - 1.0) / 11.0
        return c

    def _prepare_tabular(self, tab, T, is_future, B):
        """
        Ensures tabular tensors are shaped (B, T, num_zones + calendar_dim).
        Historical tabular may arrive as raw demand + calendar, normalized demand + calendar,
        or only demand/calendar. Future tabular may arrive as calendar only or zero-padded.
        """
        tab = torch.as_tensor(tab, dtype=torch.float32, device=self.weather_mean.device)

        if tab.dim() == 2:
            tab = tab.unsqueeze(0)
        if tab.dim() != 3:
            raise ValueError(f"Unexpected tabular shape: {tuple(tab.shape)}")

        if tab.shape[0] == 1 and B > 1:
            tab = tab.expand(B, -1, -1)

        if tab.shape[0] != B or tab.shape[1] != T:
            raise ValueError(f"Expected tabular shape (B={B}, T={T}, D), got {tuple(tab.shape)}")

        D = tab.shape[-1]
        full_dim = self.num_zones + self.calendar_dim

        if D == full_dim:
            demand_part = tab[..., :self.num_zones]
            cal_part = tab[..., self.num_zones:]
        elif D == self.calendar_dim:
            demand_part = torch.zeros(B, T, self.num_zones, dtype=torch.float32, device=tab.device)
            cal_part = tab
        elif D == self.num_zones:
            demand_part = tab
            cal_part = torch.zeros(B, T, self.calendar_dim, dtype=torch.float32, device=tab.device)
        else:
            raise ValueError(f"Expected tabular dim {full_dim}, {self.calendar_dim}, or {self.num_zones}; got {D}")

        cal_part = self._normalize_calendar_if_needed(cal_part)

        if is_future:
            # Future demand is unavailable by spec, so keep it masked as zero.
            demand_part = torch.zeros_like(demand_part)
        else:
            # If historical demand looks raw MWh, z-score it. If it already looks normalized,
            # leave it unchanged. This prevents double normalization during training.
            if demand_part.abs().median() > 20:
                demand_part = (demand_part - self.demand_mean) / (self.demand_std + EPS)

        return torch.cat([demand_part, cal_part], dim=-1)

    def forward(self, hist_w, fut_w, hist_tab, fut_tab):
        hist_w = self._standardize_weather_shape(hist_w, "hist_w")
        fut_w = self._standardize_weather_shape(fut_w, "fut_w")

        hist_w = self._normalize_weather(hist_w)
        fut_w = self._normalize_weather(fut_w)

        B = hist_w.shape[0]
        S = hist_w.shape[1]
        F = fut_w.shape[1]

        hist_tab = self._prepare_tabular(hist_tab, S, is_future=False, B=B)
        fut_tab = self._prepare_tabular(fut_tab, F, is_future=True, B=B)

        hist_spatial = self.cnn(hist_w)
        fut_spatial = self.cnn(fut_w)

        # -------------------------
        # Combine weather over time
        # -------------------------
        spatial = torch.cat([hist_spatial, fut_spatial], dim=1)
        B, T, P, D = spatial.shape

        # -------------------------
        # Tabular tokens
        # -------------------------
        tab = torch.cat([hist_tab, fut_tab], dim=1)
        tab = self.tab(tab).unsqueeze(2)

        # -------------------------
        # Sequence assembly
        # -------------------------
        seq = torch.cat([spatial, tab], dim=2)

        if T > self.max_steps or (P + 1) > (self.max_patches + 1):
            raise ValueError(f"T={T}, P+1={P + 1} exceed positional embedding limits")

        seq = seq + self.time_pos[:, :T, :, :] + self.token_pos[:, :, :P + 1, :]
        seq = seq.reshape(B, T * (P + 1), D)

        # -------------------------
        # Transformer
        # -------------------------
        out = self.transformer(seq)

        # -------------------------
        # Future tab token states
        # -------------------------
        out = out.reshape(B, T, P + 1, D)
        future_tab_states = out[:, -F:, -1, :]

        pred = self.head(future_tab_states)

        if self.return_raw:
            pred = pred * self.demand_std + self.demand_mean

        return pred

    def adapt_inputs(self, hist_weather, fut_weather, hist_tabular, fut_tabular):
        """Optional compatibility helper for evaluators that call adapt_inputs."""
        hist_weather = self._standardize_weather_shape(hist_weather, "hist_weather")
        fut_weather = self._standardize_weather_shape(fut_weather, "fut_weather")
        B = hist_weather.shape[0]
        S = hist_weather.shape[1]
        F = fut_weather.shape[1]
        hist_tabular = self._prepare_tabular(hist_tabular, S, is_future=False, B=B)
        fut_tabular = self._prepare_tabular(fut_tabular, F, is_future=True, B=B)
        return hist_weather, fut_weather, hist_tabular, fut_tabular


# ================= METRIC =================
def mape(pred, target):
    return torch.mean(torch.abs((target - pred) / (target + EPS))) * 100


def save_mape_plot(epoch_values, mape_values, path=MAPE_PLOT_PATH):
    plt.figure(figsize=(7, 4.5))
    plt.plot(epoch_values, mape_values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAPE (%)")
    plt.title("Validation MAPE per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved MAPE convergence plot to {path}")


# ================= TRAIN =================
def train():
    from torch.utils.data import random_split

    dataset = EnergyDataset(S=12)

    temp_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    w_mean, w_std = compute_weather_norm(temp_loader, max_batches=10)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Sample to save compute time
    train_dataset = Subset(train_dataset, range(min(len(train_dataset), 5000)))
    val_dataset = Subset(val_dataset, range(min(len(val_dataset), 1000)))

    temp_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    sample = dataset[0]
    model = Model(
        weather_channels=sample[0].shape[1],
        tab_dim=sample[2].shape[-1],
        num_zones=sample[-1].shape[-1],
    ).to(DEVICE)

    model.set_normalization(
        weather_mean=w_mean,
        weather_std=w_std,
        demand_mean=dataset.demand_mean,
        demand_std=dataset.demand_std,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_mape = float("inf")
    epoch_history = []
    val_mape_history = []

    print("Training...")

    for epoch in range(5):
        model.train()
        model.return_raw = False
        total_loss = 0.0

        for hist_w, fut_w, hist_tab, fut_tab, target in tqdm(train_loader):
            hist_w = hist_w.to(DEVICE)
            fut_w = fut_w.to(DEVICE)
            hist_tab = hist_tab.to(DEVICE)
            fut_tab = fut_tab.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(hist_w, fut_w, hist_tab, fut_tab)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")

        model.eval()
        model.return_raw = True
        total_mape = 0.0
        count = 0

        with torch.no_grad():
            for hist_w, fut_w, hist_tab, fut_tab, target in val_loader:
                hist_w = hist_w.to(DEVICE)
                fut_w = fut_w.to(DEVICE)
                hist_tab = hist_tab.to(DEVICE)
                fut_tab = fut_tab.to(DEVICE)
                target = target.to(DEVICE)

                pred_raw = model(hist_w, fut_w, hist_tab, fut_tab)
                target_raw = target * model.demand_std + model.demand_mean

                total_mape += mape(pred_raw, target_raw).item()
                count += 1

        val_mape = total_mape / count
        epoch_history.append(epoch + 1)
        val_mape_history.append(val_mape)
        print(f"Validation MAPE: {val_mape:.4f}")

        save_mape_plot(epoch_history, val_mape_history, MAPE_PLOT_PATH)

        if val_mape < best_mape:
            best_mape = val_mape
            torch.save({
                "model_state_dict": model.state_dict(),
                "weather_mean": model.weather_mean.detach().cpu(),
                "weather_std": model.weather_std.detach().cpu(),
                "demand_mean": model.demand_mean.detach().cpu(),
                "demand_std": model.demand_std.detach().cpu(),
                "val_mape_history": val_mape_history,
                "epoch_history": epoch_history,
            }, CHECKPOINT_PATH)
            print("Saved best model")

    print(f"Best Validation MAPE: {best_mape:.4f}")
    print(f"Final MAPE plot: {MAPE_PLOT_PATH}")


# ================= MODEL LOADER FOR EVALUATION =================
def get_model(config):
    weather_channels = 7
    tab_dim = 11
    num_zones = 8

    model = Model(
        weather_channels=weather_channels,
        tab_dim=tab_dim,
        num_zones=num_zones,
    )

    model_path = os.path.join(os.path.dirname(__file__), "best_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.set_normalization(
            weather_mean=checkpoint.get("weather_mean", model.weather_mean),
            weather_std=checkpoint.get("weather_std", model.weather_std),
            demand_mean=checkpoint.get("demand_mean", model.demand_mean),
            demand_std=checkpoint.get("demand_std", model.demand_std),
        )
    else:
        # Backward compatibility with old plain state_dict checkpoints.
        model.load_state_dict(checkpoint, strict=False)

        # Fall back to computing demand normalization from training CSVs.
        dfs = []
        for f in sorted(glob.glob(os.path.join(DEMAND_PATH, "*.csv"))):
            dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True)
        demand = df.iloc[:, 1:].to_numpy(dtype=np.float32)
        model.demand_mean.copy_(torch.tensor(demand.mean(axis=0), dtype=torch.float32))
        model.demand_std.copy_(torch.tensor(demand.std(axis=0) + EPS, dtype=torch.float32))

    # Evaluation should receive MWh-scale predictions for MAPE.
    model.return_raw = True
    model.eval()
    return model


# ================= MAIN =================
if __name__ == "__main__":
    train()
