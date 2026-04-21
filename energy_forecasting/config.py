"""
All hyperparameters in one place. Pass CONFIG (or a modified copy) to
get_model(), EnergyDataset, and train().
"""

DATA_BASE = "/cluster/tufts/c26sp1cs0137/data/assignment3_data"

CONFIG = {
    # ── Sequence ────────────────────────────────────────────────────────────
    # S: how many historical hours each sample uses.
    # The evaluator always feeds 168 hours of history; adapt_inputs() takes the
    # last S of those, so S can be anything ≤ 168.
    "S": 72,            # historical lookback (hours)
    "H_forecast": 24,  # forecast horizon (hours) — fixed by the problem

    # ── Demand zones ────────────────────────────────────────────────────────
    "Z": 8,  # ISO-NE zones: ME, NH, VT, CT, RI, SEMA, WCMA, NEMA_BOST

    # ── Weather map ─────────────────────────────────────────────────────────
    "map_H": 450,   # raw weather map height (pixels)
    "map_W": 449,   # raw weather map width  (pixels)
    "C": 7,         # number of weather channels per pixel

    # ── Spatial tokens ──────────────────────────────────────────────────────
    # The raw (450 × 449) map is spatially pooled to (grid_size × grid_size)
    # before entering the CNN feature extractor, both during dataset loading
    # and inside adapt_inputs() at evaluation time.
    "grid_size": 5,    # CNN output spatial resolution
    "P": 25,           # spatial tokens per timestep = grid_size ** 2

    # ── Calendar features ───────────────────────────────────────────────────
    # 8 features: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
    #             is_weekend, is_holiday
    "num_calendar_features": 8,

    # ── Model architecture ──────────────────────────────────────────────────
    "D": 128,                  # token embedding dimension
    "cnn_channels": [32, 64],  # CNN intermediate channels (2-layer feature extractor)
    "num_transformer_layers": 4,
    "num_heads": 8,
    "transformer_dropout": 0.25,
    "mlp_hidden_dim": 256,     # prediction-head hidden size

    # ── Training ────────────────────────────────────────────────────────────
    # sequence length = (S+24)*(P+1) = 96*26 = 2496 tokens.
    "lr": 3e-4,
    "lr_min": 1e-5,     # cosine annealing floor
    "weight_decay": 1e-2,
    "batch_size": 16,
    "num_epochs": 50,
    "gradient_clip": 1.0,
    "loss": "mae",   # "mae" or "mse"
    "num_workers": 1,

    # ── Data paths ───────────────────────────────────────────────────────────
    "weather_base_path": f"{DATA_BASE}/weather_data",
    "demand_dir": f"{DATA_BASE}/energy_demand_data",
    "train_years": [2019, 2020, 2021, 2022],
    "val_years": [2023],

    # ── Checkpointing ────────────────────────────────────────────────────────
    "checkpoint_dir": "checkpoints/lookback",
    "checkpoint_name": "best_model_lookback.pt",
}
