"""
Multi-lag variant: same architecture as the lookback model, but the future
tabular embedder receives TWO lag tensors instead of one:
    - 24-hour lag (yesterday-same-hour demand)
    - 168-hour lag (one-week-prior demand, the original)

Both lags fit inside the evaluator's 168 h history contract, so the model
remains compatible with evaluate.py.
"""

DATA_BASE = "/cluster/tufts/c26sp1cs0137/data/assignment3_data"

CONFIG = {
    # ── Sequence ────────────────────────────────────────────────────────────
    "S": 72,
    "H_forecast": 24,

    # ── Demand zones ────────────────────────────────────────────────────────
    "Z": 8,

    # ── Weather map ─────────────────────────────────────────────────────────
    "map_H": 450,
    "map_W": 449,
    "C": 7,

    # ── Spatial tokens ──────────────────────────────────────────────────────
    "grid_size": 5,
    "P": 25,

    # ── Calendar features ───────────────────────────────────────────────────
    "num_calendar_features": 8,

    # ── Multi-lag-specific ──────────────────────────────────────────────────
    # Number of lag tensors fed into the future tabular embedder.
    # Each contributes Z dims, so future_embedder input is num_lags*Z + F_cal.
    "num_lags": 2,

    # ── Model architecture ──────────────────────────────────────────────────
    "D": 128,
    "cnn_channels": [32, 64],
    "num_transformer_layers": 4,
    "num_heads": 8,
    "transformer_dropout": 0.25,
    "mlp_hidden_dim": 256,

    # ── Training ────────────────────────────────────────────────────────────
    "lr": 3e-4,
    "lr_min": 1e-5,
    "weight_decay": 1e-2,
    "batch_size": 16,
    "num_epochs": 50,
    "gradient_clip": 1.0,
    "loss": "mae",
    "num_workers": 1,

    # ── Data paths ───────────────────────────────────────────────────────────
    "weather_base_path": f"{DATA_BASE}/weather_data",
    "demand_dir": f"{DATA_BASE}/energy_demand_data",
    # 2022 held out as a clean test set (course evaluator runs on test_year=2022).
    "train_years": [2019, 2020, 2021],
    "val_years": [2023],

    # ── Checkpointing ────────────────────────────────────────────────────────
    "checkpoint_dir": "checkpoints/multilag_clean",
    "checkpoint_name": "best_model_multilag_clean.pt",
}
