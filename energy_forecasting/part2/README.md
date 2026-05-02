# Hybrid CNN-Transformer — ISO-NE Day-Ahead Energy Demand Forecasting

## File overview

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters in one dict (`CONFIG`) |
| `dataset.py` | `EnergyDataset` + `compute_calendar_features()` helper |
| `model.py` | `HybridCNNTransformer`, `CNNFeatureExtractor`, `get_model()` |
| `train.py` | Training loop with AdamW, MAE loss, and val-loss checkpointing |

## Architecture summary

```
For each of S+24 hours
  ┌─────────────────────────────────────────────┐
  │  Weather map (pre-pooled to G×G)            │
  │    → shared CNN feature extractor           │
  │    → P = G² spatial tokens of dim D        │
  │  Demand + calendar                          │
  │    → linear embedder → 1 tabular token      │
  └─────────────────────────────────────────────┘
  + learnable spatial positional embedding (P, D)
  + learnable timestep encoding (S+24, D)
  
Concatenate chronologically → seq of (S+24)×(P+1) tokens
    ↓  nn.TransformerEncoder (4 layers, 8 heads, D=128)
    ↓  slice future tabular tokens → (B, 24, D)
    ↓  2-layer MLP → (B, 24, Z)  [denormalised MWh]
```

Demand inputs are z-score normalised internally; predictions are
denormalised before return so the loss and outputs are always in raw MWh.

## How to run training

```bash
# activate the course venv
source /cluster/tufts/c26sp1cs0137/cgolem01/venv/bin/activate

cd /cluster/tufts/c26sp1cs0137/cgolem01/energy_forecasting

# train with defaults (S=72, grid=10, D=128, 50 epochs)
python train.py

# override individual hyperparameters
python train.py --batch_size 4 --lr 1e-4 --num_epochs 30
```

The best checkpoint (by val MAE) is saved to `checkpoints/best_model.pt`.

### SLURM batch job

```bash
#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
module load class/default cs137/2026spring
source /cluster/tufts/c26sp1cs0137/cgolem01/venv/bin/activate
cd /cluster/tufts/c26sp1cs0137/cgolem01/energy_forecasting
python train.py --batch_size 2
```

## Hooking into the evaluation harness

Copy (or symlink) the four `.py` files into a named folder under
`evaluation/`:

```bash
EVAL=/cluster/tufts/c26sp1cs0137/data/assignment3_data/evaluation
mkdir -p $EVAL/cgolem01
cp config.py dataset.py model.py $EVAL/cgolem01/

# The evaluator loads a checkpoint and sets demand stats at inference time.
# Add a line to get_model() or load them in the model __init__ if needed.
```

Then run:

```bash
cd $EVAL
python evaluate.py cgolem01 30
```

## Memory notes

With default settings (S=72, P=100) the Transformer sequence length is
**9 696 tokens**.  PyTorch ≥ 2.0 uses a fused, memory-efficient
`scaled_dot_product_attention` kernel so no dense (seq × seq) attention
matrix is materialised, but activation checkpointing may still be needed
for large batches.

Quick fixes if you hit OOM:
- `--batch_size 1`
- Reduce `grid_size` in `config.py` (e.g. 5 → P=25, seq_len=2450)
- Reduce `S` (e.g. 24 → seq_len=2424)

## Key design decisions

- **Shared CNN weights**: `CNNFeatureExtractor` is instantiated once and
  called on all `B*T` weather maps simultaneously via a reshape trick.
- **Pre-pooling**: weather is pooled from (450×449) to (G×G) in the
  dataset and in `adapt_inputs()` — not inside the CNN — so the CNN sees
  consistent (G×G) inputs in both training and evaluation.
- **Calendar features** (7-dim cyclical encoding) are derived from
  hours-since-epoch in `compute_calendar_features()`, shared between the
  dataset loader and `adapt_inputs()`.
- **Demand normalisation** is applied inside `forward()` using buffers
  set by `set_demand_stats()`; predictions are denormalised before return.
