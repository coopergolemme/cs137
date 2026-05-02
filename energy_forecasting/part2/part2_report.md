# Part 2: Architecture Search & Beating the Baseline

## Motivation

The Part 1 baseline achieved an overall MAPE of **12.47%** on the 2022 test set. Several
structural weaknesses motivate the new design:

- **Spatial-temporal bottleneck.** The flat architecture concatenates all (S+24)×(P+1) tokens
  into a single sequence. Spatial tokens from different hours and temporal tokens compete for
  the same low-capacity attention layers (D=32, 4 heads), even though learning *which grid
  cells matter within an hour* and *how hours relate over time* are structurally different tasks.

- **No future demand prior.** Future tabular tokens zero-mask the demand dimension. ISO-NE
  demand has a strong weekly cycle, so the 168-hour same-hour-last-week value is a highly
  informative prior that the baseline discards entirely.

- **Limited capacity and training data.** D=32, 4 layers, 3,000 training samples, and 20
  epochs is a small fraction of available model capacity and the four-year dataset.

---

## New Architecture: Hierarchical CNN-Transformer

The core change is factorizing attention into two dedicated stages rather than mixing spatial
and temporal signals in one flat sequence.

### Stage 1 — Spatial Transformer (per timestep)

A shared-weight two-layer CNN (channels [32, 64], kernel size 3, BatchNorm + GELU) extracts
P = G² = 25 spatial tokens from each pooled 5×5 weather grid for every timestep. Learnable
spatial positional embeddings (shape P × D) are added to distinguish grid cells. A 2-layer
TransformerEncoder (8 heads, D=128, pre-LN, GELU) then runs over the P tokens
**independently for each timestep** — the time dimension is folded into the batch so the layer
operates on shape (B×T, 25, 128). The P output tokens are mean-pooled into a single
**weather summary vector** per timestep: (B, T, 128). This distills each hour's full spatial
context into one fixed-size representation before any temporal reasoning occurs.

### Stage 2 — Temporal Transformer

For each of the T = S+24 = 96 timesteps, the weather summary vector is added element-wise
to a tabular token embedding (demand + calendar projected to D=128 via a linear layer), and a
learnable timestep positional embedding is added. A 4-layer TransformerEncoder (8 heads,
D=128, pre-LN, GELU, dropout 0.35) then runs over all T=96 timestep tokens: (B, 96, 128).
The 24 future timestep outputs are decoded by a two-layer MLP (hidden 256, GELU) into the
final (B, 24, Z) predictions, which are denormalized to raw MWh.

The temporal transformer sees **96 tokens** rather than the ~2,496-token flat sequence of an
equivalent baseline, and every token in Stage 2 represents a complete hour-level summary
rather than a mix of spatial patches and tabular entries.

### Feature Engineering

**Weekly demand lag.** Each of the 24 future tabular tokens includes the actual demand
observed 168 hours prior (same wall-clock hour, one week earlier), extracted from
`history_energy[:, :24, :]` at inference time. During training it is read from
`demand_values[start + S - 168 : start + S - 144]`. This encodes weekly periodicity as a
direct input instead of requiring the model to recover it from the historical sequence.

**Holiday flag.** An 8th calendar feature (alongside hour_sin/cos, dow_sin/cos,
month_sin/cos, is_weekend) marks US federal holidays via the `holidays.US()` library.
Days such as Christmas, Thanksgiving, and July 4th suppress demand in ways that
`is_weekend` and day-of-week sinusoids cannot represent.

### Hyperparameters

| Setting | Part 1 Baseline | Part 2 Model |
|---|---|---|
| Embedding dim D | 32 | 128 |
| Transformer layers | 4 (single flat stage) | 2 spatial + 4 temporal |
| Attention heads | 4 | 8 |
| Dropout | — | 0.35 |
| Trainable parameters | — | 1,273,608 |
| Historical lookback S | — | 72 h |
| Spatial grid G | — | 5×5 (P = 25) |
| Optimizer | Adam, lr=1×10⁻⁴ | AdamW, lr=3×10⁻⁴, wd=0.05 |
| LR schedule | None | Cosine annealing to 1×10⁻⁵ |
| Loss | MSE (normalized) | MAE (MW) |
| Batch size | 8 | 16 |
| Epochs | 20 | 50 |
| Training samples | 3,000 | 34,873 |
| Training years | — | 2019, 2020, 2021, 2023 |

---

## Training Behavior

Training ran for 50 epochs on the Tufts HPC cluster (~73 s/epoch on a single GPU).
Validation loss fell steadily from 66 MW MAE (epoch 1) to a best of **41.94 MW MAE at
epoch 25**, then plateaued through epoch 50 (oscillating in the 42.0–42.3 MW range).
Train loss reached 13.05 MW by epoch 50, indicating a persistent train/val gap that reflects
the difficulty of generalizing across diverse weather and demand regimes in a held-out year.

---

## Results

Test set: December 2022 (31 days, 2022-12-01 → 2022-12-31), evaluated against the same
harness and zones as Part 1.

| Zone | Part 1 Baseline | Part 2 Hierarchical | Improvement |
|---|:-:|:-:|:-:|
| ME (Maine) | 10.97 % | 2.79 % | −8.18 pp |
| NH (New Hampshire) | 12.38 % | 1.86 % | −10.52 pp |
| VT (Vermont) | 11.80 % | 3.26 % | −8.54 pp |
| CT (Connecticut) | 10.42 % | 1.97 % | −8.45 pp |
| RI (Rhode Island) | 13.53 % | 2.24 % | −11.29 pp |
| SEMA | 16.81 % | 2.39 % | −14.42 pp |
| WCMA | 10.71 % | 1.60 % | −9.11 pp |
| NEMA_BOST | 13.10 % | 1.47 % | −11.63 pp |
| **Overall** | **12.47 %** | **2.20 %** | **−10.27 pp** |

The Part 2 model reduces overall MAPE by **10.27 percentage points (82% relative
reduction)**. Every zone improved. The largest absolute gain was SEMA (−14.42 pp), which
had been the weakest zone under the baseline. The best-performing zone is NEMA_BOST
at 1.47%, closely followed by WCMA at 1.60%.

---

## Discussion: Why the New Architecture Performed Better

**Factorized attention removes the spatial-temporal bottleneck.** The flat baseline mixes P
spatial patch tokens and 1 tabular token across all T timesteps in one sequence, forcing the
same 4-head, D=32 layers to simultaneously parse weather patterns within an hour and
demand dynamics across hours. The hierarchical design assigns each task its own dedicated
stage. The spatial transformer works over 25 tokens per hour in isolation; the temporal
transformer works over 96 hour-level summaries. Each stage's attention is semantically
coherent and sequence lengths are short enough to be fully covered by a small number of
layers.

**The weekly lag anchors the forecast.** ISO-NE demand on a given December Tuesday
afternoon is highly predictable from the same Tuesday afternoon one week prior. Giving the
temporal transformer this prior as a direct input means it only needs to correct for residual
deviations — weather anomalies, holidays, load growth — rather than reconstruct the weekly
demand level from the historical sequence. This single feature likely accounts for a
substantial share of the improvement over the baseline.

**Larger model and full-dataset training.** Scaling from D=32 to D=128 and from 3,000 to
34,873 training samples allows the model to represent fine-grained seasonal, geographic,
and weather-driven patterns. The Part 1 baseline, trained on a small sample subset, was
underfitting the multi-year demand seasonality visible across all eight zones.

**MAE loss and cosine LR schedule.** MSE loss disproportionately penalizes large errors,
which can over-weight rare extreme weather events at the cost of typical-day accuracy. MAE
produces better-calibrated average forecasts. Cosine annealing allows large parameter
updates early in training and careful refinement late, compared to the flat learning rate of the
baseline.

**Holiday flag.** Federal holidays that fall on weekdays cause demand drops that neither
`is_weekend` nor day-of-week sinusoids can encode. The explicit holiday feature directly
signals these calendar anomalies during both training and inference.
