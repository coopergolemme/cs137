# Weather Prediction Model: Improvements & Fixes (Part 2)

This document describes the iterative changes made to the model architecture and training procedure following the baseline run (Job 383605). Changes were motivated by diagnosing specific failure modes observed in the training logs.

---

## Baseline Diagnosis (Job 383605)

The original model achieved a best validation AUC of **0.6361**, with high epoch-to-epoch instability (oscillating between 0.55 and 0.64). Three root causes were identified:

1. **Binary task starved of gradient signal.** The BCE focal loss value was ~0.006, and with a weight of `10.0`, its contribution to the total loss was ~0.06 — roughly 8% of the total gradient. The model had almost no incentive to learn the binary classification task.

2. **Excessive L2 regularization.** `weight_decay=1e-2` is 100× larger than the standard `1e-4`. This suppressed the binary head weights toward zero, reinforcing the trivial all-negative prediction strategy (viable given the 2% positive rate).

3. **Premature LR reduction and early stopping.** The `ReduceLROnPlateau` scheduler had `patience=2`, meaning it halved the learning rate after just 2 epochs of noisy AUC — which occurred due to natural variance, not true plateau. Similarly, `early_stop_patience=5` terminated training before the model had time to converge.

4. **Shared FC bottleneck.** All three output heads (base regression, precipitation, binary) competed for the same 128-dimensional shared representation, causing gradient interference between tasks with very different loss scales.

---

## Round 1 Changes (Jobs 383605 → 385607)

### Architecture (`model.py`)

**Replaced shared FC with task-specific branches.**

Previously, a single `shared_fc` layer fed all three output heads. This was replaced with two independent branches:

* **`reg_fc`** (128 → 128, ReLU, Dropout 0.3): feeds the base regression head.
* **`bin_fc`** (128 → 64 → 32, ReLU, Dropout 0.3): a dedicated two-layer MLP for the binary head.

This prevents the regression gradients (which dominate by scale) from overriding the binary classification signal in the shared representation.

### Loss Function (`train.py`)

| Component | Before | After | Rationale |
|---|---|---|---|
| BCE weight | `10.0` | `100.0` | Binary task contribution raised from ~8% to ~39% of total gradient |
| APCP weight | `0.02` | `0.05` | Minor increase to improve precipitation generalization |

With BCE weight = 100, the focal loss term contributes ~0.6 to the total loss (vs. Base MSE ~0.7), giving the binary head a meaningful learning signal.

### Optimization & Training (`train.py`)

| Hyperparameter | Before | After | Rationale |
|---|---|---|---|
| `weight_decay` | `1e-2` | `1e-4` | Removes excessive regularization that suppressed binary head weights |
| LR scheduler `patience` | `2` | `4` | Prevents premature LR reduction from AUC noise |
| `early_stop_patience` | `5` | `8` | Gives the model more time to converge |
| Max `epochs` | `15` | `30` | Extends training window to match the new patience settings |

### Result (Job 385607)

Best validation AUC improved to **0.7254** (Epoch 13), up from 0.6361. BCE values rose to ~0.013–0.015, confirming the binary head was now learning. However, the validation total loss showed a diverging trend (2.1 → 3.3 by Epoch 17) while training loss stayed flat, indicating **overfitting on the precipitation regression task**.

---

## Round 2 Changes (Post-385607)

### Architecture (`model.py`)

**Added dedicated APCP branch.**

The precipitation head previously shared `reg_fc` with the base variables. A new `apcp_fc` branch (128 → 32, ReLU, Dropout **0.4**) was added, with the `apcp_head` now consuming its 32-dimensional output. The higher dropout rate (0.4 vs. 0.3) directly addresses the observed overfitting on precipitation.

All three tasks now have fully independent feature pathways from the pooled representation:

* **`reg_fc`** → `base_head` (5 outputs)
* **`apcp_fc`** → `apcp_head` (1 output)
* **`bin_fc`** → `bin_head` (1 output)

### Loss Function (`train.py`)

**Removed the `> 2.0` precipitation mask from training loss.**

Previously, the APCP Smooth L1 loss was computed only on samples where the unnormalized precipitation target exceeded 2.0 mm. This created a train/val distribution mismatch: training optimized only on high-precipitation samples, but the val total loss included all samples. Removing the threshold makes the loss consistent across both splits and forces the model to generalize to dry and light-precipitation cases as well.

> Note: the `> 2.0` threshold is retained in the validation RMSE reporting metric for interpretability, since that metric specifically tracks performance on significant precipitation events.

### Optimization (`train.py`)

**Learning rate halved from `1e-4` to `5e-5`.** The diverging val loss from Epoch 8 onward suggested the model was taking steps too large to generalize. A lower LR slows down weight updates and is more compatible with the existing `ReduceLROnPlateau` scheduler.

### Summary of All Current Hyperparameters

| Parameter | Original | Current |
|---|---|---|
| Learning rate | `1e-4` | `5e-5` |
| Weight decay | `1e-2` | `1e-4` |
| BCE loss weight | `10.0` | `100.0` |
| APCP loss weight | `0.02` | `0.05` |
| APCP training mask | `> 2.0 mm` | All non-NaN |
| LR scheduler patience | `2` | `4` |
| Early stop patience | `5` | `8` |
| Max epochs | `15` | `30` |
| Binary head | Shared FC | Dedicated `bin_fc` (128→64→32) |
| APCP head | Shared FC | Dedicated `apcp_fc` (128→32, Dropout 0.4) |
| Base head | Shared FC | Dedicated `reg_fc` (128→128) |
