# Part 1: CNN Architecture and Training Procedure

## Model Architecture (`model.py`)
The model is designed as a deep Convolutional Neural Network (CNN) similar to a VGG-style feature extractor, structured to map spatial weather grids to point forecasts.
- **Input shape:** `(Batch, 42, 450, 449)`
  - Re-mapped from the raw `[450, 449, 42]` structure by permuting dimensions so PyTorch can process standard channels-first representations, where `C=42` captures 35 different atmospheric pressures/variables alongside initial $t=0$ target states.

The CNN applies sequential blocks combining the following mechanisms:
1. **Convolutions (`Conv2d`):** Extraction of spatial patterns across weather layers via overlapping `3x3` convolutions. The channels progressively expand `42 -> 64 -> 128 -> 256 -> 512 -> 512` mapping abstract meteorological indicators like thermal fronts.
2. **Standardization (`BatchNorm2d`):** Batch normalization prevents vanishing/exploding gradients in high-dimensional weather regression, stabilizing the deeper layers.
3. **Pooling (`MaxPool2d`):** Repeated spatial downsampling halves the map size repeatedly to distil regional data into coarser but semantically denser representations until adapting via `AdaptiveAvgPool2d(1, 1)` to obtain a single comprehensive 512-dimensional vector.

The classifier head uses dense linear layers (`512 -> 256 -> 7`) bundled with Dropout layers (`p=0.5`) to prevent overfitting. It outputs raw logits spanning 7 indices (6 continuous weather measurements + 1 probability logit for precipitation > 2mm). 

## Training Procedure (`train.py`)
- **Dataset Splitting:** The pipeline loads valid `.pt` records across 2018 and 2019 for the training set, and holds out the year 2020 explicitly purely for evaluation to prevent data leakage.
- **Data preprocessing:** Inputs contain sparse NaN encodings for sensor failures, which are zero-imputed uniformly to prevent propagation of missing gradients.
- **Loss Computation:** The model optimizes a compound gradient:
  - Base `MSELoss()` is calculated purely on the first 5 weather conditions (TMP, RH, UGRD, VGRD, GUST).
  - A conditional `MSELoss()` evaluates the 6th layer (APCP numerical total) strictly exclusively when the target exceeds a baseline 2mm metric.
  - Finally, `BCEWithLogitsLoss()` anchors the 7th node explicitly towards probability scores for predicting significant precipitation likelihood.
- **Optimization:** We use the Adam optimizer mapped to a constant sub-learning rate `1e-4` trained across shuffled minibatches (`16`), regularly checkpointing the model dict state caching the best ROC-AUC results logged during intermediate validation loops on the 2020 layout data.
