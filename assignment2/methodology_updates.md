# Methodology Updates for Predictive Modeling

Based on the analysis of training metrics exhibiting signs of negative transfer and loss imbalance—specifically that continuous targets (`Base MSE` and `APCP MSE`) were improving while binary classification (`BCE`) remained stagnant or worsened—the following methodological enhancements were implemented:

## 1. Multi-Head Architecture (Mitigating Negative Transfer)
Previously, a single fully-connected (`Linear`) layer projected the high-dimensional feature maps directly into the 7 outputs, forcing all tasks to share identical final representations. Because the regression and classification tasks require orthogonally different features, this resulted in *negative transfer*, where the dominating gradients from continuous attributes dismantled features learned for classification.
- **Change:** The model was refactored into a Multi-Task Learning (MTL) architecture. The shared representation is now passed into three independent linear heads: `base_head` (5 continuous outputs), `apcp_head` (1 skewed continuous output), and `bin_head` (1 binary classification output). 
- **Impact:** Decoupling the heads permits the model to learn task-exclusive non-linear combinations of the shared feature map, preserving classification performance.

## 2. Gradient and Loss Re-weighting
During training, the `mse_apcp` hovered around 40-50 whereas `mse_base` was ~1.0 and `BCE` was ~6.5 (post multiplier). Approximately 86% of the gradient updates were driven exclusively by the `APCP` target, instructing the optimizer to ignore classification.
- **Change:** Down-scaled the `mse_apcp` factor in the total weighted loss to suppress its disproportionately massive variance:
  `weighted_loss = mse_base + (mse_apcp * 0.1) + (bce * 10.0)`
- **Impact:** Aligns the backpropagated gradients to roughly the same order of magnitude so the optimizer attends equitably to the BCE classification target.

## 3. Focal Loss for Class Imbalance
The dataset exhibits extreme class imbalance. Originally, `BCEWithLogitsLoss` utilized a static `pos_weight=11.5`. However, making an incorrect but highly confident prediction during early epochs with such a multiplier causes massive destabilizing spikes in BCE loss.
- **Change:** Replaced the standard BCE loss with **Focal Loss** ($\alpha=0.75, \gamma=2.0$). 
- **Impact:** Focal loss dynamically reduces the relative gradient weight of easily classified negative examples and smoothly concentrates training on challenging, rare positive cases without inducing erratic gradient explosions.

## 4. Learning Rate Optimization
The learning rate was originally clamped at an excessively conservative `1e-6` in an effort to manually curb the chaotic gradients stemming from the loss scaling discrepancies.
- **Change:** With the gradients mathematically stabilized via Focal Loss and re-weighting, the learning rate was restored to a more exploratory `1e-4`.
- **Impact:** Accelerates convergence and empowers the model to actually traverse local minima and escape the ~0.50 (random-guessing) AUC plateau for extreme events.

## 5. Learning Rate Scheduler (ReduceLROnPlateau)
*Evidence:* During the 5-epoch training run, the Validation AUC exhibited high volatility—jumping from 0.6040 (Epoch 2) down to 0.5320 (Epoch 4) and back up to 0.6307 (Epoch 5). This erratic oscillation indicates the optimizer was bouncing around the local minimum instead of smoothly converging, pointing to a learning rate (`1e-4`) that was too high for the final stages of fine-tuning.
- **Change:** Implemented a `ReduceLROnPlateau` scheduler monitoring `val_auc`, set to halve the learning rate (`factor=0.5`) if the metric stalls for 2 consecutive epochs.
- **Impact:** Dampens the optimizer's step size as training progresses, allowing the network to settle into a deeper, smoother minimum without destabilizing the previously learned classification boundaries.

## 6. Spatial Data Augmentations (Geometric Regularization)
*Evidence:* The training metrics vividly demonstrated overfitting. While the Train Total Loss continuously decreased (1.5185 at Epoch 2 $\rightarrow$ 1.3897 at Epoch 5), the Validation Total Loss reached its lowest point at Epoch 2 (1.2467) and then continuously increased for the remainder of the run (peaking at 1.5531).
- **Change:** Integrated random spatial transformations (Horizontal and Vertical Flips with 50% probability) dynamically onto the input tensors during the training loop.
- **Impact:** Since the meteorological data represents spatially anchored 2D weather fronts, these symmetrical flips effectively multiply the geographic diversity of the dataset. This heavily penalizes the model for memorizing specific pixel sequences and forces it to learn shift-invariant topological features of extreme weather.

## 7. Extended Training Horizon with Early Stopping
*Evidence:* The previous training loop was hardcoded to stop at 5 epochs. However, the model achieved its absolute best classification performance (AUC: 0.6307) precisely on its final epoch. This suggests the binary classification head had not fully completed its learning trajectory and was artificially interrupted.
- **Change:** Increased the maximum training budget to 15 epochs and introduced an Early Stopping patience mechanism (5 epochs) tied to monitoring the validation AUC.
- **Impact:** Grants the model the necessary runway to fully optimize the challenging rare-event classification task while providing a robust, automated abort switch to halt training if the plateau inevitably resolves into terminal overfitting.
