# Weather Prediction Model Architecture & Training Procedure

This document provides a summary of the model design and the training pipeline implemented for the gridded weather forecasting task. 

## 1. Model Architecture: `WeatherCNN`
The model is a spatially-aware Convolutional Neural Network designed to handle 35-channel 2D spatial inputs. Its architecture intentionally avoids premature global pooling (typical in models like VGG) to preserve location-specific information deeper into the network. Note that the total number of predicted outputs is 7.

### **Network Breakdown**
* **Stem:** 
  The network begins with a $5\times5$ Convolution (stride 2, padding 2) that downsamples the input spatial resolution and maps the 35 input channels to 32 feature channels. This is followed by Batch Normalization and a ReLU activation.
* **Encoders (Feature Extraction):** 
  Three consecutive Convolutional Blocks extract higher-level features. Each block contains two $3\times3$ convolutions with Batch Normalization and ReLU, followed by spatial Dropout. Downsampling happens after each block via a $2\times2$ MaxPooling layer.
  * **Encoder 1:** $32 \rightarrow 64$ channels (Dropout: 0.05)
  * **Encoder 2:** $64 \rightarrow 128$ channels (Dropout: 0.10)
  * **Encoder 3:** $128 \rightarrow 256$ channels (Dropout: 0.15)
* **Bottleneck & Head Conv:** 
  A bottleneck containing two $3\times3$ convolutions processes the 256-channel feature maps. A $1\times1$ convolution then compresses the channels from $256 \rightarrow 128$ over the coarse spatial layout.
* **Global Pooling & Shared FC:** 
  Adaptive Average Pooling reduces the final spatial dimensions to $1\times1$. The representations are flattened and passed through a shared Linear layer (128 units) with ReLU and a $0.3$ Dropout rate.
* **Output Heads (Multi-Task):**
  The network splits into three separate output heads:
  1. **Base Head:** A linear layer predicting 5 base regression targets.
  2. **Precipitation (APCP) Head:** A linear layer outputting 1 target, passed through a `Softplus` activation to ensure non-negative precipitation predictions.
  3. **Binary Head:** A linear layer outputting 1 raw logit for binary classification task (e.g., event occurrence).

---

## 2. Training Procedure
The model is trained as a multi-task learning problem, balancing 5 base regression variables, 1 precipitation regression variable, and 1 binary classification variable. 

### **Data Handling**
* **Train/Val Split:** The dataset is split temporally. Years 2018 and 2019 are used for training, while the year 2020 is held out for validation.
* **Loaders:** `DataLoader` feeds the network using a batch size of 16. Missing (NaN) values are automatically filtered out during the forward pass.

### **Loss Function**
A custom composite multi-task loss is used to jointly optimize the three heads:
1. **Base MSE:** Mean Squared Error for the 5 continuous base targets.
2. **APCP Loss:** Smooth L1 Loss specifically computed for true unnormalized target precipitation values $> 2.0$. It is scaled down heavily by a weight of `0.02`. 
3. **Binary Loss:** A Focal Loss (Alpha $= 0.98$, Gamma $= 2.0$) operating on the raw logits to address severe class imbalance. It is scaled up heavily by a weight of `10.0`.
   
**Total Loss:** `Base_MSE + (0.02 * APCP_Smooth_L1) + (10.0 * Binary_Focal_Loss)`

### **Optimization & Hyperparameters**
* **Optimizer:** Adam optimizer with a learning rate of $1\times 10^{-4}$ and an L2 weight decay of $1\times 10^{-2}$.
* **Gradient Clipping:** Gradients are clipped to a maximum norm of $1.0$ to prevent explosive gradients and stabilize multi-task training.
* **Learning Rate Scheduler:** Uses `ReduceLROnPlateau` monitoring the validation AUC (mode `"max"`). If the AUC plateaus for 2 epochs, the learning rate is halved (`factor=0.5`).

### **Evaluation & Early Stopping**
* **Validation Metrics:** Total validation loss, exact scale Base RMSE, exact scale APCP RMSE, and ROC-AUC. 
* **Model Checkpointing:** Over a maximum of 15 epochs, the best model weights are saved based exclusively on the highest validation ROC-AUC score.
* **Early Stopping:** Training terminates early if the validation AUC stops improving for 5 consecutive epochs.
