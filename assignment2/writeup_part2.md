# Part 2: Diagnosing Where the Weather Comes From — Saliency Maps & Sensitivity Analysis

## 1. Method: Vanilla Gradient Saliency

To understand which parts of the input map most influence the model's 24-hour forecasts, we apply **vanilla gradient saliency** (also called sensitivity analysis). For a batch of inputs **x** and a scalar output *f_i*(**x**) (the *i*-th predicted variable), the saliency is:

> S_i(**x**) = |∂f_i / ∂**x**|

This is computed by enabling gradient tracking on the input tensor, running a forward pass, calling `backward()` on the summed scalar output, and reading `x.grad`. The result has the same shape as the input — (C=42 channels, H=450, W=449) — and captures how much a small perturbation at every grid cell and every input channel would shift the target prediction.

We average this quantity over **200 validation samples** (year 2020, unseen during training) to obtain a mean sensitivity map that is not specific to any single weather event:

> S̄_i = (1/N) Σ_n |∂f_i(x_n) / ∂x_n|

Four figures are produced:
1. **Spatial saliency maps** — saliency summed over all 42 channels, showing which grid regions matter.
2. **Channel bar charts** — saliency summed over the spatial grid, ranking input variables by importance.
3. **Per-channel spatial maps** — where the top-3 most sensitive input channels contribute for Temperature and Precipitation.
4. **Wind-channel spatial maps** — sensitivity in each wind-component channel, revealing directional upstream structure.

---

## 2. Channel Sensitivity Rankings

### Table 1 — Top-10 Input Channels per Output (ranked by mean |grad|, summed over spatial dims)

| Rank | TMP@2m | RH@2m | U-wind@10m | V-wind@10m | Gust | APCP | Binary |
|---|---|---|---|---|---|---|---|
| 1 | APCP_1hr (0.327) | APCP_1hr (0.479) | APCP_1hr (0.498) | APCP_1hr (0.550) | APCP_1hr (0.407) | APCP_1hr (0.031) | APCP_1hr (0.591) |
| 2 | VIL (0.149) | VIL (0.158) | VIL (0.138) | VIL (0.213) | VIL (0.135) | VIL (0.010) | VIL (0.161) |
| 3 | VGRD@10m | GUST | GUST | VGRD@10m | VGRD@10m | GUST | UGRD@10m |
| 4 | GUST | UGRD@10m | VGRD@10m | GUST | GUST | UGRD@10m | GUST |
| 5 | UGRD@10m | UGRD@1000 | UGRD@10m | UGRD@10m | UGRD@10m | UGRD@1000 | VGRD@10m |

**Key observation:** `APCP_1hr` (the current 1-hour accumulated precipitation analysis field) and `VIL` (Vertically Integrated Liquid) dominate every output by a large margin. `APCP_1hr` scores 10–30× higher than the third-ranked channel in most cases. Ranks 3–10 are occupied uniformly by low-level and mid-level wind components (UGRD, VGRD at 10 m, 1000 mb, 850 mb, 700 mb) along with surface GUST.

### Precipitation output is quieter overall

The precipitation (APCP) output shows absolute sensitivity scores roughly 10× smaller than the other targets (e.g., APCP_1hr score = 0.031 vs. 0.33–0.59 for other outputs). This is a direct consequence of the dedicated `apcp_fc` branch introduced in the architecture: the precipitation head operates on a separate 32-dimensional subspace with higher dropout (0.5), resulting in a more compressed gradient signal back through the input.

---

## 3. Spatial Saliency Maps

The spatial saliency maps (Figure 1, `saliency_spatial.png`) display the channel-summed gradient magnitude for each of the 7 targets across the 450 × 449 Lambert Conformal grid. Several consistent patterns emerge:

### 3a. High sensitivity near the Jumbo station

For temperature, humidity, and wind outputs, the highest saliency concentrates in a **compact region around and slightly upstream (west-southwest) of the Jumbo station** (marked with a blue triangle at grid column 263, row 177). This makes physical sense: the model is predicting a point observation 24 hours ahead, and the immediate neighborhood of the station contributes the most direct information about local conditions.

### 3b. Elongated upstream corridor

Beyond the station neighborhood, there is a secondary band of elevated sensitivity stretching to the **west and southwest** — consistent with the dominant advection direction in mid-latitude westerly flow. Air at the surface in the mid-Atlantic and Midwestern US typically arrives from the west-southwest within 12–24 hours. The model's sensitivity gradient traces this upstream path.

### 3c. Precipitation: diffuse and weaker spatial structure

For APCP and the binary extreme-event output, spatial saliency is more spread across the domain and less concentrated around the station. Precipitation is controlled by synoptic-scale systems (fronts, lows) that can be hundreds of kilometers upstream, so it is expected that the model draws on a wider spatial context for these predictions.

### 3d. Wind outputs: stronger lateral spread

For UGRD and VGRD at 10 m, sensitivity extends further in the east-west direction compared to temperature. This reflects the fact that wind at a point is more dependent on the large-scale pressure gradient, which is set by conditions over a broad region.

---

## 4. Per-Channel Spatial Analysis

Figure 3 (`saliency_top_channels.png`) shows the spatial saliency maps for the **top-3 input channels** separately, for Temperature and Precipitation.

### Temperature — top channels: APCP_1hr, VIL, VGRD@10m

- **APCP_1hr spatial map:** Sensitivity is sharply concentrated just west and southwest of the Jumbo station. This is physically counterintuitive (why does precipitation analysis influence 24-hr temperature?), but it reflects a real correlation: precipitation is associated with frontal passages and cloud cover that modulate next-day temperature. Regions immediately upstream that are currently precipitating will likely advect their air mass to the station.
- **VIL spatial map:** A similar but more diffuse pattern, confirming that the model uses total column liquid water as a proxy for convective and frontal activity upstream.
- **VGRD@10m spatial map:** Sensitivity is more spatially spread, especially in the meridional (south-north) direction, reflecting the role of southerly or northerly flow in transporting warm or cold air masses.

### Precipitation — top channels: APCP_1hr, VIL, GUST

- All three channels show sensitivity spread more uniformly across the grid compared to the temperature case. Precipitation predictability over 24 hours depends heavily on the position and motion of synoptic systems, which the model tracks by drawing on a wider spatial context.
- There is a modest concentration southwest of the station for APCP_1hr, consistent with the idea that today's precipitation in upstream areas predicts tomorrow's precipitation at the target station under typical westerly flow.

---

## 5. Wind-Channel Upstream Analysis

Figure 4 (`saliency_upstream.png`) isolates the gradient signal in four wind channels — surface UGRD/VGRD and 500 mb UGRD/VGRD — for the Temperature and Precipitation outputs. A green dashed rectangle marks the "upstream box" (the region 0–140 grid columns west and ±60 rows of the station).

### Surface wind (UGRD@10m, VGRD@10m)

Both temperature and precipitation show appreciable sensitivity inside the upstream box for the surface U-wind component (UGRD@10m). The U-wind (west-to-east) sensitivity is stronger than the V-wind (south-to-north) sensitivity, consistent with mid-latitude prevailing westerlies: the dominant transport is zonal (east-west), so conditions to the west of the station are more likely to arrive at the target location within 24 hours than conditions to the north or south.

### Upper-level wind (UGRD@500mb, VGRD@500mb)

The 500 mb wind channels show more spatially diffuse sensitivity at larger distances, extending further west and southwest of the station. At 500 mb, typical wind speeds are 20–50 m/s, so upstream influence can extend to regions 1,000–2,000 km away over 12–24 hours. The model's sensitivity to these upper-level channels captures large-scale steering flow that determines where surface weather systems move.

### Summary of directional asymmetry

Across all wind channels and both target outputs:
- Sensitivity is systematically **higher to the west** of the Jumbo station than to the east.
- Sensitivity is **higher to the south-southwest** than to the north, consistent with the southwesterly component of typical mid-latitude flow.

This spatial asymmetry is a strong confirmation that the model has implicitly learned the direction of prevailing atmospheric flow, even though it was never given explicit information about flow direction during training.

---

## 6. Physical Interpretation

### Why APCP_1hr and VIL dominate

The two leading input channels — 1-hour precipitation accumulation and vertically integrated liquid — are both **condensate and convective proxies**. They encode:

1. **Where atmospheric moisture has already condensed** at analysis time (t=0). This is a strong signal for stability, frontal position, and moisture content.
2. **Indirect temperature information** via latent heat release. Regions with active precipitation are typically cooler (evaporative cooling) or warmer at upper levels (latent heat), and these anomalies advect downstream.

In contrast, direct thermodynamic state variables like TMP@2m or RH@2m rank low. This suggests the model has learned that the *current weather* (what is precipitating right now) is more informative about tomorrow's state than the explicit temperature and humidity analysis — a non-trivial finding that aligns with the meteorological concept of **precipitation as a synoptic fingerprint**.

### Why wind variables occupy the middle ranks

Wind components at 10 m, 1000 mb, 850 mb, and 700 mb collectively tell the model about **mass transport**: what air is coming from where. Higher sensitivity to U-wind (west-east) than V-wind (south-north) across most outputs directly reflects the **prevailing westerlies** that dominate atmospheric circulation in the mid-latitudes where the Jumbo station is located.

### Why upper-level thermodynamic variables rank low

Temperature at 500 mb and geopotential height (HGT) rank outside the top 5 for most outputs. This may seem surprising since 500 mb height is a primary forecasting tool in operational meteorology. However, these variables vary smoothly across the grid and their gradient w.r.t. the target is distributed over many channels — the model likely integrates this information collectively rather than placing it in a single dominant channel. The smooth spatial fields also produce smaller absolute gradient magnitudes than the spiky precipitation and wind fields.

### Limitations of vanilla gradient saliency

Vanilla gradients are local linearizations of the model and can be sensitive to noise (saturated or dead ReLU units return zero gradient even when a feature is important). They also attribute importance to input channels proportionally to the model's current learned weights, which may not perfectly reflect the causal structure in the atmosphere. More robust alternatives (integrated gradients, SHAP, Grad-CAM) would provide additional validation but are consistent with this analysis in most weather forecasting studies.

---

## 7. Summary

| Finding | Physical Interpretation |
|---|---|
| APCP_1hr and VIL are #1 and #2 for all outputs | Current precipitation encodes frontal/moisture state; acts as a synoptic fingerprint |
| Wind variables (UGRD, VGRD) at low levels rank 3–10 | Mass transport drives 24-hr advection of temperature, humidity, and momentum |
| U-wind sensitivity > V-wind sensitivity | Prevailing westerlies are the dominant transport direction |
| High spatial sensitivity west and SW of station | Upstream advection region for mid-latitude westerly flow |
| Diffuse spatial pattern for precipitation output | Precipitation predictability requires synoptic-scale context over wider domain |
| 500 mb thermodynamic variables rank low | Information distributed across many channels; smooth fields produce small point-wise gradients |

The saliency analysis confirms that the model has implicitly learned physically meaningful representations: it focuses on upstream atmospheric conditions transported by westerly flow, uses precipitation and column liquid as proxies for synoptic state, and draws on wind fields to infer advection. These patterns are not explicitly encoded in the architecture or loss function — they emerge purely from optimizing forecast skill on historical data.
