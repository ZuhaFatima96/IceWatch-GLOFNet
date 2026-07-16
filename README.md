# IceWatch

**A multimodal deep-learning framework for glacial lake outburst flood (GLOF) early warning**

Shisper Glacier · Hunza Valley · Karakoram · Pakistan

[![Paper](https://img.shields.io/badge/paper-NHESS%20(under%20review)-b31b1b)](#-citation)
[![Dataset](https://img.shields.io/badge/dataset-GLOFNet-orange)](https://doi.org/10.1109/ICoDT269104.2025.11360730)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TerraFlow-ee4c2c)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-RiskFlow%20%7C%20TempFlow-ff6f00)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## Overview

Glacial lake outburst floods release millions of cubic metres of water within hours, threatening communities, roads and hydropower infrastructure across high-mountain Asia. Operational monitoring still leans on hydrological breach models, threshold-based water indices and manual satellite interpretation; methods that update slowly, need expert time, and degrade badly under cloud cover.

**IceWatch** takes a different route. It runs two independent streams over the same glacier and only raises an alert when they agree:

- **Vision stream** — CNN reads Sentinel-2 multispectral imagery for the visual signature of lake formation and outburst conditions.
- **Tabular stream** — Transformer estimates the glacier's kinematic state from NASA ITS_LIVE velocity records, while a stacked bidirectional LSTM forecasts surface temperature from MODIS observations.

Neither stream is trusted alone. Cloud remnants and fresh snow can fool a vision model; a thermal anomaly on its own means little. **FusionFlow** issues a high-confidence alert only when the visual evidence *and* the physical evidence are elevated at once; otherwise the case goes to a human analyst. On the May 2022 Shisper outburst sequence, both streams independently flagged both events, and thermal forecast skill held out to a **72–96 hour horizon**: days of warning rather than hours.

---


https://github.com/user-attachments/assets/7982dd40-490e-47ba-a460-78fbe162fd99


## Architecture

<div align="center">

<img width="2280" height="1760" alt="icewatch_architecture" src="https://github.com/user-attachments/assets/a9b8dd21-5004-49a7-9b66-871aabcf218a" />

</div>

### The four modules

| Module | Type | Input | Output | Key configuration |
|---|---|---|---|---|
| **RiskFlow** | CNN classifier | Sentinel-2 patch, 128 × 128 × 6 | GLOF probability `P_vis` | Conv 32 → Conv 64 → Dense 64 → sigmoid · 3.71 M params · BCE loss |
| **TerraFlow** | Transformer encoder | 32-day ITS_LIVE window | Peak surface velocity | 4 layers · 8 heads · d = 256 · 5.5 M params · L1 (median) loss |
| **TempFlow** | Stacked BiLSTM | 30-day MODIS LST window | Surface-temperature forecast | BiLSTM 128 → LSTM 64 → LSTM 32 · dropout 0.4/0.4/0.3 · MSE loss |
| **FusionFlow** | Decision-level fusion | `P_vis`, `P_tab` | Alert / review / quiescent | Agreement rule at θ = 0.5 |

### The decision rule

```
alert      ⟺  P_vis > 0.5  AND  P_tab > 0.5     → issue warning
review     ⟺  exactly one stream > 0.5          → analyst adjudicates
quiescent  ⟺  neither stream > 0.5              → no action
```

Why the rule matters: on **5 April 2022** the tabular score reached 0.657 — higher than on either real event date — while the vision stream saw essentially nothing. The streams disagreed, no alert fired, and the day was correctly quiet. That is the false-alarm mode the dual-stream design exists to contain.

---

## Results

| Module | Metric | Value |
|---|---|---|
| RiskFlow | Accuracy (n = 71) | **0.94** |
| RiskFlow | GLOF-class recall | **0.96** (95 % CI 0.79–0.99) |
| RiskFlow | GLOF-class precision / F1 | 0.88 / 0.92 |
| TerraFlow | Test MAE (spatial holdout) | **20.5 m yr⁻¹** |
| TempFlow | MAE / RMSE / R² (N = 19 748) | **1.34 K** / 2.00 K / **0.956** |
| System | Reliable forecast horizon | **72–96 h** |

**May 2022 case study**: Both events flagged, no false alarms across the evaluated sequence:

| Date | TerraFlow (m yr⁻¹) | TempFlow (°C) | `P_tab` | `P_vis` | Decision | Actual |
|---|---|---|---|---|---|---|
| 5 Apr | 12.9 | −5.0 | 0.657 | 0.00 % | no alert | — |
| 10 Apr | 53.3 | −4.8 | 0.457 | 0.01 % | no alert | — |
| 15 Apr | 16.7 | −6.1 | 0.488 | 0.00 % | no alert | — |
| **10 May** | **455.3** | **−0.2** | **0.616** | **83.2 %** | **🚨 ALERT** | **GLOF** |
| **25 May** | **204.7** | **−1.1** | **0.566** | **89.4 %** | **🚨 ALERT** | **GLOF** |
| 14 Jun | 13.4 | −5.6 | 0.491 | 1.4 % | no alert | — |

### Scope and limitations

We would rather you read these here than discover them later:

- Evaluation rests on **one confirmed outburst sequence at Shisper glacier** ; transferability is untested.
- Image augmentation precedes the train/val/test split, so vision-stream metrics reflect augmented views of a small set of source scenes and are **optimistic** with respect to fully independent scenes.
- TerraFlow's velocity split is **spatial** (unseen grid points, same melt season), not temporal.
- TerraFlow **estimates** concurrent peak velocity from mean velocity; it does not extrapolate forward. Anticipatory skill comes from TempFlow.
- Median (L1) regression trades accuracy in the extreme surge tail for robust baseline tracking.

---

## Demo

https://icewatch.streamlit.app/



https://github.com/user-attachments/assets/e26c1115-ac7c-4790-8f74-64bc8e37143c



---

## Dataset

IceWatch is built entirely on **GLOFNet**, our separately published multimodal dataset for Shisper Glacier. GLOFNet harmonises three Earth-observation streams in space and time:

| Source | Content | Coverage |
|---|---|---|
| Sentinel-2 MSI | 6 bands (B2, B3, B4, B8, B11, B12), 128 × 128 patches | 10–20 m |
| NASA ITS_LIVE | 73.6 M → 11.9 M velocity records | 1988–2024 |
| MODIS MOD11A1 v6.1 | Daily land-surface temperature | 2000–2024, 1 km |

All acquisition, cloud masking, quality filtering, labelling and harmonisation procedures are documented in the GLOFNet paper; **this repository does not duplicate them**.

> 📄 Fatima, Z., Sohaib, M. A., Talha, M., Sultana, S., Kanwal, A., and Perwaiz, N.: *GLOFNet — a multimodal dataset for GLOF monitoring and prediction*, ICoDT2, 2025. [DOI: 10.1109/ICoDT269104.2025.11360730](https://doi.org/10.1109/ICoDT269104.2025.11360730)

---



Download the GLOFNet dataset (link in the paper above), then run the notebooks in `notebooks/`. RiskFlow and TempFlow use TensorFlow/Keras; TerraFlow uses PyTorch with mixed-precision training and expects a CUDA device.

---

## Citation

If you use this code or build on the framework, please cite the IceWatch paper:

```bibtex
@article{fatima2026icewatch,
  author  = {Fatima, Zuha and Sohaib, Muhammad Anser and Talha, Muhammad and
             Kanwal, Ayesha and Sultana, Sidra and Perwaiz, Nazia},
  title   = {{IceWatch}: a multimodal deep-learning framework for early warning of
             glacial lake outburst floods, demonstrated at {Shisper} {Glacier}, {Karakoram}},
  journal = {Natural Hazards and Earth System Sciences},
  note    = {Under review},
  year    = {2026}
}
```

and the dataset paper:

```bibtex
@inproceedings{fatima2025glofnet,
  author    = {Fatima, Zuha and Sohaib, Muhammad Anser and Talha, Muhammad and
               Sultana, Sidra and Kanwal, Ayesha and Perwaiz, Nazia},
  title     = {{GLOFNet} -- a multimodal dataset for {GLOF} monitoring and prediction},
  booktitle = {Proc. Int. Conf. on Digital Futures and Transformative Technologies (ICoDT2)},
  pages     = {1--8},
  doi       = {10.1109/ICoDT269104.2025.11360730},
  year      = {2025}
}
```

---

## Authors

Zuha Fatima · Muhammad Anser Sohaib · Muhammad Talha · Ayesha Kanwal · Sidra Sultana · **Nazia Perwaiz** (corresponding)

School of Electrical Engineering and Computer Science (SEECS)
National University of Sciences and Technology (NUST), Islamabad, Pakistan

---

## License

Released under the MIT License — see [LICENSE](LICENSE).

Sentinel-2 data © European Union / ESA / Copernicus. ITS_LIVE and MODIS products courtesy of NASA.

**Satellite Imagery:** MODIS <br /><br />
**Ice Velocity Data:** ITS_LIVE <br /> https://nsidc.org/apps/itslive/ <br /><br />
**Landsat Climate Data:** Local meteorological stations <br /><br />
**Hydrological Data:** River discharge and glacial lake water levels <br /><br />


