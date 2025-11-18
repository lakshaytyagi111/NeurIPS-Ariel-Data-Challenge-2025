

# Exoplanet Atmospheric Spectroscopy Pipeline

## NeurIPS Ariel Data Challenge 2025

A machine learning pipeline for extracting clean atmospheric spectra from noisy telescope observations of exoplanet transits.



## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Key Terminology](#key-terminology)
- [Pipeline Overview](#pipeline-overview)
- [Technical Approach](#technical-approach)
- [Results](#results)
- [Installation & Usage](#installation--usage)

---

## Problem Statement

When a planet passes in front of its star (called a **transit**), it blocks a tiny amount of light (~1%). The planet's atmosphere absorbs specific wavelengths differently, creating a unique "fingerprint" that reveals its chemical composition.

### Our Challenge
- Extract this weak signal from noisy detector data  
- Predict atmospheric composition across 283 wavelengths  
- Quantify uncertainty in our predictions  

### Why It Matters
This prepares for ESA's **Ariel mission (launching 2029)** which will characterize 1,000+ exoplanets to search for water, methane, and potential biosignatures.

---

## Dataset

### Training Data
- **456 planets** with known atmospheric spectra  
- **Two instruments:**
  - **AIRS-CH0**: Infrared spectrograph (32×356 pixels, 11,250 frames)
  - **FGS1**: Fine guidance sensor (32×32 pixels, 135,000 frames)

### Input Files
| File Type | Description | Example |
|-----------|-------------|---------|
| Signal files | Raw detector readings | `AIRS-CH0_signal_0.parquet`, `FGS1_signal_0.parquet` |
| Calibration | Dark current, flat field, dead pixels | `dark.parquet`, `flat.parquet`, `dead.parquet` |
| Metadata | Star properties (radius, mass, temperature) | `train_star_info.csv` |
| Ground truth | Target atmospheric spectra | `train.csv` (283 wavelengths) |

### Output
- **283 wavelength values** per planet (atmospheric spectrum)  
- **283 uncertainty values** (confidence intervals)

---

## Key Terminology

### Astronomical Concepts

**Transit** — When a planet passes in front of its star, blocking light  
**Transit Depth** — (Planet Radius / Star Radius)²  
**Ingress** — Planet enters star’s disk  
**Egress** — Planet exits star’s disk  
**Transit Duration**  
- **T14**: Total duration  
- **T23**: Flat bottom duration  

### Signal Processing

- **ADC** — Analog-to-digital conversion  
- **CDS** — Correlated double sampling  
- **Flat Field** — Pixel sensitivity correction  
- **Dark Current** — Thermal noise  
- **Hot/Dead Pixels** — Faulty detectors  

### Machine Learning

- **Ridge Regression** — Linear model with L2 regularization  
- **Bootstrap** — Resampling-based uncertainty estimation  
- **Outliers** — Poor-quality or extreme data  

---

## Pipeline Overview

```

Raw Detector Data (11,250 AIRS frames + 135,000 FGS1 frames)
↓
[1. PREPROCESSING]
├─ ADC correction
├─ Hot/dead pixel masking
├─ Linearity correction
├─ Dark current subtraction
├─ CDS
├─ Flat field correction
├─ Binning (15:1 AIRS, 180:1 FGS1)
└─ NaN interpolation
↓
[2. TRANSIT DETECTION]
6-phase boundary detection via 2nd derivative
↓
[3. FEATURE ENGINEERING]
├─ Transit depths
├─ Slopes, curvatures
├─ Durations, symmetry
└─ Stellar parameters
↓
[4. MODELING]
Dual Ridge Regression (clean vs outliers)
↓
[5. UNCERTAINTY ESTIMATION]
Bootstrap + residual RMSE
↓
OUTPUT: 283 wavelengths + 283 uncertainties per planet

````

---

## Technical Approach

### 1. Preprocessing

Standard astrophysical cleaning pipeline:

- **ADC Correction**  
  `signal = (signal - offset) * gain`
- **Hot/Dead Pixel Masking** using σ-clipping  
- **Linearity Correction** (Polynomial)  
- **Dark Current Subtraction**  
- **Correlated Double Sampling (CDS)**  
- **Flat Field Correction**  
- **Temporal Binning**

### 2. Transit Phase Detection — *6 Phases*

- Phase 1 & 4: Inflection points  
- Phase 2 & 3: Full contact points  
- Phase 5: Steepest descent  
- Phase 6: Steepest ascent  

### 3. Feature Engineering

~200 features:  
- Transit geometry: *t14, t23, symmetry*  
- Depth features: *average, mid-depth, wavelength bins*  
- Shape features: *slopes, curvatures, percentiles*  
- Stellar params: *Rs, Ms, Ts, log g, P*  

**SNR-Weighted Averaging**
```python
weights = signal_mean / signal_std
weights /= weights.sum()
signal_clean = signal @ weights
````

**Adaptive Polynomial Detrending**

```python
penalty = degree ** (1 - n_points/max_points)
error = RMSE * penalty
```

### 4. Dual Ridge Regression

| Tier     | Condition          | Model  | Alpha |
| -------- | ------------------ | ------ | ----- |
| Clean    | Valid              | Ridge  | 0.03  |
| Outlier  | Invalid boundaries | Ridge  | 0.3   |
| Very Bad | Catastrophic       | High σ | 0.003 |

### 5. Uncertainty Quantification

**Bootstrap (1000×)** + **Residual RMSE**

```python
sigma_final = 0.75 * sigma_bootstrap + 0.25 * sigma_residual
sigma_final[outliers] *= 1.5
sigma_final[very_bad] = 0.003
```

---

## Results

### Performance

| Metric   | Value  | Meaning              |
| -------- | ------ | -------------------- |
| R²       | 0.91   | Variance explained   |
| RMSE     | 0.0089 | Per-wavelength error |
| CV Folds | 100    | Robust validation    |

### Ablation

| Variant           | R²       |
| ----------------- | -------- |
| Single Ridge      | 0.87     |
| Fixed detrending  | 0.88     |
| 4-phase           | 0.89     |
| **Full Pipeline** | **0.91** |

---

## Installation & Usage

### Requirements

```bash
Python 3.8+
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
torch>=1.10
scipy>=1.7
matplotlib>=3.4
```

### Install

```bash
pip install -r requirements.txt
```

---

## Key Contributions

1. **6-Phase Transit Detection**
2. **Adaptive Polynomial Detrending**
3. **SNR-Weighted Averaging**
4. **Dual Ridge Architecture**
5. **Hybrid Uncertainty Estimation**

---

## Limitations

* Bootstrap cost (~2 hours)
* Linear model limitations
* No stellar activity modeling

---

## Acknowledgments

* ESA Ariel Mission Team
* NeurIPS 2025 Organizers
* School of Engineering, JNU

---

**Last Updated:** November 2025

```
