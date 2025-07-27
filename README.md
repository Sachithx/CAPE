# 🌐 CAPE-TST: Context-Aware Dynamic Patch Encoder for Time Series Transformers

This repository contains the official implementation of the paper:

> **Context-Aware Dynamic Patch Encoder for Time Series Transformers**  
> _Under review at AAAI 2026_  
> 📌 **Authors**: Sachith Abeywickrama, Emadeldeen Edele, Min Wu, Xiaoli Li, Yuen Chau

---

<img src="https://github.com/user-attachments/assets/998c0d5f-8ebf-410e-95f5-7a32c7b11465" width="100%" />

---

## ✨ Overview

**CAPE-TST** introduces a novel framework for dynamic patching and representation learning in time series forecasting using Transformer models. Unlike conventional fixed-length or context-agnostic patching schemes, our method dynamically determines patch boundaries using entropy and encodes each patch using an adaptive encoder.

---

## 🚀 Key Contributions

- ✅ **Entropy-Based Dynamic Patching**  
  Uses information-theoretic entropy to identify meaningful patch boundaries based on temporal uncertainty and fluctuations.

- ✅ **Adaptive Patch Encoder**  
  Applies iterative attention mechanisms to encode variable-length patches into fixed-size latent vectors.

- ✅ **Context-Aware Feature Representation**  
  Preserves both short-term and long-term dependencies by respecting local temporal structures.

- ✅ **Plug-and-Play Encoder Module**  
  Our encoder block can be integrated into existing time series Transformer architectures with minimal modification.

- ✅ **SOTA Performance**  
  Achieves strong results across multiple long-term forecasting benchmarks including ETT, Electricity, and Exchange Rate datasets.

---

## 📂 Repository Structure

```bash
├── src/                 # Model and training code
│   ├── models/          # CAPE-TST model components
│   ├── utils/           # Helper functions
│   └── train.py         # Training script
├── data/                # Preprocessed datasets and loaders
├── experiments/         # Configs for experiments
├── results/             # Output logs and visualizations
└── README.md            # Project overview
