# Mamba vs Transformer: Comparison on Long Text Classification

> Systematic comparison of Mamba-family architectures (Mamba, Mamba-2, Mamba-3) and Transformer on byte-level text classification task from Long Range Arena (LRA) benchmark.

## 📌 Overview

This project evaluates State Space Models (SSM) against Transformer architectures on the challenging task of classifying long text sequences (up to 4096 characters) at the character/byte level using IMDb reviews. The study follows the **Self-Pretraining (SPT)** approach [2] for fair comparison, training all models from scratch without external data.

### Key Findings
- **Mamba-1** achieves **87.56%** validation accuracy, surpassing Transformer with local attention (window size 64)
- All Mamba models are comparable to Transformer (window 128) in quality while being **~3x faster** on inference
- Mamba-3 demonstrates the fastest inference speed among all tested models

## 🏗️ Models Evaluated

| Model | Parameters | Architecture Highlights |
|-------|------------|------------------------|
| **Transformer** | ~155K | LocalAttention + RoPE/XPos embeddings |
| **Mamba-1** | ~143K | Selective State Space Model (SSM) |
| **Mamba-2** | ~123K | Structured State Space Duality (SSD) |
| **Mamba-3** | ~126K | Complex states + MIMO updates |

## 📊 Results

| Model | Params | Val Acc, % | Val Time, s |
|-------|--------|------------|-------------|
| **Mamba-1** | 142,656 | **87.56** | 72 |
| Mamba-2 | 123,352 | 86.61 | 99 |
| Mamba-3 | 125,776 | 86.83 | 61 |
| Transformer (window 64) | 155,520 | 81.34 | 180 |
| Transformer (window 128) | 155,520 | 87.31 | 144 |
| Transformer (window 256) | 155,520 | 81.34 | 194 |

## Getting Started

### Prerequisites
```bash
pip install torch mamba-ssm
```

### run experiments
```bash
python 'model'/experiment_'model'.py
```
