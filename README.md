# VTE — Volatility Transmutation Engine
### Bitcoin Microstructure Alpha Strategy

Quantitative trading strategy developed for the **Itaú Quant Challenge**, exploring microstructural inefficiencies in Bitcoin price formation using Machine Learning.

---

## The Problem

Traditional trading models analyze only final transaction prices, ignoring the process behind each trade. Who bought aggressively? Who was waiting in the queue? What was the real directional pressure before the price moved?

This information exists in Level 2 orderbook data, but it is massive, noisy, and computationally expensive to process. The hypothesis behind VTE is that persistent imbalances between buyers and sellers, visible in microstructure data, can anticipate the direction of the next price movement before it appears in the final price.

---

## Data

- **Source:** Bybit exchange — Level 2 tick-by-tick trade data (publicly available)
- **Asset:** Bitcoin (BTCUSD perpetual)
- **Period:** January 2020 to October 2025
- **Volume:** 200+ GB of raw tick data, 186 parquet files
- **Sample used:** 10 million most recent records (~3 GB in memory)

### Data Cleaning

- Unix timestamps converted to datetime with second precision
- Outliers clipped at 1st and 99th percentiles to avoid distortion
- Missing values handled via forward-fill to maintain temporal continuity
- Strict temporal train/test split to prevent look-ahead bias

---

## Feature Engineering

Five microstructural indicators were engineered from raw trade data:

| Feature | Description |
|---|---|
| **CVD** (Cumulative Volume Delta) | Cumulative sum of signed volume (buys minus sells). Captures directional capital flow momentum. |
| **OFI** (Order Flow Imbalance) | Rolling mean of signed volume over 50 periods. Measures intensity of aggressive orders vs passive orders. |
| **Spread Proxy** | Normalized rolling mean of absolute price changes. Indicates market friction and transaction costs. |
| **Volume Intensity** | Rolling mean of trade size over 30 periods. Captures unusual activity levels. |
| **Micro-Volatility** | Log-return standard deviation over 25 periods. Measures short-term market regime. |

---

## Model

**Logistic Regression with L2 Regularization**

```
P(uptick) = 1 / (1 + e^(-βX))
```

- Input: 5 normalized microstructural features
- Target: Binary next-tick direction (1 = price up, 0 = price down)
- Regularization: L2 (C=0.1) to penalize excessive coefficients and reduce overfitting
- Class balancing: Adjusted weights to compensate for class imbalance

### Why Logistic Regression

Logistic regression was chosen over more complex models intentionally. With 10M+ training samples at tick frequency, a simpler model with strong regularization is less prone to overfitting noise, and each coefficient remains economically interpretable — allowing validation against market microstructure theory.

---

## Results

| Metric | Value | Note |
|---|---|---|
| AUC Score | 0.619 | Moderate predictive capacity above random |
| Win Rate | ~52% | Directional edge over 50% baseline |
| Sharpe Ratio | 15.71 | Requires additional validation |
| Max Drawdown | -0.32% | vs -2.9% Buy & Hold |
| Total Trades | 149,950 | Ultra-high frequency |

### Honest Assessment

The Sharpe Ratio of 15.71 exceeds top-tier hedge funds and is inconsistent with Bitcoin's known volatility. This was flagged and investigated during development. Possible causes include subtle look-ahead bias, overfitting to a specific 67-day regime, or underestimated transaction costs at this trade frequency.

The AUC of 0.619 and win rate of ~52% are more credible signals of a real directional edge.

---

## Limitations

- **Sample period:** 67 days of backtest may not capture regime diversity
- **Transaction costs:** 149K trades imply HFT-level execution costs not fully modeled
- **Single asset:** Concentration in Bitcoin limits generalization
- **Frequency:** Strategy operates at tick level, which is difficult to execute in practice

---

## Infrastructure

Processing 200+ GB of tick data required migrating from local environment to cloud virtual machines (Google Colab with high-memory runtime). Data was loaded in chunks to avoid RAM crashes, with explicit garbage collection between processing steps.

---

## Stack

```
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
Google Colab (cloud VM) · Parquet · PyArrow
```

---

## Status

Active development. Next steps include expanding the backtest period, implementing realistic transaction costs, and testing alternative rebalancing frequencies to reduce trade count to executable levels.

---

*Project developed independently as part of the Itaú Quant Challenge.*
