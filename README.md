# Surrogate-Assisted Monte Carlo for Portfolio Tail Risk (VaR / CVaR)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-GP%20%7C%20MLP-orange)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy)](https://numpy.org/)
[![Paper](https://img.shields.io/badge/Research-Paper-green)](#paper)

---

## What this is

Running Monte Carlo (MC) simulations to estimate portfolio tail risk (VaR, CVaR) is accurate — but expensive, requiring 10M–100M samples for stable tail statistics. This project asks: **can surrogate models replace most of those evaluations without sacrificing accuracy?**

We train Gaussian Process (GP) and Neural Network (NN) ensemble surrogates on a small subset of MC-evaluated points, then use active learning to strategically query the most informative samples near the tail. The result: **sub-1% error on VaR and CVaR estimates at a fraction of the compute cost.**

This work is accompanied by a co-authored research paper (Manipal University Jaipur, 2025).

---

## Key results

| Metric | GP Surrogate | NN Ensemble |
|---|---|---|
| VaR error (99%) | **< 1%** | ~2–4% (low-sample regime) |
| CVaR error (99%) | **< 1%** | Higher variance, slower convergence |
| Convergence | ~100 evaluations | ~250–300 evaluations |
| Ground truth baseline | 100M MC samples | 100M MC samples |

**GP outperformed NN** in both accuracy and convergence speed. The NN gap is partly a data constraint — GPs are more sample-efficient in low-budget regimes, which is exactly where surrogates matter most.

---

## How it works

**Loss function** (nonlinear portfolio loss with linear, quadratic, and sinusoidal components):

$$L(\mathbf{x}) = -\left(\mathbf{w}^\top\mathbf{x} + \frac{1}{2}\mathbf{x}^\top\Sigma\mathbf{x} + 0.05\sin(2x_1) + 0.02x_2^2\right)$$

**Active learning acquisition function** (queries uncertain points near current VaR estimate):

$$U(x) = \frac{\sigma(x)}{1 + |\mu(x) - \widehat{\text{VaR}}|}$$

**Pipeline:**
1. Simulate 100M MC scenarios → compute ground truth VaR/CVaR at 99%
2. Initialize surrogate with 60 training points
3. Iterate 12 rounds × 20 queries using acquisition function above
4. Estimate VaR/CVaR from surrogate means over full candidate pool
5. Track error convergence vs. number of true evaluations

---

## Results

**VaR convergence — GP vs NN:**

![VaR Comparison](plots/Comparison_var.png)

**CVaR convergence — GP vs NN:**

![CVaR Comparison](plots/Comparison_cvar.png)

**GP predicted loss distribution vs ground truth:**

![GP Distribution](plots/100m_gp_predicted_portfolio.png)

**Ground truth loss distribution (100M samples):**

![Ground Truth](plots/ground_truth.png)

> GP accurately captures tail shape near VaR/CVaR thresholds. NN captures the central region well but deviates in the tail — leading to overestimated risk in early iterations.

---

## Surrogates

**Gaussian Process:** Matérn kernel with automatic relevance determination + white noise kernel. Provides calibrated uncertainty estimates used directly in acquisition.

**NN Ensemble:** 5 MLPs with (150, 75) hidden layers, ReLU, bootstrapped training, L2 regularization, early stopping. Ensemble std used as uncertainty proxy.

---

## How to run
```bash
git clone https://github.com/Blazingshadows/surrogate-var-cvar
cd surrogate-var-cvar
pip install numpy pandas matplotlib scikit-learn
python src/surrogate.py
```

Switch surrogate method in `src/surrogate.py`:
```python
SURROGATE = 'gp'   # or 'nn'
```

Reduce `N_candidates` and `n_iterations` for a faster demo run.

---

## Paper

> **Neural and Gaussian Process Surrogates for Fast Monte Carlo VaR and CVaR Estimation**  
> Aditya Dixit, Aadit Datta — Manipal University Jaipur, 2025

Available in [`paper/`](paper/).

---

## Stack

Python · NumPy · Scikit-learn · Matplotlib · Pandas
