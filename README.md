# Surrogate-assisted Monte Carlo for Tail Risk (VaR / CVaR)

This repository contains a small research/demo script that shows how to use a surrogate model (Gaussian Process or a Neural Network ensemble) together with active learning to estimate tail risk metrics — Value at Risk (VaR) and Conditional Value at Risk (CVaR) — efficiently without evaluating the expensive oracle everywhere.

## Overview

- Goal: estimate high-quantile tail statistics (VaR and CVaR at level `alpha`, e.g. 0.99) for a loss function while minimizing the number of expensive oracle evaluations.
- Approach: sample a Monte Carlo pool of candidate inputs, evaluate an initial small training set (oracle), fit a surrogate model, use an acquisition function to actively query informative points (near estimated VaR and with high uncertainty), retrain, and iterate.

This approach is useful when the true loss/pricing function is expensive to evaluate but you can draw a large candidate pool (Monte Carlo) cheaply.

## Key terms and formulas

- Monte Carlo pool: a large set of candidate inputs `X_candidates` (shape: `[N_candidates, dim]`) used as the population for estimating quantiles.

- Oracle / true loss: the expensive function `L(x)` (in the demo code this is `true_loss(x)`). For each candidate we can (optionally) compute the true loss; in practice we avoid evaluating it everywhere.

- Value at Risk (VaR) at level $\alpha$:

  - Definition: the $\alpha$-quantile of the loss distribution.
  - Empirical estimator used in the code (for a sample of losses $\{l_i\}_{i=1}^n$): sort values ascending and take the index
    $$\text{idx} = \lceil \alpha n \rceil - 1,\qquad \text{VaR}_\alpha = l_{(\text{idx})}$$
  - (Note: other quantile interpolation rules exist; the script uses a simple empirical index.)

- Conditional Value at Risk (CVaR) at level $\alpha$ (also called Expected Shortfall):

  - Definition: the expected loss conditional on loss exceeding the VaR.
  - Formula used in the script (empirical):
    $$\text{CVaR}_\alpha = \frac{1}{n - \text{idx}}\sum_{j=\text{idx}}^{n-1} l_{(j)} = \text{mean of tail values at or above VaR}.$$ 
  - Intuition: while VaR gives a threshold, CVaR measures the average severity of losses in the worst $(1-\alpha)$ fraction.

- Gaussian Process (GP) surrogate:
  - GP returns predictive mean $\mu(x)$ and predictive standard deviation (uncertainty) $\sigma(x)$ at new inputs.
  - Kernel: Matern kernel with a Constant scale and a small WhiteKernel noise term is used; kernel hyperparameters are optimized (few restarts by default for speed).

- Neural Network (NN) ensemble surrogate:
  - An ensemble of MLP regressors is trained on bootstrap resamples. Ensemble mean approximates $\mu(x)$ and ensemble standard deviation approximates predictive uncertainty.

- Acquisition function used for active learning (code formula):

  $$U(x) = \frac{\sigma(x)}{1 + |\mu(x) - \widehat{\text{VaR}}|},$$

  where $\widehat{\text{VaR}}$ is the current surrogate VaR estimate computed from $\mu(x)$ over the MC pool. Intuition: prefer points that are both uncertain (high $\sigma$) and whose predicted mean is near the current VaR estimate.

## Algorithm (high level)

1. Build a large MC candidate pool `X_candidates`.
2. (For demonstration) compute `Y_true_full = L(X_candidates)` once to have a reference.
3. Draw `n_init` initial training points, evaluate the oracle, and fit a surrogate (`gp` or `nn`).
4. Repeat for `n_iterations`:
   - Sample an acquisition subset from the remaining pool to limit compute.
   - Predict surrogate mean and std on that subset.
   - Compute the surrogate VaR using surrogate means over the full pool.
   - Compute acquisition score `U(x)` and pick the top `queries_per_iter` points.
   - Query the oracle at those points, add them to the training set, retrain the surrogate.
   - Compute estimated VaR/CVaR from surrogate means over the pool and record relative errors vs. the reference.
5. Plot convergence (error vs number of true evaluations) and distribution comparison.

## Files

- `Code File/surrogate.py` - main script. Key variables to know:
  - `N_candidates` - size of MC pool
  - `n_init` - number of initial true evaluations
  - `n_iterations`, `queries_per_iter` - active-learning loop settings
  - `SURROGATE` - string flag `'gp'` or `'nn'` to choose the surrogate method

## How to run (environment)

Requirements (minimum):

- Python 3.8+ (script used scikit-learn, numpy, pandas, matplotlib)
- pip install:

```powershell
pip install numpy pandas matplotlib scikit-learn
```

Run the script from the project root:

```powershell
python "Code File/surrogate.py"
```

Notes:
- For a quick demo reduce `N_candidates` and `n_iterations` / `queries_per_iter` to keep runtime low.
- To switch to the NN ensemble surrogate set `SURROGATE = 'nn'` in the script.

## Interpreting outputs

- Convergence plot (error vs number of true evaluations): shows how the estimated VaR/CVaR get closer to the reference as we add oracle evaluations.
  - Faster decline = more sample-efficient surrogate + acquisition.
  - CVaR typically converges slower than VaR because it depends on very extreme tail samples.

- Distribution histogram: compares the empirical loss distribution (from the pool) and the surrogate mean predictions. Look at the upper tail overlap to assess how well the surrogate captures extremes.

## Practical advice and next steps

- Calibration: GPs provide principled uncertainty estimates; verify coverage (percent of true values inside mean ± k*sigma). For NN ensembles this is a heuristic.
- If CVaR estimates are unstable, consider:
  - stronger tail-focused acquisition (more queries near VaR),
  - importance sampling/stratified sampling to enrich tails,
  - sampling from GP posterior to get confidence bands for VaR/CVaR.
- For reproducibility run multiple trials (different seeds) and show mean ± std of error curves.

## Formulas summary

- Empirical VaR (alpha):
  $$\text{VaR}_\alpha = l_{(\lceil \alpha n \rceil - 1)}$$

- Empirical CVaR (alpha):
  $$\text{CVaR}_\alpha = \frac{1}{n - \text{idx}}\sum_{j=\text{idx}}^{n-1} l_{(j)}\quad\text{with }\text{idx}=\lceil\alpha n\rceil-1$$

- Acquisition score:
  $$U(x) = \frac{\sigma(x)}{1 + |\mu(x) - \widehat{\text{VaR}}|}$$
