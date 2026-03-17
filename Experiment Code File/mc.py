"""
real_mc_sim.py
Generates the true Monte Carlo dataset for surrogate modeling experiments.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import time
import math
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Download market data (robust method)
# ---------------------------------------
tickers = ['^GSPC', '^IXIC', '^VIX', '^TNX']
start, end = '2018-01-01', '2024-12-31'

print("Downloading market data...")
df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=True)


def extract_prices_from_yf(df):
    """Robustly extract price columns (close/adj close) from a yfinance DataFrame.

    Handles MultiIndex (Ticker, Field) and single-index DataFrames. Returns a DataFrame
    with one column per ticker (or the original DataFrame if no close-like field is found).
    """
    # debug print to help diagnose column layout when things fail
    # print('DEBUG df.columns:', df.columns)
    if isinstance(df.columns, pd.MultiIndex):
        # normalize second-level names to lowercase strings for matching
        lvl1 = [str(x).lower() for x in df.columns.get_level_values(1)]
        # try common candidates
        for cand in ('close', 'adj close', 'adjusted close', 'adjclose'):
            matches = [v for v in df.columns.get_level_values(1) if cand in str(v).lower()]
            if matches:
                # select that sublevel across tickers
                return df.xs(matches[0], axis=1, level=1)
        # fallback: choose the first numeric subcolumn for each ticker
        try:
            cols = df.columns.get_level_values(0)
            unique_tickers = pd.Index(cols).unique()
            result = pd.DataFrame(index=df.index)
            for tk in unique_tickers:
                sub = df[tk]
                # pick first numeric column
                numeric_cols = [c for c in sub.columns if pd.api.types.is_numeric_dtype(sub[c])]
                if numeric_cols:
                    result[tk] = sub[numeric_cols[0]]
                else:
                    # as a last resort, take the first column
                    result[tk] = sub.iloc[:, 0]
            return result
        except Exception:
            return df
    else:
        # single-level columns: look for close-like names
        cols = list(df.columns)
        lower = [str(c).lower() for c in cols]
        for cand in ('close', 'adj close', 'adjusted close', 'adjclose'):
            matches = [c for c, lc in zip(cols, lower) if cand in lc]
            if matches:
                return df[matches]
        # if no close-like field, but columns appear numeric, assume they are prices
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            return df[numeric_cols]
        # otherwise return original
        return df


prices = extract_prices_from_yf(df)
if isinstance(prices, pd.Series):
    prices = prices.to_frame()
prices = prices.dropna()
returns = prices.pct_change().dropna()
returns.to_csv("market_returns.csv")
print(f"Saved market returns with shape {returns.shape}")

# ---------------------------------------
# 2. Portfolio oracle setup (robust)
# ---------------------------------------
# keep only numeric columns (drop any non-numeric if present)
returns_num = returns.select_dtypes(include=[np.number]).copy()
if returns_num.shape[1] == 0:
    raise RuntimeError("No numeric return columns available after download")

# drop columns with (near-)zero std to avoid division by zero
stds = returns_num.std(ddof=1)
valid_cols = stds[stds > 1e-12].index.tolist()
if len(valid_cols) < returns_num.shape[1]:
    print(f"Dropping {returns_num.shape[1]-len(valid_cols)} constant or non-varying columns:",
          list(set(returns_num.columns) - set(valid_cols)))
returns_num = returns_num[valid_cols]

factors = (returns_num - returns_num.mean()) / returns_num.std(ddof=1)
X_data = factors.values

# empirical covariance and mean of the (standardized) factors
cov = np.cov(X_data.T)
# regularize covariance to ensure positive-definiteness for sampling
cov = 0.5 * (cov + cov.T)
eps = 1e-8
for _ in range(10):
    try:
        np.linalg.cholesky(cov + eps * np.eye(cov.shape[0]))
        cov += eps * np.eye(cov.shape[0])
        break
    except np.linalg.LinAlgError:
        eps *= 10
else:
    # final fallback: add a small ridge
    eps = 1e-4
    cov += eps * np.eye(cov.shape[0])

mu_emp = X_data.mean(axis=0)
rng = np.random.default_rng(42)
w = rng.normal(size=X_data.shape[1])
w /= np.linalg.norm(w)

def portfolio_loss(x):
    linear = np.dot(w, x)
    quadratic = 0.5 * (x @ cov @ x)
    nonlinear = 0.05 * np.sin(2*x[0]) + 0.02 * (x[1]**2)
    return -(linear + quadratic + nonlinear)

# ---------------------------------------
# 3. Monte Carlo simulation
# ---------------------------------------
N_mc = 100_000
print(f"Running Monte Carlo simulation ({N_mc} samples)...")
X_mc = np.random.multivariate_normal(mu_emp, cov, size=N_mc)

t0 = time.time()
Y_true = np.array([portfolio_loss(x) for x in X_mc])
oracle_time = time.time() - t0

def compute_var_cvar(values, alpha=0.99):
    sorted_vals = np.sort(values)
    idx = int(math.ceil(alpha * len(values)) - 1)
    var = sorted_vals[idx]
    cvar = np.mean(sorted_vals[idx:])
    return var, cvar

alpha = 0.99
true_var, true_cvar = compute_var_cvar(Y_true, alpha)
print(f"VaR{alpha:.2f}={true_var:.4f}, CVaR={true_cvar:.4f}, time={oracle_time:.2f}s")

# ---------------------------------------
# 4. Save results for surrogate file
# ---------------------------------------
np.savez("real_mc_data.npz", X_mc=X_mc, Y_true=Y_true, mu_emp=mu_emp, cov=cov,
         true_var=true_var, true_cvar=true_cvar, oracle_time=oracle_time)
print("Saved dataset to real_mc_data.npz")

# -------------------------------------------------
# Additional: simulate portfolio paths over a multi-day horizon
# starting from initial capital, showing progression over days.
# We treat Y_true as daily log-returns (additive across days) sampled IID.
# -------------------------------------------------
initial_capital = 10000.0
horizon_days = 10
n_path_sims = 20000

# sample IID daily returns for each path (shape: n_path_sims x horizon_days)
rng = np.random.default_rng(123)
picks = rng.integers(0, Y_true.shape[0], size=(n_path_sims, horizon_days))
daily_returns = Y_true[picks]

# cumulative log-return per path (days 1..horizon); include day 0 as zero
cum_returns = np.cumsum(daily_returns, axis=1)
logcum = np.hstack([np.zeros((n_path_sims, 1)), cum_returns])  # shape (n_path_sims, horizon_days+1)

# portfolio values per day
values = initial_capital * np.exp(logcum)

# compute percentiles across simulated paths for each day
days = np.arange(0, horizon_days + 1)
pcts = [1, 5, 25, 50, 75, 95, 99]
perc_vals = {p: np.percentile(values, p, axis=0) for p in pcts}

plt.figure(figsize=(10,6))
# shaded interquartile
plt.fill_between(days, perc_vals[25], perc_vals[75], color='C0', alpha=0.2, label='25-75%')
# shaded 5-95
plt.fill_between(days, perc_vals[5], perc_vals[95], color='C0', alpha=0.12, label='5-95%')
# median line
plt.plot(days, perc_vals[50], color='C0', linewidth=2, label='Median')
# plot a few sample paths
for i in range(10):
    plt.plot(days, values[i], color='grey', alpha=0.18)

plt.xlabel('Days')
plt.ylabel('Portfolio value (USD)')
plt.title(f'Portfolio value progression over {horizon_days} days (initial ${initial_capital:,.0f})')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------
# Additional plot: mean progression with ±1 std band
# -------------------------------------------------
mean_vals = values.mean(axis=0)
std_vals = values.std(axis=0)

plt.figure(figsize=(10,6))
plt.plot(days, mean_vals, color='C1', linewidth=2, label='Mean')
plt.fill_between(days, mean_vals - std_vals, mean_vals + std_vals, color='C1', alpha=0.2, label='±1 std')
plt.plot(days, perc_vals[50], color='C0', linestyle='--', linewidth=1.5, label='Median')
plt.xlabel('Days')
plt.ylabel('Portfolio value (USD)')
plt.title(f'Mean portfolio value and ±1 std over {horizon_days} days')
plt.legend()
plt.grid(True)
plt.show()
