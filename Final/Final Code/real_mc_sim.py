"""
real_mc_sim.py
Generates true Monte Carlo risk simulation for a fixed portfolio.
Highlights computational cost to motivate surrogate modeling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

tickers = ['^GSPC', '^IXIC', '^VIX', '^TNX']  
start, end = '2018-01-01', '2024-11-01'

print("Downloading historical data...")
raw = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
raw = raw.dropna().pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()


returns = (raw - raw.mean()) / raw.std()
X = returns.values
tickers = raw.columns.tolist()
n_assets = len(tickers)


mu = X.mean(axis=0)
cov = np.cov(X.T)
cov = 0.5 * (cov + cov.T) + 1e-6 * np.eye(n_assets)  

rng = np.random.default_rng(42)
weights = rng.normal(size=n_assets)
weights /= np.linalg.norm(weights) 

def portfolio_loss(x):
    """Custom nonlinear loss function for portfolio scenario x"""
    linear = np.dot(weights, x)
    quadratic = 0.5 * x @ cov @ x
    nonlinear = 0.05 * np.sin(2 * x[0]) + 0.02 * (x[1] ** 2)
    return -(linear + quadratic + nonlinear)


N_mc = 100_000_000
print(f"Running Monte Carlo with {N_mc} samples...")

start_time = time.time()
scenarios = np.random.multivariate_normal(mu, cov, size=N_mc)
losses = np.array([portfolio_loss(x) for x in scenarios])
elapsed = time.time() - start_time

def compute_var_cvar(samples, alpha=0.99):
    sorted_losses = np.sort(samples)
    var_idx = int((1 - alpha) * len(samples))
    var = sorted_losses[var_idx]
    cvar = sorted_losses[:var_idx].mean()
    return var, cvar

alpha = 0.99
VaR, CVaR = compute_var_cvar(losses, alpha)

print(f"\nResults at {alpha*100:.0f}% confidence:")
print(f"VaR  = {VaR:.4f}")
print(f"CVaR = {CVaR:.4f}")
print(f"Simulation time = {elapsed:.2f} seconds")


plt.figure(figsize=(10, 6))
plt.hist(losses, bins=100, alpha=0.6, color='skyblue', density=True)
plt.axvline(VaR, color='red', linestyle='--', label=f"VaR ({alpha*100:.0f}%)")
plt.axvline(CVaR, color='darkred', linestyle='-', label=f"CVaR")
plt.title("Simulated Portfolio Loss Distribution")
plt.xlabel("Loss")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# ---------------------------------------
# 5. Save results
# ---------------------------------------
np.savez("mc_ground_truth.npz",
         X=scenarios, losses=losses, weights=weights,
         mu=mu, cov=cov, VaR=VaR, CVaR=CVaR, time=elapsed)

print("Saved results to mc_ground_truth.npz")


# parameters
initial_capital = 10000.0
horizon_days = 10
n_path_sims = 10000

# un-normalize scenarios (scenarios were drawn from normalized returns X)
raw_mean = raw.mean().values
raw_std = raw.std().values
actual_scenarios = scenarios * raw_std + raw_mean  # shape (N_mc, n_assets)

# compute portfolio daily return for each scenario
port_daily_returns = actual_scenarios @ weights  # shape (N_mc,)

# sample IID daily returns for each simulated path
rng = np.random.default_rng(123)
picks = rng.integers(0, port_daily_returns.shape[0], size=(n_path_sims, horizon_days))
sampled_daily = port_daily_returns[picks]  # shape (n_path_sims, horizon_days)

# convert to cumulative log-returns (use log1p for small returns)
cum_log = np.cumsum(np.log1p(sampled_daily), axis=1)
logcum = np.hstack([np.zeros((n_path_sims, 1)), cum_log])  # include day 0

# portfolio values per day
values = initial_capital * np.exp(logcum)

# compute percentiles across simulated paths for each day
days = np.arange(0, horizon_days + 1)
pcts = [1, 5, 25, 50, 75, 95, 99]
perc_vals = {p: np.percentile(values, p, axis=0) for p in pcts}

# plot trajectories (median and bands) with distinct percentile lines and baseline
plt.figure(figsize=(10, 6))
# color mapping for percentiles
colors = {1: 'darkred', 5: 'orangered', 25: 'C3', 50: 'C0', 75: 'C2', 95: 'orangered', 99: 'darkred'}

# shaded bands (use median color for the band fill)
plt.fill_between(days, perc_vals[25], perc_vals[75], color=colors[50], alpha=0.18, label='25-75%')
plt.fill_between(days, perc_vals[5], perc_vals[95], color=colors[50], alpha=0.10, label='5-95%')

# percentile lines
plt.plot(days, perc_vals[1], color=colors[1], linestyle=':', linewidth=1, label='1%')
plt.plot(days, perc_vals[5], color=colors[5], linestyle='--', linewidth=1, label='5%')
plt.plot(days, perc_vals[25], color=colors[25], linestyle='-.', linewidth=1, label='25%')
plt.plot(days, perc_vals[50], color=colors[50], linewidth=2, label='Median (50%)')
plt.plot(days, perc_vals[75], color=colors[75], linestyle='-.', linewidth=1, label='75%')
plt.plot(days, perc_vals[95], color=colors[95], linestyle='--', linewidth=1, label='95%')
plt.plot(days, perc_vals[99], color=colors[99], linestyle=':', linewidth=1, label='99%')

# sample paths (light grey)
for i in range(10):
    plt.plot(days, values[i], color='grey', alpha=0.18)

# baseline initial capital
plt.axhline(initial_capital, color='k', linestyle=':', linewidth=1.5, label='Initial capital')

plt.xlabel('Days')
plt.ylabel('Portfolio value (USD)')
plt.title(f'Portfolio value progression over {horizon_days} days (initial ${initial_capital:,.0f})')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
