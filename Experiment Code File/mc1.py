import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
start_date, end_date = '2020-01-01', '2024-01-01'
confidence_level = 0.95
n_portfolios = 1000
n_simulations = 5000

# Step 1: Download data
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True)
if isinstance(data.columns, pd.MultiIndex):
    level_names = [lvl.lower() for lvl in data.columns.names]
    if 'price' in level_names[0]:
        prices = data.xs('Close', axis=1, level=0)
    else:
        prices = data.xs('Close', axis=1, level=1)
else:
    prices = data[['Close']] if 'Close' in data.columns else data
returns = data.pct_change().dropna()
mu = returns.mean().values
cov = returns.cov().values
n_assets = len(tickers)

# Step 2: Monte Carlo simulation function
def simulate_var_cvar(weights, mu, cov, n_sim=5000, alpha=0.95):
    sim_returns = np.random.multivariate_normal(mu, cov, n_sim)
    port_returns = sim_returns @ weights
    losses = -port_returns
    var = np.percentile(losses, 100 * alpha)
    cvar = losses[losses >= var].mean()
    return var, cvar

# Step 3: Generate training data
X, Y = [], []
for _ in range(n_portfolios):
    w = np.random.dirichlet(np.ones(n_assets))  # random weights summing to 1
    var, cvar = simulate_var_cvar(w, mu, cov, n_simulations, confidence_level)
    X.append(w)
    Y.append([var, cvar])

X = np.array(X)
Y = np.array(Y)

# Optional: Visualize VaR vs CVaR
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)
plt.xlabel("VaR (95%)")
plt.ylabel("CVaR (95%)")
plt.title("Monte Carlo Simulated VaR vs CVaR")
plt.grid(True)
plt.show()
