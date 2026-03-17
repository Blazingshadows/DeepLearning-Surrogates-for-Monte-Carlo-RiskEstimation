"""
surrogate_models.py
Trains and evaluates GP, NN, and MF-GP surrogates using data from real_mc_sim.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor

# ----------------------------
# Load saved Monte Carlo data
# ----------------------------
data = np.load("real_mc_data.npz")
X_mc, Y_true = data["X_mc"], data["Y_true"]
mu_emp, cov = data["mu_emp"], data["cov"]
true_var, true_cvar = data["true_var"], data["true_cvar"]
oracle_time = data["oracle_time"]
dim = X_mc.shape[1]
alpha = 0.99
print(f"Loaded data: {X_mc.shape}, oracle_time={oracle_time:.2f}s")

# ----------------------------
# Surrogate definitions
# ----------------------------
def compute_var_cvar(values, alpha=0.99):
    sorted_vals = np.sort(values)
    idx = int(math.ceil(alpha * len(values)) - 1)
    var = sorted_vals[idx]
    cvar = np.mean(sorted_vals[idx:])
    return var, cvar

def gp_predict_in_batches(model, X, batch_size=2000):
    n = X.shape[0]
    mus, sigs = np.empty(n), np.empty(n)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        mu, sigma = model.predict(X[i:j], return_std=True)
        mus[i:j], sigs[i:j] = mu, sigma
    return mus, sigs

def train_nn_ensemble(X, y, n_models=5, hidden_layer_sizes=(100, 50)):
    models = []
    for i in range(n_models):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=400,
                             activation='relu', solver='adam', early_stopping=True,
                             random_state=42+i)
        model.fit(X[idx], y[idx])
        models.append(model)
    return models

def nn_predict_in_batches(models, X, batch_size=2000):
    n = X.shape[0]
    mus, sigs = np.empty(n), np.empty(n)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        preds = np.column_stack([m.predict(X[i:j]) for m in models])
        mus[i:j], sigs[i:j] = preds.mean(axis=1), preds.std(axis=1)
    return mus, sigs

# ---------------------------------------
# Active learning framework (shared)
# ---------------------------------------
def run_active_learning(surrogate, X_candidates, Y_true_full, idx_init,
                        n_iterations=10, queries_per_iter=25, pool_size=6000):
    X_train = X_candidates[idx_init].copy()
    Y_train = Y_true_full[idx_init].copy()
    pool_idx = np.setdiff1d(np.arange(len(X_candidates)), idx_init)

    if surrogate == "gp":
        kernel = C(1.0) * Matern(length_scale=np.ones(dim), nu=1.5) + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
        model.fit(X_train, Y_train)
        predict = lambda X: gp_predict_in_batches(model, X)
        retrain = lambda X, y: model.fit(X, y)
    elif surrogate == "nn":
        model = train_nn_ensemble(X_train, Y_train)
        predict = lambda X: nn_predict_in_batches(model, X)
        retrain = lambda X, y: train_nn_ensemble(X, y)
    else:
        raise ValueError("Invalid surrogate")

    history = []
    for it in range(n_iterations):
        subset = np.random.choice(pool_idx, size=min(pool_size, len(pool_idx)), replace=False)
        mu_acq, sigma_acq = predict(X_candidates[subset])
        mu_all, _ = predict(X_candidates)
        est_var, _ = compute_var_cvar(mu_all, alpha)
        U = sigma_acq / (1.0 + np.abs(mu_acq - est_var))
        chosen = subset[np.argsort(-U)[:queries_per_iter]]
        X_new, Y_new = X_candidates[chosen], Y_true_full[chosen]
        X_train, Y_train = np.vstack([X_train, X_new]), np.concatenate([Y_train, Y_new])
        pool_idx = np.setdiff1d(pool_idx, chosen)
        retrain(X_train, Y_train)
        mu_all, _ = predict(X_candidates)
        est_var, est_cvar = compute_var_cvar(mu_all, alpha)
        err_var = abs(est_var - true_var) / abs(true_var) * 100
        err_cvar = abs(est_cvar - true_cvar) / abs(true_cvar) * 100
        history.append({"iter": it+1, "n_train": len(X_train),
                        "VaR": est_var, "CVaR": est_cvar,
                        "err_var(%)": err_var, "err_cvar(%)": err_cvar})
        print(f"[{surrogate.upper()}] Iter {it+1:2d} n={len(X_train)} VaR={est_var:.4f} err={err_var:.2f}% CVaR={est_cvar:.4f} err={err_cvar:.2f}%")

    return pd.DataFrame(history)

# ---------------------------------------
# Run both surrogates
# ---------------------------------------
n_init = 80
idx_init = np.random.choice(len(X_mc), size=n_init, replace=False)

print("\nRunning GP...")
t0 = time.time()
df_gp = run_active_learning("gp", X_mc, Y_true, idx_init)
gp_time = time.time() - t0

print("\nRunning NN...")
t0 = time.time()
df_nn = run_active_learning("nn", X_mc, Y_true, idx_init)
nn_time = time.time() - t0

# ---------------------------------------
# Plot & compare
# ---------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(df_gp["n_train"], df_gp["err_var(%)"], "-o", label="GP")
plt.plot(df_nn["n_train"], df_nn["err_var(%)"], "-o", label="NN")
plt.xlabel("Training points"); plt.ylabel("VaR error (%)")
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(df_gp["n_train"], df_gp["err_cvar(%)"], "-o", label="GP")
plt.plot(df_nn["n_train"], df_nn["err_cvar(%)"], "-o", label="NN")
plt.xlabel("Training points"); plt.ylabel("CVaR error (%)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

print("\nFinal GP:", df_gp.iloc[-1])
print("Final NN:", df_nn.iloc[-1])
print(f"\nOracle time: {oracle_time:.2f}s, GP time: {gp_time:.2f}s, NN time: {nn_time:.2f}s")
