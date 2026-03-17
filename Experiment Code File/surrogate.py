

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
import pandas as pd
import math
import time

np.random.seed(42)


def true_loss(x):
    s1 = np.maximum(0.0, x[..., 0] * 1.5 + 0.5 * x[..., 1] - 0.2) ** 2
    s2 = np.maximum(0.0, -0.8 * x[..., 2] + 0.3 * x[..., 3] - 0.1) ** 1.5
    interaction = 0.5 * np.sin(x[..., 0] * x[..., 2]) + 0.3 * (x[..., 1] * x[..., 3])
    heavy = 0.6 * (np.abs(x[..., 0]) ** 1.8)
    base = 2.0 + s1 + 0.8 * s2 + interaction + heavy
    return base

def low_fidelity_loss(x):
    """Cheap but correlated version of true_loss."""
    return 0.85 * true_loss(x) + 0.3 * np.sin(0.5 * x[..., 0]) + np.random.normal(0, 0.05, size=x.shape[0])



N_candidates = 15000   
dim = 4
X_candidates = np.random.normal(scale=1.5, size=(N_candidates, dim))
Y_true_full = true_loss(X_candidates)

alpha = 0.99
def compute_var_cvar(values, alpha=0.99):
    sorted_vals = np.sort(values)
    idx = int(math.ceil(alpha * len(values)) - 1)
    var = sorted_vals[idx]
    cvar = np.mean(sorted_vals[idx:])
    return var, cvar

true_var, true_cvar = compute_var_cvar(Y_true_full, alpha=alpha)
print(f"Reference True VaR_{alpha}: {true_var:.4f}, CVaR: {true_cvar:.4f}")


n_init = 60
idx_init = np.random.choice(N_candidates, size=n_init, replace=False)
X_train = X_candidates[idx_init].copy()
Y_train = Y_true_full[idx_init].copy()
pool_idx = np.setdiff1d(np.arange(N_candidates), idx_init)


kernel = C(1.0) * Matern(length_scale=np.ones(dim), nu=1.5) + WhiteKernel(noise_level=1e-6)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=2, random_state=0)

gp.fit(X_train, Y_train)


def train_nn_ensemble(X, y, n_models=5, hidden_layer_sizes=(100, 50)):
    """Train an ensemble of sklearn MLPRegressor models on bootstrap resamples.
    Returns a list of fitted models. Uses different random_state for diversity.
    """
    models = []
    n = X.shape[0]
    for i in range(n_models):
        
        idx = np.random.choice(n, size=n, replace=True)
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                             activation='relu', solver='adam', max_iter=500,
                             early_stopping=True, n_iter_no_change=10,
                             random_state=42 + i)
        model.fit(X[idx], y[idx])
        models.append(model)
    return models

def nn_predict_in_batches(models, X, batch_size=2000):
    """Predict ensemble mean and std in batches to mirror GP predict interface.
    Returns (mu, sigma) arrays of shape (n_samples,).
    """
    n = X.shape[0]
    mus = np.empty(n)
    sigs = np.empty(n)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        
        preds = np.column_stack([m.predict(X[i:j]) for m in models])
        mu = preds.mean(axis=1)
        sigma = preds.std(axis=1, ddof=0)
        mus[i:j] = mu
        sigs[i:j] = sigma
    return mus, sigs


SURROGATE = 'nn'  # options: 'gp' or 'nn'


nn_models = None
if SURROGATE == 'nn':
    nn_models = train_nn_ensemble(X_train, Y_train, n_models=5)


def gp_predict_in_batches(model, X, batch_size=2000):
    """Predict mean and std in batches to reduce memory usage."""
    n = X.shape[0]
    mus = np.empty(n)
    sigs = np.empty(n)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        mu, sigma = model.predict(X[i:j], return_std=True)
        mus[i:j] = mu
        sigs[i:j] = sigma
    return mus, sigs


def run_active_learning(surrogate_name, X_candidates, Y_true_full, idx_init,
                        n_iterations=12, queries_per_iter=20, acquisition_pool_size=5000,
                        nn_ensemble_size=5, nn_hidden=(100,50)):
    """Run the active-learning loop for a given surrogate ('gp' or 'nn').
    Returns a pandas DataFrame with history entries.
    """

    X_train_loc = X_candidates[idx_init].copy()
    Y_train_loc = Y_true_full[idx_init].copy()
    pool_idx_loc = np.setdiff1d(np.arange(X_candidates.shape[0]), idx_init)

    if surrogate_name == 'gp':
        kernel_loc = C(1.0) * Matern(length_scale=np.ones(dim), nu=1.5) + WhiteKernel(noise_level=1e-6)
        model_loc = GaussianProcessRegressor(kernel=kernel_loc, alpha=0.0, normalize_y=True,
                                             n_restarts_optimizer=2, random_state=0)
        model_loc.fit(X_train_loc, Y_train_loc)
        predict_fn = lambda X: gp_predict_in_batches(model_loc, X, batch_size=2000)
        retrain_fn = lambda X, y: model_loc.fit(X, y)
    elif surrogate_name == 'nn':
        model_loc = train_nn_ensemble(X_train_loc, Y_train_loc, n_models=nn_ensemble_size,
                                     hidden_layer_sizes=nn_hidden)
        predict_fn = lambda X: nn_predict_in_batches(model_loc, X, batch_size=2000)
        retrain_fn = lambda X, y: train_nn_ensemble(X, y, n_models=nn_ensemble_size, hidden_layer_sizes=nn_hidden)
    else:
        raise ValueError("surrogate_name must be 'gp' or 'nn'")

    history_loc = []
    for it in range(n_iterations):
        
        if len(pool_idx_loc) <= acquisition_pool_size:
            acquisition_idx = pool_idx_loc.copy()
        else:
            acquisition_idx = np.random.choice(pool_idx_loc, size=acquisition_pool_size, replace=False)
        X_acq = X_candidates[acquisition_idx]

        
        mu_acq, sigma_acq = predict_fn(X_acq)

        
        mu_full, _ = predict_fn(X_candidates)
        est_var_full, est_cvar_full = compute_var_cvar(mu_full, alpha=alpha)

        
        U = sigma_acq / (1.0 + np.abs(mu_acq - est_var_full))
        top_local = np.argsort(-U)[:queries_per_iter]
        selected_global_idx = acquisition_idx[top_local]

        
        X_new = X_candidates[selected_global_idx]
        Y_new = Y_true_full[selected_global_idx]

        
        X_train_loc = np.vstack([X_train_loc, X_new])
        Y_train_loc = np.concatenate([Y_train_loc, Y_new])
        pool_idx_loc = np.setdiff1d(pool_idx_loc, selected_global_idx)

        
        if surrogate_name == 'gp':
            retrain_fn(X_train_loc, Y_train_loc)
        else:
            model_loc = retrain_fn(X_train_loc, Y_train_loc)

        
        mu_full, _ = (gp_predict_in_batches(model_loc, X_candidates, batch_size=3000)
                      if surrogate_name == 'gp' else
                      nn_predict_in_batches(model_loc, X_candidates, batch_size=3000))
        est_var_all, est_cvar_all = compute_var_cvar(mu_full, alpha=alpha)
        err_var = 100.0 * abs(est_var_all - true_var) / (abs(true_var) + 1e-12)
        err_cvar = 100.0 * abs(est_cvar_all - true_cvar) / (abs(true_cvar) + 1e-12)

        history_loc.append({
            'iteration': it + 1,
            'n_train': len(X_train_loc),
            'est_var': est_var_all,
            'est_cvar': est_cvar_all,
            'err_var_pct': err_var,
            'err_cvar_pct': err_cvar
        })
        print(f"[{surrogate_name}] Iter {it+1:2d}: n_train={len(X_train_loc):4d}, VaR_est={est_var_all:.4f}, err_var={err_var:.3f}%, CVaR_est={est_cvar_all:.4f}, err_cvar={err_cvar:.3f}%")

    return pd.DataFrame(history_loc)



n_iterations = 12
queries_per_iter = 20
acquisition_pool_size = 5000


np.random.seed(42)
idx_init_shared = np.random.choice(N_candidates, size=n_init, replace=False)

print('\nRunning GP experiment...')
df_gp = run_active_learning('gp', X_candidates, Y_true_full, idx_init_shared,
                            n_iterations=n_iterations, queries_per_iter=queries_per_iter,
                            acquisition_pool_size=acquisition_pool_size)

print('\nRunning NN experiment...')
df_nn = run_active_learning('nn', X_candidates, Y_true_full, idx_init_shared,
                            n_iterations=n_iterations, queries_per_iter=queries_per_iter,
                            acquisition_pool_size=acquisition_pool_size, nn_ensemble_size=5)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(df_gp['n_train'], df_gp['err_var_pct'], marker='o', label='GP VaR error')
plt.plot(df_nn['n_train'], df_nn['err_var_pct'], marker='o', label='NN VaR error')
plt.xlabel('Number of true evaluations (training points)')
plt.ylabel('VaR relative error (%)')
plt.title(f'VaR error comparison (alpha={alpha})')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(df_gp['n_train'], df_gp['err_cvar_pct'], marker='s', label='GP CVaR error')
plt.plot(df_nn['n_train'], df_nn['err_cvar_pct'], marker='s', label='NN CVaR error')
plt.xlabel('Number of true evaluations (training points)')
plt.ylabel('CVaR relative error (%)')
plt.title(f'CVaR error comparison (alpha={alpha})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print('\nFinal comparison (last iteration):')
print('GP final errors (%):', df_gp.iloc[-1]['err_var_pct'], df_gp.iloc[-1]['err_cvar_pct'])
print('NN final errors (%):', df_nn.iloc[-1]['err_var_pct'], df_nn.iloc[-1]['err_cvar_pct'])
