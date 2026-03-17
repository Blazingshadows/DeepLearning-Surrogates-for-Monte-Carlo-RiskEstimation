"""Surrogate comparison script

Runs active-learning surrogate experiments (GP vs NN ensemble) for the portfolio
used in `real_mc_sim.py`. Uses the same fitted parameters (mu, cov, weights) saved
by the ground-truth run in `mc_ground_truth.npz` when available. Compares VaR/CVaR
estimates, error/accuracy, and runtime against the real Monte Carlo solution.
"""

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
GT_FILE = ROOT / 'mc_ground_truth.npz'

if not GT_FILE.exists():
    raise FileNotFoundError(f"Ground-truth file not found: {GT_FILE}. Run real_mc_sim.py first.")

data = np.load(GT_FILE)
mu = data['mu']
cov = data['cov']
weights = data['weights']
true_VaR = float(data['VaR'])
true_CVaR = float(data['CVaR'])
real_mc_time = float(data.get('time', np.nan))

dim = len(mu)


def portfolio_loss_batch(X, weights, cov):
    # X: (n, dim)
    linear = X @ weights
    # quadratic term: 0.5 * diag(X @ cov @ X.T)
    quad = 0.5 * np.einsum('ij,jk,ik->i', X, cov, X)
    nonlinear = 0.05 * np.sin(2 * X[:, 0]) + 0.02 * (X[:, 1] ** 2)
    return -(linear + quad + nonlinear)


def compute_var_cvar(samples, alpha=0.99):
    # Match the definition used in real_mc_sim.py: use the lower tail index
    sorted_losses = np.sort(samples)
    var_idx = int((1 - alpha) * len(samples))
    var = sorted_losses[var_idx]
    cvar = sorted_losses[:var_idx].mean()
    return var, cvar


# --- surrogate helpers
def gp_predict_in_batches(model, X, batch_size=2000):
    n = X.shape[0]
    mus = np.empty(n)
    sigs = np.empty(n)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        mu, sigma = model.predict(X[i:j], return_std=True)
        mus[i:j] = mu
        sigs[i:j] = sigma
    return mus, sigs


def train_nn_ensemble(X, y, n_models=5, hidden_layer_sizes=(150, 75)):
    scaler  = StandardScaler()
    Xs = scaler.fit_transform(X)
    models = []
    n = Xs.shape[0]
    for i in range(n_models):
        idx = np.random.choice(n, size=n, replace=True)
        m = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,activation='relu', max_iter=2500,
                         early_stopping=True, n_iter_no_change=20,learning_rate_init=5e-4,alpha=0.001, random_state=42 + i)
        m.fit(Xs[idx], y[idx])
        models.append(m)
    return models, scaler


def nn_predict_in_batches(models, scaler, X, batch_size=2000):
    Xs = scaler.transform(X)
    n = Xs.shape[0]
    mus = np.empty(n)
    sigs = np.empty(n)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        preds = np.column_stack([m.predict(Xs[i:j]) for m in models])
        mus[i:j] = preds.mean(axis=1)
        sigs[i:j] = preds.std(axis=1, ddof=0)
    return mus, sigs


def run_surrogate(method='gp', seed=42,
                  N_candidates=15000, n_init=60, n_iterations=12,
                  queries_per_iter=20, acquisition_pool_size=5000):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # build candidate pool from the same parametric model (mu,cov)
    X_candidates = rng.multivariate_normal(mu, cov, size=N_candidates)
    Y_true_full = portfolio_loss_batch(X_candidates, weights, cov)

    # reference true VaR/CVaR from ground-truth (full real MC)
    ref_var = true_VaR
    ref_cvar = true_CVaR

    # initial design
    idx_init = np.random.choice(N_candidates, size=n_init, replace=False)
    X_train = X_candidates[idx_init].copy()
    Y_train = Y_true_full[idx_init].copy()
    pool_idx = np.setdiff1d(np.arange(N_candidates), idx_init)

    # surrogate init
    if method == 'gp':
        kernel = C(1.0) * Matern(length_scale=np.ones(dim), nu=1.5) + WhiteKernel(noise_level=1e-6)
        model = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True,
                                         n_restarts_optimizer=2, random_state=0)
        t0 = time.time()
        model.fit(X_train, Y_train)
        t_fit = time.time() - t0
        predict = lambda X: gp_predict_in_batches(model, X)
    else:
        # NN ensemble
        t0 = time.time()
        models, scaler = train_nn_ensemble(X_train, Y_train, n_models=5)
        t_fit = time.time() - t0
        predict = lambda X: nn_predict_in_batches(models, scaler,  X)

    history = []
    total_time = t_fit

    for it in range(n_iterations):
        t_iter_start = time.time()
        # sample acquisition subset
        if len(pool_idx) <= acquisition_pool_size:
            acquisition_idx = pool_idx.copy()
        else:
            acquisition_idx = np.random.choice(pool_idx, size=acquisition_pool_size, replace=False)
        X_acq = X_candidates[acquisition_idx]

        # predict on acquisition subset
        mu_acq, sigma_acq = predict(X_acq)

        # predict on full pool to estimate VaR (surrogate mean)
        mu_full, sigma_full = predict(X_candidates)
        est_var_full, est_cvar_full = compute_var_cvar(mu_full, alpha=0.99)

        U = sigma_acq / (1.0 + np.abs(mu_acq - est_var_full))
        top_local = np.argsort(-U)[:queries_per_iter]
        selected_global_idx = acquisition_idx[top_local]

        # query oracle
        X_new = X_candidates[selected_global_idx]
        Y_new = Y_true_full[selected_global_idx]

        # update training
        X_train = np.vstack([X_train, X_new])
        Y_train = np.concatenate([Y_train, Y_new])
        pool_idx = np.setdiff1d(pool_idx, selected_global_idx)

        # retrain
        t_fit0 = time.time()
        if method == 'gp':
            model.fit(X_train, Y_train)
        else:
            models, scaler = train_nn_ensemble(X_train, Y_train, n_models=5)
        t_fit_iter = time.time() - t_fit0
        total_time += t_fit_iter

        # evaluate
        mu_full, sigma_full = predict(X_candidates)
        est_var_all, est_cvar_all = compute_var_cvar(mu_full, alpha=0.99)
        err_var = 100.0 * abs(est_var_all - ref_var) / (abs(ref_var) + 1e-12)
        err_cvar = 100.0 * abs(est_cvar_all - ref_cvar) / (abs(ref_cvar) + 1e-12)

        history.append({'iteration': it + 1,
                        'n_train': len(X_train),
                        'est_var': est_var_all,
                        'est_cvar': est_cvar_all,
                        'err_var_pct': err_var,
                        'err_cvar_pct': err_cvar})

        total_time += (time.time() - t_iter_start - t_fit_iter)

    # final full-pool predictions to return for plotting/analysis
    mu_full, sigma_full = predict(X_candidates)
    return history, total_time, X_candidates, Y_true_full, mu_full, sigma_full
    # (unreachable) keep signature consistent



def plot_results(hist_gp, hist_nn, ref_var, ref_cvar):
    df_gp = pd.DataFrame(hist_gp)
    df_nn = pd.DataFrame(hist_nn)

    plt.figure(figsize=(10, 5))
    plt.plot(df_gp['n_train'], df_gp['est_var'], '-o', label='GP est VaR')
    plt.plot(df_nn['n_train'], df_nn['est_var'], '-s', label='NN est VaR')
    plt.hlines(ref_var, df_gp['n_train'].min(), df_gp['n_train'].max(), colors='k', linestyles='--', label='True VaR')
    plt.xlabel('Number of true evaluations')
    plt.ylabel('VaR estimate')
    plt.legend()
    plt.grid(True)
    plt.title('VaR estimates: GP vs NN')

    plt.figure(figsize=(10, 5))
    plt.plot(df_gp['n_train'], df_gp['est_cvar'], '-o', label='GP est CVaR')
    plt.plot(df_nn['n_train'], df_nn['est_cvar'], '-s', label='NN est CVaR')
    plt.hlines(ref_cvar, df_gp['n_train'].min(), df_gp['n_train'].max(), colors='k', linestyles='--', label='True CVaR')
    plt.xlabel('Number of true evaluations')
    plt.ylabel('CVaR estimate')
    plt.legend()
    plt.grid(True)
    plt.title('CVaR estimates: GP vs NN')

    plt.show()


def main():
    params = dict(N_candidates=15000, n_init=60, n_iterations=12, queries_per_iter=20, acquisition_pool_size=5000)

    print('Running GP surrogate...')
    t0 = time.time()
    hist_gp, time_gp, X_cand_gp, Y_true_gp, mu_gp, sigma_gp = run_surrogate('gp', **params)
    t_gp = time.time() - t0

    print('Running NN surrogate...')
    t0 = time.time()
    hist_nn, time_nn, X_cand_nn, Y_true_nn, mu_nn, sigma_nn = run_surrogate('nn', **params)
    t_nn = time.time() - t0

    print('\nSummary:')
    print(f'Real MC time (ground truth): {real_mc_time:.3f} s')
    print(f'GP surrogate wall-clock time (including retrain/predict): {t_gp:.3f} s')
    print(f'NN surrogate wall-clock time (including retrain/predict): {t_nn:.3f} s')

    # final errors and accuracy
    df_gp = pd.DataFrame(hist_gp)
    df_nn = pd.DataFrame(hist_nn)
    final_gp = df_gp.iloc[-1]
    final_nn = df_nn.iloc[-1]

    err_var_gp = final_gp['err_var_pct']
    err_cvar_gp = final_gp['err_cvar_pct']
    err_var_nn = final_nn['err_var_pct']
    err_cvar_nn = final_nn['err_cvar_pct']

    acc_var_gp = max(0.0, 100.0 - err_var_gp)
    acc_cvar_gp = max(0.0, 100.0 - err_cvar_gp)
    acc_var_nn = max(0.0, 100.0 - err_var_nn)
    acc_cvar_nn = max(0.0, 100.0 - err_cvar_nn)

    print('\nFinal errors (%):')
    print(f'GP: VaR err = {err_var_gp:.3f}%, CVaR err = {err_cvar_gp:.3f}%  -> accuracy approx VaR {acc_var_gp:.2f}%, CVaR {acc_cvar_gp:.2f}%')
    print(f'NN: VaR err = {err_var_nn:.3f}%, CVaR err = {err_cvar_nn:.3f}%  -> accuracy approx VaR {acc_var_nn:.2f}%, CVaR {acc_cvar_nn:.2f}%')

    plot_results(hist_gp, hist_nn, true_VaR, true_CVaR)

    # additional comparison plots: density vs loss and portfolio value over days
    def gaussian_mixture_pdf(x, mus, sigs, eps=1e-8):
        sigs = np.clip(sigs, eps, None)
        coeff = 1.0 / (np.sqrt(2 * np.pi) * sigs)
        exps = np.exp(-0.5 * ((x[:, None] - mus[None, :]) / sigs[None, :]) ** 2)
        pdf = (coeff[None, :] * exps).mean(axis=1)
        return pdf

    def plot_density_vs_loss(Y_true, mu_pred, sigma_pred, label_prefix='Surrogate'):
        # histogram of true losses
        plt.figure(figsize=(10, 5))
        counts, bins, _ = plt.hist(Y_true, bins=80, density=True, alpha=0.4, label='True loss histogram')
        xs = np.linspace(bins.min(), bins.max(), 1000)
        # surrogate predicted density (Gaussian mixture)
        pdf_sur = gaussian_mixture_pdf(xs, mu_pred, sigma_pred)
        plt.plot(xs, pdf_sur, '-', lw=2, label=f'{label_prefix} predicted density')

        # overlay VaR/CVaR lines
        true_v, true_cv = compute_var_cvar(Y_true, alpha=0.99)
        est_v, est_cv = compute_var_cvar(mu_pred, alpha=0.99)
        plt.axvline(true_v, color='k', linestyle='--', label='True VaR')
        plt.axvline(true_cv, color='k', linestyle=':', label='True CVaR')
        plt.axvline(est_v, color='r', linestyle='--', label=f'{label_prefix} est VaR')
        plt.axvline(est_cv, color='r', linestyle=':', label=f'{label_prefix} est CVaR')

        plt.xlabel('Loss')
        plt.ylabel('Density')
        plt.title('Density of losses: true vs surrogate predicted')
        plt.legend()
        plt.grid(True)
        plt.show()

    def simulate_portfolio_paths(mu, cov, weights, n_days=10, n_paths=2000, initial_value=10000, seed=0):
        rng = np.random.default_rng(seed)
        dim = len(mu)
        # draw (n_paths, n_days, dim)
        draws = rng.multivariate_normal(mu, cov, size=(n_paths, n_days))
        # portfolio daily returns for each path/day
        port_rets = draws @ weights
        # cumulative portfolio value (multiplicative returns)
        values = np.empty((n_paths, n_days + 1))
        values[:, 0] = initial_value
        for t in range(n_days):
            values[:, t + 1] = values[:, t] * (1.0 + port_rets[:, t])
        return values

    def plot_portfolio_value_progression(mu, cov, weights, initial_value=10000):
        vals = simulate_portfolio_paths(mu, cov, weights, n_days=10, n_paths=2000, initial_value=initial_value, seed=123)
        days = np.arange(vals.shape[1])
        p10 = np.percentile(vals, 10, axis=0)
        p50 = np.percentile(vals, 50, axis=0)
        p90 = np.percentile(vals, 90, axis=0)

        plt.figure(figsize=(10, 6))
        plt.fill_between(days, p10, p90, color='lightgray', label='10-90 percentile')
        plt.plot(days, p50, '-k', lw=2, label='Median portfolio value')
        plt.xlabel('Days')
        plt.ylabel('Portfolio value')
        plt.title('Portfolio value progression (parametric trajectories)')
        plt.grid(True)
        plt.legend()
        plt.show()

    # show density plots: use GP predictions vs true
    plot_density_vs_loss(Y_true_gp, mu_gp, sigma_gp, label_prefix='GP')
    plot_density_vs_loss(Y_true_nn, mu_nn, sigma_nn, label_prefix='NN')

    # portfolio progression plot (parametric simulation)
    plot_portfolio_value_progression(mu, cov, weights, initial_value=10000)


if __name__ == '__main__':
    main()
