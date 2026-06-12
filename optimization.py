"""Markowitz mean-variance optimization: min-variance, max-Sharpe, efficient frontier.

Long-only portfolios (weights >= 0, sum to 1), solved with scipy SLSQP.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def annualized_stats(returns: pd.DataFrame):
    """Annualized mean return vector and covariance matrix from daily returns."""
    mu = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    return mu.values, cov.values


def portfolio_return(w, mu):
    return w @ mu


def portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)


def _solve(objective, n, extra_constraints=()):
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra_constraints]
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1 / n)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return res.x


def min_variance_weights(cov):
    n = cov.shape[0]
    return _solve(lambda w: w @ cov @ w, n)


def max_sharpe_weights(mu, cov, rf=0.0):
    n = len(mu)
    def neg_sharpe(w):
        vol = portfolio_vol(w, cov)
        return -(w @ mu - rf) / vol if vol > 0 else 0.0
    return _solve(neg_sharpe, n)


def efficient_frontier(mu, cov, n_points=50):
    """Min-vol portfolio for each target return between min-var return and max asset return."""
    w_minvar = min_variance_weights(cov)
    r_lo = portfolio_return(w_minvar, mu)
    r_hi = mu.max()
    targets = np.linspace(r_lo, r_hi, n_points)
    vols, rets = [], []
    n = len(mu)
    for target in targets:
        cons = ({"type": "eq", "fun": lambda w, t=target: w @ mu - t},)
        try:
            w = _solve(lambda w: w @ cov @ w, n, cons)
        except RuntimeError:
            continue
        vols.append(portfolio_vol(w, cov))
        rets.append(portfolio_return(w, mu))
    return np.array(vols), np.array(rets)
