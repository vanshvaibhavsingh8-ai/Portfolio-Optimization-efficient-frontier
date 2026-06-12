"""Walk-forward out-of-sample backtest.

Each year: estimate mu/cov on the trailing 3 years of daily returns,
form weights (equal-weight, min-variance, max-Sharpe), hold them for the
next year. No lookahead: weights for year T use only data through year T-1.
"""

import numpy as np
import pandas as pd

from optimization import annualized_stats, min_variance_weights, max_sharpe_weights

TRADING_DAYS = 252
LOOKBACK_YEARS = 3
RF = 0.02  # assumed constant annual risk-free rate


def walk_forward(returns: pd.DataFrame):
    """Returns dict of strategy name -> daily out-of-sample return series,
    plus a DataFrame of the weights chosen at each rebalance."""
    years = sorted(returns.index.year.unique())
    test_years = years[LOOKBACK_YEARS:]

    oos = {"Equal-weight": [], "Min-variance": [], "Max-Sharpe": []}
    weight_records = []
    n = returns.shape[1]

    for year in test_years:
        train = returns[(returns.index.year >= year - LOOKBACK_YEARS)
                        & (returns.index.year < year)]
        test = returns[returns.index.year == year]
        if len(test) == 0 or len(train) < TRADING_DAYS:
            continue

        mu, cov = annualized_stats(train)
        weights = {
            "Equal-weight": np.full(n, 1 / n),
            "Min-variance": min_variance_weights(cov),
            "Max-Sharpe": max_sharpe_weights(mu, cov, rf=RF),
        }
        for name, w in weights.items():
            oos[name].append(test @ w)
            weight_records.append(
                {"year": year, "strategy": name,
                 **dict(zip(returns.columns, np.round(w, 4)))})

    oos_returns = pd.DataFrame({name: pd.concat(parts) for name, parts in oos.items()})
    return oos_returns, pd.DataFrame(weight_records)


def performance_summary(oos_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized return/vol, Sharpe, and max drawdown per strategy."""
    rows = {}
    for name, r in oos_returns.items():
        ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        wealth = (1 + r).cumprod()
        max_dd = (wealth / wealth.cummax() - 1).min()
        rows[name] = {
            "Ann. return": ann_ret,
            "Ann. volatility": ann_vol,
            "Sharpe (rf=2%)": (ann_ret - RF) / ann_vol,
            "Max drawdown": max_dd,
        }
    return pd.DataFrame(rows).T.round(4)
