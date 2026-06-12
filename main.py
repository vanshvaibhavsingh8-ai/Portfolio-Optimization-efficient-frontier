"""Run the full analysis: download data, plot the efficient frontier,
backtest equal-weight vs. min-variance vs. max-Sharpe out-of-sample."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import download_prices, daily_returns
from optimization import (annualized_stats, efficient_frontier,
                          min_variance_weights, max_sharpe_weights,
                          portfolio_return, portfolio_vol)
from backtest import walk_forward, performance_summary, RF

OUT = "output"


def plot_frontier(returns):
    """Efficient frontier on the full sample, with individual assets and key portfolios."""
    mu, cov = annualized_stats(returns)
    vols, rets = efficient_frontier(mu, cov)

    w_mv = min_variance_weights(cov)
    w_ms = max_sharpe_weights(mu, cov, rf=RF)
    w_eq = np.full(len(mu), 1 / len(mu))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(vols, rets, "b-", lw=2, label="Efficient frontier")

    asset_vols = np.sqrt(np.diag(cov))
    ax.scatter(asset_vols, mu, c="gray", s=40, alpha=0.7)
    for ticker, v, r in zip(returns.columns, asset_vols, mu):
        ax.annotate(ticker, (v, r), fontsize=8, xytext=(5, 3),
                    textcoords="offset points")

    for w, label, color in [(w_eq, "Equal-weight", "orange"),
                            (w_mv, "Min-variance", "green"),
                            (w_ms, "Max-Sharpe", "red")]:
        ax.scatter(portfolio_vol(w, cov), portfolio_return(w, mu),
                   c=color, s=120, marker="*", label=label, zorder=5)

    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.set_title("Efficient frontier (full sample, long-only)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/efficient_frontier.png", dpi=150)
    plt.close(fig)


def plot_backtest(oos_returns):
    wealth = (1 + oos_returns).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    wealth.plot(ax=ax)
    ax.set_ylabel("Growth of $1")
    ax.set_title("Out-of-sample wealth curves (annual rebalance, 3y lookback)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/backtest_wealth.png", dpi=150)
    plt.close(fig)


def main():
    print("Downloading prices...")
    prices = download_prices()
    returns = daily_returns(prices)
    print(f"{returns.shape[1]} assets, {len(returns)} days "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")

    print("Plotting efficient frontier...")
    plot_frontier(returns)

    print("Running walk-forward backtest...")
    oos_returns, weights = walk_forward(returns)
    plot_backtest(oos_returns)

    summary = performance_summary(oos_returns)
    summary.to_csv(f"{OUT}/performance_summary.csv")
    weights.to_csv(f"{OUT}/rebalance_weights.csv", index=False)

    print("\n=== Out-of-sample performance "
          f"({oos_returns.index[0].date()} to {oos_returns.index[-1].date()}) ===")
    print(summary.to_string())
    print(f"\nOutputs saved to {OUT}/")


if __name__ == "__main__":
    main()
