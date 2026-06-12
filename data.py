"""Download price data and compute returns."""

import pandas as pd
import yfinance as yf

# Diversified universe across sectors (US large caps + a bond and gold ETF)
TICKERS = [
    "AAPL",  # tech
    "MSFT",  # tech
    "JPM",   # financials
    "JNJ",   # healthcare
    "PG",    # consumer staples
    "XOM",   # energy
    "CAT",   # industrials
    "WMT",   # consumer retail
    "NEE",   # utilities
    "DIS",   # communication/media
    "TLT",   # long-term treasuries ETF
    "GLD",   # gold ETF
]

START = "2014-01-01"


def download_prices(tickers=TICKERS, start=START, end=None) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"].dropna(how="all")
    # Drop tickers with too much missing history, then drop remaining NaN rows
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.95)).dropna()
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()
