import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from utils import save_dataframe_cache, load_dataframe_cache

logger = logging.getLogger(__name__)

BANK_TICKERS = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank Negara Indonesia",
    "BRIS.JK": "Bank Syariah Indonesia",
}

MARKET_TICKERS = {"ihsg": "^JKSE", "usdidr": "IDR=X"}


def _download_raw_stock_data(ticker: str, period_years: int = 10) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365 + 120)

    logger.info(f"Fetching RAW {ticker} {start_date.date()} -> {end_date.date()}")
    raw = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False, auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"Tidak ada data untuk '{ticker}'. Periksa simbol.")
    return raw



def fetch_raw_stock_data(ticker: str, period_years: int = 10) -> pd.DataFrame:
    cache_key = f"raw_{ticker}_{period_years}y"
    cached = load_dataframe_cache(cache_key, max_age_hours=6)
    if cached is not None:
        return cached
    try:
        raw = _download_raw_stock_data(ticker, period_years)
    except Exception as e:
        stale = load_dataframe_cache(cache_key, max_age_hours=None)
        if stale is not None:
            logger.warning(f"Unduhan {ticker} gagal ({e}); memakai cache lama.")
            return stale
        raise
    save_dataframe_cache(cache_key, raw)
    return raw

def fetch_market_context(period_years: int = 10) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365 + 120)
    out = pd.DataFrame()
    for name, symbol in MARKET_TICKERS.items():
        try:
            raw = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"),
                              end=end_date.strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index)
            ret = np.log(close).diff()
            out[f"{name}_ret"] = ret
        except Exception as e:
            logger.warning(f"Gagal mengambil konteks pasar {symbol}: {e}")
    if not out.empty:
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
    return out
