"""
data_acquisition.py
═══════════════════
Modul khusus untuk mengakuisisi data mentah dari Yahoo Finance.
TIDAK ADA proses pembersihan (preprocessing) di sini.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

BANK_TICKERS = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank Negara Indonesia",
    "BRIS.JK": "Bank Syariah Indonesia",
}

def fetch_raw_stock_data(ticker: str, period_years: int = 5) -> pd.DataFrame:
    """Mengambil data OHLCV historis mentah dari Yahoo Finance."""
    end_date   = datetime.today()
    # Buffer hari ditambahkan agar saat preprocessing, MA50/indikator lain tidak kekurangan data
    start_date = end_date - timedelta(days=period_years * 365 + 100) 

    logger.info(f"Fetching RAW data {ticker} dari {start_date.date()} ke {end_date.date()}")

    raw = yf.download(
        ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True, # Normalisasi dasar terhadap aksi korporasi
    )

    if raw.empty:
        raise ValueError(f"Tidak ada data untuk ticker '{ticker}'. Pastikan simbol benar.")

    return raw
