"""
data_acquisition.py
═══════════════════
Modul untuk mengambil data historis saham dari Yahoo Finance menggunakan yfinance.
Menangani missing values, outlier, dan validasi data secara otomatis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════════════════════
BANK_TICKERS = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank Negara Indonesia",
    "BRIS.JK": "Bank Syariah Indonesia",
}

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ══════════════════════════════════════════════════════════════════════════════
def fetch_stock_data(ticker: str, period_years: int = 5) -> pd.DataFrame:
    """
    Mengambil data OHLCV historis dari Yahoo Finance.

    Args:
        ticker      : Simbol saham (contoh: 'BBCA.JK')
        period_years: Jumlah tahun data historis yang diambil

    Returns:
        DataFrame dengan kolom lowercase OHLCV + Adj Close yang sudah dibersihkan
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365 + 30)  # buffer 30 hari

    logger.info(f"Fetching {ticker} from {start_date.date()} to {end_date.date()}")

    # ── Download data ──────────────────────────────────────────────────────
    raw = yf.download(
        ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True,  # adjusted prices secara otomatis
    )

    # ── Validasi data kosong ───────────────────────────────────────────────
    if raw.empty:
        raise ValueError(f"Tidak ada data untuk ticker '{ticker}'. "
                         "Pastikan simbol benar dan pasar aktif.")

    # ── Flatten MultiIndex columns jika ada ───────────────────────────────
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    # ── Seleksi dan rename kolom ke lowercase ─────────────────────────────
    available = [c for c in REQUIRED_COLUMNS if c in raw.columns]
    if len(available) < 4:
        raise ValueError(f"Kolom tidak lengkap. Tersedia: {list(raw.columns)}")

    df = raw[available].copy()
    df.columns = df.columns.str.lower()

    # ── Tangani Missing Values ─────────────────────────────────────────────
    df = _handle_missing_values(df)

    # ── Hapus baris dengan harga 0 atau negatif ────────────────────────────
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df = df[df[col] > 0]

    # ── Pastikan indeks adalah DatetimeIndex ──────────────────────────────
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # ── Buang data duplikat ────────────────────────────────────────────────
    df = df[~df.index.duplicated(keep='last')]

    # ── Trim ke periode yang diminta ──────────────────────────────────────
    cutoff = end_date - timedelta(days=period_years * 365)
    df = df[df.index >= pd.Timestamp(cutoff)]

    logger.info(f"Data berhasil diambil: {len(df)} baris, "
                f"{df.index.min().date()} → {df.index.max().date()}")

    return df


def fetch_all_banks(period_years: int = 5) -> dict[str, pd.DataFrame]:
    """
    Mengambil data semua 5 bank Indonesia secara bersamaan.

    Returns:
        Dict {ticker: DataFrame}
    """
    results = {}
    for ticker in BANK_TICKERS:
        try:
            results[ticker] = fetch_stock_data(ticker, period_years)
            logger.info(f"✅ {ticker} berhasil diambil")
        except Exception as e:
            logger.warning(f"⚠️ {ticker} gagal: {e}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategi penanganan missing values untuk data OHLCV:
    1. Forward fill untuk harga (hari libur bursa)
    2. Backward fill untuk NaN di awal
    3. Interpolasi linear untuk gap kecil (< 5 hari)
    4. Drop baris yang masih NaN setelah semua metode
    """
    # Identifikasi jumlah NaN sebelum
    nan_count_before = df.isnull().sum().sum()
    if nan_count_before > 0:
        logger.info(f"Menangani {nan_count_before} missing values...")

    # Hitung konsekutif NaN untuk tiap kolom
    for col in df.columns:
        # Interpolasi linear untuk gap kecil (≤5 hari berurutan)
        mask = df[col].isnull()
        if mask.any():
            # Hitung panjang setiap blok NaN
            groups = mask.ne(mask.shift()).cumsum()
            gap_sizes = df[col].isnull().groupby(groups).transform('sum')
            small_gaps = mask & (gap_sizes <= 5)
            df.loc[small_gaps, col] = df[col].interpolate(method='linear')[small_gaps]

    # Forward fill (harga terakhir yang diketahui)
    df = df.ffill()

    # Backward fill (untuk NaN di awal dataset)
    df = df.bfill()

    # Drop sisa NaN yang tidak bisa diselesaikan
    df = df.dropna()

    nan_count_after = df.isnull().sum().sum()
    if nan_count_before > 0:
        logger.info(f"Missing values setelah penanganan: {nan_count_after}")

    return df


def validate_data_quality(df: pd.DataFrame, ticker: str) -> dict:
    """
    Mengevaluasi kualitas data yang diambil.

    Returns:
        Dict berisi statistik kualitas data
    """
    return {
        "ticker":        ticker,
        "n_rows":        len(df),
        "date_start":    df.index.min().strftime('%Y-%m-%d'),
        "date_end":      df.index.max().strftime('%Y-%m-%d'),
        "n_missing":     df.isnull().sum().sum(),
        "close_min":     df['close'].min(),
        "close_max":     df['close'].max(),
        "close_mean":    df['close'].mean(),
        "daily_return_std": df['close'].pct_change().std(),
        "total_volume":  df['volume'].sum() if 'volume' in df.columns else None,
    }
