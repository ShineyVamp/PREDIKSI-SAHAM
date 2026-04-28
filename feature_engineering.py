"""
feature_engineering.py
══════════════════════
Modul untuk membangun semua fitur yang dibutuhkan model TFT:
  · Fitur teknis  : RSI, Moving Averages, Volatilitas, MACD, Bollinger Bands
  · Fitur temporal: day_of_week, month, quarter, is_month_end, dll.
  · Static covariate: ticker_id (encoded)
  · Observed inputs : harga dan volume historis

TFT membedakan 3 jenis input:
  1. Static Covariates   — tidak berubah terhadap waktu (e.g. ticker ID)
  2. Known Future Inputs — diketahui di masa depan (e.g. fitur kalender)
  3. Observed Inputs     — hanya diketahui di masa lalu (e.g. harga, RSI)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Mapping ticker ke integer ID untuk static covariate
TICKER_ID_MAP = {
    "BBCA.JK": 0,
    "BBRI.JK": 1,
    "BMRI.JK": 2,
    "BBNI.JK": 3,
    "BRIS.JK": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ══════════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Pipeline feature engineering lengkap.

    Args:
        df     : DataFrame OHLCV dari data_acquisition
        ticker : Simbol saham

    Returns:
        DataFrame dengan semua fitur siap pakai
    """
    result = df.copy()

    # ── 1. Fitur Teknis ───────────────────────────────────────────────────
    result = _add_technical_features(result)

    # ── 2. Fitur Temporal (Known Future) ──────────────────────────────────
    result = _add_temporal_features(result)

    # ── 3. Fitur Harga Turunan ────────────────────────────────────────────
    result = _add_price_derived_features(result)

    # ── 4. Static Covariate ───────────────────────────────────────────────
    result['ticker_id'] = TICKER_ID_MAP.get(ticker, len(TICKER_ID_MAP))
    result['ticker']    = ticker

    # ── 5. Time Index untuk pytorch-forecasting ───────────────────────────
    result = result.reset_index()
    result.rename(columns={'Date': 'date', 'index': 'date'}, inplace=True)
    if 'date' not in result.columns and 'Date' not in result.columns:
        result['date'] = result.index

    # Pastikan kolom date ada dan bersih
    date_col = 'date' if 'date' in result.columns else 'Date'
    result[date_col] = pd.to_datetime(result[date_col])
    result = result.set_index(date_col)

    # ── 6. Hapus NaN yang dihasilkan dari perhitungan indikator ───────────
    # (MA50 menghasilkan 49 NaN di awal, RSI menghasilkan 14 NaN, dll.)
    initial_len = len(result)
    result = result.dropna()
    dropped = initial_len - len(result)
    if dropped > 0:
        import logging
        logging.getLogger(__name__).info(
            f"Membuang {dropped} baris awal akibat kalkulasi indikator teknis."
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FITUR TEKNIS
# ══════════════════════════════════════════════════════════════════════════════
def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan indikator teknis standar:
    - RSI (14), MA20, MA50, EMA12, EMA26
    - MACD dan Signal Line
    - Bollinger Bands (upper, lower, width)
    - ATR (Average True Range)
    - Volatilitas (std return 20 hari)
    - OBV (On-Balance Volume)
    """
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df.get('volume', pd.Series(0, index=df.index))

    # ── RSI (14) ──────────────────────────────────────────────────────────
    df['rsi'] = _compute_rsi(close, period=14)

    # ── Moving Averages ───────────────────────────────────────────────────
    df['ma_20'] = close.rolling(window=20).mean()
    df['ma_50'] = close.rolling(window=50).mean()
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()

    # ── MACD ──────────────────────────────────────────────────────────────
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────────
    bb_std         = close.rolling(20).std()
    df['bb_upper'] = df['ma_20'] + 2 * bb_std
    df['bb_lower'] = df['ma_20'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma_20']
    df['bb_pct']   = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # ── ATR (14) ──────────────────────────────────────────────────────────
    df['atr'] = _compute_atr(high, low, close, period=14)

    # ── Volatilitas (std return 20 hari) ──────────────────────────────────
    daily_return       = close.pct_change()
    df['volatility']   = daily_return.rolling(20).std()
    df['daily_return'] = daily_return

    # ── Volume features ───────────────────────────────────────────────────
    if 'volume' in df.columns:
        df['volume_ma20'] = vol.rolling(20).mean()
        df['volume_ratio'] = vol / (df['volume_ma20'] + 1e-8)
        df['obv']         = _compute_obv(close, vol)

    # ── Price position features ───────────────────────────────────────────
    df['price_ma20_ratio'] = close / (df['ma_20'] + 1e-8)
    df['price_ma50_ratio'] = close / (df['ma_50'] + 1e-8)

    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Menghitung Relative Strength Index (RSI)."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(high: pd.Series, low: pd.Series,
                 close: pd.Series, period: int = 14) -> pd.Series:
    """Menghitung Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Menghitung On-Balance Volume (OBV)."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ══════════════════════════════════════════════════════════════════════════════
# FITUR TEMPORAL (Known Future Covariates)
# ══════════════════════════════════════════════════════════════════════════════
def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur kalender yang diketahui di masa depan — dapat disediakan
    untuk horizon prediksi tanpa data leak.

    Semua fitur ini dikodekan sebagai sinusoid (sin/cos) untuk
    menangkap periodesitas siklikal.
    """
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)

    df['day_of_week']    = idx.dayofweek          # 0=Senin … 4=Jumat
    df['day_of_month']   = idx.day
    df['month']          = idx.month
    df['quarter']        = idx.quarter
    df['week_of_year']   = idx.isocalendar().week.astype(int)
    df['year']           = idx.year
    df['is_month_start'] = idx.is_month_start.astype(int)
    df['is_month_end']   = idx.is_month_end.astype(int)
    df['is_quarter_end'] = idx.is_quarter_end.astype(int)

    # ── Encoding siklikal (sin/cos) ────────────────────────────────────────
    df['day_sin']   = np.sin(2 * np.pi * df['day_of_week']  / 5)
    df['day_cos']   = np.cos(2 * np.pi * df['day_of_week']  / 5)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1)  / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1)  / 12)
    df['week_sin']  = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos']  = np.cos(2 * np.pi * df['week_of_year'] / 52)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FITUR TURUNAN HARGA
# ══════════════════════════════════════════════════════════════════════════════
def _add_price_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur turunan dari harga untuk memperkaya sinyal prediktif.
    """
    close = df['close']

    # Lag features (harga N hari lalu)
    for lag in [1, 5, 10, 20]:
        df[f'close_lag_{lag}'] = close.shift(lag)

    # Return N hari
    for period in [5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)

    # High-Low spread
    df['hl_spread']     = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['oc_spread']     = (df['close'] - df['open']) / (df['open'] + 1e-8)

    # Log price (lebih stabil untuk training)
    df['log_close']     = np.log(close + 1e-8)
    df['log_volume']    = np.log(df['volume'] + 1) if 'volume' in df.columns else 0

    # Normalized close (z-score rolling 60 hari)
    roll_mean = close.rolling(60).mean()
    roll_std  = close.rolling(60).std()
    df['close_zscore']  = (close - roll_mean) / (roll_std + 1e-8)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: DEFINISI KOLOM PER KATEGORI TFT
# ══════════════════════════════════════════════════════════════════════════════
def get_feature_groups() -> dict:
    """
    Mengembalikan mapping nama kolom ke kategori input TFT.
    Digunakan saat konfigurasi model.
    """
    return {
        # Kolom target yang diprediksi
        "target": "close",

        # Static covariates — tidak berubah terhadap waktu
        "static_categoricals": ["ticker_id"],
        "static_reals":        [],

        # Known future covariates — diketahui untuk horizon mendatang
        "time_varying_known_categoricals": [],
        "time_varying_known_reals": [
            "day_of_week", "day_of_month", "month", "quarter",
            "week_of_year", "year",
            "is_month_start", "is_month_end", "is_quarter_end",
            "day_sin", "day_cos", "month_sin", "month_cos",
            "week_sin", "week_cos",
        ],

        # Observed inputs — hanya tersedia di masa lalu
        "time_varying_unknown_reals": [
            "close", "open", "high", "low",
            "rsi", "macd", "macd_signal", "macd_hist",
            "ma_20", "ma_50", "ema_12", "ema_26",
            "bb_upper", "bb_lower", "bb_width", "bb_pct",
            "atr", "volatility", "daily_return",
            "volume_ma20", "volume_ratio",
            "price_ma20_ratio", "price_ma50_ratio",
            "close_lag_1", "close_lag_5", "close_lag_10", "close_lag_20",
            "return_5d", "return_10d", "return_20d",
            "hl_spread", "oc_spread",
            "close_zscore", "log_close",
        ],
    }
