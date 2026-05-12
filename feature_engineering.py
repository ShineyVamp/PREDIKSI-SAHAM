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

TICKER_ID_MAP = {
    "BBCA.JK": 0, "BBRI.JK": 1, "BMRI.JK": 2,
    "BBNI.JK": 3, "BRIS.JK": 4,
}

# Mapping string untuk categorical embedding di model.py
TICKER_ID_MAP_STR = {k: str(v) for k, v in TICKER_ID_MAP.items()}

def build_features(cleaned_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Membangun semua fitur untuk TFT dari dataframe yang sudah bersih."""
    df = cleaned_df.copy()

    # 1. Fitur Teknis, Temporal, & Turunan
    df = _add_technical_features(df)
    df = _add_temporal_features(df)
    df = _add_price_derived_features(df)

    # 2. Static Covariate
    df['ticker_id'] = TICKER_ID_MAP.get(ticker, len(TICKER_ID_MAP))
    df['ticker']    = ticker

    # 3. Format Index untuk PyTorch Forecasting (Time Index Setup)
    df = df.reset_index()
    df.rename(columns={'Date': 'date', 'index': 'date'}, inplace=True)
    if 'date' not in df.columns:
        df['date'] = df.index
        
    date_col = 'date' if 'date' in df.columns else 'Date'
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # 4. Hapus NaN hasil perhitungan indikator (Sanitasi numerik pasca windowing)
    df = df.dropna()

    return df

def _add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df['close'], df['high'], df['low']
    vol = df.get('volume', pd.Series(0, index=df.index))

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MA, EMA, MACD
    df['ma_20'] = close.rolling(window=20).mean()
    df['ma_50'] = close.rolling(window=50).mean()
    df['ema_12'] = close.ewm(span=12, adjust=False).mean()
    df['ema_26'] = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands (Epsilon 1e-8 anti-div-by-zero)
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['ma_20'] + 2 * bb_std
    df['bb_lower'] = df['ma_20'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['ma_20'] + 1e-8)
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # ATR & Volatilitas
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    daily_return = close.pct_change()
    df['volatility'] = daily_return.rolling(20).std()
    df['daily_return'] = daily_return

    # Volume Ratio (Normalisasi volume relatif)
    if 'volume' in df.columns:
        df['volume_ma20'] = vol.rolling(20).mean()
        df['volume_ratio'] = vol / (df['volume_ma20'] + 1e-8)
    df['price_ma20_ratio'] = close / (df['ma_20'] + 1e-8)
    df['price_ma50_ratio'] = close / (df['ma_50'] + 1e-8)

    return df

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic encoding sin/cos untuk pemahaman periodisitas."""
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
    df['day_of_week'] = idx.dayofweek
    df['day_of_month'] = idx.day
    df['month'] = idx.month
    df['quarter'] = idx.quarter
    df['week_of_year'] = idx.isocalendar().week.astype(int)
    df['year'] = idx.year
    df['is_month_start'] = idx.is_month_start.astype(int)
    df['is_month_end'] = idx.is_month_end.astype(int)
    df['is_quarter_end'] = idx.is_quarter_end.astype(int)

    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    return df

def _add_price_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df['close']
    # Lag features
    for lag in [1, 5, 10, 20]:
        df[f'close_lag_{lag}'] = close.shift(lag)
    # Return features
    for period in [5, 10, 20]:
        df[f'return_{period}d'] = close.pct_change(period)

    df['hl_spread'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['oc_spread'] = (df['close'] - df['open']) / (df['open'] + 1e-8)
    
    # Transformasi Log (Stabilisasi skala)
    df['log_close'] = np.log(close + 1e-8)
    df['log_volume'] = np.log(df['volume'] + 1) if 'volume' in df.columns else 0

    # Normalisasi Z-score bergulir 60 hari
    roll_mean = close.rolling(60).mean()
    roll_std = close.rolling(60).std()
    df['close_zscore'] = (close - roll_mean) / (roll_std + 1e-8)
    return df

def get_feature_groups() -> dict:
    """Definisi kolompok fitur untuk konfigurasi TimeSeriesDataSet TFT."""
    return {
        "target": ["open", "high", "low", "close", "log_volume"],
        "static_categoricals": ["ticker_id"],
        "static_reals": [],
        "time_varying_known_categoricals": [],
        "time_varying_known_reals": [
            "day_of_week", "day_of_month", "month", "quarter",
            "week_of_year", "year", "is_month_start", "is_month_end", "is_quarter_end",
            "day_sin", "day_cos", "month_sin", "month_cos", "week_sin", "week_cos",
        ],
        "time_varying_unknown_reals": [
            "close", "open", "high", "low", "rsi", "macd", "macd_signal", "macd_hist",
            "ma_20", "ma_50", "ema_12", "ema_26", "bb_upper", "bb_lower", "bb_width", "bb_pct",
            "atr", "volatility", "daily_return", "volume_ma20", "volume_ratio",
            "price_ma20_ratio", "price_ma50_ratio", "close_lag_1", "close_lag_5", "close_lag_10", "close_lag_20",
            "return_5d", "return_10d", "return_20d", "hl_spread", "oc_spread",
            "close_zscore", "log_close",
        ],
    }
