import numpy as np
import pandas as pd

EPS = 1e-8

TICKER_ID_MAP = {
    "BBCA.JK": 0, "BBRI.JK": 1, "BMRI.JK": 2,
    "BBNI.JK": 3, "BRIS.JK": 4,
}
TICKER_ID_MAP_STR = {k: str(v) for k, v in TICKER_ID_MAP.items()}


def build_features(cleaned_df: pd.DataFrame, ticker: str,
                   market_df: pd.DataFrame = None) -> pd.DataFrame:
    df = cleaned_df.copy()

    df["log_close"] = np.log(df["close"])
    df["target"] = df["log_close"].diff()

    df = _add_trend_momentum(df)
    df = _add_volatility_range(df)
    df = _add_volume(df)
    df = _add_temporal(df)
    df = _add_lags(df)
    df = _add_market_context(df, market_df)

    df["ticker_id"] = TICKER_ID_MAP.get(ticker, len(TICKER_ID_MAP))
    df["ticker"] = ticker

    df = df.reset_index()
    rename_map = {c: "date" for c in ("Date", "index", "Datetime") if c in df.columns}
    df = df.rename(columns=rename_map)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    df = df.dropna()
    return df


def _add_market_context(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan log-return IHSG & USD/IDR (sejajar tanggal). Bila tak ada, isi 0.
    Past-only: nilai masa depan tak diketahui, jadi hanya menambah konteks encoder."""
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
    if market_df is not None and not market_df.empty:
        m = market_df.copy()
        m.index = pd.to_datetime(m.index)
        m = m[~m.index.duplicated(keep="last")].sort_index()
        aligned = m.reindex(idx).ffill()     
        for col in MARKET_COLS:
            df[col] = aligned[col].to_numpy() if col in aligned.columns else 0.0
    else:
        for col in MARKET_COLS:
            df[col] = 0.0
    return df


def _add_trend_momentum(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / (avg_loss + EPS)
    df["rsi"] = (100 - 100 / (1 + rs)) / 100.0

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["ma_20"] = ma20        
    df["ma_50"] = ma50
    df["price_ma20_gap"] = close / (ma20 + EPS) - 1.0
    df["ma_trend"] = ma20 / (ma50 + EPS) - 1.0 

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_norm"] = (ema12 - ema26) / (close + EPS)

    df["roc_10"] = close / (close.shift(10) + EPS) - 1.0
    return df


def _add_volatility_range(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]

    df["vol_20"] = df["target"].rolling(20).std()

    ma20 = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    df["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + EPS)

    df["hl_range"] = (high - low) / (close + EPS)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_norm"] = tr.rolling(14).mean() / (close + EPS)
    return df


def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        v_mean = vol.rolling(20).mean()
        v_std = vol.rolling(20).std()
        df["volume_z"] = (vol - v_mean) / (v_std + EPS) 
    else:
        df["volume_z"] = 0.0
    return df


def calendar_features(index) -> dict:
    idx = pd.DatetimeIndex(index)
    dow = idx.dayofweek.to_numpy().astype(float)
    month = idx.month.to_numpy().astype(float)
    dom = idx.day.to_numpy().astype(float)
    woy = idx.isocalendar().week.to_numpy().astype(float)
    return {
        "day_sin": np.sin(2 * np.pi * dow / 5),
        "day_cos": np.cos(2 * np.pi * dow / 5),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "dom_sin": np.sin(2 * np.pi * (dom - 1) / 31),
        "dom_cos": np.cos(2 * np.pi * (dom - 1) / 31),
        "woy_sin": np.sin(2 * np.pi * (woy - 1) / 52),
        "woy_cos": np.cos(2 * np.pi * (woy - 1) / 52),
        "is_month_end": idx.is_month_end.astype(float),
        "is_quarter_end": idx.is_quarter_end.astype(float),
    }


CALENDAR_COLS = [
    "day_sin", "day_cos", "month_sin", "month_cos",
    "dom_sin", "dom_cos", "woy_sin", "woy_cos",
    "is_month_end", "is_quarter_end",
]

MARKET_COLS = ["ihsg_ret", "usdidr_ret"]


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
    cf = calendar_features(idx)
    for col, values in cf.items():
        df[col] = values
    return df


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    for lag in (1, 2, 3, 5):
        df[f"ret_lag_{lag}"] = df["target"].shift(lag)
    return df


def get_feature_groups() -> dict:
    return {
        "target": ["target"],
        "static_categoricals": ["ticker_id"],
        "time_varying_known_reals": list(CALENDAR_COLS),
        "time_varying_unknown_reals": [
            "target",
            "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
            "rsi", "price_ma20_gap", "ma_trend", "macd_norm", "roc_10",
            "vol_20", "bb_pct", "hl_range", "atr_norm", "volume_z",
            "ihsg_ret", "usdidr_ret",
        ],
    }
