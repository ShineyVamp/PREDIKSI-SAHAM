import pandas as pd
import logging

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def preprocess_stock_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if len(available) < 4:
        raise ValueError(f"Kolom tidak lengkap. Tersedia: {list(df.columns)}")
    df = df[available]
    df.columns = df.columns.str.lower()

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = _handle_missing_values(df)

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df = df[df[col] > 0]
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df.ffill()
    df = df.dropna()
    return df
