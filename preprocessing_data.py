
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

def preprocess_stock_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Menerima dataframe mentah dan mengembalikan dataframe bersih."""
    df = raw_df.copy()

    # 1. Flatten MultiIndex columns jika format yfinance versi baru
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # 2. Seleksi kolom penting dan ubah ke lowercase
    available = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if len(available) < 4:
        raise ValueError(f"Kolom tidak lengkap. Tersedia: {list(df.columns)}")
    
    df = df[available]
    df.columns = df.columns.str.lower()

    # 3. Tangani Missing Values (Gap interpolasi, ffill, bfill)
    df = _handle_missing_values(df)

    # 4. Hapus baris dengan harga anomali (0 atau negatif)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df = df[df[col] > 0]

    # 5. Pastikan indeks adalah DatetimeIndex yang rapi dan unik
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]

    return df

def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strategi interpolasi untuk gap kecil, dan fill untuk gap besar."""
    for col in df.columns:
        mask = df[col].isnull()
        if mask.any():
            groups = mask.ne(mask.shift()).cumsum()
            gap_sizes = df[col].isnull().groupby(groups).transform('sum')
            small_gaps = mask & (gap_sizes <= 5)
            df.loc[small_gaps, col] = df[col].interpolate(method='linear')[small_gaps]

    # Forward fill lalu backward fill untuk sisa NaN, terakhir drop jika masih ada
    return df.ffill().bfill().dropna()
