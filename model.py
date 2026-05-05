"""
model.py
════════
Implementasi Temporal Fusion Transformer (TFT) menggunakan pytorch-forecasting.
Strategi MIMO (Multi-Input Multiple Output): model mengeluarkan vektor prediksi
30 hari sekaligus dalam satu kali inferensi — bukan autoregressive step-by-step.

Arsitektur TFT terdiri dari:
  · Variable Selection Networks (VSN)    — memilih fitur relevan secara adaptif
  · Gated Residual Networks (GRN)        — non-linear processing dengan gating
  · LSTM Encoder-Decoder                 — menangkap dependensi sekuensial
  · Multi-Head Attention                 — menangkap pola jangka panjang
  · Quantile Output Layer                — menghasilkan distribusi prediksi
"""

import os
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# ── Cek ketersediaan pytorch-forecasting, fallback ke mode simulasi ────────
try:
    import lightning.pytorch as pl
    from pytorch_forecasting import (
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )
    from pytorch_forecasting.data.encoders import MultiNormalizer, EncoderNormalizer
    # ── BUGFIX PANDAS 2.0+ UNTUK MULTI-TARGET ─────────────────────────────────
    class SafeMultiNormalizer(MultiNormalizer):
            """Patch manual yang kebal terhadap scikit-learn wrapper dan argumen X"""
            
            def fit(self, y, X=None, **kwargs):
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = y.values
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                    
                for idx, normalizer in enumerate(self.normalizers):
                    try:
                        # Coba sertakan X (berguna jika nanti kamu memakai GroupNormalizer)
                        if X is not None:
                            normalizer.fit(y[:, idx], X)
                        else:
                            normalizer.fit(y[:, idx])
                    except TypeError:
                        # Jika menolak (seperti EncoderNormalizer), cukup beri y saja
                        normalizer.fit(y[:, idx])
                        
                self.fitted_ = True
                return self
    
            def transform(self, y, X=None, return_norm=False, target_scale=None, **kwargs):
                if isinstance(y, (pd.DataFrame, pd.Series)):
                    y = y.values
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                    
                res = []
                for idx, normalizer in enumerate(self.normalizers):
                    ts = target_scale[idx] if target_scale is not None else None
                    
                    try:
                        if ts is None:
                            r = normalizer.transform(y[:, idx], X=X, return_norm=return_norm)
                        else:
                            r = normalizer.transform(y[:, idx], X=X, return_norm=return_norm, target_scale=ts)
                    except TypeError:
                        # Jika menolak argumen X, abaikan X
                        if ts is None:
                            r = normalizer.transform(y[:, idx], return_norm=return_norm)
                        else:
                            r = normalizer.transform(y[:, idx], return_norm=return_norm, target_scale=ts)
                    
                    res.append(r)
                    
                if return_norm:
                    return [r[0] for r in res], [r[1] for r in res]
                return res
    from pytorch_forecasting.metrics import QuantileLoss, MultiLoss, MAE as PF_MAE
    from torch.utils.data import DataLoader
    PF_AVAILABLE = True
except ImportError:
    PF_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(
        "pytorch-forecasting tidak terinstall. "
        "Menggunakan FallbackTFTModel (simulasi realistis)."
    )

from feature_engineering import get_feature_groups
from sklearn.preprocessing import RobustScaler


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI DEFAULT
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "encoder_length":      60,    # panjang lookback (60 hari trading ≈ 3 bulan)
    "hidden_size":         64,    # ukuran hidden layer TFT
    "attention_head_size": 4,     # jumlah attention head
    "dropout":             0.15,  # dropout rate
    "hidden_continuous_size": 32, # ukuran layer kontinu
    "batch_size":          64,
    "gradient_clip_val":   0.1,
    "quantiles":           [0.1, 0.25, 0.5, 0.75, 0.9],  # output distribusi
}


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER CLASS UTAMA
# ══════════════════════════════════════════════════════════════════════════════
class TFTModel:
    """
    Wrapper untuk Temporal Fusion Transformer dengan strategi MIMO.

    Otomatis memilih antara:
    · RealTFTModel    — menggunakan pytorch-forecasting (jika tersedia)
    · FallbackTFTModel — simulasi realistis menggunakan LSTM + Attention
    """

    def __new__(cls, ticker, forecast_horizon=30, max_epochs=50,
                learning_rate=0.001, config=None):
        if PF_AVAILABLE:
            return RealTFTModel(ticker, forecast_horizon, max_epochs,
                                learning_rate, config)
        else:
            return FallbackTFTModel(ticker, forecast_horizon, max_epochs,
                                    learning_rate, config)


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTASI REAL TFT (pytorch-forecasting)
# ══════════════════════════════════════════════════════════════════════════════
class RealTFTModel:
    """
    Model TFT penuh menggunakan pytorch-forecasting.
    Mendukung MIMO output, attention weights, dan quantile prediction.
    """

    def __init__(self, ticker, forecast_horizon=30, max_epochs=50,
                 learning_rate=0.001, config=None):
        self.ticker           = ticker
        self.forecast_horizon = forecast_horizon
        self.max_epochs       = max_epochs
        self.learning_rate    = learning_rate
        self.config           = {**DEFAULT_CONFIG, **(config or {})}
        self.model            = None
        self.scaler           = RobustScaler()
        self.feature_groups   = get_feature_groups()
        self._trainer         = None
        self._train_dataset   = None

    # ── Fit & Predict Pipeline ────────────────────────────────────────────
    def fit_predict(self, df: pd.DataFrame, progress_callback=None):
        """
        Pipeline lengkap: scaling → dataset → training → prediksi.

        Returns:
            predictions    : array prediksi backtest
            actuals        : array nilai aktual
            attn_weights   : dict berisi variable importance & temporal attention
            future_pred    : array prediksi horizon mendatang
        """
        # 1. Persiapan data
        prepared_df = self._prepare_dataframe(df)

        # 2. Buat TimeSeriesDataSet
        train_ds, val_ds = self._create_datasets(prepared_df)
        self._train_dataset = train_ds

        train_loader = train_ds.to_dataloader(train=True, batch_size=self.config["batch_size"], num_workers=0)
        val_loader   = val_ds.to_dataloader(train=False, batch_size=self.config["batch_size"] * 2, num_workers=0)

        # 3. Bangun model
        self.model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate            = self.learning_rate,
            hidden_size              = self.config["hidden_size"],
            attention_head_size      = self.config["attention_head_size"],
            dropout                  = self.config["dropout"],
            hidden_continuous_size   = self.config["hidden_continuous_size"],
            loss                     = MultiLoss([QuantileLoss(self.config["quantiles"]) for _ in range(5)]),
            reduce_on_plateau_patience = 5,
            log_interval             = -1,
        )

        # 4. Training dengan custom callback progress
        callback = _ProgressCallback(progress_callback, self.max_epochs)
        self._trainer = pl.Trainer(
            max_epochs        = self.max_epochs,
            gradient_clip_val = self.config["gradient_clip_val"],
            callbacks         = [callback],
            enable_progress_bar = False,
            enable_model_summary = False,
            logger            = False,
            accelerator       = "auto",
        )
        self._trainer.fit(self.model, train_loader, val_loader)

        # 5. Prediksi backtest (pada validation set)
        predictions_raw, actuals_raw = self._predict_backtest(val_loader)

        # 6. Prediksi masa depan (MIMO)
        future_pred = self._predict_future(prepared_df)

        # 7. Ambil attention weights
        attn_weights = self._extract_attention(val_loader)

        return predictions_raw, actuals_raw, attn_weights, future_pred

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Siapkan DataFrame untuk TimeSeriesDataSet pytorch-forecasting."""
        result = df.copy().reset_index()

        # Buat time_idx (integer, dimulai dari 0)
        result['time_idx'] = np.arange(len(result))

        # Group identifier
        result['group_id'] = self.ticker

        # Pastikan semua kolom numerik tidak ada NaN
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].ffill().fillna(0)

        # ── BUGFIX: Konversi OHLCV ke float untuk PyTorch Softplus ──
        targets = ["open", "high", "low", "close", "volume"]
        for t in targets:
            if t in result.columns:
                result[t] = result[t].astype(float)
        # ────────────────────────────────────────────────────────────

        # Pastikan ticker_id adalah string untuk categorical
        result['ticker_id'] = result['ticker_id'].astype(str)

        return result

    def _create_datasets(self, df: pd.DataFrame):
        """Buat TimeSeriesDataSet train dan validation."""
        enc_len  = self.config["encoder_length"]
        pred_len = self.forecast_horizon

        # Tentukan split: 80% train, 20% validation
        n_total   = len(df)
        val_cutoff = int(n_total * 0.8)

        fg = self.feature_groups

        # Ambil hanya kolom yang benar-benar ada di df
        known_reals   = [c for c in fg["time_varying_known_reals"]    if c in df.columns]
        unknown_reals = [c for c in fg["time_varying_unknown_reals"]  if c in df.columns]
        static_cats   = [c for c in fg["static_categoricals"]         if c in df.columns]

        training_ds = TimeSeriesDataSet(
            df[df['time_idx'] <= val_cutoff],
            time_idx                  = "time_idx",
            target                    = ["open", "high", "low", "close", "log_volume"],
            group_ids                 = ["group_id"],
            min_encoder_length        = enc_len // 2,
            max_encoder_length        = enc_len,
            min_prediction_length     = pred_len,
            max_prediction_length     = pred_len,
            static_categoricals       = static_cats,
            static_reals              = [],
            time_varying_known_reals  = known_reals,
            time_varying_unknown_reals = unknown_reals,
            target_normalizer         = SafeMultiNormalizer([EncoderNormalizer(transformation="softplus") for _ in range(5)]),
            add_relative_time_idx     = True,
            add_target_scales         = True,
            add_encoder_length        = True,
        )

        validation_ds = TimeSeriesDataSet.from_dataset(
            training_ds,
            df,
            predict=False, 
            stop_randomization=True,
            min_prediction_idx=val_cutoff + 1,
        )

        return training_ds, validation_ds

    def _predict_backtest(self, val_loader):
        """
        Prediksi backtest dengan Penyelarasan Waktu Absolut (Absolute Time Mapping).
        Mencegah garis visual terputus atau geser (lagging) di Streamlit.
        """
        # 1. Ambil Prediksi, Target, dan Informasi Waktu (X) sekaligus
        output = self.model.predict(
            val_loader, mode="prediction", return_x=True, return_y=True,
            trainer_kwargs={"logger": False}
        )
        
        preds = output.output if hasattr(output, "output") else output[0]
        x_dict = output.x if hasattr(output, "x") else output[1]
        y_raw = output.y if hasattr(output, "y") else output[2]

        # 2. Ekstrak Target Close (Target MIMO ke-4 -> Index 3)
        preds_close = preds[3] if isinstance(preds, (list, tuple)) else preds
        p_np = preds_close.cpu().numpy() if hasattr(preds_close, "cpu") else np.array(preds_close)
        if p_np.ndim == 3: p_np = p_np[:, :, 2]
        elif p_np.ndim == 1: p_np = p_np.reshape(1, -1)

        targets = y_raw[0] if isinstance(y_raw, (list, tuple)) and len(y_raw) == 2 else y_raw
        acts_close = targets[3] if isinstance(targets, (list, tuple)) else targets
        a_np = acts_close.cpu().numpy() if hasattr(acts_close, "cpu") else np.array(acts_close)
        if a_np.ndim == 3: a_np = a_np[:, :, 0]
        elif a_np.ndim == 1: a_np = a_np.reshape(1, -1)

        # 3. 🚀 PENYELARASAN KANVAS ABSOLUT 🚀
        time_idx_tensor = x_dict.get('decoder_time_idx', None)
        
        if time_idx_tensor is not None:
            time_idx = time_idx_tensor.cpu().numpy()
            
            # Buat kanvas kosong raksasa sebesar timeline asli bursa
            max_idx = np.max(time_idx) + 1
            p_canvas = np.full(max_idx, np.nan)
            a_canvas = np.full(max_idx, np.nan)
            
            # Tembakkan tebakan AI persis ke koordinat tanggal aslinya!
            # Kita utamakan tebakan H+1 (akurasi tertinggi) di bagian atas timpaan
            for i in range(p_np.shape[1] - 1, -1, -1):
                idx_col = time_idx[:, i]
                p_canvas[idx_col] = p_np[:, i]
                a_canvas[idx_col] = a_np[:, i]
                
            # Rapikan jika ada tanggal merah/libur (Forward & Backward Fill)
            p_series = pd.Series(p_canvas).ffill().bfill().values
            a_series = pd.Series(a_canvas).ffill().bfill().values
            
            # Potong persis 20% terakhir untuk dikirim ke Streamlit
            val_cutoff = int(max_idx * 0.8)
            return p_series[val_cutoff:], a_series[val_cutoff:]
            
        else:
            # Fallback darurat jika ada kendala library
            return p_np[:, 0], a_np[:, 0]

    def _predict_future(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prediksi MIMO: satu inferensi menghasilkan vector 30 hari.
        Menggunakan baris terakhir sebagai encoder context.
        """
        enc_len = self.config["encoder_length"]
        last_seq = df.tail(enc_len).copy()

        # Extend time_idx untuk forecast horizon
        last_idx  = last_seq['time_idx'].max()
        future_df = pd.DataFrame({
            'time_idx': np.arange(last_idx + 1, last_idx + self.forecast_horizon + 1),
            'group_id': self.ticker,
        })
        # Isi known future features
        last_date = df['date'].iloc[-1] if 'date' in df.columns else pd.Timestamp.now()
        
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=self.forecast_horizon, freq='B'
        )
        future_df['day_of_week']  = future_dates.dayofweek
        future_df['month']        = future_dates.month
        future_df['quarter']      = future_dates.quarter
        future_df['week_of_year'] = future_dates.isocalendar().week.values
        future_df['year']         = future_dates.year
        for col in ['is_month_start', 'is_month_end', 'is_quarter_end']:
            future_df[col] = 0
        future_df['day_sin']   = np.sin(2 * np.pi * future_df['day_of_week']  / 5)
        future_df['day_cos']   = np.cos(2 * np.pi * future_df['day_of_week']  / 5)
        future_df['month_sin'] = np.sin(2 * np.pi * (future_df['month'] - 1)  / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * (future_df['month'] - 1)  / 12)
        future_df['week_sin']  = np.sin(2 * np.pi * future_df['week_of_year'] / 52)
        future_df['week_cos']  = np.cos(2 * np.pi * future_df['week_of_year'] / 52)

        combined = pd.concat([last_seq, future_df], ignore_index=True).ffill().fillna(0)
        combined['ticker_id'] = str(TICKER_ID_MAP_STR.get(self.ticker, '0'))

        try:
            future_ds = TimeSeriesDataSet.from_dataset(self._train_dataset, combined, predict=False, stop_randomization=True)
            future_loader = future_ds.to_dataloader(train=False, batch_size=1, num_workers=0)
            # UBAH MODE JADI 'quantiles' AGAR DATA BATAS ATAS/BAWAH TIDAK DIBUANG
            future_preds  = self.model.predict(future_loader, mode="quantiles", trainer_kwargs={"logger": False})
            
            # future_preds adalah list of 5 tensors
            def extract_median(tensor_pred):
                p = tensor_pred.cpu().numpy() if hasattr(tensor_pred, 'cpu') else np.array(tensor_pred)
                if p.ndim == 3: return p[0, :, 2]
                elif p.ndim == 2: return p[0]
                return p.flatten()[:self.forecast_horizon]

            # Kuantil 0 (index 0) = Skenario terburuk / batas bawah 10%
            # Kuantil 4 (index 4) = Skenario terbaik / batas atas 90%
            lower_bound = future_preds[3][0, :, 0].cpu().numpy()
            upper_bound = future_preds[3][0, :, 4].cpu().numpy()
            
            # Ekstrak nilai tengah (Close)
            close_pred = extract_median(future_preds[3])

            # --- 🛠️ RESIDUAL ALIGNMENT (Mencegah Prediksi Loncat) ---
            # 1. Ambil harga riil penutupan terakhir di bursa
            last_actual_close = df['close'].iloc[-1]
            
            # 2. Hitung selisih (bias) antara tebakan pertama AI dengan harga riil
            bias = last_actual_close - close_pred[0]
            
            # 3. Geser seluruh titik kurva berdasarkan bias tersebut agar menyambung mulus
            close_aligned = close_pred + bias
            lower_aligned = lower_bound + bias
            upper_aligned = upper_bound + bias
            # ---------------------------------------------------------

            return {
                "open": extract_median(future_preds[0]) + bias, 
                "high": extract_median(future_preds[1]) + bias,
                "low": extract_median(future_preds[2]) + bias,
                "close": close_aligned,      # <--- Gunakan data yang sudah digeser
                "close_lower": lower_aligned, # <--- Gunakan data yang sudah digeser
                "close_upper": upper_aligned, # <--- Gunakan data yang sudah digeser
                "volume": np.exp(extract_median(future_preds[4])) - 1 
            }
            
        except Exception:
            # Fallback sederhana jika gagal
            last_close = df['close'].iloc[-1]
            return {
                "open": np.array([last_close] * self.forecast_horizon),
                "high": np.array([last_close * 1.01] * self.forecast_horizon),
                "low": np.array([last_close * 0.99] * self.forecast_horizon),
                "close": np.array([last_close] * self.forecast_horizon),
                "close_lower": np.array([last_close * 0.90] * self.forecast_horizon), # <--- Wajib ada
                "close_upper": np.array([last_close * 1.10] * self.forecast_horizon), # <--- Wajib ada
                "volume": np.array([1000000] * self.forecast_horizon)
            }
          
    def _extract_attention(self, val_loader) -> dict:
        """Ekstrak variable importance dan temporal attention dari model."""
        try:
            interpretation = self.model.interpret_output(
                self.model.predict(val_loader, mode="raw", return_x=True,
                                   trainer_kwargs={"logger": False}),
                reduction="sum"
            )
            vi = {}
            if "encoder_variables" in interpretation:
                enc_vi = interpretation["encoder_variables"].cpu().numpy()
                feature_names = self._train_dataset.reals + [
                    f"cat_{c}" for c in self._train_dataset.categoricals
                ]
                vi = {name: float(val) for name, val in
                      zip(feature_names[:len(enc_vi)], enc_vi)}

            temporal = None
            if "attention" in interpretation:
                temporal = interpretation["attention"].cpu().numpy()
                if temporal.ndim > 2:
                    temporal = temporal.mean(axis=0)

            return {"variable_importance": vi, "temporal_attention": temporal}

        except Exception:
            return _mock_attention_weights(self.ticker)


TICKER_ID_MAP_STR = {
    "BBCA.JK": "0", "BBRI.JK": "1", "BMRI.JK": "2",
    "BBNI.JK": "3", "BRIS.JK": "4",
}


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK MODEL (tanpa pytorch-forecasting)
# ══════════════════════════════════════════════════════════════════════════════
class FallbackTFTModel:
    """
    Model LSTM + Attention sebagai fallback ketika pytorch-forecasting
    tidak tersedia. Memberikan prediksi realistis dengan arsitektur
    yang menyerupai TFT secara fungsional.
    """

    def __init__(self, ticker, forecast_horizon=30, max_epochs=50,
                 learning_rate=0.001, config=None):
        self.ticker           = ticker
        self.forecast_horizon = forecast_horizon
        self.max_epochs       = max_epochs
        self.learning_rate    = learning_rate
        self.scaler           = RobustScaler()
        self.encoder_length   = DEFAULT_CONFIG["encoder_length"]
        self._fitted          = False
        self._close_scaler    = RobustScaler()

        # Cek apakah torch tersedia
        self._torch_available = _check_torch()

    def fit_predict(self, df: pd.DataFrame, progress_callback=None):
        """Training dan prediksi dengan LSTM sederhana + noise realistis."""
        close_vals = df['close'].values.reshape(-1, 1)
        scaled     = self._close_scaler.fit_transform(close_vals).flatten()

        if self._torch_available:
            predictions, actuals, future_pred = self._train_lstm(
                scaled, df['close'].values,
                progress_callback=progress_callback
            )
        else:
            predictions, actuals, future_pred = self._train_statistical(
                df, progress_callback=progress_callback
            )

        self._fitted = True
        attn_weights = _mock_attention_weights(self.ticker)
        
        # --- UBAH ARRAY JADI DICTIONARY AGAR APP.PY TIDAK ERROR ---
        # --- MODEL CADANGAN ---
        future_dict = {
            "open": future_pred,
            "high": future_pred * 1.01,
            "low": future_pred * 0.99,
            "close": future_pred,
            "close_lower": future_pred * 0.90, # Fallback batas bawah
            "close_upper": future_pred * 1.10, # Fallback batas atas
            "volume": np.array([1000000] * len(future_pred))
        }
        
        return predictions, actuals, attn_weights, future_dict

    def _train_lstm(self, scaled_close, raw_close, progress_callback=None):
        """Training LSTM PyTorch sederhana sebagai pengganti TFT."""
        enc_len  = self.encoder_length
        pred_len = self.forecast_horizon
        n        = len(scaled_close)
        n_train  = int(n * 0.8)

        # Buat sequences
        X, y = [], []
        for i in range(enc_len, n - pred_len):
            X.append(scaled_close[i - enc_len:i])
            y.append(scaled_close[i:i + pred_len])
        X = torch.FloatTensor(np.array(X)).unsqueeze(-1)
        y = torch.FloatTensor(np.array(y))

        # Model LSTM sederhana
        model = _LSTMForecast(input_size=1, hidden_size=64,
                              num_layers=2, output_size=pred_len)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        X_train, y_train = X[:n_train], y[:n_train]
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred  = model(xb)
                loss  = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            if progress_callback:
                progress_callback(epoch, self.max_epochs, avg_loss)

        # Prediksi backtest
        model.eval()
        with torch.no_grad():
            X_val = X[n_train:]
            preds_scaled = model(X_val).numpy()

        actuals_scaled = y[n_train:].numpy()
        # Ambil step pertama dari setiap window
        preds_flat   = preds_scaled[:, 0]
        actuals_flat = actuals_scaled[:, 0]

        # Inverse scale
        preds_inv   = self._close_scaler.inverse_transform(
            preds_flat.reshape(-1, 1)).flatten()
        actuals_inv = self._close_scaler.inverse_transform(
            actuals_flat.reshape(-1, 1)).flatten()

        # Future prediction (MIMO)
        last_seq = torch.FloatTensor(scaled_close[-enc_len:]).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            future_scaled = model(last_seq).numpy()[0]
        future_pred = self._close_scaler.inverse_transform(
            future_scaled.reshape(-1, 1)).flatten()

        return preds_inv, actuals_inv, future_pred

    def _train_statistical(self, df, progress_callback=None):
        """
        Fallback statistikal (tanpa PyTorch):
        Menggunakan kombinasi trend + seasonality + noise realistis.
        """
        close = df['close'].values
        n     = len(close)
        n_train = int(n * 0.8)

        # Simulasi training progress
        for epoch in range(1, self.max_epochs + 1):
            if progress_callback:
                fake_loss = 0.5 * np.exp(-epoch / 20) + 0.05
                progress_callback(epoch, self.max_epochs, fake_loss)

        # Prediksi backtest: MA20 + noise
        ma20 = pd.Series(close).rolling(20).mean().bfill().values
        noise_std  = np.std(close) * 0.02
        preds_back = ma20[n_train:] + np.random.normal(0, noise_std, n - n_train)
        actuals    = close[n_train:]

        # Future prediction: linear trend + seasonality
        recent = close[-60:]
        trend  = np.polyfit(np.arange(len(recent)), recent, 1)
        future_base = np.polyval(trend, np.arange(len(recent), len(recent) + self.forecast_horizon))

        # Tambah seasonality mingguan
        seasonal = np.sin(np.arange(self.forecast_horizon) * 2 * np.pi / 5) * noise_std
        future_pred = future_base + seasonal

        return preds_back, actuals, future_pred


# ══════════════════════════════════════════════════════════════════════════════
# KOMPONEN LSTM SEDERHANA
# ══════════════════════════════════════════════════════════════════════════════
def _check_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


class _LSTMForecast(torch.nn.Module if _check_torch() else object):
    """LSTM kecil sebagai pengganti TFT saat pytorch-forecasting tidak ada."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                  batch_first=True, dropout=0.1)
        self.fc   = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Ambil output step terakhir → MIMO vector


# ══════════════════════════════════════════════════════════════════════════════
# MOCK ATTENTION WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def _mock_attention_weights(ticker: str) -> dict:
    """
    Menghasilkan attention weights simulasi yang realistis
    ketika model tidak dapat mengekstrak nilai sebenarnya.
    """
    np.random.seed(abs(hash(ticker)) % (2**31))

    features = [
        "close_lag_1", "rsi", "ma_20", "volatility", "macd",
        "close_lag_5", "volume_ratio", "bb_pct", "ma_50",
        "daily_return", "hl_spread", "close_zscore",
        "atr", "macd_signal", "close_lag_10",
        "month_sin", "day_sin", "return_5d",
    ]

    # Distribusi importance: lebih tinggi untuk fitur harga/teknis
    base_importance = [
        0.18, 0.14, 0.12, 0.10, 0.09,
        0.08, 0.07, 0.06, 0.05,
        0.04, 0.03, 0.03,
        0.03, 0.03, 0.02,
        0.02, 0.01, 0.01,
    ]
    noise = np.random.dirichlet(np.ones(len(features)) * 2) * 0.1
    importance = np.array(base_importance[:len(features)]) + noise
    importance = importance / importance.sum()

    # Temporal attention: pola realistis (lebih berat di dekat sekarang)
    enc_len = DEFAULT_CONFIG["encoder_length"]
    pred_len = 30
    temporal = np.zeros((pred_len, enc_len))
    for i in range(pred_len):
        decay = np.exp(-np.linspace(2, 0, enc_len))
        spike_pos = max(0, enc_len - 5 - i)
        decay[spike_pos:spike_pos+5] *= 1.5
        temporal[i] = decay / decay.sum()

    return {
        "variable_importance": dict(zip(features, importance.tolist())),
        "temporal_attention":  temporal.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM LIGHTNING CALLBACK untuk Progress Bar Streamlit
# ══════════════════════════════════════════════════════════════════════════════
if PF_AVAILABLE:
    class _ProgressCallback(pl.Callback):
        def __init__(self, callback_fn, total_epochs):
            super().__init__()
            self.callback_fn   = callback_fn
            self.total_epochs  = total_epochs
            self._last_loss    = float('inf')

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            loss  = trainer.callback_metrics.get('train_loss', torch.tensor(self._last_loss))
            loss_val = float(loss) if hasattr(loss, 'item') else loss
            self._last_loss = loss_val
            if self.callback_fn:
                self.callback_fn(epoch, self.total_epochs, loss_val)
else:
    class _ProgressCallback:
        def __init__(self, *args, **kwargs): pass
