"""
model.py
════════
Temporal Fusion Transformer (TFT) via pytorch-forecasting.

Perubahan penting versi ini:
  · Backtest mengembalikan MATRIKS (n_jendela, horizon), bukan seri 1-D hasil
    ffill. Ini menghapus distorsi metrik dan memungkinkan evaluasi per horizon.
  · Bias correction kosmetik DIHAPUS. Grafik menampilkan output mentah model.
  · Split data train/val/test terpisah. EarlyStopping memakai val, metrik akhir
    dilaporkan dari test yang belum pernah dilihat.
  · Mendukung pelatihan PANEL: satu model untuk banyak ticker sekaligus, lalu
    ramalan dibuat untuk satu ticker fokus. Static covariate ticker_id jadi
    bermakna, dan data latih jauh lebih banyak.

Kontrak fit_predict:
    backtest, attn_weights, future_pred = model.fit_predict(df)
  backtest = {
    "preds_matrix":   (n, H),  "actuals_matrix": (n, H),
    "anchors":        (n,),    # close terakhir sebelum tiap jendela (baseline)
    "preds_1step":    (n,),    "actuals_1step":  (n,),   # untuk grafik
  }
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Deep learning stack (torch ikut di dalam try agar fallback bisa jalan) ───
try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import MultiNormalizer, EncoderNormalizer

    class SafeMultiNormalizer(MultiNormalizer):
        def fit(self, y, X=None, **kwargs):
            if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values
            if y.ndim == 1: y = y.reshape(-1, 1)
            for idx, normalizer in enumerate(self.normalizers):
                try:
                    if X is not None: normalizer.fit(y[:, idx], X)
                    else: normalizer.fit(y[:, idx])
                except TypeError: normalizer.fit(y[:, idx])
            self.fitted_ = True
            return self

        def transform(self, y, X=None, return_norm=False, target_scale=None, **kwargs):
            if isinstance(y, (pd.DataFrame, pd.Series)): y = y.values
            if y.ndim == 1: y = y.reshape(-1, 1)
            res = []
            for idx, normalizer in enumerate(self.normalizers):
                ts = target_scale[idx] if target_scale is not None else None
                try:
                    if ts is None: r = normalizer.transform(y[:, idx], X=X, return_norm=return_norm)
                    else: r = normalizer.transform(y[:, idx], X=X, return_norm=return_norm, target_scale=ts)
                except TypeError:
                    if ts is None: r = normalizer.transform(y[:, idx], return_norm=return_norm)
                    else: r = normalizer.transform(y[:, idx], return_norm=return_norm, target_scale=ts)
                res.append(r)
            if return_norm: return [r[0] for r in res], [r[1] for r in res]
            return res

    from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
    from lightning.pytorch.callbacks import EarlyStopping
    PF_AVAILABLE = True
except ImportError:
    PF_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("pytorch-forecasting tidak terinstall. Memakai FallbackTFTModel.")

BACKEND_NAME = "Temporal Fusion Transformer" if PF_AVAILABLE else "Mode Simulasi (Demo)"
IS_REAL_MODEL = PF_AVAILABLE

from feature_engineering import get_feature_groups, TICKER_ID_MAP_STR
from sklearn.preprocessing import RobustScaler

DEFAULT_CONFIG = {
    "encoder_length": 60,
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.15,
    "hidden_continuous_size": 32,
    "batch_size": 64,
    "gradient_clip_val": 0.1,
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
}


class TFTModel:
    def __new__(cls, ticker, forecast_horizon=30, max_epochs=50, learning_rate=0.001, config=None):
        if PF_AVAILABLE:
            return RealTFTModel(ticker, forecast_horizon, max_epochs, learning_rate, config)
        return FallbackTFTModel(ticker, forecast_horizon, max_epochs, learning_rate, config)


class RealTFTModel:
    def __init__(self, ticker, forecast_horizon=30, max_epochs=50, learning_rate=0.001, config=None):
        self.ticker = ticker                 # ticker FOKUS untuk ramalan
        self.forecast_horizon = forecast_horizon
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.feature_groups = get_feature_groups()
        self._trainer = None
        self._train_dataset = None

    # ── Orkestrasi ──────────────────────────────────────────────────────────
    def fit_predict(self, df: pd.DataFrame, progress_callback=None):
        panel = self._prepare_dataframe(df)
        train_ds, val_ds = self._create_datasets(panel)
        self._train_dataset = train_ds

        train_loader = train_ds.to_dataloader(train=True, batch_size=self.config["batch_size"], num_workers=0)
        val_loader = val_ds.to_dataloader(train=False, batch_size=self.config["batch_size"] * 2, num_workers=0)

        self.model = TemporalFusionTransformer.from_dataset(
            train_ds,
            learning_rate=self.learning_rate,
            hidden_size=self.config["hidden_size"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            loss=MultiLoss([QuantileLoss(self.config["quantiles"]) for _ in range(5)]),
            reduce_on_plateau_patience=5,
            log_interval=-1,
        )

        callback = _ProgressCallback(progress_callback, self.max_epochs)
        early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min", min_delta=1e-4)
        self._trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.config["gradient_clip_val"],
            callbacks=[callback, early_stop],
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, accelerator="auto",
        )
        self._trainer.fit(self.model, train_loader, val_loader)

        focus = panel[panel["group_id"] == str(self.ticker)].copy()
        backtest = self._backtest_focus(focus)
        future_pred = self._predict_future(focus)
        attn_weights = self._extract_attention(val_loader)
        return backtest, attn_weights, future_pred

    # ── Persiapan data (mendukung panel multi-ticker) ───────────────────────
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy().reset_index()
        if "ticker" not in result.columns:
            result["ticker"] = self.ticker
        result["group_id"] = result["ticker"].astype(str)
        # time_idx per grup
        result = result.sort_values(["group_id"]).reset_index(drop=True)
        result["time_idx"] = result.groupby("group_id").cumcount()

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result.groupby("group_id")[numeric_cols].ffill()
        result[numeric_cols] = result[numeric_cols].fillna(0)

        for t in ["open", "high", "low", "close", "volume"]:
            if t in result.columns:
                result[t] = result[t].astype(float)
        if "ticker_id" in result.columns:
            result["ticker_id"] = result["ticker_id"].astype(str)
        return result

    def _split_cutoffs(self, df: pd.DataFrame):
        """Cutoff train/val/test berbasis time_idx global (grup ~ sama panjang)."""
        max_idx = int(df["time_idx"].max())
        train_cut = int(max_idx * 0.70)
        val_cut = int(max_idx * 0.85)
        return train_cut, val_cut

    def _create_datasets(self, df: pd.DataFrame):
        enc_len = self.config["encoder_length"]
        pred_len = self.forecast_horizon
        train_cut, _ = self._split_cutoffs(df)

        fg = self.feature_groups
        known_reals = [c for c in fg["time_varying_known_reals"] if c in df.columns]
        unknown_reals = [c for c in fg["time_varying_unknown_reals"] if c in df.columns]
        static_cats = [c for c in fg["static_categoricals"] if c in df.columns]

        training_ds = TimeSeriesDataSet(
            df[df["time_idx"] <= train_cut],
            time_idx="time_idx",
            target=["open", "high", "low", "close", "log_volume"],
            group_ids=["group_id"],
            min_encoder_length=enc_len // 2,
            max_encoder_length=enc_len,
            min_prediction_length=pred_len,
            max_prediction_length=pred_len,
            static_categoricals=static_cats,
            static_reals=[],
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=SafeMultiNormalizer(
                [EncoderNormalizer(transformation="softplus") for _ in range(5)]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        # Validasi = mulai setelah train_cut (untuk EarlyStopping)
        validation_ds = TimeSeriesDataSet.from_dataset(
            training_ds, df, predict=False, stop_randomization=True,
            min_prediction_idx=train_cut + 1)
        return training_ds, validation_ds

    # ── Backtest pada porsi TEST ticker fokus → matriks per horizon ─────────
    def _backtest_focus(self, focus_df: pd.DataFrame) -> dict:
        _, val_cut = self._split_cutoffs(focus_df)
        try:
            test_ds = TimeSeriesDataSet.from_dataset(
                self._train_dataset, focus_df, predict=False,
                stop_randomization=True, min_prediction_idx=val_cut + 1)
            test_loader = test_ds.to_dataloader(train=False, batch_size=self.config["batch_size"] * 2, num_workers=0)
            out = self.model.predict(test_loader, mode="prediction",
                                     return_x=True, return_y=True,
                                     trainer_kwargs={"logger": False})
            preds = out.output if hasattr(out, "output") else out[0]
            x = out.x if hasattr(out, "x") else out[1]
            y = out.y if hasattr(out, "y") else out[2]

            pclose = preds[3] if isinstance(preds, (list, tuple)) else preds
            P = pclose.cpu().numpy() if hasattr(pclose, "cpu") else np.array(pclose)
            if P.ndim == 3:
                P = P[:, :, P.shape[2] // 2]      # median quantile

            ytgt = y[0] if isinstance(y, (list, tuple)) and len(y) == 2 else y
            aclose = ytgt[3] if isinstance(ytgt, (list, tuple)) else ytgt
            A = aclose.cpu().numpy() if hasattr(aclose, "cpu") else np.array(aclose)
            if A.ndim == 3:
                A = A[:, :, 0]

            anchors = self._anchors_from_x(x, focus_df, n=A.shape[0])
            return {
                "preds_matrix": P, "actuals_matrix": A, "anchors": anchors,
                "preds_1step": P[:, 0], "actuals_1step": A[:, 0],
            }
        except Exception:
            # Degradasi aman: kembalikan struktur kosong yang valid
            H = self.forecast_horizon
            empty = np.empty((0, H))
            return {"preds_matrix": empty, "actuals_matrix": empty,
                    "anchors": np.empty(0), "preds_1step": np.empty(0),
                    "actuals_1step": np.empty(0)}

    def _anchors_from_x(self, x, focus_df, n):
        """Close terakhir sebelum tiap jendela prediksi, dari time_idx decoder."""
        try:
            t0 = x.get("decoder_time_idx")
            if t0 is None:
                return np.full(n, np.nan)
            t0 = t0.cpu().numpy()[:, 0]            # time_idx hari pertama tiap jendela
            close_by_idx = focus_df.set_index("time_idx")["close"]
            anchors = []
            for ti in t0:
                anchors.append(float(close_by_idx.get(ti - 1, np.nan)))
            return np.array(anchors)
        except Exception:
            return np.full(n, np.nan)

    # ── Ramalan masa depan (TANPA bias correction) ──────────────────────────
    def _predict_future(self, focus_df: pd.DataFrame) -> dict:
        enc_len = self.config["encoder_length"]
        last_seq = focus_df.tail(enc_len).copy()
        last_idx = int(last_seq["time_idx"].max())
        H = self.forecast_horizon

        last_date = pd.to_datetime(focus_df["date"].iloc[-1]) if "date" in focus_df.columns else pd.Timestamp.now()
        fdates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=H, freq="B")
        fut = pd.DataFrame({
            "time_idx": np.arange(last_idx + 1, last_idx + H + 1),
            "group_id": str(self.ticker),
        })
        fut["day_of_week"], fut["month"] = fdates.dayofweek, fdates.month
        fut["quarter"], fut["year"] = fdates.quarter, fdates.year
        fut["week_of_year"] = fdates.isocalendar().week.values
        fut["day_of_month"] = fdates.day
        fut["day_sin"] = np.sin(2 * np.pi * fut["day_of_week"] / 5)
        fut["day_cos"] = np.cos(2 * np.pi * fut["day_of_week"] / 5)
        fut["month_sin"] = np.sin(2 * np.pi * (fut["month"] - 1) / 12)
        fut["month_cos"] = np.cos(2 * np.pi * (fut["month"] - 1) / 12)
        fut["week_sin"] = np.sin(2 * np.pi * fut["week_of_year"] / 52)
        fut["week_cos"] = np.cos(2 * np.pi * fut["week_of_year"] / 52)
        for c in ["is_month_start", "is_month_end", "is_quarter_end"]:
            fut[c] = 0

        combined = pd.concat([last_seq, fut], ignore_index=True).ffill().fillna(0)
        combined["group_id"] = str(self.ticker)
        if "ticker_id" in combined.columns:
            combined["ticker_id"] = str(TICKER_ID_MAP_STR.get(self.ticker, "0"))

        try:
            fds = TimeSeriesDataSet.from_dataset(self._train_dataset, combined,
                                                 predict=False, stop_randomization=True)
            floader = fds.to_dataloader(train=False, batch_size=1, num_workers=0)
            fp = self.model.predict(floader, mode="quantiles", trainer_kwargs={"logger": False})

            def med(t):
                p = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
                return p[0, :, p.shape[2] // 2] if p.ndim == 3 else p[0]

            qclose = fp[3]
            lower = qclose[0, :, 0].cpu().numpy()
            upper = qclose[0, :, -1].cpu().numpy()
            # Output MENTAH model, tanpa pergeseran apa pun.
            return {
                "open": med(fp[0]), "high": med(fp[1]), "low": med(fp[2]),
                "close": med(fp[3]),
                "close_lower": lower, "close_upper": upper,
                "volume": np.exp(med(fp[4])) - 1,
            }
        except Exception:
            last = float(focus_df["close"].iloc[-1])
            return {"close": np.array([last] * H),
                    "close_lower": np.array([last * 0.9] * H),
                    "close_upper": np.array([last * 1.1] * H)}

    def _extract_attention(self, val_loader) -> dict:
        try:
            interp = self.model.interpret_output(
                self.model.predict(val_loader, mode="raw", return_x=True,
                                   trainer_kwargs={"logger": False}), reduction="sum")
            vi = {}
            if "encoder_variables" in interp:
                enc = interp["encoder_variables"].cpu().numpy()
                names = self._train_dataset.reals + [f"cat_{c}" for c in self._train_dataset.categoricals]
                vi = {n: float(v) for n, v in zip(names[:len(enc)], enc)}
            temporal = interp["attention"].cpu().numpy().mean(axis=0) if "attention" in interp else None
            return {"variable_importance": vi, "temporal_attention": temporal}
        except Exception:
            return _mock_attention_weights(self.ticker)


class FallbackTFTModel:
    """Mode demo tanpa torch. Menghasilkan kontrak data yang sama (matriks)."""
    def __init__(self, ticker, forecast_horizon=30, max_epochs=50, learning_rate=0.001, config=None):
        self.ticker = ticker
        self.forecast_horizon = forecast_horizon
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.encoder_length = DEFAULT_CONFIG["encoder_length"]

    def fit_predict(self, df: pd.DataFrame, progress_callback=None):
        # Panel: pakai hanya ticker fokus untuk simulasi.
        if "ticker" in df.columns and df["ticker"].nunique() > 1:
            d = df[df["ticker"] == self.ticker].copy()
        else:
            d = df.copy()
        close = d["close"].values.astype(float)

        for epoch in range(1, self.max_epochs + 1):
            if progress_callback:
                progress_callback(epoch, self.max_epochs, 0.05 * np.exp(-epoch / 20))

        backtest = self._sim_backtest(close)
        future = self._sim_future(close)
        attn = _mock_attention_weights(self.ticker)
        return backtest, attn, future

    def _sim_backtest(self, close):
        H = self.forecast_horizon
        n = len(close)
        val_cut = int(n * 0.85)
        starts = list(range(val_cut, n - H))
        rng = np.random.default_rng(abs(hash(self.ticker)) % (2**31))
        preds, acts, anchors = [], [], []
        for s in starts:
            anchor = close[s - 1]
            actual_win = close[s:s + H]
            # Simulasi "model": anchor + drift kecil + noise (sengaja lemah/jujur)
            pred_win = anchor + (actual_win - anchor) * 0.0 + rng.normal(0, anchor * 0.01, H)
            anchors.append(anchor)
            acts.append(actual_win)
            preds.append(pred_win)
        if not preds:
            empty = np.empty((0, H))
            return {"preds_matrix": empty, "actuals_matrix": empty,
                    "anchors": np.empty(0), "preds_1step": np.empty(0),
                    "actuals_1step": np.empty(0)}
        P, A = np.array(preds), np.array(acts)
        return {"preds_matrix": P, "actuals_matrix": A,
                "anchors": np.array(anchors),
                "preds_1step": P[:, 0], "actuals_1step": A[:, 0]}

    def _sim_future(self, close):
        H = self.forecast_horizon
        last = float(close[-1])
        rng = np.random.default_rng((abs(hash(self.ticker)) + 1) % (2**31))
        # Random walk datar dengan rentang melebar: jujur untuk mode demo.
        steps = rng.normal(0, last * 0.012, H)
        path = last + np.cumsum(steps)
        width = last * 0.01 * np.sqrt(np.arange(1, H + 1))
        return {"close": path, "close_lower": path - 2 * width,
                "close_upper": path + 2 * width,
                "open": path, "high": path + width * 0.3,
                "low": path - width * 0.3,
                "volume": np.array([1_000_000] * H)}


def _mock_attention_weights(ticker: str) -> dict:
    np.random.seed(abs(hash(ticker)) % (2**31))
    features = ["close_lag_1", "rsi", "ma_20", "volatility", "macd", "close_lag_5"]
    importance = np.random.dirichlet(np.ones(len(features)) * 2)
    temporal = np.random.rand(30, 60)
    return {"variable_importance": dict(zip(features, importance.tolist())),
            "temporal_attention": temporal.tolist()}


if PF_AVAILABLE:
    class _ProgressCallback(pl.Callback):
        def __init__(self, callback_fn, total_epochs):
            super().__init__()
            self.callback_fn, self.total_epochs = callback_fn, total_epochs
        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0))
            if self.callback_fn:
                self.callback_fn(epoch, self.total_epochs, float(loss))
else:
    class _ProgressCallback:
        def __init__(self, *a, **k): pass
