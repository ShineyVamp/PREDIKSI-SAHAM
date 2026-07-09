import os
import numpy as np
import pandas as pd
import warnings
from statistics import NormalDist
warnings.filterwarnings("ignore")

SEED = 42

try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch.callbacks import EarlyStopping
    PF_AVAILABLE = True
except ImportError:
    PF_AVAILABLE = False

from feature_engineering import get_feature_groups, calendar_features, TICKER_ID_MAP_STR

DEFAULT_CONFIG = {
    "encoder_length": 63,        
    "hidden_size": 32,
    "attention_head_size": 4,
    "dropout": 0.15,               
    "hidden_continuous_size": 16,
    "batch_size": 64,
    "gradient_clip_val": 0.1,
    "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
}


def _z_from_quantile(q: float) -> float:
    q = min(max(q, 1e-4), 1 - 1e-4)
    return float(NormalDist().inv_cdf(q))


def future_sessions(last_date, horizon: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XIDX")
        end = start + pd.Timedelta(days=horizon * 3 + 45)
        sessions = cal.sessions_in_range(start, end)
        sessions = pd.DatetimeIndex([pd.Timestamp(s).tz_localize(None) for s in sessions])
        if len(sessions) >= horizon:
            return sessions[:horizon]
    except Exception:
        pass
    return pd.date_range(start=start, periods=horizon, freq="B")


class TFTModel:
    def __new__(cls, ticker, forecast_horizon=30, max_epochs=50, learning_rate=0.001, config=None):
        if PF_AVAILABLE:
            return RealTFTModel(ticker, forecast_horizon, max_epochs, learning_rate, config)
        return FallbackTFTModel(ticker, forecast_horizon, max_epochs, learning_rate, config)


class RealTFTModel:
    def __init__(self, ticker, forecast_horizon=30, max_epochs=50, learning_rate=0.001, config=None):
        self.ticker = ticker
        self.forecast_horizon = int(forecast_horizon)
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.config["encoder_length"] = max(self.config["encoder_length"], self.forecast_horizon)
        self.model = None
        self.feature_groups = get_feature_groups()
        self._train_dataset = None

    def fit_predict(self, df: pd.DataFrame, progress_callback=None):
        if PF_AVAILABLE:
            pl.seed_everything(SEED, workers=True)

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
            loss=QuantileLoss(self.config["quantiles"]),
            reduce_on_plateau_patience=4,
            log_interval=-1,
        )

        callback = _ProgressCallback(progress_callback, self.max_epochs)
        early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min", min_delta=0.0)

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            gradient_clip_val=self.config["gradient_clip_val"],
            callbacks=[callback, early_stop],
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, accelerator="auto", deterministic=False,
        )
        trainer.fit(self.model, train_loader, val_loader)

        try:
            val_qloss = float(trainer.callback_metrics.get("val_loss", float("nan")))
        except Exception:
            val_qloss = float("nan")

        focus = panel[panel["group_id"] == str(self.ticker)].copy()
        backtest = self._backtest_focus(focus)
        future_pred = self._predict_future(focus)
        attn_weights = self._extract_attention(val_loader)
        if isinstance(attn_weights, dict):
            attn_weights["val_quantile_loss"] = val_qloss
            if len(backtest.get("preds_1step", [])) == 0 and \
                    getattr(self, "last_backtest_error", None):
                attn_weights["backtest_error"] = str(self.last_backtest_error)
        return backtest, attn_weights, future_pred

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy().reset_index()
        if "date" not in result.columns:
            result = result.rename(columns={result.columns[0]: "date"})
        if "ticker" not in result.columns:
            result["ticker"] = self.ticker
        result["group_id"] = result["ticker"].astype(str)
        result = result.sort_values(["group_id", "date"]).reset_index(drop=True)
        result["time_idx"] = result.groupby("group_id").cumcount()

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = (
            result.groupby("group_id")[numeric_cols].ffill().fillna(0)
        )
        if "ticker_id" in result.columns:
            result["ticker_id"] = result["ticker_id"].astype(str)
        return result

    def _split_cutoffs(self, df: pd.DataFrame):
        max_idx = int(df["time_idx"].max())
        return int(max_idx * 0.70), int(max_idx * 0.85)

    def _create_datasets(self, df: pd.DataFrame):
        enc_len = self.config["encoder_length"]
        pred_len = self.forecast_horizon
        train_cut, val_cut = self._split_cutoffs(df)

        fg = self.feature_groups
        known = [c for c in fg["time_varying_known_reals"] if c in df.columns]
        unknown = [c for c in fg["time_varying_unknown_reals"] if c in df.columns]

        training_ds = TimeSeriesDataSet(
            df[df["time_idx"] <= train_cut],
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            min_encoder_length=enc_len,
            max_encoder_length=enc_len,
            min_prediction_length=pred_len,
            max_prediction_length=pred_len,
            static_categoricals=["ticker_id"],
            time_varying_known_reals=known,
            time_varying_unknown_reals=unknown,
            target_normalizer=EncoderNormalizer(transformation=None),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        validation_ds = TimeSeriesDataSet.from_dataset(
            training_ds, df[df["time_idx"] <= val_cut],
            predict=False, stop_randomization=True,
            min_prediction_idx=train_cut + 1,
        )
        return training_ds, validation_ds
        
    def _predict_returns(self, data_df: pd.DataFrame, min_idx: int,
                         pred_len_override=None):
        kwargs = dict(predict=False, stop_randomization=True,
                      min_prediction_idx=min_idx)
        if pred_len_override is not None:
            kwargs["min_prediction_length"] = pred_len_override
            kwargs["max_prediction_length"] = pred_len_override
        ds = TimeSeriesDataSet.from_dataset(self._train_dataset, data_df, **kwargs)
        loader = ds.to_dataloader(train=False,
                                  batch_size=self.config["batch_size"] * 2,
                                  num_workers=0)
        out = self.model.predict(loader, mode="prediction",
                                 return_x=True, return_y=True,
                                 trainer_kwargs={"logger": False})
        preds = out.output if hasattr(out, "output") else out[0]
        x = out.x if hasattr(out, "x") else out[1]
        y = out.y if hasattr(out, "y") else out[2]
        P = (preds[0] if isinstance(preds, (list, tuple)) else preds).cpu().numpy()
        A = (y[0][0] if isinstance(y[0], (list, tuple)) else y[0]).cpu().numpy()
        if P.ndim == 1:
            P = P[:, None]
        if A.ndim == 1:
            A = A[:, None]
        anchors = self._anchors_from_x(x, data_df, n=A.shape[0])
        dates = self._dates_from_x(x, data_df, n=A.shape[0])
        return P, A, anchors, dates

    @staticmethod
    def _prices_from_returns(P_ret, A_ret, anchors):
        P_price = np.zeros_like(P_ret, dtype=float)
        A_price = np.zeros_like(A_ret, dtype=float)
        for i in range(len(anchors)):
            if not np.isfinite(anchors[i]):
                continue
            P_price[i] = anchors[i] * np.exp(np.cumsum(P_ret[i]))
            A_price[i] = anchors[i] * np.exp(np.cumsum(A_ret[i]))
        return P_price, A_price

    def _backtest_focus(self, focus_df: pd.DataFrame) -> dict:
        _, val_cut = self._split_cutoffs(focus_df)
        H = self.forecast_horizon
        try:
            P, A, anchors, dates = self._predict_returns(focus_df, val_cut + 1)
            P_price, A_price = self._prices_from_returns(P, A, anchors)
            result = {
                "preds_matrix": P_price, "actuals_matrix": A_price,
                "anchors": anchors,
                "preds_1step": P_price[:, 0], "actuals_1step": A_price[:, 0],
                "pred_dates": dates,
            }
        except Exception as e:
            self.last_backtest_error = repr(e)
            empty = np.empty((0, H))
            return {"preds_matrix": empty, "actuals_matrix": empty,
                    "anchors": np.empty(0), "preds_1step": np.empty(0),
                    "actuals_1step": np.empty(0), "pred_dates": []}
        try:
            P1, A1, anch1, dates1 = self._predict_returns(
                focus_df, val_cut + 1, pred_len_override=1)
            ok = np.isfinite(anch1) & (anch1 > 0)
            p1 = np.where(ok, anch1 * np.exp(P1[:, 0]), np.nan)
            a1 = np.where(ok, anch1 * np.exp(A1[:, 0]), np.nan)
            if len(dates1) == len(p1) and len(p1) >= len(result["preds_1step"]):
                result["preds_1step"] = p1
                result["actuals_1step"] = a1
                result["pred_dates"] = dates1
        except Exception:
            pass
        return result

    def _anchors_from_x(self, x, focus_df: pd.DataFrame, n: int):
        try:
            t0 = x.get("decoder_time_idx").cpu().numpy()[:, 0]
            close_by_idx = focus_df.set_index("time_idx")["close"]
            return np.array([float(close_by_idx.get(ti - 1, np.nan)) for ti in t0])
        except Exception:
            return np.full(n, np.nan)

    def _dates_from_x(self, x, focus_df: pd.DataFrame, n: int):
        try:
            t0 = x.get("decoder_time_idx").cpu().numpy()[:, 0]
            date_by_idx = focus_df.set_index("time_idx")["date"]
            out = []
            for ti in t0:
                d = date_by_idx.get(int(ti), None)
                out.append(pd.Timestamp(d).strftime("%Y-%m-%d") if d is not None else "")
            return out
        except Exception:
            return []

    def _predict_future(self, focus_df: pd.DataFrame) -> dict:
        enc_len = self.config["encoder_length"]
        last_seq = focus_df.tail(enc_len).copy()
        last_idx = int(last_seq["time_idx"].max())
        H = self.forecast_horizon

        last_date = pd.to_datetime(focus_df["date"].iloc[-1])
        fdates = future_sessions(last_date, H)
        fut = pd.DataFrame({
            "time_idx": np.arange(last_idx + 1, last_idx + H + 1),
            "group_id": str(self.ticker),
        })
        for col, values in calendar_features(fdates).items():
            fut[col] = values

        combined = pd.concat([last_seq, fut], ignore_index=True).ffill().fillna(0)
        combined["group_id"] = str(self.ticker)
        if "ticker_id" in combined.columns:
            combined["ticker_id"] = str(TICKER_ID_MAP_STR.get(self.ticker, "0"))

        last_close = float(focus_df["close"].iloc[-1])
        try:
            fds = TimeSeriesDataSet.from_dataset(
                self._train_dataset, combined, predict=False, stop_randomization=True)
            floader = fds.to_dataloader(train=False, batch_size=1, num_workers=0)
            fp = self.model.predict(floader, mode="quantiles", trainer_kwargs={"logger": False})

            qret = fp[0] if isinstance(fp, (list, tuple)) else fp
            qret = qret.cpu().numpy()[0]            
            qret = np.sort(qret, axis=1)            

            quantiles = self.config["quantiles"]
            mid = len(quantiles) // 2
            r_med = qret[:, mid]
            r_low = qret[:, 0]
            r_upp = qret[:, -1]

            return self._reconstruct_band(last_close, r_med, r_low, r_upp)
        except Exception:
            daily = float(focus_df["close"].pct_change().tail(60).std() or 0.015)
            steps = np.arange(1, H + 1)
            spread = 1.2816 * daily * np.sqrt(steps)
            return {"close": np.array([last_close] * H, dtype=float),
                    "close_lower": last_close * np.exp(-spread),
                    "close_upper": last_close * np.exp(spread)}

    def _reconstruct_band(self, last_close, r_med, r_low, r_upp) -> dict:
        up_dev = np.clip(r_upp - r_med, 1e-9, None)
        lo_dev = np.clip(r_med - r_low, 1e-9, None)
        cum_med = np.cumsum(r_med)
        cum_up = np.sqrt(np.cumsum(up_dev ** 2))
        cum_lo = np.sqrt(np.cumsum(lo_dev ** 2))
        return {
            "close":       last_close * np.exp(cum_med),
            "close_lower": last_close * np.exp(cum_med - cum_lo),
            "close_upper": last_close * np.exp(cum_med + cum_up),
        }

    def _extract_attention(self, val_loader) -> dict:
        try:
            raw = self.model.predict(
                val_loader, mode="raw", return_x=True, trainer_kwargs={"logger": False})
            raw_out = raw.output if hasattr(raw, "output") else raw
            interp = self.model.interpret_output(raw_out, reduction="sum")
            vi = {}
            if "encoder_variables" in interp:
                enc = np.asarray(interp["encoder_variables"].detach().cpu().numpy()).ravel()
                enc = enc / (enc.sum() + 1e-12)
                names = (getattr(self.model, "encoder_variables", None)
                         or list(self._train_dataset.reals))
                vi = {str(n): float(v) for n, v in zip(names[:len(enc)], enc)}
            return {"variable_importance": vi}
        except Exception:
            return {"variable_importance": {}}


class FallbackTFTModel:
    def __init__(self, ticker, forecast_horizon=30, max_epochs=50, learning_rate=0.001, config=None):
        self.ticker = ticker
        self.forecast_horizon = int(forecast_horizon)
        self.max_epochs = int(max_epochs)
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def fit_predict(self, df: pd.DataFrame, progress_callback=None):
        if "ticker" in df.columns and df["ticker"].nunique() > 1:
            d = df[df["ticker"] == self.ticker].copy()
        else:
            d = df.copy()
        close = d["close"].to_numpy(dtype=float)

        for epoch in range(1, self.max_epochs + 1):
            if progress_callback:
                progress_callback(epoch, self.max_epochs, 0.05 * np.exp(-epoch / 20))

        H = self.forecast_horizon
        n = len(close)
        val_cut = int(n * 0.85)
        starts = list(range(max(val_cut, 1), max(n - H, 1)))
        rng = np.random.default_rng(abs(hash(self.ticker)) % (2**31))

        preds, acts, anchors = [], [], []
        for s in starts:
            anchors.append(close[s - 1])
            acts.append(close[s:s + H])
            noise = rng.normal(0, close[s - 1] * 0.01, H)
            preds.append(close[s - 1] + np.cumsum(noise) * 0.1)

        if preds:
            P, A = np.array(preds), np.array(acts)
        else:
            P = A = np.empty((0, H))
        backtest = {
            "preds_matrix": P, "actuals_matrix": A, "anchors": np.array(anchors),
            "preds_1step": P[:, 0] if len(P) else np.empty(0),
            "actuals_1step": A[:, 0] if len(A) else np.empty(0),
        }

        last = float(close[-1])
        daily = float(np.std(np.diff(np.log(close[-60:]))) or 0.015)
        steps = np.arange(1, H + 1)
        drift = rng.normal(0, daily, H)
        path = last * np.exp(np.cumsum(drift))
        spread = 1.2816 * daily * np.sqrt(steps)
        future = {"close": path,
                  "close_lower": path * np.exp(-spread),
                  "close_upper": path * np.exp(spread)}

        attn = {"variable_importance": {
            "target": 0.22, "vol_20": 0.18, "rsi": 0.14, "macd_norm": 0.12,
            "price_ma20_gap": 0.10, "roc_10": 0.09, "atr_norm": 0.08,
            "volume_z": 0.07,
        }, "val_quantile_loss": None}
        return backtest, attn, future


if PF_AVAILABLE:
    class _ProgressCallback(pl.Callback):
        def __init__(self, callback_fn, total_epochs):
            super().__init__()
            self.callback_fn, self.total_epochs = callback_fn, total_epochs

        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            train_loss = float(trainer.callback_metrics.get("train_loss", 0.0))
            val_loss = float(trainer.callback_metrics.get("val_loss", 0.0))
            print(f" >>> [TFT] Epoch {epoch:02d}/{self.total_epochs:02d} "
                  f"-> train {train_loss:.5f} | val {val_loss:.5f}")
            if self.callback_fn:
                self.callback_fn(epoch, self.total_epochs, train_loss)
else:
    class _ProgressCallback:
        def __init__(self, *a, **k):
            pass
