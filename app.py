import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from data_acquisition import fetch_raw_stock_data, fetch_market_context
from preprocessing_data import preprocess_stock_data
from feature_engineering import build_features
from model import TFTModel, BACKEND_NAME, future_sessions
from evaluation import (evaluate_with_baseline, evaluate_per_horizon,
                        return_space_metrics, aggregate_horizon_metrics)
from utils import (load_model_cache, save_model_cache,
                   load_panel_cache, save_panel_cache, clear_all_cache)

st.set_page_config(
    page_title="Prediksi Saham Bank — TFT",
    page_icon="📈", layout="wide", initial_sidebar_state="expanded",
)

PALETTE = {
    "primary": "#6366f1", "primary2": "#a5b4fc", "success": "#10b981",
    "danger": "#ef4444", "warning": "#f59e0b", "purple": "#a78bfa", "neutral": "#94a3b8",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, .stApp {{ font-family: 'Inter', sans-serif; }}
    .block-container {{ padding-top: 2.2rem; }}
    .hero {{ border-radius:16px; padding:1.4rem 1.6rem;
        background:linear-gradient(135deg,{PALETTE['primary']}14 0%,{PALETTE['success']}10 100%);
        border:1px solid {PALETTE['primary']}26; margin-bottom:1.4rem; }}
    .hero-title {{ font-size:1.6rem; font-weight:800; letter-spacing:-0.02em; margin:0; }}
    .hero-sub {{ font-size:0.92rem; opacity:0.7; margin-top:0.25rem; }}
    .pill {{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:0.72rem; font-weight:600; }}
    .side-label {{ font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
        color:{PALETTE['primary']}; margin:0.4rem 0 0.3rem 0; }}
    .stButton button[kind="primary"] {{ background:{PALETTE['primary']} !important; border:none !important;
        border-radius:10px !important; font-weight:600 !important; padding:0.55rem 1rem !important; }}
    div[data-testid="stMetricLabel"] p {{ font-size:0.82rem; opacity:0.8; }}
    div[data-testid="stMetricValue"] {{ font-weight:700; }}
</style>
""", unsafe_allow_html=True)

BANK_OPTIONS = {
    "BCA — BBCA": "BBCA.JK", "BRI — BBRI": "BBRI.JK", "Mandiri — BMRI": "BMRI.JK",
    "BNI — BBNI": "BBNI.JK", "BSI — BRIS": "BRIS.JK",
}

TRAINING_PRESETS = {
    "Cepat":    {"epochs": 15, "lr": 0.01,  "note": "Tercepat, akurasi paling rendah."},
    "Seimbang": {"epochs": 40, "lr": 0.003, "note": "Rekomendasi. Seimbang waktu dan kualitas."},
    "Akurat":   {"epochs": 80, "lr": 0.001, "note": "Paling lama. Untuk hasil lebih stabil."},
}

PRESET_MODEL_CONFIG = {
    "Cepat":    {"hidden_size": 16, "attention_head_size": 2, "dropout": 0.20,
                 "hidden_continuous_size": 8},
    "Seimbang": {"hidden_size": 32, "attention_head_size": 4, "dropout": 0.15,
                 "hidden_continuous_size": 16},
    "Akurat":   {"hidden_size": 48, "attention_head_size": 4, "dropout": 0.10,
                 "hidden_continuous_size": 24},
}

HELP_MAE = ("Mean Absolute Error. Rata-rata selisih harga prediksi vs harga asli (Rupiah). "
            "Makin kecil makin baik.")
HELP_DA = ("Directional Accuracy. Persentase hari saat arah (naik/turun) ditebak benar. "
           "50% setara koin. Di atas 55% baru lumayan.")


def money(v) -> str:
    return f"Rp {v:,.0f}"


def pill(label: str, color: str) -> str:
    return (f'<span class="pill" style="color:{color};background:{color}1a;'
            f'border:1px solid {color}33;">{label}</span>')


def interpret_da(v: float):
    p = v * 100
    if p >= 55: return ("Di atas acak", PALETTE["success"])
    if p >= 50: return ("Setara acak", PALETTE["warning"])
    return ("Di bawah acak", PALETTE["danger"])


def simulate_scenarios(last_close, median, upper, lower, n_paths=8, seed=0):
    median = np.asarray(median, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower = np.asarray(lower, dtype=float)
    H = len(median)
    log_mid = np.log(np.clip(median, 1e-9, None))
    cum_up = np.clip(np.log(np.clip(upper, 1e-9, None)) - log_mid, 1e-9, None)
    cum_lo = np.clip(log_mid - np.log(np.clip(lower, 1e-9, None)), 1e-9, None)
    prev_log = np.concatenate([[np.log(max(last_close, 1e-9))], log_mid[:-1]])
    drift = log_mid - prev_log
    up_var = np.clip(cum_up ** 2 - np.concatenate([[0.0], cum_up[:-1] ** 2]), 0.0, None)
    lo_var = np.clip(cum_lo ** 2 - np.concatenate([[0.0], cum_lo[:-1] ** 2]), 0.0, None)
    step_up, step_lo = np.sqrt(up_var), np.sqrt(lo_var)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, H))
    for k in range(n_paths):
        u = rng.normal(0.0, 1.0, H)
        shock = np.where(u >= 0, u * step_up, u * step_lo)
        paths[k] = last_close * np.exp(np.cumsum(drift + shock))
    return paths

with st.sidebar:
    st.markdown('<div class="side-label">Saham Bank</div>', unsafe_allow_html=True)
    selected_bank_label = st.selectbox("Pilih bank", list(BANK_OPTIONS.keys()),
                                       label_visibility="collapsed")
    ticker = BANK_OPTIONS[selected_bank_label]

    st.markdown('<div class="side-label">Prediksi Berapa Hari ke Depan</div>', unsafe_allow_html=True)
    forecast_days = st.slider("Horizon", 7, 30, 30, 1, label_visibility="collapsed",
                              help="Hari kerja ke depan. Makin jauh, makin tidak pasti.")

    st.markdown('<div class="side-label">Mode Pelatihan</div>', unsafe_allow_html=True)
    preset_name = st.radio("Mode", list(TRAINING_PRESETS.keys()), index=1, horizontal=True,
                           label_visibility="collapsed",
                           help="Seberapa lama dan teliti model belajar.")
    st.caption(TRAINING_PRESETS[preset_name]["note"])
    epochs = TRAINING_PRESETS[preset_name]["epochs"]
    learning_rate = TRAINING_PRESETS[preset_name]["lr"]

    with st.expander("Pengaturan lanjutan (opsional)"):
        st.caption("Semua opsional. Default sudah aman; ubah hanya jika paham efeknya.")

        manual = st.checkbox("Atur learning rate, epoch & dropout manual", value=False)
        st.caption("Buka kendali ahli di bawah. Jika mati, ketiganya mengikuti Mode Pelatihan.")
        dropout_override = None
        if manual:
            epochs = st.slider("Epoch (maksimum)", 5, 100, epochs, 5,
                               help="Batas berapa kali model membaca seluruh data latih.")
            st.caption("**Epoch** = jumlah putaran belajar maksimum. Pelatihan berhenti otomatis "
                       "(early stopping) bila tak membaik, jadi angka aktual bisa lebih kecil.")
            learning_rate = st.select_slider("Learning rate",
                                             options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
                                             value=learning_rate)
            st.caption("**Learning rate** = besar langkah tiap pembaruan bobot. Terlalu besar = "
                       "tidak stabil; terlalu kecil = lambat belajar.")
            dropout_override = st.select_slider(
                "Dropout", options=[0.05, 0.10, 0.15, 0.20, 0.30],
                value=PRESET_MODEL_CONFIG[preset_name]["dropout"])
            st.caption("**Dropout** = porsi neuron dimatikan acak saat latih untuk cegah overfitting. "
                       "Turunkan bila model terlalu kaku (underfitting); naikkan bila overfitting.")

        st.divider()
        show_scenarios = st.checkbox("Tampilkan skenario ilustratif (Monte Carlo)", value=True)
        st.caption("**Skenario** = 8 lintasan acak yang konsisten dengan rentang ketidakpastian model "
                   "(garis ungu tipis di grafik Ringkasan). Untuk intuisi sebaran, BUKAN prediksi arah.")

        use_cache = st.checkbox("Gunakan hasil tersimpan bila ada", value=True)
        st.caption("**Cache** = pakai ulang hasil 24 jam terakhir untuk pengaturan yang sama persis, "
                   "agar tidak melatih ulang. Matikan untuk memaksa pelatihan baru.")

        panel_mode = st.checkbox("Latih model gabungan 5 bank (panel)", value=False)
        st.caption("**Mode panel** = satu model dilatih pada kelima bank sekaligus (~5x data latih). "
                   "Mengatasi keterbatasan data per saham, tapi pelatihan lebih lama.")

        st.divider()
        if st.button("Hapus semua cache", use_container_width=True):
            n = clear_all_cache()
            st.success(f"{n} berkas cache dihapus.")
        st.caption("Menghapus semua hasil tersimpan di disk. Analisis berikutnya akan melatih dari awal.")

    st.divider()
    run_button = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

    st.markdown(
        f"""<div style="margin-top:1.5rem; font-size:0.72rem; opacity:0.75; line-height:1.7;">
        <b>Mesin:</b> {BACKEND_NAME}<br>
        <b>Target:</b> log-return harian (direkonstruksi ke harga)<br>
        <b>Strategi:</b> MIMO (prediksi {forecast_days} hari sekaligus)<br>
        <b>Sumber data:</b> Yahoo Finance, 10 tahun<br>
        </div>""", unsafe_allow_html=True)

st.markdown(
    f"""<div class="hero">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem;">
            <div>
                <p class="hero-title">Prediksi Harga Saham Bank Indonesia</p>
                <p class="hero-sub">{selected_bank_label} · {forecast_days} hari ke depan · Temporal Fusion Transformer</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_market_context(years: int) -> pd.DataFrame:
    """Unduh IHSG & USD/IDR sekali, lalu cache. Mencegah unduh ulang tiap rerun."""
    try:
        return fetch_market_context(years)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_featured(ticker: str, years: int, _market_sig: int) -> pd.DataFrame:
    """Unduh + bersihkan + rekayasa fitur satu ticker, lalu cache by argumen.
    Tab switch / interaksi UI tidak lagi memicu unduh ulang 10 tahun data."""
    market = _load_market_context(years)
    return build_features(preprocess_stock_data(fetch_raw_stock_data(ticker, years)),
                          ticker, market_df=market)


def _build_training_frame(focus_ticker, panel_mode, status_writer):
    market = _load_market_context(10)
    market_sig = 1 if (market is not None and not market.empty) else 0
    tickers = list(BANK_OPTIONS.values()) if panel_mode else [focus_ticker]
    frames, focus_featured = [], None
    for tk in tickers:
        try:
            feat = _load_featured(tk, 10, market_sig)
            frames.append(feat)
            if tk == focus_ticker:
                focus_featured = feat
        except Exception as e:
            status_writer(f"⚠️ Lewati {tk}: {e}")
    if focus_featured is None:
        focus_featured = _load_featured(focus_ticker, 10, market_sig)
        frames = [focus_featured]
    panel = pd.concat(frames) if len(frames) > 1 else focus_featured
    return panel, focus_featured, len(frames)


def run_pipeline(ticker, forecast_days, epochs, learning_rate, use_cache,
                 preset_name, panel_mode, model_config):
    with st.status("Memproses analisis...", expanded=True) as status:
        scope = "5 bank (panel)" if panel_mode else "1 bank"
        st.write("📥 Menyiapkan data bank fokus...")
        try:
            market = _load_market_context(10)
            market_sig = 1 if (market is not None and not market.empty) else 0
            featured_df = _load_featured(ticker, 10, market_sig)
        except Exception as e:
            status.update(label="Gagal menyiapkan data", state="error")
            st.error(f"Gagal menyiapkan data: {e}")
            return None
        n_features = featured_df.shape[1]
        cfg_tag = f"h{model_config['hidden_size']}d{model_config['dropout']}"
        if panel_mode:
            cache_key = f"PANEL{len(BANK_OPTIONS)}_{forecast_days}_{epochs}_{learning_rate}_{cfg_tag}_v14"
        else:
            cache_key = f"{ticker}_{forecast_days}_{epochs}_{learning_rate}_single_{cfg_tag}_v14"
        if use_cache:
            cached = load_panel_cache(cache_key) if panel_mode else load_model_cache(cache_key)
        else:
            cached = None

        backtest = attn_weights = future_pred = None
        n_banks = len(BANK_OPTIONS) if panel_mode else 1
        from_cache = False

        if cached is not None and panel_mode:
            per = cached.get("per_ticker", {})
            if ticker in per:
                st.write("⚡ Memuat hasil dari cache (model panel, tanpa latih ulang)...")
                backtest = per[ticker]["backtest"]
                future_pred = per[ticker]["future"]
                attn_weights = cached.get("attention", {})
                n_banks = int(cached.get("n_banks", len(per)))
                from_cache = True
            else:
                cached = None      
        elif cached is not None:
            st.write("⚡ Memuat hasil dari cache...")
            backtest = cached["backtest"]
            attn_weights = cached["attention"]
            future_pred = cached["future"]
            n_banks = 1
            from_cache = True

        if not from_cache:
            progress = st.progress(0, text="Inisialisasi...")
            trained = {"epochs": 0}

            def _cb(epoch, total, loss):
                trained["epochs"] = int(epoch)
                pct = min(int(epoch / max(total, 1) * 100), 100)
                progress.progress(pct, text=f"Epoch {epoch}/{total} — loss: {loss:.4f}")

            try:
                if panel_mode:
                    st.write("📥 Menggabungkan data lima bank (panel)...")
                    panel_df, _focus_feat, n_banks = _build_training_frame(ticker, True, st.write)
                else:
                    panel_df, n_banks = featured_df, 1

                st.write(f"🧠 Melatih model ({preset_name}: maks {epochs} epoch, {scope})...")
                model = TFTModel(ticker=ticker, forecast_horizon=forecast_days,
                                 max_epochs=epochs, learning_rate=learning_rate,
                                 config=model_config)
                backtest, attn_weights, future_pred = model.fit_predict(panel_df, progress_callback=_cb)
                progress.progress(100, text="Selesai")
                actual_epochs = trained["epochs"] or epochs
                if isinstance(attn_weights, dict):
                    attn_weights["actual_epochs"] = actual_epochs

                if panel_mode:
                    per_ticker = {str(ticker): {"backtest": backtest, "future": future_pred}}
                    can_extract = all(hasattr(model, m) for m in
                                      ("_prepare_dataframe", "_backtest_focus", "_predict_future"))
                    if can_extract:
                        st.write("🔁 Menyiapkan hasil untuk seluruh bank panel...")
                        try:
                            prepared = model._prepare_dataframe(panel_df)
                            orig_focus = model.ticker
                            for tk in prepared["group_id"].unique():
                                if str(tk) == str(ticker):
                                    continue
                                fdf = prepared[prepared["group_id"] == str(tk)].copy()
                                model.ticker = str(tk)
                                try:
                                    per_ticker[str(tk)] = {
                                        "backtest": model._backtest_focus(fdf),
                                        "future": model._predict_future(fdf),
                                    }
                                except Exception:
                                    pass   
                            model.ticker = orig_focus
                        except Exception:
                            pass              
                    n_banks = len(per_ticker)
                    save_panel_cache(cache_key, attn_weights, per_ticker, n_banks)
                else:
                    n_banks = 1
                    save_model_cache(cache_key, {"backtest": backtest,
                                                 "attention": attn_weights, "future": future_pred})
            except Exception as e:
                status.update(label="Pelatihan gagal", state="error")
                st.error(f"Pelatihan gagal: {e}")
                import traceback
                with st.expander("Detail teknis error"):
                    st.code(traceback.format_exc())
                return None

        if from_cache:
            actual_epochs = int(attn_weights.get("actual_epochs", epochs)) if isinstance(attn_weights, dict) else epochs

        status.update(label="Analisis selesai", state="complete", expanded=False)

    preds_1 = np.asarray(backtest.get("preds_1step", []), dtype=float)
    acts_1 = np.asarray(backtest.get("actuals_1step", []), dtype=float)
    P = np.asarray(backtest.get("preds_matrix", np.empty((0, forecast_days))), dtype=float)
    A = np.asarray(backtest.get("actuals_matrix", np.empty((0, forecast_days))), dtype=float)
    anchors = np.asarray(backtest.get("anchors", []), dtype=float)

    baseline = evaluate_with_baseline(acts_1, preds_1)
    per_horizon = evaluate_per_horizon(P, A, anchors)
    ret_metrics = return_space_metrics(acts_1, preds_1)
    agg = aggregate_horizon_metrics(P, A, anchors)

    last_date = featured_df.index[-1]
    last_close_val = float(featured_df["close"].iloc[-1])
    H = len(future_pred["close"])
    fdates = future_sessions(last_date, H)

    all_close = pd.concat([featured_df["close"],
                           pd.Series(np.asarray(future_pred["close"]), index=fdates)])
    delta = all_close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    all_rsi = 100 - (100 / (1 + rs))
    future_rsi = all_rsi.reindex(fdates).to_numpy()
    future_ma20 = all_close.rolling(20).mean().reindex(fdates).to_numpy()
    future_ma50 = all_close.rolling(50).mean().reindex(fdates).to_numpy()
    future_vol = all_close.pct_change().rolling(20).std().reindex(fdates).to_numpy()

    rng = np.random.default_rng(abs(hash((ticker, forecast_days))) % (2**32))
    recent_vol = float(featured_df["close"].pct_change().tail(30).std() or 0.015)
    sim_open, sim_high, sim_low, prev_c = [], [], [], last_close_val
    for c in np.asarray(future_pred["close"]):
        o = prev_c + prev_c * rng.uniform(-recent_vol * 0.4, recent_vol * 0.4)
        hi = max(o, c) + c * rng.uniform(0.001, recent_vol * 1.5)
        lo = min(o, c) - c * rng.uniform(0.001, recent_vol * 1.5)
        sim_open.append(o); sim_high.append(hi); sim_low.append(lo); prev_c = c

    has_market = any(c in featured_df.columns and featured_df[c].abs().sum() > 0
                     for c in ("ihsg_ret", "usdidr_ret"))

    return {
        "ticker": ticker, "bank_label": selected_bank_label,
        "forecast_days": forecast_days, "from_cache": from_cache,
        "panel_mode": panel_mode, "n_banks": n_banks,
        "epochs": epochs, "lr": learning_rate, "n_features": n_features,
        "actual_epochs": int(actual_epochs), "max_epochs": int(epochs),
        "rows": len(featured_df), "has_market": has_market,
        "val_quantile_loss": (attn_weights.get("val_quantile_loss")
                              if isinstance(attn_weights, dict) else None),
        "date_min": featured_df.index.min().date(), "date_max": featured_df.index.max().date(),
        "featured_df": featured_df, "backtest": backtest,
        "preds_1step": preds_1, "actuals_1step": acts_1,
        "pred_dates": backtest.get("pred_dates", []),
        "attention": attn_weights if isinstance(attn_weights, dict) else {},
        "attn_weights": attn_weights, "future_pred": future_pred,
        "baseline": baseline, "per_horizon": per_horizon, "ret_metrics": ret_metrics, "agg": agg,
        "future_dates": fdates,
        "future_rsi": future_rsi, "future_ma20": future_ma20, "future_ma50": future_ma50,
        "future_vol": future_vol,
        "sim_open": np.array(sim_open), "sim_high": np.array(sim_high), "sim_low": np.array(sim_low),
    }

FEATURE_CATALOG = [
    ("target", "Target", "Unknown real", "Log-return harian — yang diprediksi model; nilai lampaunya juga jadi input."),
    ("ret_lag_1", "Momentum", "Unknown real", "Log-return 1 hari lalu."),
    ("ret_lag_2", "Momentum", "Unknown real", "Log-return 2 hari lalu."),
    ("ret_lag_3", "Momentum", "Unknown real", "Log-return 3 hari lalu."),
    ("ret_lag_5", "Momentum", "Unknown real", "Log-return 5 hari lalu (sepekan bursa)."),
    ("rsi", "Momentum", "Unknown real", "Relative Strength Index (14), skala 0–1. >0,7 jenuh beli; <0,3 jenuh jual."),
    ("roc_10", "Momentum", "Unknown real", "Rate of Change 10 hari: persentase perubahan harga."),
    ("macd_norm", "Tren", "Unknown real", "MACD dinormalisasi terhadap harga: selisih EMA12 dan EMA26."),
    ("price_ma20_gap", "Tren", "Unknown real", "Jarak harga dari rata-rata 20 hari (sinyal mean-reversion)."),
    ("ma_trend", "Tren", "Unknown real", "Rasio MA20 terhadap MA50: arah tren menengah."),
    ("vol_20", "Volatilitas", "Unknown real", "Volatilitas realized: simpangan baku log-return 20 hari."),
    ("bb_pct", "Volatilitas", "Unknown real", "Posisi harga dalam Bollinger Band (0=bawah, 1=atas)."),
    ("hl_range", "Volatilitas", "Unknown real", "Rentang intraday (high−low) relatif terhadap harga."),
    ("atr_norm", "Volatilitas", "Unknown real", "Average True Range (14) dinormalisasi terhadap harga."),
    ("volume_z", "Volume", "Unknown real", "Z-score volume: lonjakan volume relatif terhadap rata-rata 20 hari."),
    ("ihsg_ret", "Konteks pasar", "Unknown real", "Log-return IHSG (^JKSE): arah pasar modal Indonesia."),
    ("usdidr_ret", "Konteks pasar", "Unknown real", "Log-return kurs USD/IDR: proksi arus modal asing."),
    ("day_sin", "Kalender", "Known future", "Komponen siklik hari dalam pekan (sinus)."),
    ("day_cos", "Kalender", "Known future", "Komponen siklik hari dalam pekan (kosinus)."),
    ("month_sin", "Kalender", "Known future", "Komponen siklik bulan (sinus)."),
    ("month_cos", "Kalender", "Known future", "Komponen siklik bulan (kosinus)."),
    ("dom_sin", "Kalender", "Known future", "Komponen siklik tanggal dalam bulan (sinus)."),
    ("dom_cos", "Kalender", "Known future", "Komponen siklik tanggal dalam bulan (kosinus)."),
    ("woy_sin", "Kalender", "Known future", "Komponen siklik pekan dalam tahun (sinus)."),
    ("woy_cos", "Kalender", "Known future", "Komponen siklik pekan dalam tahun (kosinus)."),
    ("is_month_end", "Kalender", "Known future", "Penanda akhir bulan (efek turn-of-month)."),
    ("is_quarter_end", "Kalender", "Known future", "Penanda akhir kuartal (rebalancing/window dressing)."),
    ("ticker_id", "Identitas", "Static", "Identitas bank (embedding statis pada mode panel)."),
]


def render_results(R):
    featured_df = R["featured_df"]
    predictions = R["preds_1step"]
    actuals = R["actuals_1step"]
    attn_weights = R["attn_weights"]
    future_pred = R["future_pred"]
    baseline = R["baseline"]
    per_horizon = R["per_horizon"]
    ret_metrics = R["ret_metrics"]
    agg = R.get("agg", {})
    forecast_days = R["forecast_days"]

    last_date = featured_df.index[-1]
    last_close = float(featured_df["close"].iloc[-1])
    future_dates = pd.DatetimeIndex(R["future_dates"])

    act_ep, max_ep = R.get("actual_epochs", R["epochs"]), R.get("max_epochs", R["epochs"])
    ep_txt = f"{act_ep} epoch" + (f", berhenti dini dari maks {max_ep}" if act_ep < max_ep else f" (maks {max_ep})")
    src = "diambil dari cache" if R["from_cache"] else f"baru dilatih ({ep_txt})"
    scope = f"panel {R['n_banks']} bank" if R.get("panel_mode") else "1 bank"
    market_note = "konteks pasar (IHSG, USD/IDR) aktif" if R.get("has_market") else "tanpa konteks pasar"
    st.caption(f"Data: {R['rows']:,} hari ({R['date_min']} → {R['date_max']}) · "
               f"{R['n_features']} kolom fitur · {market_note} · dilatih pada {scope} · Model {src}.")

    tab_ring, tab_val, tab_tek, tab_int, tab_data = st.tabs(
        ["📊 Ringkasan", "✅ Validasi Model", "📈 Analisis Teknikal",
         "🔍 Interpretasi Model", "🗂️ Data & Tabel"])

    with tab_ring:
        pred_last = float(future_pred["close"][-1])
        change_pct = (pred_last - last_close) / last_close * 100
        arah = "datar" if abs(change_pct) < 0.05 else ("naik" if change_pct >= 0 else "turun")
        arah_txt = "**mendatar** (perubahan median mendekati 0%)" if arah == "datar" else f"**{arah} {abs(change_pct):.2f}%**"

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric("Harga terakhir", money(last_close))
        with c2:
            with st.container(border=True):
                st.metric(f"Prediksi {forecast_days} hari lagi", money(pred_last), f"{change_pct:+.2f}%")
        with c3:
            with st.container(border=True):
                st.metric("Rentang prediksi (akhir)", money(float(future_pred["close_lower"][-1])))
                st.caption(f"hingga {money(float(future_pred['close_upper'][-1]))}")

        st.markdown(f"Model memperkirakan harga cenderung {arah_txt} dalam "
                    f"{forecast_days} hari kerja. Ini perkiraan, bukan kepastian.")

        st.markdown("##### Harga: aktual, backtest, dan prediksi")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=featured_df.index[-505:], y=featured_df["close"][-505:],
                                 mode="lines", name="Harga asli (historis)",
                                 line=dict(color=PALETTE["neutral"], width=1.5), opacity=0.7))
        if len(actuals) > 0:
            _pd = R.get("pred_dates", [])
            if _pd is not None and len(_pd) == len(actuals) and all(_pd):
                dates_hist = pd.to_datetime(list(_pd))
            else:
                dates_hist = featured_df.index[-len(actuals):]
            pred_plot = np.asarray(predictions, dtype=float)
            pred_plot = np.where(np.isfinite(pred_plot) & (pred_plot > 0), pred_plot, np.nan)
            fig.add_vrect(x0=dates_hist[0], x1=dates_hist[-1],
                          fillcolor=PALETTE["primary"], opacity=0.05, line_width=0,
                          annotation_text="wilayah uji (holdout)", annotation_position="top left",
                          annotation_font_size=10, annotation_font_color=PALETTE["primary"])
            fig.add_trace(go.Scatter(x=dates_hist, y=pred_plot, mode="lines", connectgaps=False,
                                     name="Backtest: prediksi 1 hari ke depan (digulir harian)",
                                     line=dict(color=PALETTE["primary"], width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=[last_date, future_dates[0]],
                                 y=[last_close, float(future_pred["close"][0])],
                                 mode="lines", line=dict(color=PALETTE["success"], width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=list(future_dates), y=list(future_pred["close"]),
                                 mode="lines", name=f"Median prediksi {forecast_days} hari",
                                 line=dict(color=PALETTE["success"], width=2.5)))
        upper, lower = list(future_pred["close_upper"]), list(future_pred["close_lower"])
        fig.add_trace(go.Scatter(x=list(future_dates) + list(future_dates[::-1]),
                                 y=upper + lower[::-1], fill="toself",
                                 fillcolor="rgba(16,185,129,0.10)", line=dict(color="rgba(0,0,0,0)"),
                                 name="Rentang keyakinan (5–95%)"))

        if globals().get("show_scenarios", False):
            paths = simulate_scenarios(last_close, future_pred["close"], future_pred["close_upper"],
                                       future_pred["close_lower"], n_paths=8,
                                       seed=abs(hash(R["ticker"])) % (2**31))
            for k, p in enumerate(paths):
                fig.add_trace(go.Scatter(
                    x=list(future_dates), y=list(p), mode="lines",
                    line=dict(color=PALETTE["purple"], width=1), opacity=0.28,
                    legendgroup="scenarios", name="Skenario ilustratif (8 lintasan)",
                    showlegend=(k == 0), hoverinfo="skip"))

        fig.add_vline(x=last_date.timestamp() * 1000, line_dash="dash",
                      line_color=PALETTE["warning"], line_width=1.5,
                      annotation_text="  hari ini", annotation_font_color=PALETTE["warning"])
        fig.update_layout(height=460, font=dict(family="Inter"),
                          legend=dict(orientation="h", y=-0.18, groupclick="togglegroup"),
                          margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        with st.expander("📖 Cara membaca grafik ini"):
            guide = pd.DataFrame({
                "Elemen": [
                    "Garis abu-abu", "Area biru muda + garis biru putus-putus",
                    "Garis kuning putus-putus", "Garis hijau tebal",
                    "Area hijau", "Garis ungu tipis (opsional)",
                ],
                "Arti": [
                    "Harga penutupan asli (historis).",
                    "Wilayah uji (holdout) di luar data latih. Garis biru = backtest: tiap titik adalah "
                    "prediksi 1 hari ke depan yang digulir setiap hari, lalu dibandingkan dengan harga asli.",
                    "Batas 'hari ini': kiri = masa lalu, kanan = prediksi.",
                    "Median prediksi ke depan (jalur paling mungkin menurut model).",
                    "Rentang keyakinan 5–95%, asimetris (sisi bawah bisa lebih panjang = risiko turun) "
                    "dan melebar ~akar horizon.",
                    "Skenario acak yang konsisten dengan ketidakpastian model. Ilustrasi sebaran, "
                    "BUKAN prediksi arah.",
                ],
            })
            st.table(guide)
            st.caption("Catatan: garis backtest hanya ada di wilayah holdout, jadi wajar bila lebih "
                       "pendek dari garis abu-abu. Median yang nyaris datar itu BENAR: untuk return "
                       "harian, tebakan optimal mendekati nol; grafik yang bergelombang tajam justru "
                       "menandakan model mengarang pola.")

    with tab_val:
        st.markdown("#### Seberapa andal model ini?")
        st.markdown(
            "Halaman ini menguji prediksi model pada **data uji (holdout)** yang tidak pernah dilihat "
            "saat pelatihan, lalu menjawab dua hal:")
        st.markdown(
            "1. **Apakah model lebih baik dari tebakan sederhana?** Pembanding kami adalah baseline "
            "*harga datar*: menebak harga ke depan = harga terakhir. Untuk saham, ini sulit dikalahkan. "
            "Kalau model tidak mengungguli baseline, model tidak menambah nilai.\n"
            "2. **Bagaimana performa di SELURUH horizon?** Model memprediksi "
            f"{forecast_days} hari sekaligus. Maka metrik utama di bawah adalah **rata-rata semua "
            f"{forecast_days} hari**, bukan hanya hari pertama. Hari pertama hampir selalu terlihat "
            "bagus (harga besok mirip hari ini), jadi menampilkannya sendirian akan menyesatkan.")
        st.divider()
        mdl, nv, sk = baseline["model"], baseline["naive"], baseline["skill"]
        n_eval = int(agg.get("n_samples", 0))

        if n_eval < 3:
            st.info("Data backtest belum cukup. Coba horizon lebih pendek atau periode lebih panjang.")
            _att = R.get("attention", {})
            _bt_err = _att.get("backtest_error") if isinstance(_att, dict) else None
            if _bt_err:
                with st.expander("Detail teknis (penyebab backtest kosong)"):
                    st.code(_bt_err)
        else:
            beats = agg.get("beats_naive", False)
            da_all = agg.get("DA", 0.0)
            if beats and da_all > 0.5:
                vcolor, vlabel, vtext = PALETTE["success"], "Lulus", ("Pada seluruh horizon, model "
                    "**mengungguli** baseline harga-datar dan menebak arah benar di atas 50%. Tetap "
                    "waspada: keunggulan kecil bisa hilang setelah biaya transaksi nyata.")
            elif beats:
                vcolor, vlabel, vtext = PALETTE["warning"], "Lemah", ("Model **sedikit** mengungguli "
                    "baseline pada error harga, tapi akurasi arahnya belum konsisten di atas 50%. "
                    "Bukti keunggulan masih lemah.")
            else:
                vcolor, vlabel, vtext = PALETTE["danger"], "Tidak lulus", ("Pada seluruh horizon, model "
                    "**tidak** mengungguli baseline harga-datar. Untuk saham ini, menahan harga terakhir "
                    "sama baik atau lebih baik. Jangan dipakai untuk bertaruh.")
            st.markdown(f"<div style='padding:0.8rem 1rem;border-radius:10px;background:{vcolor}14;"
                        f"border:1px solid {vcolor}40;font-size:0.9rem;'><b>Putusan: {vlabel}.</b> "
                        f"{vtext}</div>", unsafe_allow_html=True)
            st.write("")

            st.markdown(f"**Metrik utama — rata-rata seluruh {agg.get('n_horizons', forecast_days)} horizon**")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                with st.container(border=True):
                    st.metric("MAE (semua horizon)", money(agg.get("MAE", 0)), help=HELP_MAE)
                    st.caption(f"Baseline datar: {money(agg.get('MAE_naive', 0))}")
            with m2:
                with st.container(border=True):
                    st.metric("RMSE (semua horizon)", money(agg.get("RMSE", 0)),
                              help="Root Mean Squared Error: akar rata-rata galat kuadrat. "
                                   "Lebih menghukum galat besar daripada MAE.")
            with m3:
                with st.container(border=True):
                    st.metric("MAPE (semua horizon)", f"{agg.get('MAPE', 0):.2f}%",
                              help="Rata-rata galat persentase di seluruh horizon.")
            with m4:
                with st.container(border=True):
                    qloss = R.get("val_quantile_loss")
                    if qloss is not None and np.isfinite(qloss):
                        st.metric("Quantile Loss (validasi)", f"{qloss:.5f}",
                                  help="Pinball loss yang diminimalkan TFT saat pelatihan, diukur pada "
                                       "data validasi. Makin kecil makin baik.")
                    else:
                        st.metric("Quantile Loss (validasi)", "—",
                                  help="Hanya tersedia saat model TFT asli aktif (bukan mode demo).")
                        st.caption("Tersedia saat model TFT aktif.")

            n2a, n2b = st.columns(2)
            with n2a:
                with st.container(border=True):
                    skill_pct = agg.get("mae_skill", 0) * 100
                    st.metric("Skill vs baseline", f"{skill_pct:+.1f}%",
                              help="Pengurangan error rata-rata semua horizon vs harga datar. Positif = lebih baik.")
                    st.markdown(pill("unggul" if skill_pct > 0 else "tidak unggul",
                                     PALETTE["success"] if skill_pct > 0 else PALETTE["danger"]),
                                unsafe_allow_html=True)
            with n2b:
                with st.container(border=True):
                    da_lbl, da_col = interpret_da(da_all)
                    st.metric("Akurasi arah (semua horizon)", f"{da_all*100:.1f}%",
                              help="Apakah arah pergerakan dari titik awal sampai tiap horizon tertebak benar. 50% = koin.")
                    st.markdown(pill(da_lbl, da_col), unsafe_allow_html=True)

            with st.expander("Pembanding: performa hanya hari ke-1 (h=1)"):
                st.caption("Angka 1-langkah biasanya jauh lebih bagus karena harga besok mirip hari ini. "
                           "Inilah kenapa metrik ini TIDAK dijadikan headline.")
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("MAE (h=1)", money(mdl["MAE"]))
                s2.metric("RMSE (h=1)", money(mdl["RMSE"]))
                s3.metric("Skill (h=1)", f"{sk['mae_skill']*100:+.1f}%")
                s4.metric("Akurasi arah (h=1)", f"{sk['da']*100:.1f}%")
                s5.metric("Arah ruang return (h=1)", f"{ret_metrics['return_DA']*100:.1f}%")

            if per_horizon is not None and len(per_horizon) > 0:
                st.markdown("##### Error membesar seiring jarak prediksi")
                ph = per_horizon.reset_index()
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(x=ph["horizon_day"], y=ph["MAE_model"],
                                           mode="lines+markers", name="MAE model",
                                           line=dict(color=PALETTE["primary"], width=2)))
                if "MAE_naive" in ph.columns and ph["MAE_naive"].notna().any():
                    fig_h.add_trace(go.Scatter(x=ph["horizon_day"], y=ph["MAE_naive"], mode="lines",
                                               name="MAE baseline (harga datar)",
                                               line=dict(color=PALETTE["neutral"], width=2, dash="dash")))
                fig_h.update_layout(height=300, font=dict(family="Inter", size=11),
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    legend=dict(orientation="h", y=-0.25),
                                    xaxis_title="Hari ke depan", yaxis_title="MAE (Rp)")
                st.plotly_chart(fig_h, use_container_width=True, theme="streamlit")
                st.caption("Garis model di bawah baseline berarti model berguna pada horizon itu. "
                           "Bila menempel, model tidak menambah nilai.")

        with st.expander("Kenapa membandingkan dengan baseline itu wajib?"):
            st.markdown(
                "- Pada **level harga**, MAPE dan R² hampir selalu terlihat bagus karena harga besok "
                "mirip harga hari ini. Menebak 'harga ke depan = harga terakhir' saja sudah memberi "
                "MAPE rendah dan R² tinggi. Jadi angka itu **bukan** bukti model pintar.\n"
                "- Yang berarti adalah **selisih** terhadap baseline harga-datar. Kalau model tidak "
                "mengungguli baseline, model tidak berguna untuk memprediksi pergerakan.\n"
                "- **Akurasi arah** mengukur seberapa sering arah naik/turun ditebak benar. 50% setara "
                "koin. Prediksi arah harga harian sangat sulit. Bersikaplah skeptis.")

    with tab_tek:
        st.markdown("Indikator teknikal umum beserta proyeksinya (garis putus-putus).")
        future_rsi = R["future_rsi"]
        future_ma20 = R["future_ma20"]
        future_ma50 = R["future_ma50"]
        future_vol = R["future_vol"]
        sim_open, sim_high, sim_low = R["sim_open"], R["sim_high"], R["sim_low"]

        fig_t = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
                              vertical_spacing=0.05,
                              subplot_titles=("Harga + rata-rata bergerak", "RSI (14)", "Volatilitas harian (%)"))
        plot_df = featured_df.tail(252)
        fig_t.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["open"], high=plot_df["high"],
                                       low=plot_df["low"], close=plot_df["close"], name="Harga asli",
                                       increasing_line_color=PALETTE["success"],
                                       decreasing_line_color=PALETTE["danger"]), row=1, col=1)
        for col_name, color, label in [("ma_20", PALETTE["warning"], "MA20"),
                                       ("ma_50", PALETTE["primary"], "MA50")]:
            if col_name in plot_df.columns:
                fig_t.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_name], mode="lines",
                                           name=label, line=dict(color=color, width=1.5)), row=1, col=1)

        fig_t.add_trace(go.Candlestick(x=future_dates, open=sim_open, high=sim_high, low=sim_low,
                                       close=np.asarray(future_pred["close"]), name="Prediksi",
                                       increasing_line_color="rgba(16,185,129,0.85)",
                                       decreasing_line_color="rgba(239,68,68,0.85)"), row=1, col=1)
        if "ma_20" in featured_df.columns:
            fig_t.add_trace(go.Scatter(x=[last_date] + list(future_dates),
                                       y=[featured_df["ma_20"].iloc[-1]] + list(future_ma20),
                                       mode="lines", name="MA20 proyeksi",
                                       line=dict(color=PALETTE["warning"], width=1.5, dash="dot")), row=1, col=1)
        if "ma_50" in featured_df.columns:
            fig_t.add_trace(go.Scatter(x=[last_date] + list(future_dates),
                                       y=[featured_df["ma_50"].iloc[-1]] + list(future_ma50),
                                       mode="lines", name="MA50 proyeksi",
                                       line=dict(color=PALETTE["primary"], width=1.5, dash="dot")), row=1, col=1)

        hist_rsi = (featured_df["rsi"] * 100) if "rsi" in featured_df.columns else None
        if hist_rsi is not None:
            fig_t.add_trace(go.Scatter(x=plot_df.index, y=hist_rsi.tail(252), mode="lines", name="RSI",
                                       line=dict(color=PALETTE["purple"], width=1.5)), row=2, col=1)
            fig_t.add_hline(y=70, line_dash="dash", line_color=PALETTE["danger"], row=2, col=1)
            fig_t.add_hline(y=30, line_dash="dash", line_color=PALETTE["success"], row=2, col=1)
            last_rsi = float(hist_rsi.iloc[-1])
            fig_t.add_trace(go.Scatter(x=[last_date] + list(future_dates),
                                       y=[last_rsi] + list(future_rsi), mode="lines",
                                       name="RSI proyeksi", line=dict(color=PALETTE["success"], width=2, dash="dot")),
                            row=2, col=1)

        if "vol_20" in featured_df.columns:
            fig_t.add_trace(go.Bar(x=plot_df.index, y=plot_df["vol_20"] * 100, name="Volatilitas",
                                   marker_color=PALETTE["warning"], opacity=0.7), row=3, col=1)
        fig_t.add_trace(go.Bar(x=future_dates, y=future_vol * 100, name="Volatilitas proyeksi",
                               marker_color=PALETTE["success"], opacity=0.4), row=3, col=1)

        fig_t.add_vline(x=last_date.timestamp() * 1000, line_dash="dash",
                        line_color=PALETTE["warning"], line_width=1.5, row="all", col=1)
        fig_t.update_layout(height=680, showlegend=True, legend=dict(orientation="h", y=-0.08),
                            font=dict(family="Inter", size=11), xaxis_rangeslider_visible=False,
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_t, use_container_width=True, theme="streamlit")

        cexp = st.columns(3)
        with cexp[0]:
            st.markdown("**Rata-rata bergerak (MA)**")
            st.caption("Harga rata-rata 20/50 hari terakhir. Menunjukkan arah tren tanpa terganggu "
                       "naik-turun harian.")
        with cexp[1]:
            st.markdown("**RSI (14)**")
            st.caption("Jenuh beli (>70) atau jenuh jual (<30). Garis merah/hijau adalah batasnya.")
        with cexp[2]:
            st.markdown("**Volatilitas**")
            st.caption("Seberapa liar harga bergerak 20 hari terakhir. Makin tinggi, makin besar risiko.")
          
    with tab_int:
        st.markdown("TFT memiliki *variable selection network* yang menunjukkan fitur mana paling "
                    "berpengaruh. Ini membantu memahami *alasan* di balik prediksi.")
        vi = attn_weights.get("variable_importance", {}) if isinstance(attn_weights, dict) else {}
        if vi:
            names, vals = list(vi.keys()), list(vi.values())
            order = np.argsort(vals)[::-1]
            fig_v = go.Figure(go.Bar(x=[vals[i] for i in order], y=[names[i] for i in order],
                                     orientation="h",
                                     marker=dict(color=[vals[i] for i in order],
                                                 colorscale=[[0, PALETTE["primary2"]], [1, PALETTE["primary"]]],
                                                 showscale=False)))
            fig_v.update_layout(title=dict(text="Pengaruh tiap fitur (proporsi)", font=dict(size=13), x=0),
                                height=460, margin=dict(l=10, r=20, t=40, b=10),
                                font=dict(family="Inter", size=11))
            st.plotly_chart(fig_v, use_container_width=True, theme="streamlit")
            st.caption("Batang lebih panjang = fitur lebih berpengaruh. Hanya fitur encoder "
                       "(masa lalu) yang muncul di sini; rincian semua variabel ada di tabel bawah.")
        else:
            st.info("Grafik pengaruh fitur tidak tersedia untuk hasil ini (mode demo atau "
                    "interpretasi gagal). Tabel variabel di bawah tetap berlaku.")

        st.markdown("##### Kamus seluruh variabel model")
        st.caption("Peran: **Unknown real** = hanya diketahui di masa lalu (dipakai encoder). "
                   "**Known future** = pasti diketahui di masa depan, mis. kalender (dipakai decoder). "
                   "**Static** = identitas tetap per saham.")
        catalog_df = pd.DataFrame(
            [{"Variabel": n, "Kelompok": g, "Peran": role, "Penjelasan": desc,
              "Pengaruh": (f"{vi[n]*100:.1f}%" if n in vi else "—")}
             for (n, g, role, desc) in FEATURE_CATALOG])
        st.dataframe(catalog_df, use_container_width=True, hide_index=True)
        st.caption("Kolom Pengaruh hanya terisi untuk fitur encoder yang dinilai VSN; fitur "
                   "kalender/identitas dipakai di jalur decoder/statis sehingga tak punya skor encoder.")
      
    with tab_data:
        st.markdown("##### Tabel prediksi harian")
        forecast_df = pd.DataFrame({
            "Tanggal": future_dates.strftime("%Y-%m-%d"),
            "Prediksi": [money(p) for p in future_pred["close"]],
            "Batas bawah": [money(p) for p in future_pred["close_lower"]],
            "Batas atas": [money(p) for p in future_pred["close_upper"]],
            "Perubahan": [f"{((p - last_close)/last_close*100):+.2f}%" for p in future_pred["close"]],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.caption("Batas bawah/atas = kuantil 5% dan 95%, melebar ~akar horizon. Model "
                   "memperkirakan harga punya peluang besar berada di antaranya, tanpa jaminan.")

        st.markdown("##### Ringkasan teknis")
        _ql = R.get("val_quantile_loss")
        st.json({
            "ticker": R["ticker"], "horizon_hari": R["forecast_days"],
            "jumlah_baris_data": R["rows"], "jumlah_kolom_fitur": R["n_features"],
            "rentang_tanggal": f"{R['date_min']} s/d {R['date_max']}",
            "target": "log-return harian", "epoch_aktual": R.get("actual_epochs", R["epochs"]),
            "epoch_maks": R.get("max_epochs", R["epochs"]),
            "learning_rate": R["lr"], "konteks_pasar": bool(R.get("has_market")),
            "MAE_semua_horizon": round(float(agg.get("MAE", 0)), 2),
            "RMSE_semua_horizon": round(float(agg.get("RMSE", 0)), 2),
            "MAPE_semua_horizon_persen": round(float(agg.get("MAPE", 0)), 2),
            "quantile_loss_validasi": (round(float(_ql), 6) if (_ql is not None and np.isfinite(_ql)) else None),
            "dari_cache": R["from_cache"],
        })

    st.markdown(
        f"""<div style="margin-top:1.6rem; padding:1rem 1.4rem;
        background-color:var(--secondary-background-color);
        border:1px solid {PALETTE['warning']}40; border-radius:12px; font-size:0.85rem;">
        ⚠️ <b>Penting.</b> Prediksi ini untuk edukasi dan riset, bukan saran investasi.
        Harga saham dipengaruhi banyak faktor di luar data historis (berita, kebijakan, kondisi
        global) yang tidak diketahui model. Pergerakan harga harian sangat dekat dengan acak,
        sehingga tidak ada model yang bisa menjaminnya. Lakukan riset mandiri dan konsultasi
        dengan profesional sebelum mengambil keputusan keuangan.
        </div>""", unsafe_allow_html=True)

if run_button:
    _model_config = dict(PRESET_MODEL_CONFIG[preset_name])
    if dropout_override is not None:
        _model_config["dropout"] = dropout_override
    result = run_pipeline(ticker, forecast_days, epochs, learning_rate, use_cache,
                          preset_name, panel_mode, _model_config)
    if result is not None:
        st.session_state["results"] = result

if "results" in st.session_state:
    render_results(st.session_state["results"])
else:
    st.markdown(
        """<div style="text-align:center; padding:2.5rem 1rem;">
            <div style="font-size:3rem;">📈</div>
            <h2 style="font-weight:700; margin:0.4rem 0;">Siap menganalisis</h2>
            <p style="opacity:0.7; max-width:520px; margin:0 auto;">
                Pilih bank dan jumlah hari prediksi di panel kiri, lalu tekan
                <b>Jalankan Analisis</b>.</p>
        </div>""", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    for col, num, title, desc in [
        (g1, "1", "Pilih saham", "Lima bank terbesar Indonesia."),
        (g2, "2", "Atur prediksi", "Tentukan horizon dan mode pelatihan."),
        (g3, "3", "Baca hasil", "Grafik, validasi, dan interpretasi lengkap.")]:
        with col:
            with st.container(border=True):
                st.markdown(f"<div style='font-size:1.3rem;font-weight:800;color:{PALETTE['primary']};'>{num}</div>"
                            f"<div style='font-weight:600;margin:0.2rem 0;'>{title}</div>"
                            f"<div style='font-size:0.84rem;opacity:0.7;'>{desc}</div>",
                            unsafe_allow_html=True)
