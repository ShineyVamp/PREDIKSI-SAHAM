import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import pipeline modules
from data_acquisition import fetch_raw_stock_data
from preprocessing_data import preprocess_stock_data
from feature_engineering import build_features
from model import TFTModel, BACKEND_NAME, IS_REAL_MODEL
from evaluation import (evaluate_with_baseline, evaluate_per_horizon,
                        return_space_metrics)
from decision_engine import (generate_signal, backtest_strategy,
                             position_size_suggestion, round_trip_cost)
from utils import load_model_cache, save_model_cache, clear_all_cache

# ════════════════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prediksi Saham Bank — TFT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet warna terpusat. Satu sumber kebenaran agar grafik konsisten.
PALETTE = {
    "primary":  "#6366f1",   # indigo
    "primary2": "#a5b4fc",
    "success":  "#10b981",   # emerald
    "danger":   "#ef4444",   # red
    "warning":  "#f59e0b",   # amber
    "purple":   "#a78bfa",
    "neutral":  "#94a3b8",   # slate
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {{ font-family: 'Inter', sans-serif; }}

    /* Kurangi padding atas default agar header lebih rapat */
    .block-container {{ padding-top: 2.2rem; }}

    /* Kartu hero di bagian atas */
    .hero {{
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        background: linear-gradient(135deg, {PALETTE['primary']}14 0%, {PALETTE['success']}10 100%);
        border: 1px solid {PALETTE['primary']}26;
        margin-bottom: 1.4rem;
    }}
    .hero-title {{
        font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em;
        color: var(--text-color); margin: 0;
    }}
    .hero-sub {{
        font-size: 0.92rem; color: var(--text-color); opacity: 0.7; margin-top: 0.25rem;
    }}

    /* Pil status / badge */
    .pill {{
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600;
    }}

    /* Judul seksi minimalis di sidebar */
    .side-label {{
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; color: {PALETTE['primary']};
        margin: 0.4rem 0 0.3rem 0;
    }}

    /* Tombol utama */
    .stButton button[kind="primary"] {{
        background: {PALETTE['primary']} !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        padding: 0.55rem 1rem !important;
    }}
    .stButton button[kind="primary"]:hover {{ filter: brightness(1.05); transform: translateY(-1px); }}

    /* Rapikan label metric */
    div[data-testid="stMetricLabel"] p {{ font-size: 0.82rem; opacity: 0.8; }}
    div[data-testid="stMetricValue"] {{ font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# KONSTANTA & HELPER
# ════════════════════════════════════════════════════════════════════════════
BANK_OPTIONS = {
    "BCA — BBCA":      "BBCA.JK",
    "BRI — BBRI":      "BBRI.JK",
    "Mandiri — BMRI":  "BMRI.JK",
    "BNI — BBNI":      "BBNI.JK",
    "BSI — BRIS":      "BRIS.JK",
}

# Preset pelatihan. Pengguna awam cukup memilih salah satu, tanpa perlu paham
# learning rate atau epoch. Tiap preset memetakan ke nilai teknis yang masuk akal.
TRAINING_PRESETS = {
    "Cepat":    {"epochs": 15, "lr": 0.01,  "note": "Hasil tercepat, akurasi paling rendah."},
    "Seimbang": {"epochs": 40, "lr": 0.003, "note": "Rekomendasi. Keseimbangan waktu dan kualitas."},
    "Akurat":   {"epochs": 80, "lr": 0.001, "note": "Paling lama. Coba bila hasil 'Seimbang' kurang stabil."},
}

# Tooltip penjelasan tiap metrik (muncul saat kursor diarahkan ke ikon ?).
HELP_MAE  = ("Mean Absolute Error. Rata-rata selisih antara harga prediksi dan harga "
             "asli, dalam Rupiah. Nilai 0 berarti sempurna. Makin kecil makin baik.")
HELP_RMSE = ("Root Mean Squared Error. Mirip MAE, tetapi kesalahan besar dihukum lebih "
             "berat. Berguna untuk mendeteksi prediksi yang meleset jauh. Makin kecil makin baik.")
HELP_MAPE = ("Mean Absolute Percentage Error. Rata-rata kesalahan dalam persen. MAPE 5% "
             "berarti prediksi rata-rata meleset sekitar 5% dari harga asli. Makin kecil makin baik.")
HELP_DA   = ("Directional Accuracy. Persentase hari saat model menebak arah (naik atau turun) "
             "dengan benar. 50% setara menebak lewat lemparan koin. Di atas 55% baru bisa disebut lumayan.")


def money(v) -> str:
    return f"Rp {v:,.0f}"


def pill(label: str, color: str) -> str:
    """HTML pil berwarna untuk badge interpretasi."""
    return (f'<span class="pill" style="color:{color};background:{color}1a;'
            f'border:1px solid {color}33;">{label}</span>')


def interpret_pct(v: float):
    """Interpretasi untuk metrik berbasis persen kesalahan (makin kecil makin baik)."""
    if v < 3:  return ("Baik", PALETTE["success"])
    if v < 7:  return ("Cukup", PALETTE["warning"])
    return ("Kurang", PALETTE["danger"])


def interpret_da(v: float):
    """Interpretasi Directional Accuracy. v dalam rentang 0..1."""
    p = v * 100
    if p >= 55: return ("Di atas acak", PALETTE["success"])
    if p >= 50: return ("Setara acak", PALETTE["warning"])
    return ("Di bawah acak", PALETTE["danger"])


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — INPUT PENGGUNA
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="side-label">Saham Bank</div>', unsafe_allow_html=True)
    selected_bank_label = st.selectbox(
        "Pilih bank", list(BANK_OPTIONS.keys()), label_visibility="collapsed",
    )
    ticker = BANK_OPTIONS[selected_bank_label]

    st.markdown('<div class="side-label">Prediksi Berapa Hari ke Depan</div>', unsafe_allow_html=True)
    forecast_days = st.slider(
        "Horizon", min_value=7, max_value=30, value=30, step=1,
        label_visibility="collapsed",
        help="Berapa hari kerja ke depan yang ingin diprediksi. Makin jauh, makin tidak pasti.",
    )

    st.markdown('<div class="side-label">Mode Pelatihan</div>', unsafe_allow_html=True)
    preset_name = st.radio(
        "Mode", list(TRAINING_PRESETS.keys()), index=1, horizontal=True,
        label_visibility="collapsed",
        help="Mengatur seberapa lama dan teliti model belajar. Tidak perlu paham istilah teknis.",
    )
    st.caption(TRAINING_PRESETS[preset_name]["note"])

    # Nilai teknis default berasal dari preset.
    epochs = TRAINING_PRESETS[preset_name]["epochs"]
    learning_rate = TRAINING_PRESETS[preset_name]["lr"]

    st.markdown('<div class="side-label">Keputusan (DSS)</div>', unsafe_allow_html=True)
    signal_threshold = st.slider(
        "Ambang sinyal (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
        label_visibility="collapsed",
        help="Return bersih minimal (setelah biaya transaksi) agar sistem "
             "mengeluarkan sinyal Beli. Makin tinggi, makin konservatif. "
             "Default 1%.",
    )
    st.caption(f"Sinyal Beli muncul bila perkiraan untung bersih melebihi {signal_threshold:.1f}%.")

    # Mode lanjutan: hanya untuk yang paham. Tersembunyi secara default.
    with st.expander("Pengaturan lanjutan (opsional)"):
        st.caption(
            "Abaikan bagian ini jika Anda tidak yakin. Nilai default sudah aman "
            "untuk sebagian besar kasus."
        )
        manual = st.checkbox("Atur learning rate & epoch secara manual", value=False)
        if manual:
            epochs = st.slider(
                "Epoch (jumlah putaran belajar)", 5, 100, epochs, 5,
                help="Berapa kali model membaca seluruh data saat berlatih. "
                     "Terlalu sedikit: kurang pintar. Terlalu banyak: hafalan (overfitting). "
                     "Pelatihan berhenti otomatis jika sudah tidak membaik.",
            )
            learning_rate = st.select_slider(
                "Learning rate (kecepatan belajar)",
                options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
                value=learning_rate,
                help="Seberapa besar langkah model saat memperbaiki diri. "
                     "Terlalu besar: tidak stabil. Terlalu kecil: lambat.",
            )
        use_cache = st.checkbox(
            "Gunakan hasil tersimpan bila ada", value=True,
            help="Jika menyala dan kombinasi pengaturan sama persis pernah dijalankan "
                 "dalam 24 jam terakhir, hasil dipakai ulang tanpa melatih ulang. "
                 "Jauh lebih cepat.",
        )
        panel_mode = st.checkbox(
            "Latih model gabungan 5 bank (disarankan)", value=False,
            help="Melatih satu model TFT memakai data kelima bank sekaligus, lalu "
                 "membuat ramalan untuk bank yang dipilih. Lebih banyak data latih "
                 "dan membuat fitur ticker bermakna. Konsekuensinya: pelatihan jauh "
                 "lebih lama. Di Mode Demo, opsi ini tidak mengubah hasil.",
        )
        if st.button("Hapus semua cache", use_container_width=True):
            n = clear_all_cache()
            st.success(f"{n} berkas cache dihapus.")

    st.divider()
    run_button = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

    st.markdown(
        f"""<div style="margin-top:1.5rem; font-size:0.72rem; opacity:0.75; line-height:1.7;">
        <b>Mesin:</b> {BACKEND_NAME}<br>
        <b>Strategi:</b> MIMO (prediksi {forecast_days} hari sekaligus)<br>
        <b>Sumber data:</b> Yahoo Finance, 10 tahun<br>
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER UTAMA + STATUS MESIN
# ════════════════════════════════════════════════════════════════════════════
if IS_REAL_MODEL:
    engine_pill = pill("● Model aktif", PALETTE["success"])
else:
    engine_pill = pill("● Mode Demo (simulasi)", PALETTE["warning"])

st.markdown(
    f"""<div class="hero">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.6rem;">
            <div>
                <p class="hero-title">Prediksi Harga Saham Bank Indonesia</p>
                <p class="hero-sub">{selected_bank_label} · prediksi {forecast_days} hari ke depan · model Temporal Fusion Transformer</p>
            </div>
            <div>{engine_pill}</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# Peringatan jujur bila berjalan di mode simulasi.
if not IS_REAL_MODEL:
    st.error(
        "**Mode Demo aktif.** Library model (`pytorch-forecasting`) tidak ditemukan, "
        "jadi aplikasi memakai **data simulasi**. Semua angka dan grafik di halaman ini "
        "**bukan** hasil prediksi sungguhan. Jangan dipakai untuk keputusan apa pun. "
        "Untuk mengaktifkan model asli, jalankan: `pip install -r requirements.txt`."
    )


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE — DIJALANKAN SAAT TOMBOL DITEKAN
# ════════════════════════════════════════════════════════════════════════════
def _build_training_frame(focus_ticker, panel_mode, status_writer):
    """Bangun dataframe latih. Panel: gabung 5 bank. Single: hanya fokus."""
    tickers = list(BANK_OPTIONS.values()) if panel_mode else [focus_ticker]
    frames, focus_featured = [], None
    for tk in tickers:
        try:
            raw = fetch_raw_stock_data(tk, period_years=10)
            clean = preprocess_stock_data(raw)
            feat = build_features(clean, tk)
            frames.append(feat)
            if tk == focus_ticker:
                focus_featured = feat
        except Exception as e:
            status_writer(f"⚠️ Lewati {tk}: {e}")
    if focus_featured is None:
        # Fokus wajib ada. Kalau gagal di panel, jatuhkan ke single.
        raw = fetch_raw_stock_data(focus_ticker, period_years=10)
        focus_featured = build_features(preprocess_stock_data(raw), focus_ticker)
        frames = [focus_featured]
    panel = pd.concat(frames) if len(frames) > 1 else focus_featured
    return panel, focus_featured, len([f for f in frames])


def run_pipeline(ticker, forecast_days, epochs, learning_rate, use_cache,
                 preset_name, signal_threshold, panel_mode):
    """Jalankan seluruh proses dan kembalikan dict hasil, atau None bila gagal."""
    with st.status("Memproses analisis...", expanded=True) as status:

        scope = "5 bank (panel)" if panel_mode else "1 bank"
        st.write(f"📥 Mengambil & menyiapkan data ({scope})...")
        try:
            panel_df, featured_df, n_banks = _build_training_frame(
                ticker, panel_mode, st.write)
        except Exception as e:
            status.update(label="Gagal menyiapkan data", state="error")
            st.error(f"Gagal menyiapkan data: {e}")
            return None
        n_features = featured_df.shape[1]

        # Latih model (atau ambil dari cache)
        pflag = f"panel{n_banks}" if panel_mode else "single"
        cache_key = f"{ticker}_{forecast_days}_{epochs}_{learning_rate}_{pflag}_s5"
        cached = load_model_cache(cache_key) if use_cache else None

        if cached is not None:
            st.write("⚡ Memuat hasil dari cache (melewati pelatihan)...")
            backtest     = cached["backtest"]
            attn_weights = cached["attention"]
            future_pred  = cached["future"]
            from_cache   = True
        else:
            st.write(f"🧠 Melatih model ({preset_name}: maksimal {epochs} epoch, {scope})...")
            progress = st.progress(0, text="Inisialisasi model...")

            def _cb(epoch, total, loss):
                pct = min(int(epoch / max(total, 1) * 100), 100)
                progress.progress(pct, text=f"Epoch {epoch}/{total} — loss: {loss:.4f}")

            try:
                model = TFTModel(
                    ticker=ticker, forecast_horizon=forecast_days,
                    max_epochs=epochs, learning_rate=learning_rate,
                )
                backtest, attn_weights, future_pred = model.fit_predict(
                    panel_df, progress_callback=_cb,
                )
                progress.progress(100, text="Pelatihan selesai")
                save_model_cache(cache_key, {
                    "backtest": backtest,
                    "attention": attn_weights, "future": future_pred,
                })
            except Exception as e:
                status.update(label="Pelatihan gagal", state="error")
                st.error(f"Pelatihan gagal: {e}")
                import traceback
                with st.expander("Detail teknis error"):
                    st.code(traceback.format_exc())
                return None
            from_cache = False

        status.update(label="Analisis selesai", state="complete", expanded=False)

    # ── Evaluasi jujur: model vs baseline random walk ───────────────────────
    preds_1 = np.asarray(backtest.get("preds_1step", []), dtype=float)
    acts_1 = np.asarray(backtest.get("actuals_1step", []), dtype=float)
    P = np.asarray(backtest.get("preds_matrix", np.empty((0, forecast_days))), dtype=float)
    A = np.asarray(backtest.get("actuals_matrix", np.empty((0, forecast_days))), dtype=float)
    anchors = np.asarray(backtest.get("anchors", []), dtype=float)

    baseline = evaluate_with_baseline(acts_1, preds_1)
    per_horizon = evaluate_per_horizon(P, A, anchors)
    ret_metrics = return_space_metrics(acts_1, preds_1)

    # ── Keputusan + backtest strategi ───────────────────────────────────────
    last_close = float(featured_df['close'].iloc[-1])
    decision = generate_signal(
        last_close=last_close,
        forecast_close_end=float(future_pred['close'][-1]),
        lower_end=float(future_pred['close_lower'][-1]),
        upper_end=float(future_pred['close_upper'][-1]),
        horizon_days=forecast_days,
        threshold_pct=signal_threshold,
    ).to_dict()
    strategy = backtest_strategy(P, A, anchors, forecast_days, threshold_pct=signal_threshold)
    daily_vol = float(featured_df['close'].pct_change().tail(60).std())
    pos_size = position_size_suggestion(daily_vol)

    return {
        "ticker": ticker, "bank_label": selected_bank_label,
        "forecast_days": forecast_days, "from_cache": from_cache,
        "panel_mode": panel_mode, "n_banks": n_banks,
        "preset": preset_name if not (use_cache and cached is not None) else "cache",
        "epochs": epochs, "lr": learning_rate, "n_features": n_features,
        "rows": len(featured_df),
        "date_min": featured_df.index.min().date(),
        "date_max": featured_df.index.max().date(),
        "featured_df": featured_df,
        "backtest": backtest,
        "preds_1step": preds_1, "actuals_1step": acts_1,
        "attn_weights": attn_weights,
        "future_pred": future_pred,
        "baseline": baseline,
        "per_horizon": per_horizon,
        "ret_metrics": ret_metrics,
        "decision": decision,
        "strategy": strategy,
        "pos_size": pos_size,
        "signal_threshold": signal_threshold,
    }


# ════════════════════════════════════════════════════════════════════════════
# RENDER HASIL
# ════════════════════════════════════════════════════════════════════════════
def render_results(R):
    featured_df  = R["featured_df"]
    predictions  = R["preds_1step"]
    actuals      = R["actuals_1step"]
    attn_weights = R["attn_weights"]
    future_pred  = R["future_pred"]
    baseline     = R["baseline"]
    per_horizon  = R["per_horizon"]
    ret_metrics  = R["ret_metrics"]
    decision     = R["decision"]
    strategy     = R["strategy"]
    pos_size     = R["pos_size"]
    forecast_days = R["forecast_days"]

    last_date  = featured_df.index[-1]
    last_close = float(featured_df['close'].iloc[-1])

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(future_pred['close']), freq='B',
    )

    # Sumber & status baris ringkas
    src = "diambil dari cache" if R["from_cache"] else f"baru dilatih ({R['epochs']} epoch)"
    scope = f"panel {R['n_banks']} bank" if R.get("panel_mode") else "1 bank"
    st.caption(
        f"Data: {R['rows']:,} hari ({R['date_min']} → {R['date_max']}) · "
        f"{R['n_features']} fitur · dilatih pada {scope} · Model {src}."
    )

    tab_ring, tab_dss, tab_tek, tab_int, tab_data = st.tabs(
        ["📊 Ringkasan", "🎯 Keputusan (DSS)", "📈 Analisis Teknikal",
         "🔍 Interpretasi Model", "🗂️ Data & Tabel"]
    )

    # ───────────────────────── TAB 1: RINGKASAN ─────────────────────────
    with tab_ring:
        # Ringkasan arah prediksi
        pred_last = float(future_pred['close'][-1])
        change_pct = (pred_last - last_close) / last_close * 100
        arah = "naik" if change_pct >= 0 else "turun"
        arah_color = PALETTE["success"] if change_pct >= 0 else PALETTE["danger"]

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric("Harga terakhir", money(last_close))
        with c2:
            with st.container(border=True):
                st.metric(
                    f"Prediksi {forecast_days} hari lagi", money(pred_last),
                    f"{change_pct:+.2f}%",
                )
        with c3:
            with st.container(border=True):
                st.metric("Rentang prediksi (akhir)",
                          f"{money(float(future_pred['close_lower'][-1]))}")
                st.caption(f"hingga {money(float(future_pred['close_upper'][-1]))}")

        st.markdown(
            f"Model memperkirakan harga cenderung **{arah} {abs(change_pct):.2f}%** "
            f"dalam {forecast_days} hari kerja ke depan. Angka ini adalah perkiraan, "
            f"bukan kepastian."
        )

        # Grafik utama: aktual vs backtest vs forecast + rentang keyakinan
        st.markdown("##### Harga: aktual, backtest, dan prediksi")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=featured_df.index[-505:], y=featured_df['close'][-505:],
            mode='lines', name='Harga asli (historis)',
            line=dict(color=PALETTE["neutral"], width=1.5), opacity=0.7,
        ))
        if len(actuals) > 0:
            dates_hist = featured_df.index[-len(actuals):]
            fig.add_trace(go.Scatter(
                x=dates_hist, y=predictions, mode='lines',
                name='Tebakan model (backtest, 1 hari)',
                line=dict(color=PALETTE["primary"], width=2, dash='dot'),
            ))
        # Penghubung tipis dari harga terakhir ke ramalan hari pertama.
        # Tidak menggeser angka model (bias correction sudah dihapus), jadi
        # bisa muncul lompatan kecil. Itu jujur: ramalan hari-1 model memang
        # belum tentu sama persis dengan harga penutupan terakhir.
        fig.add_trace(go.Scatter(
            x=[last_date, future_dates[0]],
            y=[last_close, float(future_pred['close'][0])],
            mode='lines', line=dict(color=PALETTE["success"], width=1, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=list(future_dates),
            y=list(future_pred['close']),
            mode='lines', name=f'Prediksi {forecast_days} hari',
            line=dict(color=PALETTE["success"], width=2.5),
        ))
        upper = list(future_pred['close_upper'])
        lower = list(future_pred['close_lower'])
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=upper + lower[::-1], fill='toself',
            fillcolor='rgba(16,185,129,0.10)', line=dict(color='rgba(0,0,0,0)'),
            name='Rentang keyakinan (10–90%)',
        ))
        fig.add_vline(
            x=last_date.timestamp() * 1000, line_dash="dash",
            line_color=PALETTE["warning"], line_width=1.5,
            annotation_text="  hari ini", annotation_font_color=PALETTE["warning"],
        )
        fig.update_layout(
            height=460, font=dict(family='Inter'),
            legend=dict(orientation='h', y=-0.18),
            margin=dict(l=10, r=10, t=20, b=10), hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        st.caption(
            "**Cara membaca:** garis abu-abu adalah harga asli. Garis biru putus-putus "
            "adalah tebakan model 1 hari ke depan pada data uji (backtest). Garis hijau "
            "adalah prediksi ke depan; area hijau muda adalah rentang kemungkinan. "
            "Lompatan kecil di awal garis hijau itu wajar: angka model tidak lagi "
            "dipaksa menempel ke harga terakhir."
        )

        # ── Metrik akurasi: model VS baseline random walk ───────────────────
        st.markdown("##### Apakah model mengalahkan tebakan acak?")
        mdl = baseline["model"]
        nv = baseline["naive"]
        sk = baseline["skill"]
        n_eval = int(mdl.get("n_samples", 0))

        if n_eval < 3:
            st.info(
                "Data backtest belum cukup untuk menilai. Coba horizon lebih pendek "
                "atau periode data lebih panjang."
            )
        else:
            beats = sk["beats_naive"]
            ret_beats = ret_metrics["beats_zero_return"]
            # Verdict utama. Ini pertanyaan terpenting untuk sebuah DSS.
            if beats and sk["da_edge"] > 0:
                vcolor, vtext = PALETTE["success"], (
                    "Model **mengalahkan** baseline random walk pada data uji. "
                    "Tetap waspada: keunggulan kecil bisa hilang setelah biaya.")
            elif beats:
                vcolor, vtext = PALETTE["warning"], (
                    "Model **sedikit** mengalahkan baseline pada error harga, tapi "
                    "arah geraknya belum konsisten di atas 50%. Bukti keunggulan lemah.")
            else:
                vcolor, vtext = PALETTE["danger"], (
                    "Model **TIDAK** mengalahkan baseline random walk. Artinya "
                    "'tebak harga besok = harga hari ini' sama bagus atau lebih baik. "
                    "Jangan pakai prediksi ini untuk bertaruh.")
            st.markdown(
                f"<div style='padding:0.7rem 1rem;border-radius:10px;"
                f"background:{vcolor}14;border:1px solid {vcolor}40;"
                f"font-size:0.9rem;'>{vtext}</div>",
                unsafe_allow_html=True,
            )
            st.write("")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                with st.container(border=True):
                    st.metric("MAE model", money(mdl["MAE"]), help=HELP_MAE)
                    st.caption(f"Baseline naive: {money(nv['MAE'])}")
            with m2:
                with st.container(border=True):
                    skill_pct = sk["mae_skill"] * 100
                    st.metric("Skill vs baseline", f"{skill_pct:+.1f}%",
                              help="Pengurangan error dibanding random walk. "
                                   "Positif = lebih baik dari menebak harga kemarin. "
                                   "Nol atau negatif = tidak berguna.")
                    st.markdown(
                        pill("unggul" if skill_pct > 0 else "tidak unggul",
                             PALETTE["success"] if skill_pct > 0 else PALETTE["danger"]),
                        unsafe_allow_html=True,
                    )
            with m3:
                with st.container(border=True):
                    da = sk["da"]
                    da_lbl, da_col = interpret_da(da)
                    st.metric("Akurasi arah", f"{da*100:.1f}%", help=HELP_DA)
                    st.markdown(
                        pill(da_lbl, da_col) +
                        f"<div style='font-size:0.76rem;opacity:0.75;margin-top:6px;'>"
                        f"{sk['da_edge']*100:+.1f} poin vs koin (50%).</div>",
                        unsafe_allow_html=True,
                    )
            with m4:
                with st.container(border=True):
                    rd = ret_metrics["return_DA"]
                    st.metric("Arah (ruang return)", f"{rd*100:.1f}%",
                              help="Akurasi arah dihitung pada return harian, bukan "
                                   "level harga. Lebih sulit dimanipulasi.")
                    st.markdown(
                        pill("kalahkan nol" if ret_beats else "kalah dari nol",
                             PALETTE["success"] if ret_beats else PALETTE["danger"]),
                        unsafe_allow_html=True,
                    )

            # ── Error per horizon: hari ke-1 jauh beda dengan hari ke-30 ─────
            if per_horizon is not None and len(per_horizon) > 0:
                st.markdown("##### Error membesar seiring jarak prediksi")
                ph = per_horizon.reset_index()
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=ph["horizon_day"], y=ph["MAE_model"], mode='lines+markers',
                    name='MAE model', line=dict(color=PALETTE["primary"], width=2),
                ))
                if "MAE_naive" in ph.columns and ph["MAE_naive"].notna().any():
                    fig_h.add_trace(go.Scatter(
                        x=ph["horizon_day"], y=ph["MAE_naive"], mode='lines',
                        name='MAE baseline (harga datar)',
                        line=dict(color=PALETTE["neutral"], width=2, dash='dash'),
                    ))
                fig_h.update_layout(
                    height=300, font=dict(family='Inter', size=11),
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation='h', y=-0.25),
                    xaxis_title="Hari ke depan", yaxis_title="MAE (Rp)",
                )
                st.plotly_chart(fig_h, use_container_width=True, theme="streamlit")
                st.caption(
                    "Garis model di bawah garis baseline berarti model berguna pada "
                    "horizon itu. Bila keduanya menempel, model tidak menambah nilai."
                )

        with st.expander("Kenapa membandingkan dengan baseline itu wajib?"):
            st.markdown(
                "- Pada **level harga**, MAPE dan R² hampir selalu terlihat bagus, "
                "karena harga besok mirip harga hari ini. Menebak 'harga besok = "
                "harga hari ini' (random walk) saja sudah memberi MAPE ~1% dan R² ~0.98. "
                "Jadi angka itu **bukan** bukti model pintar.\n"
                "- Yang berarti adalah **selisih** terhadap baseline tersebut. Kalau "
                "model tidak mengalahkan random walk, model tidak berguna untuk trading.\n"
                "- **Akurasi arah** mengukur seberapa sering arah naik/turun ditebak "
                "benar. 50% setara lemparan koin. Prediksi arah harga harian sangat "
                "sulit, bahkan untuk model canggih. Bersikaplah skeptis."
            )

    # ───────────────────────── TAB 2: KEPUTUSAN (DSS) ─────────────────────────
    with tab_dss:
        d = decision
        sig = d["signal"]
        sig_color = (PALETTE["success"] if sig == "Beli"
                     else PALETTE["danger"] if sig.startswith("Hindari")
                     else PALETTE["neutral"])

        st.markdown(
            f"<div style='padding:1.1rem 1.4rem;border-radius:14px;"
            f"background:{sig_color}14;border:1px solid {sig_color}45;'>"
            f"<div style='font-size:0.8rem;opacity:0.7;text-transform:uppercase;"
            f"letter-spacing:0.05em;'>Sinyal untuk {forecast_days} hari ke depan</div>"
            f"<div style='font-size:2rem;font-weight:800;color:{sig_color};'>{sig}</div>"
            f"<div style='font-size:0.9rem;opacity:0.85;'>Keyakinan: {d['strength']} · "
            f"interval keyakinan: {d['confidence']}</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric("Perkiraan untung kotor", f"{d['expected_return_pct']:+.2f}%")
                st.caption("Sebelum biaya transaksi.")
        with c2:
            with st.container(border=True):
                st.metric("Untung bersih perkiraan", f"{d['net_return_pct']:+.2f}%")
                st.caption(f"Setelah biaya ~{d['cost_pct']:.2f}% (beli + jual).")
        with c3:
            with st.container(border=True):
                st.metric("Ambang sinyal", f"{R['signal_threshold']:.1f}%")
                st.caption("Bisa diatur di panel kiri.")

        st.markdown(
            f"<div style='font-size:0.9rem;padding:0.6rem 0;'>{d['rationale']}</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Backtest strategi vs buy-and-hold ───────────────────────────────
        st.markdown("##### Uji strategi: ikuti sinyal vs beli lalu diamkan")
        s = strategy
        if s["n_windows"] < 2:
            st.info(
                "Data uji belum cukup untuk mensimulasikan strategi. Perlu lebih "
                "banyak riwayat atau horizon lebih pendek."
            )
        else:
            verdict_col = PALETTE["success"] if s["beats_buyhold"] else PALETTE["danger"]
            verdict_txt = ("Strategi berbasis model **mengalahkan** buy-and-hold pada "
                           "periode uji."
                           if s["beats_buyhold"] else
                           "Strategi berbasis model **kalah** dari sekadar beli lalu "
                           "diamkan. Untuk saham ini, trading aktif tidak terbukti "
                           "lebih baik.")
            st.markdown(
                f"<div style='padding:0.6rem 1rem;border-radius:10px;"
                f"background:{verdict_col}14;border:1px solid {verdict_col}40;"
                f"font-size:0.88rem;'>{verdict_txt}</div>",
                unsafe_allow_html=True,
            )
            st.write("")

            eq_s = s["strategy_equity"]
            eq_b = s["buyhold_equity"]
            if eq_s and eq_b:
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(
                    y=[(v - 1) * 100 for v in eq_s], mode='lines',
                    name='Strategi (ikuti sinyal)',
                    line=dict(color=PALETTE["primary"], width=2.5),
                ))
                fig_e.add_trace(go.Scatter(
                    y=[(v - 1) * 100 for v in eq_b], mode='lines',
                    name='Buy and hold',
                    line=dict(color=PALETTE["neutral"], width=2, dash='dash'),
                ))
                fig_e.update_layout(
                    height=320, font=dict(family='Inter', size=11),
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation='h', y=-0.22),
                    xaxis_title="Transaksi ke-", yaxis_title="Imbal hasil kumulatif (%)",
                )
                st.plotly_chart(fig_e, use_container_width=True, theme="streamlit")

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                with st.container(border=True):
                    st.metric("Hasil strategi", f"{s['strategy_total_return_pct']:+.1f}%")
                    st.caption(f"Buy-hold: {s['buyhold_total_return_pct']:+.1f}%")
            with k2:
                with st.container(border=True):
                    st.metric("Sharpe strategi", f"{s['strategy_sharpe']:.2f}",
                              help="Imbal hasil disesuaikan risiko. Di atas 1 bagus, "
                                   "di bawah 0 buruk.")
                    st.caption(f"Buy-hold: {s['buyhold_sharpe']:.2f}")
            with k3:
                with st.container(border=True):
                    st.metric("Penurunan terdalam", f"{s['strategy_max_drawdown_pct']:.1f}%",
                              help="Kerugian terbesar dari puncak ke lembah. Makin "
                                   "dangkal makin baik.")
                    st.caption(f"Buy-hold: {s['buyhold_max_drawdown_pct']:.1f}%")
            with k4:
                with st.container(border=True):
                    st.metric("Akurasi transaksi", f"{s['hit_rate']*100:.0f}%",
                              help="Persentase transaksi yang menguntungkan.")
                    st.caption(f"{s['n_trades']} transaksi dari {s['n_windows']} peluang")

        st.divider()

        # ── Ukuran posisi berbasis risiko ──────────────────────────────────
        st.markdown("##### Saran ukuran posisi (manajemen risiko)")
        with st.container(border=True):
            st.markdown(f"**{pos_size.get('note', '-')}**")
            st.caption(
                "Ini kerangka sederhana berbasis volatilitas, bukan perintah. "
                "Posisi lebih kecil pada saham yang lebih bergejolak."
            )

        st.markdown(
            f"""<div style="margin-top:1rem; padding:0.9rem 1.2rem;
            background-color:var(--secondary-background-color);
            border:1px solid {PALETTE['warning']}40; border-radius:10px; font-size:0.84rem;">
            ⚠️ Sinyal dan backtest ini bersifat edukatif. Backtest memakai data masa lalu
            dan tidak menjamin masa depan. Biaya nyata (slippage, spread, pajak) bisa lebih
            besar. Untuk lima bank likuid dengan horizon mingguan, keunggulan nyata setelah
            biaya cenderung mendekati nol. Sering kali keputusan paling rasional adalah menahan.
            </div>""",
            unsafe_allow_html=True,
        )

    # ───────────────────────── TAB 3: ANALISIS TEKNIKAL ─────────────────────────
    with tab_tek:
        st.markdown(
            "Bagian ini menampilkan indikator teknikal yang umum dipakai analis, "
            "lengkap dengan proyeksinya ke depan (garis putus-putus)."
        )

        # Hitung proyeksi RSI, MA, dan volatilitas dengan menyambung harga prediksi.
        all_close = pd.concat([
            featured_df['close'],
            pd.Series(np.asarray(future_pred['close']), index=future_dates),
        ])
        delta = all_close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        all_rsi = 100 - (100 / (1 + rs))
        future_rsi = all_rsi[future_dates]
        future_ma20 = all_close.rolling(20).mean()[future_dates]
        future_ma50 = all_close.rolling(50).mean()[future_dates]
        all_vol = all_close.pct_change().rolling(20).std()
        future_vol = all_vol[future_dates]

        fig_t = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.05,
            subplot_titles=("Harga + rata-rata bergerak", "RSI (14)", "Volatilitas harian (%)"),
        )
        plot_df = featured_df.tail(252)

        fig_t.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df['open'], high=plot_df['high'],
            low=plot_df['low'], close=plot_df['close'], name='Harga asli',
            increasing_line_color=PALETTE["success"], decreasing_line_color=PALETTE["danger"],
        ), row=1, col=1)

        for col_name, color, label in [('ma_20', PALETTE["warning"], 'MA20'),
                                       ('ma_50', PALETTE["primary"], 'MA50')]:
            if col_name in plot_df.columns:
                fig_t.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df[col_name], mode='lines', name=label,
                    line=dict(color=color, width=1.5),
                ), row=1, col=1)

        # Simulasi OHLC masa depan. RNG di-seed agar grafik stabil antar-refresh.
        np.random.seed(abs(hash((R["ticker"], forecast_days))) % (2**32))
        recent_vol = featured_df['close'].pct_change().tail(30).std()
        sim_open, sim_high, sim_low = [], [], []
        prev_c = last_close
        for c in np.asarray(future_pred['close']):
            o = prev_c + prev_c * np.random.uniform(-recent_vol * 0.4, recent_vol * 0.4)
            h = max(o, c) + c * np.random.uniform(0.001, recent_vol * 1.5)
            l = min(o, c) - c * np.random.uniform(0.001, recent_vol * 1.5)
            sim_open.append(o); sim_high.append(h); sim_low.append(l)
            prev_c = c

        fig_t.add_trace(go.Candlestick(
            x=future_dates, open=sim_open, high=sim_high, low=sim_low,
            close=np.asarray(future_pred['close']), name='Prediksi',
            increasing_line_color='rgba(16,185,129,0.85)',
            decreasing_line_color='rgba(239,68,68,0.85)',
        ), row=1, col=1)

        last_ma20 = featured_df['ma_20'].iloc[-1]
        fig_t.add_trace(go.Scatter(
            x=[last_date] + list(future_dates), y=[last_ma20] + list(future_ma20),
            mode='lines', name='MA20 proyeksi',
            line=dict(color=PALETTE["warning"], width=1.5, dash='dot'),
        ), row=1, col=1)
        last_ma50 = featured_df['ma_50'].iloc[-1]
        fig_t.add_trace(go.Scatter(
            x=[last_date] + list(future_dates), y=[last_ma50] + list(future_ma50),
            mode='lines', name='MA50 proyeksi',
            line=dict(color=PALETTE["primary"], width=1.5, dash='dot'),
        ), row=1, col=1)

        if 'rsi' in plot_df.columns:
            fig_t.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df['rsi'], mode='lines', name='RSI',
                line=dict(color=PALETTE["purple"], width=1.5),
            ), row=2, col=1)
            fig_t.add_hline(y=70, line_dash='dash', line_color=PALETTE["danger"], row=2, col=1)
            fig_t.add_hline(y=30, line_dash='dash', line_color=PALETTE["success"], row=2, col=1)
        last_rsi = featured_df['rsi'].iloc[-1]
        fig_t.add_trace(go.Scatter(
            x=[last_date] + list(future_dates), y=[last_rsi] + list(future_rsi),
            mode='lines', name='RSI proyeksi',
            line=dict(color=PALETTE["success"], width=2, dash='dot'),
        ), row=2, col=1)

        if 'volatility' in plot_df.columns:
            fig_t.add_trace(go.Bar(
                x=plot_df.index, y=plot_df['volatility'] * 100, name='Volatilitas',
                marker_color=PALETTE["warning"], opacity=0.7,
            ), row=3, col=1)
        fig_t.add_trace(go.Bar(
            x=future_dates, y=future_vol * 100, name='Volatilitas proyeksi',
            marker_color=PALETTE["success"], opacity=0.4,
        ), row=3, col=1)

        fig_t.add_vline(x=last_date.timestamp() * 1000, line_dash="dash",
                        line_color=PALETTE["warning"], line_width=1.5, row='all', col=1)
        fig_t.update_layout(
            height=680, showlegend=True, legend=dict(orientation='h', y=-0.08),
            font=dict(family='Inter', size=11), xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_t, use_container_width=True, theme="streamlit")

        cexp = st.columns(3)
        with cexp[0]:
            st.markdown("**Rata-rata bergerak (MA)**")
            st.caption(
                "Harga rata-rata 20 atau 50 hari terakhir. Dipakai untuk melihat arah "
                "tren tanpa terganggu naik-turun harian. Harga di atas MA biasanya "
                "dianggap tren naik."
            )
        with cexp[1]:
            st.markdown("**RSI (14)**")
            st.caption(
                "Mengukur apakah saham 'kemahalan' (di atas 70, jenuh beli) atau "
                "'kemurahan' (di bawah 30, jenuh jual). Garis merah dan hijau adalah "
                "batas tersebut."
            )
        with cexp[2]:
            st.markdown("**Volatilitas**")
            st.caption(
                "Seberapa liar harga bergerak dalam 20 hari terakhir. Makin tinggi "
                "batangnya, makin besar risiko dan ketidakpastian."
            )

    # ───────────────────────── TAB 3: INTERPRETASI MODEL ─────────────────────────
    with tab_int:
        st.markdown(
            "Model TFT bisa menunjukkan fitur mana yang paling memengaruhi prediksinya. "
            "Ini membantu kita memahami *alasan* di balik angka, bukan sekadar menerima hasilnya."
        )
        vi = attn_weights.get("variable_importance", {}) if isinstance(attn_weights, dict) else {}
        if vi:
            names = list(vi.keys())
            vals = list(vi.values())
            order = np.argsort(vals)[::-1]
            fig_v = go.Figure(go.Bar(
                x=[vals[i] for i in order], y=[names[i] for i in order],
                orientation='h',
                marker=dict(
                    color=[vals[i] for i in order],
                    colorscale=[[0, PALETTE["primary2"]], [1, PALETTE["primary"]]],
                    showscale=False,
                ),
            ))
            fig_v.update_layout(
                title=dict(text="Pengaruh tiap fitur terhadap prediksi", font=dict(size=13), x=0),
                height=420, margin=dict(l=10, r=20, t=40, b=10),
                font=dict(family='Inter', size=11),
            )
            st.plotly_chart(fig_v, use_container_width=True, theme="streamlit")
            st.caption(
                "**Cara membaca:** batang lebih panjang berarti fitur tersebut lebih "
                "berpengaruh pada prediksi. Contoh: `close_lag_1` berarti harga penutupan "
                "kemarin; `rsi` adalah indikator jenuh beli/jual; `ma_20` adalah rata-rata "
                "20 hari. Wajar bila harga-harga terbaru paling berpengaruh."
            )
        else:
            st.info("Data interpretasi tidak tersedia untuk hasil ini.")

    # ───────────────────────── TAB 4: DATA & TABEL ─────────────────────────
    with tab_data:
        st.markdown("##### Tabel prediksi harian")
        forecast_df = pd.DataFrame({
            'Tanggal': future_dates.strftime('%Y-%m-%d'),
            'Prediksi': [money(p) for p in future_pred['close']],
            'Batas bawah': [money(p) for p in future_pred['close_lower']],
            'Batas atas': [money(p) for p in future_pred['close_upper']],
            'Perubahan': [f"{((p - last_close)/last_close*100):+.2f}%" for p in future_pred['close']],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.caption(
            "Batas bawah dan atas berasal dari kuantil 10% dan 90%. Artinya: model "
            "memperkirakan harga punya peluang besar berada di antara kedua angka itu, "
            "tetapi tidak menjamin."
        )

        st.markdown("##### Ringkasan teknis")
        st.json({
            "ticker": R["ticker"],
            "horizon_hari": R["forecast_days"],
            "jumlah_baris_data": R["rows"],
            "jumlah_fitur": R["n_features"],
            "rentang_tanggal": f"{R['date_min']} s/d {R['date_max']}",
            "epoch": R["epochs"],
            "learning_rate": R["lr"],
            "dari_cache": R["from_cache"],
        })

    # Disclaimer di bawah semua tab
    st.markdown(
        f"""<div style="margin-top:1.6rem; padding:1rem 1.4rem;
        background-color:var(--secondary-background-color);
        border:1px solid {PALETTE['warning']}40; border-radius:12px; font-size:0.85rem;">
        ⚠️ <b>Penting.</b> Prediksi ini untuk edukasi dan riset, bukan saran investasi.
        Harga saham dipengaruhi banyak faktor di luar data historis (berita, kebijakan,
        kondisi global) yang tidak diketahui model. Pergerakan harga harian sangat dekat
        dengan acak, sehingga tidak ada model yang bisa menjaminnya. Selalu lakukan riset
        mandiri dan konsultasi dengan profesional sebelum mengambil keputusan keuangan.
        </div>""",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# ALUR UTAMA
# ════════════════════════════════════════════════════════════════════════════
if run_button:
    result = run_pipeline(ticker, forecast_days, epochs, learning_rate, use_cache,
                          preset_name, signal_threshold, panel_mode)
    if result is not None:
        # Simpan ke session_state agar hasil tidak hilang saat halaman dimuat ulang
        # (misalnya ketika pengguna membuka tab atau mengubah widget lain).
        st.session_state["results"] = result

if "results" in st.session_state:
    render_results(st.session_state["results"])
else:
    # Tampilan awal / onboarding
    st.markdown(
        """<div style="text-align:center; padding:2.5rem 1rem;">
            <div style="font-size:3rem;">📈</div>
            <h2 style="font-weight:700; margin:0.4rem 0;">Siap menganalisis</h2>
            <p style="opacity:0.7; max-width:520px; margin:0 auto;">
                Pilih bank dan jumlah hari prediksi di panel kiri, lalu tekan
                <b>Jalankan Analisis</b>. Aplikasi akan mengambil data, melatih model,
                dan menampilkan hasilnya dengan penjelasan.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
    g1, g2, g3 = st.columns(3)
    for col, num, title, desc in [
        (g1, "1", "Pilih saham", "Lima bank terbesar Indonesia tersedia di panel kiri."),
        (g2, "2", "Atur prediksi", "Tentukan berapa hari ke depan dan mode pelatihan."),
        (g3, "3", "Baca hasil", "Grafik dan metrik lengkap dengan penjelasan untuk pemula."),
    ]:
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-size:1.3rem;font-weight:800;color:{PALETTE['primary']};'>{num}</div>"
                    f"<div style='font-weight:600;margin:0.2rem 0;'>{title}</div>"
                    f"<div style='font-size:0.84rem;opacity:0.7;'>{desc}</div>",
                    unsafe_allow_html=True,
                )
