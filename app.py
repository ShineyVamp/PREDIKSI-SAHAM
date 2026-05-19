import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import
from data_acquisition import fetch_raw_stock_data
from preprocessing_data import preprocess_stock_data
from feature_engineering import build_features
from model import TFTModel
from evaluation import evaluate_predictions
from utils import load_model_cache, save_model_cache

# konfigurasi halaman
st.set_page_config(
    page_title="TFT Bank Indonesia Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root styling menggunakan variabel bawaan Streamlit */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header & Text */
    .main-header {
        font-weight: 700;
        font-size: 1.8rem;
        color: var(--text-color);
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-weight: 400;
        font-size: 0.95rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 2rem;
    }

    /* Card Adaptif */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Section Title Minimalis */
    .section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #3b82f6; /* Aksen biru tetap konsisten */
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
    }
    .section-title::after {
        content: "";
        flex: 1;
        margin-left: 10px;
        height: 1px;
        background-color: rgba(128, 128, 128, 0.2);
    }

    /* Tombol Modern */
    .stButton button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Box Informasi Adaptif */
    .info-box {
        background-color: var(--secondary-background-color);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: var(--text-color);
    }

    /* Sidebar minimalis */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Bagian Sidebar
BANK_OPTIONS = {
    "BCA (BBCA.JK)":  "BBCA.JK",
    "BRI (BBRI.JK)":  "BBRI.JK",
    "Mandiri (BMRI.JK)": "BMRI.JK",
    "BNI (BBNI.JK)":  "BBNI.JK",
    "BSI (BRIS.JK)":  "BRIS.JK",
}

with st.sidebar:
    st.markdown('<div class="main-header" style="font-size:1.3rem">Prediksi Saham dengan TFT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Temporal Fusion Transformer for<br>Top 5 Indonesian Bank Prediction Engine</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-title">Pilih Saham</div>', unsafe_allow_html=True)
    selected_bank_label = st.selectbox(
        "Bank", list(BANK_OPTIONS.keys()), label_visibility="collapsed"
    )
    ticker = BANK_OPTIONS[selected_bank_label]

    st.markdown('<div class="section-title">Horizon Prediksi</div>', unsafe_allow_html=True)
    forecast_days = st.slider("Hari ke depan", min_value=7, max_value=30, value=30, step=7)

    st.markdown('<div class="section-title">Konfigurasi Model</div>', unsafe_allow_html=True)
    max_epochs = st.slider("Max Epochs", 1, 100, 50, 10)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0003, 0.001, 0.003, 0.01],
        value=0.001
    )
    use_cache = st.checkbox("Gunakan model tersimpan (jika ada)", value=True)

    st.divider()
    run_button = st.button("Jalankan Analisis", use_container_width=True)
    
    st.markdown("""
    <div style="margin-top:2rem; font-family:'Inter',sans-serif; font-size:0.72rem; color:var(--text-color); opacity:0.8; line-height:1.6;">
    <b>Model:</b> Temporal Fusion Transformer<br>
    <b>Strategi:</b> MIMO (Multi-Input Multi-Output)<br>
    <b>Data:</b> 10 Tahun historis via yfinance<br>
    </div>
    """, unsafe_allow_html=True)
    
# HEADER UTAMA
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">STOCK PREDICTION ENGINE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{ticker} · {forecast_days}-Day Horizon</div>', unsafe_allow_html=True)

# Konfigurasi proses utama
if run_button:

    # 1.DATA ACQUISITION
    st.markdown('<div class="section-title">01 · Data Acquisition</div>', unsafe_allow_html=True)
    with st.spinner(f"Mengambil data historis {ticker} dari Yahoo Finance"):
        try:
            raw_df = fetch_raw_stock_data(ticker, period_years=10)
            st.markdown(f'<div class="info-box">Berhasil mengambil <b>{len(raw_df):,}</b> baris data historis untuk <b>{ticker}</b> ({raw_df.index.min().date()} → {raw_df.index.max().date()})</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gagal mengambil data: {e}")
            st.stop()

    # 2.PRE PROCESSING
    st.markdown('<div class="section-title">02 · Pre-processing Data</div>', unsafe_allow_html=True)
    with st.spinner(f"Memproses data historis {ticker}..."):
        try:
            cleaned_df = preprocess_stock_data(raw_df)
            st.markdown(f'<div class="info-box">Data berhasil melalu proses pre processing</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gagal preprocessing data: {e}")
            st.stop()
    
    # 2.FEATURE ENGINEERING
    st.markdown('<div class="section-title">03 · Feature Engineering</div>', unsafe_allow_html=True)
    with st.spinner("Membangun fitur teknis dan temporal..."):
        try:
            featured_df = build_features(cleaned_df, ticker)
            n_features = len(featured_df.columns)
            st.markdown(f'<div class="info-box">Feature engineering selesai: <b>{n_features}</b> fitur dibuat (teknis + temporal + static covariates)</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Feature engineering gagal: {e}")
            st.stop()

    # 3.MODEL TRAINING
    st.markdown('<div class="section-title">03 · Model Training (TFT · MIMO)</div>', unsafe_allow_html=True)

    cache_key = f"{ticker}_{forecast_days}_{max_epochs}_{learning_rate}_v3"
    cached = load_model_cache(cache_key) if use_cache else None

    if cached:
        st.markdown('<div class="info-box">Model dimuat dari cache — melewati training.</div>', unsafe_allow_html=True)
        tft_model    = cached["model"]
        predictions  = cached["predictions"]
        actuals      = cached["actuals"]
        attn_weights = cached["attention"]
        future_pred  = cached["future"]
    else:
        progress_bar = st.progress(0, text="Inisialisasi model TFT...")
        status_box   = st.empty()

        def _progress_callback(epoch, total, loss):
            pct = int((epoch / total) * 100)
            progress_bar.progress(pct, text=f"Epoch {epoch}/{total} — Loss: {loss:.4f}")
            status_box.markdown(f'<div class="info-box" style="padding:0.4rem 0.8rem; font-size:0.78rem;">Training... Epoch <b>{epoch}/{total}</b> · Train Loss: <b>{loss:.4f}</b></div>', unsafe_allow_html=True)

        try:
            tft_model = TFTModel(
                ticker=ticker,
                forecast_horizon=forecast_days,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
            )
            predictions, actuals, attn_weights, future_pred = tft_model.fit_predict(
                featured_df,
                progress_callback=_progress_callback
            )
            progress_bar.progress(100, text="Training selesai")
            status_box.empty()

            save_model_cache(cache_key, {
                "model": tft_model, "predictions": predictions,
                "actuals": actuals,  "attention": attn_weights,
                "future": future_pred
            })
        except Exception as e:
            st.error(f"Training gagal: {e}")
            import traceback; st.code(traceback.format_exc())
            st.stop()
            
     # 4.EVALUATION METRICS
    st.markdown('<div class="section-title">04 · Evaluation Metrics</div>', unsafe_allow_html=True)
    
    act_close = np.array(actuals).flatten()
    pred_close = np.array(predictions).flatten()

    metrics = evaluate_predictions(act_close, pred_close)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
    with col2:
        st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
    with col3:
        st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
    with col4:
        st.metric("Directional Accuracy", f"{metrics.get('Directional_Accuracy', 0):.2f}%")

    # 5.CHART AKTUAL vs PREDIKSI
    st.markdown('<div class="section-title">05 · Aktual vs Prediksi (Backtest)</div>', unsafe_allow_html=True)

    last_date = featured_df.index[-1]
    last_close = featured_df['close'].iloc[-1]

    dates_hist = featured_df.index[-len(actuals):]    

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(future_pred['close']), freq='B'
    )

    fig_main = go.Figure()

    # Harga aktual historis (full)
    fig_main.add_trace(go.Scatter(
        x=featured_df.index[-505:],
        y=featured_df['close'][-505:],
        mode='lines', name='Aktual (Historis)',
        line=dict(color='#64748b', width=1.5),
        opacity=0.6
    ))
    # Prediksi backtest
    fig_main.add_trace(go.Scatter(
        x=dates_hist, y=predictions,
        mode='lines', name='Prediksi (Backtest)',
        line=dict(color='#3b82f6', width=2, dash='dot')
    ))
    # Prediksi masa depan
    fig_main.add_trace(go.Scatter(
        x=[last_date] + list(future_dates),
        y=[last_close] + list(future_pred['close']),
        mode='lines', name=f'Forecast {forecast_days}H',
        line=dict(color='#00ff88', width=2.5)
    ))
    # Area bayangan uncertainty (DIAMBIL LANGSUNG DARI QUANTILE LOSS)
    upper = list(future_pred['close_upper'])
    lower = list(future_pred['close_lower'])
    fig_main.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=upper + lower[::-1],
        fill='toself', fillcolor='rgba(0,255,136,0.06)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band (±5%)',
        showlegend=True
    ))
    # Garis vertikal pemisah
    fig_main.add_vline(
        x=featured_df.index[-1].timestamp() * 1000, line_dash="dash",
        line_color="#f59e0b", line_width=1.5,
        annotation_text="  Today", annotation_font_color="#f59e0b"
    )

    # PERBAIKAN: Menambahkan update_layout dan menghapus warna hardcode
    fig_main.update_layout(
        height=480,
        font=dict(family='Inter'), 
        legend=dict(
            orientation='h', y=-0.15
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode='x unified'
    )
    # PERBAIKAN: Menambahkan theme="streamlit"
    st.plotly_chart(fig_main, use_container_width=True, theme="streamlit")

    # 6.TABEL PREDIKSI MASA DEPAN
    with st.expander("Detail Tabel Forecast", expanded=False):
        forecast_df = pd.DataFrame({
            'Tanggal':   future_dates.strftime('%Y-%m-%d'),
            'Prediksi (IDR)': [f"Rp {p:,.0f}" for p in future_pred['close']],
            'Batas Atas': [f"Rp {p:,.0f}" for p in future_pred['close_upper']], 
            'Batas Bawah': [f"Rp {p:,.0f}" for p in future_pred['close_lower']], 
            'Change (%)': [f"{((p - last_close)/last_close*100):+.2f}%" for p in future_pred['close']],
        })
        st.dataframe(
            forecast_df, use_container_width=True, hide_index=True,
            column_config={
                "Prediksi (IDR)": st.column_config.TextColumn(width="medium"),
            }
        )

    # 7. ATTENTION WEIGHTS VISUALISASI
    st.markdown(
        '<div class="section-title">06 · Interpretabilitas — Feature Importance</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="info-box">'
        'Visualisasi di bawah menunjukkan fitur mana yang paling berpengaruh '
        'pada prediksi model TFT. Semakin tinggi bar, semakin besar kontribusi fitur tersebut.'
        '</div>',
        unsafe_allow_html=True
    )
    
    feat_names = list(attn_weights["variable_importance"].keys())
    feat_vals  = list(attn_weights["variable_importance"].values())
    
    sorted_idx = np.argsort(feat_vals)[::-1]
    
    fig_att = go.Figure(go.Bar(
        x=[feat_vals[i] for i in sorted_idx],
        y=[feat_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=[feat_vals[i] for i in sorted_idx],
            colorscale=[
                [0,   '#1e3a5f'],
                [0.5, '#3b82f6'],
                [1,   '#00ff88']
            ],
            showscale=False
        )
    ))
    
    # PERBAIKAN: Menambahkan update_layout dan menghapus warna hardcode
    fig_att.update_layout(
        title=dict(
            text="Variable Importance Score",
            font=dict(size=13),
            x=0
        ),
        height=420,
        margin=dict(l=10, r=20, t=40, b=10),
        font=dict(
            family='Inter',
            size=11
        )
    )
    
    # PERBAIKAN: Menambahkan theme="streamlit"
    st.plotly_chart(fig_att, use_container_width=True, theme="streamlit")

    # 8.ANALISIS TEKNIKAL DENGAN PROYEKSI MASA DEPAN
    st.markdown('<div class="section-title">07 · Analisis Teknikal</div>', unsafe_allow_html=True)
    
    # Hitung Proyeksi RSI dan Volatilitas Masa Depan
    all_close = pd.concat([featured_df['close'], pd.Series(future_pred['close'], index=future_dates)]) 
    
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
    all_return = all_close.pct_change()
    all_vol = all_return.rolling(20).std()
    future_vol = all_vol[future_dates]

    fig_tech = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=("Harga + Moving Averages", "RSI (14)", "Volatilitas Harian (%)")
    )
    plot_df = featured_df.tail(252)

    # --- ROW 1: HARGA HISTORIS (OHLC) ---
    fig_tech.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['open'],  high=plot_df['high'],
        low=plot_df['low'],    close=plot_df['close'],
        name='OHLC Aktual', increasing_line_color='#00ff88', decreasing_line_color='#ef4444'
    ), row=1, col=1)

    for col_name, color, label in [('ma_20','#f59e0b','MA20'), ('ma_50','#3b82f6','MA50')]:
        if col_name in plot_df.columns:
            fig_tech.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[col_name],
                mode='lines', name=label,
                line=dict(color=color, width=1.5)
            ), row=1, col=1)
            
    recent_vol = featured_df['close'].pct_change().tail(30).std()
    
    sim_open, sim_high, sim_low = [], [], []
    prev_c = last_close
    
    for c in future_pred['close']:
        gap_noise = prev_c * np.random.uniform(-recent_vol * 0.4, recent_vol * 0.4)
        o = prev_c + gap_noise
        wick_h = c * np.random.uniform(0.001, recent_vol * 1.5)
        wick_l = c * np.random.uniform(0.001, recent_vol * 1.5)
        h = max(o, c) + wick_h
        l = min(o, c) - wick_l
        sim_open.append(o)
        sim_high.append(h)
        sim_low.append(l)
        prev_c = c

    # PROYEKSI ROW 1: CANDLESTICK MASA DEPAN
    fig_tech.add_trace(go.Candlestick(
        x=future_dates,
        open=sim_open,
        high=sim_high,
        low=sim_low,
        close=future_pred['close'],
        name='Forecast OHLC', 
        increasing_line_color='rgba(0, 255, 136, 0.85)', 
        decreasing_line_color='rgba(239, 68, 68, 0.85)'  
    ), row=1, col=1)

    # PROYEKSI ROW 1: MA MASA DEPAN
    last_ma20 = featured_df['ma_20'].iloc[-1]
    fig_tech.add_trace(go.Scatter(
        x=[last_date] + list(future_dates),
        y=[last_ma20] + list(future_ma20),
        mode='lines', name='Forecast MA20',
        line=dict(color='#f59e0b', width=1.5, dash='dot')
    ), row=1, col=1)

    last_ma50 = featured_df['ma_50'].iloc[-1]
    fig_tech.add_trace(go.Scatter(
        x=[last_date] + list(future_dates),
        y=[last_ma50] + list(future_ma50),
        mode='lines', name='Forecast MA50',
        line=dict(color='#3b82f6', width=1.5, dash='dot')
    ), row=1, col=1)

    # --- ROW 2: RSI HISTORIS ---
    if 'rsi' in plot_df.columns:
        fig_tech.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['rsi'],
            mode='lines', name='RSI Aktual',
            line=dict(color='#a78bfa', width=1.5)
        ), row=2, col=1)
        fig_tech.add_hline(y=70, line_dash='dash', line_color='#ef4444', row=2, col=1)
        fig_tech.add_hline(y=30, line_dash='dash', line_color='#00ff88', row=2, col=1)

    # PROYEKSI ROW 2: GARIS RSI MASA DEPAN
    last_rsi = featured_df['rsi'].iloc[-1]
    fig_tech.add_trace(go.Scatter(
        x=[last_date] + list(future_dates),
        y=[last_rsi] + list(future_rsi),
        mode='lines', name='Forecast RSI',
        line=dict(color='#00ff88', width=2, dash='dot')
    ), row=2, col=1)

    # --- ROW 3: VOLATILITAS HISTORIS ---
    if 'volatility' in plot_df.columns:
        fig_tech.add_trace(go.Bar(
            x=plot_df.index, y=plot_df['volatility'] * 100,
            name='Volatilitas Aktual', marker_color='#f59e0b', opacity=0.7
        ), row=3, col=1)

    # PROYEKSI ROW 3: BAR VOLATILITAS MASA DEPAN
    fig_tech.add_trace(go.Bar(
        x=future_dates, y=future_vol * 100,
        name='Forecast Volatilitas', marker_color='#00ff88', opacity=0.4
    ), row=3, col=1)

    fig_tech.add_vline(x=last_date.timestamp() * 1000, line_dash="dash", line_color="#f59e0b", line_width=1.5, row='all', col=1)

    # PERBAIKAN: Menambahkan update_layout dan menghapus warna hardcode
    fig_tech.update_layout(
        height=640, showlegend=True,
        legend=dict(orientation='h', y=-0.08),
        font=dict(family='Inter', size=11),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # PERBAIKAN: Menambahkan theme="streamlit"
    st.plotly_chart(fig_tech, use_container_width=True, theme="streamlit")

    # PERBAIKAN: Menyesuaikan Disclaimer agar adaptif
    st.markdown(f"""
    <div style="margin-top:2rem; padding:1rem 1.5rem; background-color:var(--secondary-background-color); border:1px solid rgba(128,128,128,0.2);
                border-radius:12px; font-family:'Inter',sans-serif; font-size:0.85rem; color:var(--text-color);">
    ⚠️ <b>Disclaimer:</b> Prediksi ini dibuat untuk tujuan edukasi dan riset.
    Bukan merupakan saran investasi. Selalu lakukan riset mandiri sebelum berinvestasi.
    Model TFT dilatih pada data historis dan tidak menjamin hasil di masa depan.
    </div>
    """, unsafe_allow_html=True)

# Tampilan awal
else:
    # PERBAIKAN: Tampilan awal dibuat adaptif
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; min-height:60vh; flex-direction:column; gap:1rem;">
        <div style="font-size:3rem;">📈</div>
        <h2 style="color:var(--text-color); font-weight:600; font-family:'Inter', sans-serif;">Siap untuk Menganalisis?</h2>
        <p style="color:var(--text-color); opacity:0.7; text-align:center; max-width:400px; font-family:'Inter', sans-serif;">
            Pilih parameter saham di sidebar dan klik tombol <b>Jalankan Analisis</b> untuk memulai prediksi berbasis Deep Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)
