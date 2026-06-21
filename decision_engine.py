"""
decision_engine.py
══════════════════
Lapisan Decision Support System (DSS).

Prediktor harga saja bukan DSS. Modul ini melengkapi:
  1. Aturan keputusan eksplisit (Beli / Tahan / Hindari) dari ramalan + ketidakpastian.
  2. Model biaya transaksi Bursa Efek Indonesia (fee beli, fee jual, pajak).
  3. Backtest strategi berbasis model, dibandingkan dengan buy-and-hold.
  4. Metrik risiko: Sharpe ratio, maximum drawdown, hit rate.

Semua output bersifat edukatif, bukan saran investasi. Harga saham harian
mendekati acak, jadi sistem ini sengaja konservatif dan jujur soal batasnya.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# ── Biaya transaksi default (dapat disesuaikan per broker) ──────────────────
# Beli  : fee broker ~0.15%
# Jual  : fee broker ~0.25% (sudah termasuk PPh final 0.1% + levy bursa)
# Round-trip efektif ~0.4%. Ini konservatif untuk saham bank likuid.
DEFAULT_BUY_FEE = 0.0015
DEFAULT_SELL_FEE = 0.0025
TRADING_DAYS_PER_YEAR = 252


def round_trip_cost(buy_fee=DEFAULT_BUY_FEE, sell_fee=DEFAULT_SELL_FEE) -> float:
    """Total biaya beli lalu jual, dalam fraksi (mis. 0.004 = 0.4%)."""
    return buy_fee + sell_fee


# ══════════════════════════════════════════════════════════════════════════════
# 1. ATURAN KEPUTUSAN
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Decision:
    signal: str            # "Beli", "Tahan", "Hindari/Jual"
    strength: str          # "kuat", "lemah/spekulatif", "netral"
    expected_return_pct: float
    net_return_pct: float          # setelah biaya round-trip
    cost_pct: float
    confidence: str        # "tinggi", "sedang", "rendah"
    interval_crosses_zero: bool
    horizon_days: int
    rationale: str

    def to_dict(self):
        return asdict(self)


def generate_signal(last_close: float,
                    forecast_close_end: float,
                    lower_end: float,
                    upper_end: float,
                    horizon_days: int,
                    threshold_pct: float = 1.0,
                    buy_fee: float = DEFAULT_BUY_FEE,
                    sell_fee: float = DEFAULT_SELL_FEE) -> Decision:
    """
    Hasilkan sinyal keputusan dari ramalan akhir horizon + rentang keyakinan.

    threshold_pct: ambang return bersih minimal (persen) agar layak masuk.
                   Default 1% di atas biaya. Naikkan untuk lebih konservatif.
    """
    cost = round_trip_cost(buy_fee, sell_fee) * 100  # ke persen
    exp_ret = (forecast_close_end - last_close) / last_close * 100
    net_ret = exp_ret - cost

    # Ketidakpastian dari lebar rentang relatif terhadap harga ramalan.
    rel_width = (upper_end - lower_end) / forecast_close_end if forecast_close_end else 1.0
    if rel_width < 0.08:
        confidence = "tinggi"
    elif rel_width < 0.18:
        confidence = "sedang"
    else:
        confidence = "rendah"

    crosses_zero = (lower_end <= last_close <= upper_end)

    # Logika sinyal
    if net_ret >= threshold_pct and not crosses_zero and exp_ret > 0:
        signal, strength = "Beli", "kuat"
        rationale = (f"Return bersih perkiraan +{net_ret:.2f}% melebihi ambang "
                     f"{threshold_pct:.1f}%, dan batas bawah rentang masih di atas "
                     f"harga sekarang. Tetap waspada, ini perkiraan.")
    elif net_ret >= threshold_pct and crosses_zero:
        signal, strength = "Beli", "lemah/spekulatif"
        rationale = (f"Return bersih +{net_ret:.2f}% terlihat menarik, tapi rentang "
                     f"keyakinan masih melewati harga sekarang. Peluang rugi nyata.")
    elif net_ret <= -threshold_pct:
        signal, strength = "Hindari/Jual", "kuat"
        rationale = (f"Return bersih perkiraan {net_ret:.2f}% negatif setelah biaya. "
                     f"Model condong menurun untuk horizon ini.")
    else:
        signal, strength = "Tahan", "netral"
        rationale = (f"Return bersih perkiraan {net_ret:.2f}% terlalu kecil untuk "
                     f"menutup biaya {cost:.2f}%. Tidak ada keunggulan jelas. "
                     f"Menahan atau tidak masuk lebih masuk akal.")

    return Decision(
        signal=signal, strength=strength,
        expected_return_pct=round(exp_ret, 3),
        net_return_pct=round(net_ret, 3),
        cost_pct=round(cost, 3),
        confidence=confidence,
        interval_crosses_zero=bool(crosses_zero),
        horizon_days=int(horizon_days),
        rationale=rationale,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. METRIK RISIKO
# ══════════════════════════════════════════════════════════════════════════════
def sharpe_ratio(period_returns: np.ndarray, periods_per_year: float) -> float:
    """Sharpe disetahunkan. Risk-free diasumsikan 0 untuk kesederhanaan."""
    r = np.asarray(period_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Penurunan terdalam dari puncak ke lembah, sebagai fraksi negatif."""
    e = np.asarray(equity_curve, dtype=np.float64)
    if len(e) == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / peak
    return float(np.min(dd))


# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST STRATEGI BERBASIS MODEL (vs BUY-AND-HOLD)
# ══════════════════════════════════════════════════════════════════════════════
def backtest_strategy(preds_matrix: np.ndarray,
                      actuals_matrix: np.ndarray,
                      anchors: np.ndarray,
                      horizon_days: int,
                      threshold_pct: float = 1.0,
                      buy_fee: float = DEFAULT_BUY_FEE,
                      sell_fee: float = DEFAULT_SELL_FEE) -> dict:
    """
    Uji strategi long-only berbasis model pada jendela uji.

    Untuk tiap jendela non-tumpang-tindih:
      · predicted_return = (ramalan akhir - anchor) / anchor
      · jika predicted_return - biaya > ambang  -> masuk posisi, tahan H hari
      · realisasi = actual_return - biaya (hanya saat bertransaksi)
    Lalu bandingkan dengan buy-and-hold pada jendela yang sama.

    Jendela diambil setiap H hari agar tidak tumpang tindih (anti bias).
    """
    P = np.asarray(preds_matrix, dtype=np.float64)
    A = np.asarray(actuals_matrix, dtype=np.float64)
    anch = np.asarray(anchors, dtype=np.float64).flatten()
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if A.ndim == 1:
        A = A.reshape(-1, 1)

    n = min(len(P), len(A), len(anch))
    if n == 0 or horizon_days < 1:
        return _empty_backtest()

    step = max(int(horizon_days), 1)
    idx = list(range(0, n, step))
    cost = round_trip_cost(buy_fee, sell_fee)
    thr = threshold_pct / 100.0

    strat_rets, bh_rets, took_trade, wins = [], [], 0, 0
    for i in idx:
        anchor = anch[i]
        if not np.isfinite(anchor) or anchor <= 0:
            continue
        pred_end = P[i, -1]
        act_end = A[i, -1]
        if not (np.isfinite(pred_end) and np.isfinite(act_end)):
            continue

        pred_ret = (pred_end - anchor) / anchor
        act_ret = (act_end - anchor) / anchor
        bh_rets.append(act_ret)  # buy-and-hold selalu memegang

        if pred_ret - cost > thr:           # sinyal masuk
            realized = act_ret - cost
            took_trade += 1
            if realized > 0:
                wins += 1
        else:
            realized = 0.0                  # tidak masuk, return 0
        strat_rets.append(realized)

    if len(strat_rets) == 0:
        return _empty_backtest()

    strat_rets = np.array(strat_rets)
    bh_rets = np.array(bh_rets)

    strat_equity = np.cumprod(1 + strat_rets)
    bh_equity = np.cumprod(1 + bh_rets)
    periods_per_year = TRADING_DAYS_PER_YEAR / step

    return {
        "n_windows": len(strat_rets),
        "n_trades": int(took_trade),
        "hit_rate": float(wins / took_trade) if took_trade else 0.0,
        "strategy_total_return_pct": float((strat_equity[-1] - 1) * 100),
        "buyhold_total_return_pct": float((bh_equity[-1] - 1) * 100),
        "strategy_sharpe": sharpe_ratio(strat_rets, periods_per_year),
        "buyhold_sharpe": sharpe_ratio(bh_rets, periods_per_year),
        "strategy_max_drawdown_pct": float(max_drawdown(strat_equity) * 100),
        "buyhold_max_drawdown_pct": float(max_drawdown(bh_equity) * 100),
        "beats_buyhold": bool(strat_equity[-1] > bh_equity[-1]),
        "avg_trade_return_pct": float(np.mean(strat_rets[strat_rets != 0]) * 100) if np.any(strat_rets != 0) else 0.0,
        "strategy_equity": strat_equity.tolist(),
        "buyhold_equity": bh_equity.tolist(),
        "cost_per_roundtrip_pct": float(cost * 100),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. UKURAN POSISI BERBASIS VOLATILITAS (saran sederhana)
# ══════════════════════════════════════════════════════════════════════════════
def position_size_suggestion(realized_daily_vol: float,
                             target_annual_vol: float = 0.20) -> dict:
    """
    Saran bobot posisi agar volatilitas portofolio mendekati target.
    Bukan perintah, hanya kerangka manajemen risiko.
    """
    if not np.isfinite(realized_daily_vol) or realized_daily_vol <= 0:
        return {"suggested_weight_pct": 0.0, "note": "Volatilitas tidak valid."}
    annual_vol = realized_daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    weight = min(target_annual_vol / annual_vol, 1.0)
    return {
        "suggested_weight_pct": round(weight * 100, 1),
        "annualized_vol_pct": round(annual_vol * 100, 1),
        "note": (f"Volatilitas tahunan ~{annual_vol*100:.0f}%. Untuk target risiko "
                 f"{target_annual_vol*100:.0f}%, batasi posisi sekitar "
                 f"{weight*100:.0f}% modal."),
    }


def _empty_backtest() -> dict:
    return {"n_windows": 0, "n_trades": 0, "hit_rate": 0.0,
            "strategy_total_return_pct": 0.0, "buyhold_total_return_pct": 0.0,
            "strategy_sharpe": 0.0, "buyhold_sharpe": 0.0,
            "strategy_max_drawdown_pct": 0.0, "buyhold_max_drawdown_pct": 0.0,
            "beats_buyhold": False, "avg_trade_return_pct": 0.0,
            "strategy_equity": [], "buyhold_equity": [],
            "cost_per_roundtrip_pct": round(round_trip_cost() * 100, 3)}
