"""
evaluation.py
═════════════
Modul evaluasi model prediksi saham.

Filosofi baru: sebuah angka error tidak berarti apa-apa tanpa pembanding.
Karena itu setiap evaluasi WAJIB membandingkan model dengan baseline naive
(prediksi = harga terakhir / random walk). Kalau model tidak mengalahkan
baseline, sistem harus mengatakannya terang-terangan.

Metrik magnitudo (MAE, RMSE, MAPE) dihitung pada level harga DAN diukur
relatif terhadap baseline. Akurasi arah dan metrik ruang return mengukur
kemampuan memprediksi perubahan, yang jauh lebih relevan untuk keputusan.
"""

import numpy as np
import pandas as pd
from typing import Union

Arr = Union[np.ndarray, list]


# ══════════════════════════════════════════════════════════════════════════════
# METRIK DASAR
# ══════════════════════════════════════════════════════════════════════════════
def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred):
    denom = np.abs(y_true)
    denom = np.where(denom < 1e-8, np.nan, denom)
    val = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return float(val) if np.isfinite(val) else 0.0

def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)

def _directional_accuracy(y_true, y_pred):
    """Berapa sering arah perubahan (naik/turun) ditebak benar."""
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    # Selisih nol (harga datar) tidak boleh dihitung sebagai tebakan benar.
    mask = true_dir != 0
    if mask.sum() == 0:
        return 0.0
    correct = np.sum((true_dir == pred_dir) & mask)
    return float(correct / mask.sum())


def _clean_pair(actuals, predictions):
    y_true = np.asarray(actuals, dtype=np.float64).flatten()
    y_pred = np.asarray(predictions, dtype=np.float64).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def evaluate_predictions(actuals: Arr, predictions: Arr) -> dict:
    """Metrik standar pada satu pasang seri aktual vs prediksi."""
    y_true, y_pred = _clean_pair(actuals, predictions)
    if len(y_true) == 0:
        return _empty_metrics()
    return {
        "MAE": _mae(y_true, y_pred),
        "RMSE": _rmse(y_true, y_pred),
        "MAPE": _mape(y_true, y_pred),
        "R2": _r2(y_true, y_pred),
        "Directional_Accuracy": _directional_accuracy(y_true, y_pred),
        "Max_Error": float(np.max(np.abs(y_true - y_pred))),
        "Median_AE": float(np.median(np.abs(y_true - y_pred))),
        "n_samples": int(len(y_true)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE NAIVE (RANDOM WALK) + SKILL SCORE
# ══════════════════════════════════════════════════════════════════════════════
def naive_last_value(actuals: Arr) -> np.ndarray:
    """Prediksi naive 1-langkah: harga besok = harga hari ini."""
    y = np.asarray(actuals, dtype=np.float64).flatten()
    if len(y) == 0:
        return y
    return np.concatenate([[y[0]], y[:-1]])


def evaluate_with_baseline(actuals: Arr, predictions: Arr) -> dict:
    """
    Bandingkan model dengan baseline random walk pada prediksi 1-langkah.

    Model menebak harga[t]; baseline menebak harga[t-1].
    Keduanya dinilai terhadap harga[t] yang sama, lalu dihitung skill.
    """
    y_true, y_pred = _clean_pair(actuals, predictions)
    if len(y_true) < 3:
        return {"model": _empty_metrics(), "naive": _empty_metrics(),
                "skill": _empty_skill()}

    # Selaraskan pada t = 1..n. Baseline = nilai sebelumnya.
    a = y_true[1:]
    p_model = y_pred[1:]
    p_naive = y_true[:-1]

    model = evaluate_predictions(a, p_model)
    naive = evaluate_predictions(a, p_naive)

    mae_ratio = model["MAE"] / naive["MAE"] if naive["MAE"] > 0 else np.inf
    rmse_ratio = model["RMSE"] / naive["RMSE"] if naive["RMSE"] > 0 else np.inf

    skill = {
        "mae_ratio": float(mae_ratio),
        "mae_skill": float(1 - mae_ratio),       # > 0 berarti mengalahkan naive
        "rmse_skill": float(1 - rmse_ratio),
        "beats_naive": bool(model["MAE"] < naive["MAE"]),
        "da": model["Directional_Accuracy"],
        "da_edge": float(model["Directional_Accuracy"] - 0.5),  # > 0 di atas koin
    }
    return {"model": model, "naive": naive, "skill": skill}


# ══════════════════════════════════════════════════════════════════════════════
# METRIK RUANG RETURN (mengukur prediksi PERUBAHAN, bukan level)
# ══════════════════════════════════════════════════════════════════════════════
def return_space_metrics(actual_prices: Arr, predicted_prices: Arr) -> dict:
    """
    Ubah harga menjadi return harian, lalu nilai prediksinya.
    Baseline di ruang return = menebak return 0 (random walk tanpa drift).
    """
    a = np.asarray(actual_prices, dtype=np.float64).flatten()
    p = np.asarray(predicted_prices, dtype=np.float64).flatten()
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    if n < 3:
        return {"return_MAE_pct": 0.0, "return_RMSE_pct": 0.0,
                "return_DA": 0.0, "beats_zero_return": False, "n_samples": 0}

    r_true = np.diff(a) / a[:-1]
    r_pred = np.diff(p) / np.where(p[:-1] == 0, np.nan, p[:-1])
    m = np.isfinite(r_true) & np.isfinite(r_pred)
    r_true, r_pred = r_true[m], r_pred[m]
    if len(r_true) == 0:
        return {"return_MAE_pct": 0.0, "return_RMSE_pct": 0.0,
                "return_DA": 0.0, "beats_zero_return": False, "n_samples": 0}

    mae_model = np.mean(np.abs(r_true - r_pred))
    mae_zero = np.mean(np.abs(r_true))          # baseline: prediksi return 0
    da = float(np.mean(np.sign(r_true[r_true != 0]) ==
                       np.sign(r_pred[r_true != 0]))) if np.any(r_true != 0) else 0.0

    return {
        "return_MAE_pct": float(mae_model * 100),
        "return_RMSE_pct": float(np.sqrt(np.mean((r_true - r_pred) ** 2)) * 100),
        "return_DA": da,
        "beats_zero_return": bool(mae_model < mae_zero),
        "n_samples": int(len(r_true)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUASI PER HORIZON (hari ke-1 vs ke-30) DENGAN BASELINE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_per_horizon(preds_matrix: np.ndarray,
                         actuals_matrix: np.ndarray,
                         anchors: Arr = None) -> pd.DataFrame:
    """
    Hitung error pada tiap horizon prediksi.

    Args:
        preds_matrix   : shape (n_windows, H) harga close prediksi
        actuals_matrix : shape (n_windows, H) harga close aktual
        anchors        : shape (n_windows,) close terakhir sebelum tiap jendela.
                         Dipakai sebagai baseline naive (asumsi harga datar).

    Returns:
        DataFrame index horizon_day (1..H) dengan MAE/MAPE model, MAE naive,
        dan skill (1 - model/naive). Skill > 0 = model mengalahkan asumsi datar.
    """
    P = np.asarray(preds_matrix, dtype=np.float64)
    A = np.asarray(actuals_matrix, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    n, H = A.shape
    P = P[:, :H] if P.shape[1] >= H else np.pad(P, ((0, 0), (0, H - P.shape[1])), constant_values=np.nan)

    anch = np.asarray(anchors, dtype=np.float64).flatten() if anchors is not None else None

    rows = []
    for h in range(H):
        a_col = A[:, h]
        p_col = P[:, h]
        m = np.isfinite(a_col) & np.isfinite(p_col)
        if m.sum() == 0:
            continue
        a_v, p_v = a_col[m], p_col[m]
        mae_model = _mae(a_v, p_v)
        mape_model = _mape(a_v, p_v)

        if anch is not None and len(anch) == n:
            naive_v = anch[m]
            mae_naive = _mae(a_v, naive_v)
            skill = 1 - mae_model / mae_naive if mae_naive > 0 else 0.0
        else:
            mae_naive = np.nan
            skill = np.nan

        rows.append({
            "horizon_day": h + 1,
            "MAE_model": mae_model,
            "MAPE_model": mape_model,
            "MAE_naive": mae_naive,
            "skill": skill,
            "n": int(m.sum()),
        })

    if not rows:
        return pd.DataFrame(columns=["horizon_day", "MAE_model", "MAPE_model",
                                     "MAE_naive", "skill", "n"]).set_index("horizon_day")
    return pd.DataFrame(rows).set_index("horizon_day")


# ══════════════════════════════════════════════════════════════════════════════
# UTIL
# ══════════════════════════════════════════════════════════════════════════════
def format_metrics_for_display(metrics: dict) -> dict:
    return {
        "MAE": f"Rp {metrics.get('MAE', 0):,.0f}",
        "RMSE": f"Rp {metrics.get('RMSE', 0):,.0f}",
        "MAPE": f"{metrics.get('MAPE', 0):.2f}%",
        "R²": f"{metrics.get('R2', 0):.4f}",
        "Dir. Acc.": f"{metrics.get('Directional_Accuracy', 0)*100:.1f}%",
    }


def _empty_metrics() -> dict:
    return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "R2": 0.0,
            "Directional_Accuracy": 0.0, "Max_Error": 0.0,
            "Median_AE": 0.0, "n_samples": 0}


def _empty_skill() -> dict:
    return {"mae_ratio": float("inf"), "mae_skill": 0.0, "rmse_skill": 0.0,
            "beats_naive": False, "da": 0.0, "da_edge": 0.0}
