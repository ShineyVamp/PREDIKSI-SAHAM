"""
evaluation.py
═════════════
Modul evaluasi model prediksi saham.
Menyediakan fungsi perhitungan metrik standar:
  · MAE  (Mean Absolute Error)
  · RMSE (Root Mean Squared Error)
  · MAPE (Mean Absolute Percentage Error)
  · R²   (Coefficient of Determination)
  · Directional Accuracy (DA) — seberapa sering arah prediksi benar
"""

import numpy as np
import pandas as pd
from typing import Union


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_predictions(
    actuals: Union[np.ndarray, list],
    predictions: Union[np.ndarray, list],
) -> dict:
    """
    Menghitung semua metrik evaluasi prediksi.

    Args:
        actuals     : Array nilai aktual harga
        predictions : Array nilai prediksi model

    Returns:
        Dict berisi semua metrik evaluasi
    """
    y_true = np.array(actuals,     dtype=np.float64).flatten()
    y_pred = np.array(predictions, dtype=np.float64).flatten()

    # Samakan panjang array
    min_len = min(len(y_true), len(y_pred))
    y_true  = y_true[:min_len]
    y_pred  = y_pred[:min_len]

    # Hapus NaN atau Inf
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return _empty_metrics()

    return {
        "MAE":                  _mae(y_true, y_pred),
        "RMSE":                 _rmse(y_true, y_pred),
        "MAPE":                 _mape(y_true, y_pred),
        "R2":                   _r2(y_true, y_pred),
        "Directional_Accuracy": _directional_accuracy(y_true, y_pred),
        "Max_Error":            float(np.max(np.abs(y_true - y_pred))),
        "Median_AE":            float(np.median(np.abs(y_true - y_pred))),
        "n_samples":            int(len(y_true)),
    }


def evaluate_multi_horizon(
    actuals_matrix: np.ndarray,
    predictions_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluasi metrik per horizon (hari ke-1, ke-5, ke-10, ke-30, dst.).

    Args:
        actuals_matrix     : shape (n_samples, horizon)
        predictions_matrix : shape (n_samples, horizon)

    Returns:
        DataFrame dengan metrik per horizon
    """
    assert actuals_matrix.shape == predictions_matrix.shape, \
        "Shape aktual dan prediksi harus sama"

    n_horizons = actuals_matrix.shape[1]
    rows = []
    for h in range(n_horizons):
        m = evaluate_predictions(actuals_matrix[:, h], predictions_matrix[:, h])
        m["horizon_day"] = h + 1
        rows.append(m)

    return pd.DataFrame(rows).set_index("horizon_day")


def format_metrics_for_display(metrics: dict) -> dict:
    """
    Format nilai metrik untuk ditampilkan di UI Streamlit.
    """
    return {
        "MAE":   f"Rp {metrics.get('MAE', 0):,.0f}",
        "RMSE":  f"Rp {metrics.get('RMSE', 0):,.0f}",
        "MAPE":  f"{metrics.get('MAPE', 0):.2f}%",
        "R²":    f"{metrics.get('R2', 0):.4f}",
        "Dir. Acc.": f"{metrics.get('Directional_Accuracy', 0)*100:.1f}%",
    }


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTASI METRIK
# ══════════════════════════════════════════════════════════════════════════════
def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Menghindari division by zero dengan menambahkan epsilon kecil.
    """
    epsilon = np.finfo(np.float64).eps
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).
    R² = 1 - SS_res / SS_tot
    Nilai mendekati 1.0 menunjukkan model sangat baik.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy: seberapa sering arah perubahan harga diprediksi benar.
    DA = jumlah prediksi arah benar / total prediksi
    """
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    correct  = np.sum(true_dir == pred_dir)
    return float(correct / len(true_dir))


def _empty_metrics() -> dict:
    """Mengembalikan metrik kosong jika tidak ada data valid."""
    return {
        "MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0,
        "R2": 0.0,  "Directional_Accuracy": 0.0,
        "Max_Error": 0.0, "Median_AE": 0.0,
        "n_samples": 0,
    }
