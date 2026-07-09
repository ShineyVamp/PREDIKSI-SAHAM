import numpy as np
import pandas as pd
from typing import Union

Arr = Union[np.ndarray, list]


def _mae(a, b):  return float(np.mean(np.abs(a - b)))
def _rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def _mape(y_true, y_pred):
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    val = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    return float(val) if np.isfinite(val) else 0.0


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def _directional_accuracy(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    td = np.sign(np.diff(y_true))
    pd_ = np.sign(np.diff(y_pred))
    mask = td != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.sum((td == pd_) & mask) / mask.sum())


def _clean_pair(actuals, predictions):
    a = np.asarray(actuals, dtype=np.float64).flatten()
    p = np.asarray(predictions, dtype=np.float64).flatten()
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    m = np.isfinite(a) & np.isfinite(p)
    return a[m], p[m]


def evaluate_predictions(actuals: Arr, predictions: Arr) -> dict:
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


def naive_last_value(actuals: Arr) -> np.ndarray:
    y = np.asarray(actuals, dtype=np.float64).flatten()
    if len(y) == 0:
        return y
    return np.concatenate([[y[0]], y[:-1]])


def evaluate_with_baseline(actuals: Arr, predictions: Arr) -> dict:
    """Bandingkan model vs random walk pada prediksi 1-langkah."""
    y_true, y_pred = _clean_pair(actuals, predictions)
    if len(y_true) < 3:
        return {"model": _empty_metrics(), "naive": _empty_metrics(), "skill": _empty_skill()}

    a = y_true[1:]
    p_model = y_pred[1:]

    model = evaluate_predictions(a, p_model)
    naive = evaluate_predictions(a, y_true[:-1])

    mae_ratio = model["MAE"] / naive["MAE"] if naive["MAE"] > 0 else np.inf
    rmse_ratio = model["RMSE"] / naive["RMSE"] if naive["RMSE"] > 0 else np.inf
    skill = {
        "mae_ratio": float(mae_ratio),
        "mae_skill": float(1 - mae_ratio),
        "rmse_skill": float(1 - rmse_ratio),
        "beats_naive": bool(model["MAE"] < naive["MAE"]),
        "da": model["Directional_Accuracy"],
        "da_edge": float(model["Directional_Accuracy"] - 0.5),
    }
    return {"model": model, "naive": naive, "skill": skill}


def return_space_metrics(actual_prices: Arr, predicted_prices: Arr) -> dict:
    """Nilai prediksi PERUBAHAN. Baseline = menebak return 0."""
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
    mae_zero = np.mean(np.abs(r_true))
    da = (float(np.mean(np.sign(r_true[r_true != 0]) == np.sign(r_pred[r_true != 0])))
          if np.any(r_true != 0) else 0.0)
    return {
        "return_MAE_pct": float(mae_model * 100),
        "return_RMSE_pct": float(np.sqrt(np.mean((r_true - r_pred) ** 2)) * 100),
        "return_DA": da,
        "beats_zero_return": bool(mae_model < mae_zero),
        "n_samples": int(len(r_true)),
    }


def evaluate_per_horizon(preds_matrix, actuals_matrix, anchors=None) -> pd.DataFrame:
    P = np.asarray(preds_matrix, dtype=np.float64)
    A = np.asarray(actuals_matrix, dtype=np.float64)
    if P.ndim == 1: P = P.reshape(-1, 1)
    if A.ndim == 1: A = A.reshape(-1, 1)
    if A.size == 0:
        return pd.DataFrame(columns=["MAE_model", "MAPE_model", "MAE_naive", "skill", "n"]).rename_axis("horizon_day")
    n, H = A.shape
    if P.shape[1] >= H:
        P = P[:, :H]
    else:
        P = np.pad(P, ((0, 0), (0, H - P.shape[1])), constant_values=np.nan)
    anch = np.asarray(anchors, dtype=np.float64).flatten() if anchors is not None else None

    rows = []
    for h in range(H):
        a_col, p_col = A[:, h], P[:, h]
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
            mae_naive, skill = np.nan, np.nan
        rows.append({"horizon_day": h + 1, "MAE_model": mae_model, "MAPE_model": mape_model,
                     "MAE_naive": mae_naive, "skill": skill, "n": int(m.sum())})
    if not rows:
        return pd.DataFrame(columns=["MAE_model", "MAPE_model", "MAE_naive", "skill", "n"]).rename_axis("horizon_day")
    return pd.DataFrame(rows).set_index("horizon_day")


def format_metrics_for_display(metrics: dict) -> dict:
    return {
        "MAE": f"Rp {metrics.get('MAE', 0):,.0f}",
        "RMSE": f"Rp {metrics.get('RMSE', 0):,.0f}",
        "MAPE": f"{metrics.get('MAPE', 0):.2f}%",
        "R2": f"{metrics.get('R2', 0):.4f}",
        "Dir. Acc.": f"{metrics.get('Directional_Accuracy', 0)*100:.1f}%",
    }


def aggregate_horizon_metrics(preds_matrix, actuals_matrix, anchors=None) -> dict:
    P = np.asarray(preds_matrix, dtype=np.float64)
    A = np.asarray(actuals_matrix, dtype=np.float64)
    if P.ndim == 1: P = P.reshape(-1, 1)
    if A.ndim == 1: A = A.reshape(-1, 1)
    if A.size == 0:
        return {"MAE": 0.0, "MAPE": 0.0, "RMSE": 0.0, "DA": 0.0,
                "MAE_naive": 0.0, "mae_skill": 0.0, "beats_naive": False,
                "n_samples": 0, "n_horizons": 0}
    n, H = A.shape
    P = P[:, :H] if P.shape[1] >= H else np.pad(P, ((0, 0), (0, H - P.shape[1])), constant_values=np.nan)

    flat_a, flat_p = A.flatten(), P.flatten()
    m = np.isfinite(flat_a) & np.isfinite(flat_p)
    flat_a, flat_p = flat_a[m], flat_p[m]
    mae = _mae(flat_a, flat_p)
    mape = _mape(flat_a, flat_p)
    rmse = _rmse(flat_a, flat_p)

    da, mae_naive = 0.0, np.nan
    if anchors is not None:
        anch = np.asarray(anchors, dtype=np.float64).flatten()
        if len(anch) == n:
            anch_col = anch.reshape(-1, 1)
            true_dir = np.sign(A - anch_col)
            pred_dir = np.sign(P - anch_col)
            dmask = np.isfinite(A) & np.isfinite(P) & (true_dir != 0)
            if dmask.sum() > 0:
                da = float(np.sum((true_dir == pred_dir) & dmask) / dmask.sum())
            naive = np.broadcast_to(anch_col, A.shape)
            nm = np.isfinite(A) & np.isfinite(P)
            mae_naive = _mae(A[nm], naive[nm])

    mae_skill = float(1 - mae / mae_naive) if (mae_naive and mae_naive > 0) else 0.0
    return {
        "MAE": mae, "MAPE": mape, "RMSE": rmse, "DA": da,
        "MAE_naive": float(mae_naive) if np.isfinite(mae_naive) else 0.0,
        "mae_skill": mae_skill, "beats_naive": bool(np.isfinite(mae_naive) and mae < mae_naive),
        "n_samples": int(len(flat_a)), "n_horizons": int(H),
    }


def _empty_metrics() -> dict:
    return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "R2": 0.0,
            "Directional_Accuracy": 0.0, "Max_Error": 0.0, "Median_AE": 0.0, "n_samples": 0}


def _empty_skill() -> dict:
    return {"mae_ratio": float("inf"), "mae_skill": 0.0, "rmse_skill": 0.0,
            "beats_naive": False, "da": 0.0, "da_edge": 0.0}
