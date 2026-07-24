"""Evaluation metrics: GEH, RMSE/MAE, and the blueprint skill score."""

import numpy as np


def geh(model: np.ndarray, count: np.ndarray) -> np.ndarray:
    """GEH statistic per element. <5 good, 5-10 review, >10 unusable (FHWA/DMRB)."""
    m = np.asarray(model, float)
    c = np.asarray(count, float)
    denom = m + c
    out = np.zeros_like(m)
    nz = denom > 0
    out[nz] = np.sqrt(2.0 * (m[nz] - c[nz]) ** 2 / denom[nz])
    return out


def geh_pass_rate(model: np.ndarray, count: np.ndarray, thr: float = 5.0) -> float:
    """Fraction with GEH < thr (DMRB calibration target is 85% below 5)."""
    return float((geh(model, count) < thr).mean())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(true, float)) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float))))


def skill_score(err_a: float, err_b: float, err_c: float) -> float:
    """(err_A - err_B) / (err_A - err_C).

    Fraction of the recoverable error (floor A -> ceiling C) that method B captures.
    1 = B matches oracle; 0 = B no better than naive; >1 = B beats the history-only
    ceiling using cross-sectional neighbor info; <0 = B worse than naive.
    """
    denom = err_a - err_c
    if abs(denom) < 1e-12:
        return float("nan")
    return (err_a - err_b) / denom


if __name__ == "__main__":
    m = np.array([100, 200, 50, 1000])
    c = np.array([110, 150, 55, 1020])
    print("GEH:", np.round(geh(m, c), 2))
    print("pass<5:", geh_pass_rate(m, c))
    print("skill(2.0,1.0,0.5):", round(skill_score(2.0, 1.0, 0.5), 3))
