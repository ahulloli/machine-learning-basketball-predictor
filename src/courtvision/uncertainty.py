"""Prediction intervals and empirical probabilities for CourtVision.

Stage 1 uncertainty: split-conformal residual calibration on top of the
existing XGBoost point model. Given a calibration season the fitted model has
never seen, we collect signed residuals (actual - predicted) and use them to:

* build a finite-sample conformal prediction interval (``point ± radius``), and
* estimate empirical over/under probabilities by re-applying the historical
  error distribution to today's projection.

Probabilities are tied to the model's *actual* historical errors rather than an
assumed normal distribution, which keeps them honest and defensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .train import MODEL_DIR


@dataclass(frozen=True)
class ResidualCalibrator:
    """Calibration information for one prediction target."""

    target: str
    coverage: float
    radius: float
    residuals: np.ndarray


def conformal_radius(
    residuals: np.ndarray,
    coverage: float = 0.80,
) -> float:
    """Return a finite-sample conformal interval radius.

    For an 80% interval, approximately 80% of future outcomes should fall
    inside ``point_prediction ± radius``, assuming future data resembles the
    calibration period.
    """
    residuals = np.asarray(residuals, dtype=float)

    if residuals.ndim != 1 or len(residuals) == 0:
        raise ValueError("residuals must be a non-empty 1D array")

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")

    absolute_errors = np.abs(residuals)
    n = len(absolute_errors)

    # Finite-sample conformal quantile.
    quantile_level = min(
        1.0,
        np.ceil((n + 1) * coverage) / n,
    )

    return float(
        np.quantile(
            absolute_errors,
            quantile_level,
            method="higher",
        )
    )


def fit_calibrator(
    *,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage: float = 0.80,
) -> ResidualCalibrator:
    """Build a calibrator from unseen calibration predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    signed_residuals = y_true - y_pred
    radius = conformal_radius(signed_residuals, coverage)

    return ResidualCalibrator(
        target=target,
        coverage=coverage,
        radius=radius,
        residuals=signed_residuals,
    )


def prediction_interval(
    projection: float,
    calibrator: ResidualCalibrator,
) -> tuple[float, float]:
    """Create a nonnegative symmetric prediction interval."""
    lower = max(0.0, projection - calibrator.radius)
    upper = max(lower, projection + calibrator.radius)

    return lower, upper


def probability_over(
    *,
    prediction: float,
    line: float,
    calibrator: ResidualCalibrator,
) -> float:
    """Estimate P(actual > line) from historical calibration errors.

    This asks: if today's model error resembles one of the calibration errors,
    how often would ``prediction + error`` exceed the requested line?
    """
    possible_outcomes = prediction + calibrator.residuals
    over_count = int(np.sum(possible_outcomes > line))
    n = len(possible_outcomes)

    # Smoothing avoids claiming exactly 0% or 100%.
    return float((over_count + 1) / (n + 2))


def probability_under(
    *,
    prediction: float,
    line: float,
    calibrator: ResidualCalibrator,
) -> float:
    return 1.0 - probability_over(
        prediction=prediction, line=line, calibrator=calibrator
    )


def _calibration_path(target: str) -> Path:
    tag = target.replace("target_", "")
    return MODEL_DIR / f"calibration_{tag}.npz"


def save_calibrator(calibrator: ResidualCalibrator) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)

    path = _calibration_path(calibrator.target)
    np.savez_compressed(
        path,
        target=np.array(calibrator.target),
        coverage=np.array(calibrator.coverage),
        radius=np.array(calibrator.radius),
        residuals=calibrator.residuals,
    )

    return path


def load_calibrator(target: str) -> ResidualCalibrator:
    path = _calibration_path(target)

    if not path.exists():
        raise FileNotFoundError(
            f"No calibration artifact at {path}. "
            "Run scripts/calibrate_uncertainty.py first."
        )

    with np.load(path) as artifact:
        return ResidualCalibrator(
            target=str(artifact["target"].item()),
            coverage=float(artifact["coverage"].item()),
            radius=float(artifact["radius"].item()),
            residuals=artifact["residuals"].astype(float),
        )
