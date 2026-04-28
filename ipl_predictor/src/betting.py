"""Optional betting-style calibration from ranking-score gaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class BettingCalibrationResult:
    """Outputs from calibration pipeline."""

    confidence_raw: float
    probability_calibrated: float | None
    threshold_suggestion: float | None
    warning: str | None


MIN_SAMPLES_ISOTONIC = 50


def normalized_top_gap(scores: np.ndarray) -> float:
    """
    Confidence = normalized gap between top two ranking scores (per match).

    Uses absolute denominator for stability when scores are negative.
    """
    if scores.size < 2:
        return 0.0
    top2 = np.sort(scores)[-2:]
    denom = max(abs(top2[-1]), 1e-9)
    return float((top2[-1] - top2[-2]) / denom)


def fit_isotonic_calibrator(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> tuple[IsotonicRegression | None, str | None]:
    """Fit isotonic regression mapping gap→P(correct) when enough validation rows exist."""
    if len(confidences) < MIN_SAMPLES_ISOTONIC:
        return None, (
            f"Insufficient calibration data ({len(confidences)} rows; "
            f"need at least {MIN_SAMPLES_ISOTONIC}). Using uncalibrated scores."
        )
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(confidences, correctness.astype(float))
    return ir, None


def optimize_threshold(confidences: np.ndarray, correctness: np.ndarray) -> float | None:
    """Pick threshold that maximizes accuracy on validation grid (simple heuristic)."""
    if len(confidences) < 10:
        return None
    qs = np.linspace(0.05, 0.95, 19)
    thr_grid = np.quantile(confidences, qs)
    best_acc, best_t = -1.0, None
    for t in thr_grid:
        pred = confidences >= t
        acc = (pred == correctness).mean()
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def apply_calibration(
    raw_gap: float,
    calibrator: IsotonicRegression | None,
    threshold: float | None,
) -> BettingCalibrationResult:
    """Combine gap, optional isotonic mapping, and threshold suggestion."""
    prob = None
    warn = None
    if calibrator is not None:
        prob = float(calibrator.predict([raw_gap])[0])
    else:
        warn = "Isotonic calibration unavailable; gap is not a calibrated probability."

    return BettingCalibrationResult(
        confidence_raw=raw_gap,
        probability_calibrated=prob,
        threshold_suggestion=threshold,
        warning=warn,
    )


def build_calibration_from_validation(
    gaps: np.ndarray,
    hits: np.ndarray,
) -> tuple[IsotonicRegression | None, float | None, str | None]:
    """Train isotonic + threshold on validation arrays of gap scores and binary hits."""
    iso, warn = fit_isotonic_calibrator(gaps, hits)
    thr = optimize_threshold(gaps, hits)
    return iso, thr, warn

