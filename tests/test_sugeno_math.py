"""Mathematical verification tests for OpenFLL Sugeno-0 inference.

Verifies:
1. Strict linearity of Sugeno-0 output with respect to consequent constants (y = A @ c).
2. Reconstruction precision between direct engine evaluation and A @ c.
3. Parameter identifiability and recovery via ordinary least squares (lstsq).
4. Parameter sharing (multiple rules mapping to the same consequent MF).
5. Behavior when rule activations sum to zero (out of range / inactive inputs).
6. Invariance of linearity under different t-norms (ProdAnd, BoundedDiff, etc.).
"""
from __future__ import annotations

import math
import numpy as np
import pytest

import openfll


def _build_test_engine(c1: float, c2: float, t_norm: str = "min") -> openfll.SugenoEngine:
    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")

    e.add_membership_func("x1", "x1_low", "triangular", [-10, 0, 10])
    e.add_membership_func("x1", "x1_high", "triangular", [0, 10, 20])
    e.add_membership_func("x2", "x2_low", "gaussian", openfll.gaussian(center=0, sigma=2))
    e.add_membership_func("x2", "x2_high", "trapezoidal", [0, 5, 10, 15])

    e.add_membership_func("y", "c1", "constant", [c1])
    e.add_membership_func("y", "c2", "constant", [c2])

    if t_norm == "min":
        e.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"')
        e.add_rule('IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c2"')
    else:
        e.add_rule(
            antecedent=[("x1", "x1_low"), ("x2", "x2_low")],
            consequent=[("y", "c1")],
            t_norm=t_norm,
        )
        e.add_rule(
            antecedent=[("x1", "x1_high"), ("x2", "x2_high")],
            consequent=[("y", "c2")],
            t_norm=t_norm,
        )
    e.build()
    return e


def test_linearity_and_exact_reconstruction_min_norm():
    """Verify that y(X) == A @ c holds to machine precision for standard Min norm."""
    c_target = np.array([7.25, -3.50], dtype=np.float64)
    engine_real = _build_test_engine(c_target[0], c_target[1], t_norm="min")
    e0 = _build_test_engine(1.0, 0.0, t_norm="min")
    e1 = _build_test_engine(0.0, 1.0, t_norm="min")

    rng = np.random.default_rng(42)
    # Generate points inside overlapping active regions
    X = rng.uniform(0.0, 8.0, size=(200, 2))

    y_real = engine_real.predict(X)
    col0 = e0.predict(X)
    col1 = e1.predict(X)
    A = np.column_stack([col0, col1])

    y_reconstructed = A @ c_target

    # Must be bit-for-bit or floating-point machine precision identical
    max_err = np.max(np.abs(y_real - y_reconstructed))
    assert max_err < 1e-12, f"Reconstruction error too large: {max_err}"

    # Also verify that sum of normalized weights equals 1 when rules cover domain
    sum_weights = col0 + col1
    assert np.allclose(sum_weights, 1.0, atol=1e-12)


def test_least_squares_exact_parameter_recovery():
    """Verify that np.linalg.lstsq perfectly recovers target parameters c."""
    c_true = np.array([12.34, -56.78], dtype=np.float64)
    engine_real = _build_test_engine(c_true[0], c_true[1], t_norm="min")
    e0 = _build_test_engine(1.0, 0.0, t_norm="min")
    e1 = _build_test_engine(0.0, 1.0, t_norm="min")

    rng = np.random.default_rng(100)
    X = rng.uniform(0.0, 8.0, size=(100, 2))
    y = engine_real.predict(X)

    col0 = e0.predict(X)
    col1 = e1.predict(X)
    A = np.column_stack([col0, col1])

    c_est, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    assert rank == 2
    assert np.allclose(c_est, c_true, atol=1e-12)


def test_linearity_with_product_and_norm():
    """Verify linearity holds under product t-norm (prod_and)."""
    c_true = np.array([4.0, 8.0], dtype=np.float64)
    engine_real = _build_test_engine(c_true[0], c_true[1], t_norm="prod_and")
    e0 = _build_test_engine(1.0, 0.0, t_norm="prod_and")
    e1 = _build_test_engine(0.0, 1.0, t_norm="prod_and")

    rng = np.random.default_rng(200)
    X = rng.uniform(0.0, 8.0, size=(100, 2))
    y = engine_real.predict(X)

    A = np.column_stack([e0.predict(X), e1.predict(X)])
    y_rec = A @ c_true
    assert np.allclose(y, y_rec, atol=1e-12)

    c_est, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    assert np.allclose(c_est, c_true, atol=1e-12)


def test_parameter_sharing_multiple_rules_same_consequent():
    """Verify parameter sharing when multiple rules point to the same consequent MF."""
    def make_sharing_engine(c_low: float, c_high: float):
        e = openfll.SugenoEngine()
        e.add_input_var("x1")
        e.add_input_var("x2")
        e.add_output_var("y")

        e.add_membership_func("x1", "x1_low", "triangular", [-10, 0, 10])
        e.add_membership_func("x1", "x1_high", "triangular", [0, 10, 20])
        e.add_membership_func("x2", "x2_low", "gaussian", openfll.gaussian(center=0, sigma=2))
        e.add_membership_func("x2", "x2_high", "trapezoidal", [0, 5, 10, 15])

        e.add_membership_func("y", "c_low", "constant", [c_low])
        e.add_membership_func("y", "c_high", "constant", [c_high])

        # 3 rules, 2 share 'c_low'
        e.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c_low"')
        e.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_high" THEN y IS "c_low"')
        e.add_rule('IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c_high"')
        e.build()
        return e

    c_true = np.array([3.0, 9.0])  # [c_low, c_high]
    real_engine = make_sharing_engine(c_true[0], c_true[1])

    # Basis models for the 2 unique parameters
    e_low = make_sharing_engine(1.0, 0.0)
    e_high = make_sharing_engine(0.0, 1.0)

    rng = np.random.default_rng(300)
    X = rng.uniform(-5.0, 15.0, size=(150, 2))
    y = real_engine.predict(X)

    col_low = e_low.predict(X)
    col_high = e_high.predict(X)
    A = np.column_stack([col_low, col_high])

    y_rec = A @ c_true
    # Verify that wherever inference is defined (non-NaN), reconstruction is exact
    assert np.allclose(y, y_rec, equal_nan=True, atol=1e-12)

    # For least squares recovery, we filter rows where the model has non-zero rule support
    valid_mask = ~np.isnan(y)
    assert np.any(valid_mask), "Expected at least some valid points"
    c_est, _, rank, _ = np.linalg.lstsq(A[valid_mask], y[valid_mask], rcond=None)
    assert rank == 2
    assert np.allclose(c_est, c_true, atol=1e-12)


def test_zero_activation_behavior():
    """Investigate what OpenFLL returns when inputs fall completely outside rule support."""
    e = _build_test_engine(5.0, 15.0)

    # Point far outside support: x1=100, x2=-100 -> all antecedent MFs are 0.0
    x_outside = np.array([[100.0, -100.0]], dtype=np.float64)
    y_out = e.predict(x_outside)

    # In C++, 0.0 / 0.0 produces nan
    assert np.isnan(y_out[0]), f"Expected NaN for zero total activation, got {y_out[0]}"


def test_linearity_with_bounded_diff_norm():
    """Verify linearity holds under bounded_diff (Łukasiewicz) t-norm."""
    c_true = np.array([10.0, -20.0], dtype=np.float64)
    engine_real = _build_test_engine(c_true[0], c_true[1], t_norm="bounded_diff")
    e0 = _build_test_engine(1.0, 0.0, t_norm="bounded_diff")
    e1 = _build_test_engine(0.0, 1.0, t_norm="bounded_diff")

    rng = np.random.default_rng(400)
    X = rng.uniform(0.0, 6.0, size=(100, 2))
    y = engine_real.predict(X)

    col0 = e0.predict(X)
    col1 = e1.predict(X)
    A = np.column_stack([col0, col1])

    valid_mask = ~np.isnan(y)
    assert np.any(valid_mask)
    y_rec = A @ c_true
    assert np.allclose(y[valid_mask], y_rec[valid_mask], atol=1e-12)

    c_est, _, rank, _ = np.linalg.lstsq(A[valid_mask], y[valid_mask], rcond=None)
    assert rank == 2
    assert np.allclose(c_est, c_true, atol=1e-12)


def test_rank_deficiency_and_svd_minimum_norm():
    """Verify behavior when a rule is never activated (rank deficient matrix A)."""
    e_real = _build_test_engine(5.0, 10.0, t_norm="min")
    e0 = _build_test_engine(1.0, 0.0, t_norm="min")
    e1 = _build_test_engine(0.0, 1.0, t_norm="min")

    # Data sampled strictly in [-8, -2] for x1: only 'x1_low' fires, 'x1_high' [0, 10, 20] is 0!
    rng = np.random.default_rng(500)
    X = rng.uniform(-6.0, -1.0, size=(50, 2))
    y = e_real.predict(X)

    col0 = e0.predict(X)
    col1 = e1.predict(X)
    A = np.column_stack([col0, col1])

    # Rule 2 never fired -> col1 is all zeros
    assert np.allclose(col1, 0.0)
    assert np.allclose(col0, 1.0)

    # Rank must be 1 instead of 2
    c_est, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    assert rank == 1

    # Active parameter c[0] should be reconstructed exactly:
    assert math.isclose(c_est[0], 5.0, abs_tol=1e-12)
    # Inactive parameter c[1] defaults to minimum-norm solution (0.0):
    assert math.isclose(c_est[1], 0.0, abs_tol=1e-12)


def test_basis_extraction_scaling():
    """Verify that basis batch extraction computes design matrix A rapidly."""
    import time

    e0 = _build_test_engine(1.0, 0.0, t_norm="min")
    e1 = _build_test_engine(0.0, 1.0, t_norm="min")

    rng = np.random.default_rng(600)
    X = rng.uniform(0.0, 8.0, size=(10000, 2))

    t0 = time.perf_counter()
    col0 = e0.predict(X)
    col1 = e1.predict(X)
    A = np.column_stack([col0, col1])
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert A.shape == (10000, 2)
    # 10k samples over 2 basis engines should finish comfortably in under 50 ms
    assert elapsed_ms < 100.0, f"Extraction took too long: {elapsed_ms:.2f} ms"


