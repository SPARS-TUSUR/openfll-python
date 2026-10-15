"""Comprehensive scikit-learn compatibility verification for SugenoRegressor.

Tests:
1. sklearn.base.clone preserves hyperparameters and wipes fitted state.
2. get_params / set_params conform strictly to sklearn conventions.
3. n_features_in_ and feature_names_in_ behavior with NumPy arrays and pandas DataFrames.
4. KFold and cross_val_score without data/state leakage across folds.
5. GridSearchCV over real hyperparameters (alpha, solver).
6. Integration with standard metrics (mean_squared_error, mean_absolute_error, r2_score).
7. Analysis of check_estimator expectations vs model_spec design constraints.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.estimator_checks import check_estimator
except ImportError:
    clone = None
    NotFittedError = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    GridSearchCV = None
    KFold = None
    cross_val_score = None
    Pipeline = None
    StandardScaler = None
    check_estimator = None

from openfll import SugenoEngine
from openfll.sklearn import SugenoModelSpec, SugenoRegressor


def _build_standard_spec() -> SugenoModelSpec:
    return SugenoModelSpec(
        inputs={
            "x1": [
                ("x1_low", "triangular", [-10.0, 0.0, 10.0]),
                ("x1_high", "triangular", [0.0, 10.0, 20.0]),
            ],
            "x2": [
                ("x2_low", "triangular", [-10.0, 0.0, 10.0]),
                ("x2_high", "triangular", [0.0, 10.0, 20.0]),
            ],
        },
        rules=[
            'IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"',
            'IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c2"',
        ],
        output_name="y",
    )


def test_clone_preserves_hyperparams_and_cleans_fitted_state():
    """clone(reg) reproduces constructor parameters without carrying over fitted state."""
    if clone is None:
        pytest.skip("scikit-learn is not available")

    spec = _build_standard_spec()
    reg = SugenoRegressor(
        model_spec=spec,
        default_t_norm="min",
        default_s_norm="max",
        solver="ridge",
        alpha=0.25,
    )

    rng = np.random.default_rng(101)
    X = rng.uniform(0.0, 8.0, size=(40, 2))
    y = 5.0 * X[:, 0] - 2.0 * X[:, 1]
    reg.fit(X, y)

    # Verify reg is fitted
    assert hasattr(reg, "engine_")
    assert hasattr(reg, "coef_")
    assert hasattr(reg, "n_features_in_")

    # Clone the fitted regressor
    cloned = clone(reg)

    # Cloned instance must not be identical
    assert cloned is not reg

    # Hyperparameters must match exactly
    assert cloned.model_spec == spec
    assert cloned.default_t_norm == "min"
    assert cloned.default_s_norm == "max"
    assert cloned.solver == "ridge"
    assert cloned.alpha == 0.25

    # Fitted attributes must NOT exist on the cloned instance
    assert not hasattr(cloned, "engine_")
    assert not hasattr(cloned, "coef_")
    assert not hasattr(cloned, "consequents_")
    assert not hasattr(cloned, "n_features_in_")
    assert not hasattr(cloned, "rank_")

    # Calling predict on cloned instance raises NotFittedError
    with pytest.raises(Exception) as exc_info:
        cloned.predict(X)
    assert "not fitted" in str(exc_info.value).lower()


def test_get_params_and_set_params_conventions():
    """get_params and set_params conform strictly to sklearn conventions."""
    spec = _build_standard_spec()
    reg = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)

    params = reg.get_params(deep=True)
    expected_keys = {"model_spec", "default_t_norm", "default_s_norm", "solver", "alpha"}
    assert set(params.keys()) == expected_keys
    assert params["solver"] == "lstsq"
    assert params["alpha"] == 0.0

    # Fitted attributes are never exposed as hyperparameters
    rng = np.random.default_rng(102)
    X = rng.uniform(0.0, 8.0, size=(20, 2))
    y = np.ones(20)
    reg.fit(X, y)

    params_after_fit = reg.get_params(deep=True)
    assert set(params_after_fit.keys()) == expected_keys
    assert "coef_" not in params_after_fit
    assert "engine_" not in params_after_fit

    # set_params modifies attributes and returns self
    ret = reg.set_params(alpha=1.5, solver="ridge")
    assert ret is reg
    assert reg.alpha == 1.5
    assert reg.solver == "ridge"


def test_n_features_in_and_feature_names_in():
    """n_features_in_ and feature_names_in_ behavior with NumPy arrays and pandas DataFrames."""
    spec = _build_standard_spec()
    reg = SugenoRegressor(model_spec=spec)

    rng = np.random.default_rng(103)
    X_np = rng.uniform(0.0, 8.0, size=(30, 2))
    y = 3.0 * X_np[:, 0] + 1.0

    # 1. Fit with pure NumPy array
    reg.fit(X_np, y)
    assert reg.n_features_in_ == 2
    assert not hasattr(reg, "feature_names_in_")

    # 2. Fit with pandas DataFrame (if pandas is installed)
    if pd is not None:
        df = pd.DataFrame(X_np, columns=["x1", "x2"])
        reg.fit(df, y)
        assert reg.n_features_in_ == 2
        assert hasattr(reg, "feature_names_in_")
        assert list(reg.feature_names_in_) == ["x1", "x2"]

        # Predict with same column names succeeds
        pred = reg.predict(df)
        assert pred.shape == (30,)

        # Predict with wrong column names raises ValueError
        df_wrong = pd.DataFrame(X_np, columns=["feat_a", "feat_b"])
        with pytest.raises(ValueError, match="The feature names should match"):
            reg.predict(df_wrong)

        # Fit with wrong column names raises ValueError
        with pytest.raises(ValueError, match="Feature names in X"):
            reg.fit(df_wrong, y)


def test_kfold_cross_validation_no_leakage():
    """KFold and cross_val_score run without data or state leakage across folds."""
    if cross_val_score is None or KFold is None:
        pytest.skip("scikit-learn is not available")

    spec = _build_standard_spec()
    reg = SugenoRegressor(model_spec=spec, solver="lstsq")

    rng = np.random.default_rng(104)
    X = rng.uniform(1.0, 7.0, size=(100, 2))
    # Target function inside rule coverage: y = 4.0 * w1 + 10.0 * w2
    y = 4.0 + 0.75 * X[:, 0] + 0.25 * X[:, 1]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")

    assert len(scores) == 5
    assert all(np.isfinite(scores))

    # Also verify that reg itself was not fitted by cross_val_score (clone safety)
    assert not hasattr(reg, "engine_")


def test_grid_search_cv_on_real_hyperparameters():
    """GridSearchCV works cleanly on actual hyperparameters (solver, alpha)."""
    if GridSearchCV is None:
        pytest.skip("scikit-learn is not available")

    spec = _build_standard_spec()
    reg = SugenoRegressor(model_spec=spec)

    rng = np.random.default_rng(105)
    X = rng.uniform(1.0, 7.0, size=(60, 2))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1]

    param_grid = {
        "solver": ["lstsq", "ridge"],
        "alpha": [0.0, 0.05, 0.5],
    }

    grid = GridSearchCV(reg, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid.fit(X, y)

    assert grid.best_estimator_ is not None
    assert hasattr(grid.best_estimator_, "engine_")
    assert hasattr(grid.best_estimator_, "coef_")
    assert grid.best_params_["solver"] in ["lstsq", "ridge"]
    assert grid.best_params_["alpha"] in [0.0, 0.05, 0.5]

    y_pred = grid.predict(X)
    assert y_pred.shape == (60,)
    assert np.all(np.isfinite(y_pred))


def test_standard_sklearn_metrics_evaluation():
    """SugenoRegressor predictions evaluate cleanly with MSE, MAE, and R2."""
    if mean_squared_error is None:
        pytest.skip("scikit-learn is not available")

    spec = _build_standard_spec()
    reg = SugenoRegressor(model_spec=spec)

    # Reference ground truth model
    ref = SugenoEngine()
    ref.add_input_var("x1")
    ref.add_input_var("x2")
    ref.add_output_var("y")
    ref.add_membership_func("x1", "x1_low", "triangular", [-10.0, 0.0, 10.0])
    ref.add_membership_func("x1", "x1_high", "triangular", [0.0, 10.0, 20.0])
    ref.add_membership_func("x2", "x2_low", "triangular", [-10.0, 0.0, 10.0])
    ref.add_membership_func("x2", "x2_high", "triangular", [0.0, 10.0, 20.0])
    ref.add_membership_func("y", "c1", "constant", [8.0])
    ref.add_membership_func("y", "c2", "constant", [-4.0])
    ref.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"')
    ref.add_rule('IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c2"')
    ref.build()

    rng = np.random.default_rng(106)
    X_train = rng.uniform(0.0, 8.0, size=(100, 2))
    y_train = ref.predict_batch(X_train, input_names=["x1", "x2"])

    X_test = rng.uniform(0.5, 7.5, size=(50, 2))
    y_test = ref.predict_batch(X_test, input_names=["x1", "x2"])

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    assert mse < 1e-12
    assert mae < 1e-12
    assert math.isclose(r2, 1.0, abs_tol=1e-10)


def test_check_estimator_behavior_documentation():
    """Document why standard check_estimator requires model_spec.

    check_estimator tests generic estimators by calling fit(X, y) with
    arbitrary random data and arbitrary numbers of features (1, 2, 10 features).
    SugenoRegressor requires a defined model_spec specifying the fuzzy partitions
    and rule base for the specific feature space. Calling fit() with default
    model_spec=None raises ValueError("model_spec must be specified before calling fit()"),
    which is the expected domain-specific contract for fuzzy systems.
    """
    reg = SugenoRegressor()
    with pytest.raises(ValueError, match="model_spec must be specified"):
        reg.fit(np.array([[1.0, 2.0]]), np.array([1.0]))
