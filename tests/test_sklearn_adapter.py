"""Unit and integration tests for SugenoRegressor scikit-learn adapter."""
from __future__ import annotations

import math
import pickle
import numpy as np
import pytest

try:
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError
    from sklearn.model_selection import cross_val_score
except ImportError:
    clone = None
    NotFittedError = None
    cross_val_score = None

from openfll import SugenoEngine
from openfll.sklearn import SugenoModelSpec, SugenoRegressor


def _make_sample_spec() -> SugenoModelSpec:
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


def test_sugeno_engine_is_not_polluted_by_sklearn():
    """SugenoEngine remains a flat engine and is not modified by sklearn integration."""
    engine = SugenoEngine()
    assert not hasattr(engine, "fit")
    assert not hasattr(engine, "get_params")
    assert not hasattr(engine, "set_params")


def test_sugeno_regressor_init_defaults():
    """SugenoRegressor initializes with clone-safe default parameters."""
    reg = SugenoRegressor()
    assert reg.model_spec is None
    assert reg.default_t_norm is None
    assert reg.default_s_norm is None
    assert reg.solver == "lstsq"
    assert reg.alpha == 0.0


def test_sugeno_regressor_clone_safety():
    """SugenoRegressor is clone-safe according to scikit-learn conventions."""
    if clone is None:
        pytest.skip("scikit-learn is not available in the test environment")

    spec = _make_sample_spec()
    reg = SugenoRegressor(
        model_spec=spec,
        default_t_norm="min",
        default_s_norm="max",
        solver="lstsq",
        alpha=0.1,
    )

    cloned = clone(reg)
    assert cloned is not reg
    assert cloned.model_spec == spec
    assert cloned.default_t_norm == "min"
    assert cloned.default_s_norm == "max"
    assert cloned.solver == "lstsq"
    assert cloned.alpha == 0.1

    params = reg.get_params()
    assert params["alpha"] == 0.1
    reg.set_params(alpha=0.5)
    assert reg.alpha == 0.5


def test_sugeno_regressor_predict_before_fit_raises_not_fitted_error():
    """Calling predict before fit raises NotFittedError."""
    reg = SugenoRegressor(model_spec=_make_sample_spec())
    with pytest.raises(Exception) as exc_info:
        reg.predict([[1.0, 2.0]])

    assert "not fitted" in str(exc_info.value).lower()


def test_fit_and_exact_parameter_recovery():
    """fit() finds exact parameters on data generated from ground truth Sugeno model."""
    spec = _make_sample_spec()
    c_true = [7.5, -3.25]

    # Generate synthetic target via known Sugeno model
    ref_engine = SugenoEngine()
    ref_engine.add_input_var("x1")
    ref_engine.add_input_var("x2")
    ref_engine.add_output_var("y")
    ref_engine.add_membership_func("x1", "x1_low", "triangular", [-10.0, 0.0, 10.0])
    ref_engine.add_membership_func("x1", "x1_high", "triangular", [0.0, 10.0, 20.0])
    ref_engine.add_membership_func("x2", "x2_low", "triangular", [-10.0, 0.0, 10.0])
    ref_engine.add_membership_func("x2", "x2_high", "triangular", [0.0, 10.0, 20.0])
    ref_engine.add_membership_func("y", "c1", "constant", [c_true[0]])
    ref_engine.add_membership_func("y", "c2", "constant", [c_true[1]])
    ref_engine.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"')
    ref_engine.add_rule('IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c2"')
    ref_engine.build()

    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 8.0, size=(100, 2))
    y = ref_engine.predict_batch(X, input_names=["x1", "x2"])

    reg = SugenoRegressor(model_spec=spec)
    fitted_reg = reg.fit(X, y)

    # Returns self
    assert fitted_reg is reg

    # Fitted attributes exist and follow sklearn trailing underscore convention
    assert hasattr(reg, "engine_")
    assert hasattr(reg, "coef_")
    assert hasattr(reg, "consequents_")
    assert hasattr(reg, "n_features_in_")
    assert hasattr(reg, "rank_")

    assert reg.n_features_in_ == 2
    assert reg.consequents_ == ["c1", "c2"]
    assert reg.rank_ == 2
    assert np.allclose(reg.coef_, c_true, atol=1e-10)

    # Predictions match y to machine precision
    y_pred = reg.predict(X)
    assert np.allclose(y_pred, y, atol=1e-10)

    # R2 score is 1.0
    score = reg.score(X, y)
    assert math.isclose(score, 1.0, abs_tol=1e-10)


def test_refit_resets_and_updates_state():
    """Subsequent fit() calls overwrite previous state cleanly."""
    spec = _make_sample_spec()
    reg = SugenoRegressor(model_spec=spec)

    rng = np.random.default_rng(123)
    X = rng.uniform(0.0, 8.0, size=(50, 2))
    y1 = np.ones(50, dtype=np.float64) * 10.0
    y2 = np.ones(50, dtype=np.float64) * -5.0

    reg.fit(X, y1)
    assert np.allclose(reg.coef_, [10.0, 10.0], atol=1e-10)

    reg.fit(X, y2)
    assert np.allclose(reg.coef_, [-5.0, -5.0], atol=1e-10)


def test_fit_with_parameter_sharing():
    """fit() handles parameter sharing across multiple rules."""
    spec = SugenoModelSpec(
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
            'IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c_shared"',
            'IF x1 IS "x1_low" AND x2 IS "x2_high" THEN y IS "c_shared"',
            'IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c_other"',
        ],
        output_name="y",
    )

    rng = np.random.default_rng(777)
    X = rng.uniform(0.0, 8.0, size=(100, 2))

    # Real targets with c_shared=4.0, c_other=-8.0
    ref_engine = SugenoEngine()
    ref_engine.add_input_var("x1")
    ref_engine.add_input_var("x2")
    ref_engine.add_output_var("y")
    ref_engine.add_membership_func("x1", "x1_low", "triangular", [-10.0, 0.0, 10.0])
    ref_engine.add_membership_func("x1", "x1_high", "triangular", [0.0, 10.0, 20.0])
    ref_engine.add_membership_func("x2", "x2_low", "triangular", [-10.0, 0.0, 10.0])
    ref_engine.add_membership_func("x2", "x2_high", "triangular", [0.0, 10.0, 20.0])
    ref_engine.add_membership_func("y", "c_shared", "constant", [4.0])
    ref_engine.add_membership_func("y", "c_other", "constant", [-8.0])
    ref_engine.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c_shared"')
    ref_engine.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_high" THEN y IS "c_shared"')
    ref_engine.add_rule('IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c_other"')
    ref_engine.build()

    y = ref_engine.predict_batch(X, input_names=["x1", "x2"])

    reg = SugenoRegressor(model_spec=spec)
    reg.fit(X, y)

    assert reg.consequents_ == ["c_shared", "c_other"]
    assert len(reg.coef_) == 2
    assert np.allclose(reg.coef_, [4.0, -8.0], atol=1e-10)


def test_fit_with_dict_model_spec():
    """model_spec can be passed as a standard dict."""
    dict_spec = {
        "inputs": {
            "x1": [("x1_low", "triangular", [-10.0, 0.0, 10.0]), ("x1_high", "triangular", [0.0, 10.0, 20.0])],
            "x2": [("x2_low", "triangular", [-10.0, 0.0, 10.0]), ("x2_high", "triangular", [0.0, 10.0, 20.0])],
        },
        "rules": [
            'IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"',
            'IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c2"',
        ],
        "output_name": "y",
    }
    reg = SugenoRegressor(model_spec=dict_spec)

    X = np.array([[-5.0, -5.0], [0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [15.0, 15.0]])
    y = np.array([2.0, 2.0, 6.0, 10.0, 10.0])

    reg.fit(X, y)
    assert reg.n_features_in_ == 2
    assert len(reg.coef_) == 2


def test_fit_with_ridge_solver():
    """fit() with solver='ridge' and alpha > 0 works properly."""
    spec = _make_sample_spec()
    reg = SugenoRegressor(model_spec=spec, solver="ridge", alpha=1.0)

    rng = np.random.default_rng(888)
    X = rng.uniform(0.0, 8.0, size=(50, 2))
    y = rng.normal(size=50)

    reg.fit(X, y)
    assert hasattr(reg, "coef_")
    assert reg.coef_.shape == (2,)


def test_fit_zero_activation_raises_informative_error():
    """Samples completely outside rule support raise a clear ValueError."""
    spec = _make_sample_spec()
    reg = SugenoRegressor(model_spec=spec)

    # Point at [100.0, -100.0] has zero rule activation
    X_bad = np.array([[100.0, -100.0], [2.0, 2.0]])
    y_bad = np.array([1.0, 2.0])

    with pytest.raises(ValueError) as exc_info:
        reg.fit(X_bad, y_bad)

    msg = str(exc_info.value).lower()
    assert "support of the fuzzy rule base" in msg or "total rule activation is zero" in msg


def test_fit_input_validations():
    """fit() validates input dimensions, missing spec, inf/nan and feature counts."""
    # 1. Missing model_spec
    with pytest.raises(ValueError, match="model_spec must be specified"):
        SugenoRegressor().fit([[1.0, 2.0]], [1.0])

    spec = _make_sample_spec()
    reg = SugenoRegressor(model_spec=spec)

    # 2. Wrong feature count
    with pytest.raises(ValueError, match="X has 1 features, but model_spec expects 2"):
        reg.fit([[1.0]], [1.0])

    # 3. 1D X
    with pytest.raises(ValueError):
        reg.fit([1.0, 2.0], [1.0, 2.0])

    # 4. Inconsistent sample counts
    with pytest.raises(ValueError):
        reg.fit([[1.0, 2.0], [3.0, 4.0]], [1.0])

    # 5. Non-finite values
    with pytest.raises(ValueError):
        reg.fit([[np.nan, 2.0]], [1.0])

    # 6. Negative alpha
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        SugenoRegressor(model_spec=spec, alpha=-0.5).fit([[1.0, 2.0]], [1.0])


def test_fit_with_tuple_rules():
    """Tuple rules can be used in model_spec."""
    spec = SugenoModelSpec(
        inputs={
            "x1": [("x1_low", "triangular", [-10.0, 0.0, 10.0])],
            "x2": [("x2_low", "triangular", [-10.0, 0.0, 10.0])],
        },
        rules=[
            ([("x1", "x1_low"), ("x2", "x2_low")], [("y", "c_out")], "min"),
        ],
        output_name="y",
    )
    reg = SugenoRegressor(model_spec=spec)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([3.0, 3.0, 3.0])

    reg.fit(X, y)
    assert np.allclose(reg.coef_, [3.0], atol=1e-10)




def test_scikit_learn_cross_val_score_integration():
    """SugenoRegressor works seamlessly inside scikit-learn cross_val_score."""
    if cross_val_score is None:
        pytest.skip("scikit-learn is not available in the test environment")

    spec = _make_sample_spec()
    reg = SugenoRegressor(model_spec=spec)

    rng = np.random.default_rng(999)
    X = rng.uniform(1.0, 7.0, size=(60, 2))
    # Target = 2.0 * col0 + 5.0 * col1
    y = 2.0 + 0.5 * X[:, 0]

    scores = cross_val_score(reg, X, y, cv=3)
    assert len(scores) == 3
    # Check that scores are finite numbers
    assert all(np.isfinite(scores))


def test_pickle_serialization_roundtrip():
    """SugenoRegressor serializes cleanly with pickle both unfitted and fitted."""
    spec = _make_sample_spec()

    # 1. Unfitted roundtrip
    unfitted_reg = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)
    data = pickle.dumps(unfitted_reg)
    restored_unfitted = pickle.loads(data)
    assert not hasattr(restored_unfitted, "engine_")
    assert not hasattr(restored_unfitted, "coef_")
    assert restored_unfitted.solver == "lstsq"

    # 2. Fitted roundtrip
    reg = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array([1.5, 4.5])
    reg.fit(X, y)

    orig_pred = reg.predict(X)
    data = pickle.dumps(reg)
    restored = pickle.loads(data)

    assert hasattr(restored, "engine_")
    assert hasattr(restored, "coef_")
    assert hasattr(restored, "rule_names_")
    assert np.allclose(restored.coef_, reg.coef_)
    assert restored.rule_names_ == reg.rule_names_

    restored_pred = restored.predict(X)
    assert np.allclose(restored_pred, orig_pred)


def test_solver_lstsq_and_ridge_strict_semantics():
    """Strict semantics: solver='lstsq' requires alpha=0.0, solver='ridge' requires alpha>0.0."""
    spec = _make_sample_spec()
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array([1.5, 4.5])

    # solver='lstsq' with alpha > 0 fails
    with pytest.raises(ValueError, match="solver='lstsq' does not support L2 regularization"):
        SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.5).fit(X, y)

    # solver='ridge' with alpha == 0 fails
    with pytest.raises(ValueError, match="solver='ridge' requires alpha > 0.0"):
        SugenoRegressor(model_spec=spec, solver="ridge", alpha=0.0).fit(X, y)

    # solver='ridge' with alpha < 0 fails
    with pytest.raises(ValueError, match="alpha must be non-negative and finite"):
        SugenoRegressor(model_spec=spec, solver="ridge", alpha=-0.1).fit(X, y)

    # Valid solver runs succeed
    reg_lstsq = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0).fit(X, y)
    assert reg_lstsq.coef_ is not None

    reg_ridge = SugenoRegressor(model_spec=spec, solver="ridge", alpha=0.1).fit(X, y)
    assert reg_ridge.coef_ is not None


def test_alpha_nan_inf_validation():
    """Non-finite or boolean values for alpha raise informative ValueError."""
    spec = _make_sample_spec()
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array([1.5, 4.5])

    for bad_alpha in [np.nan, np.inf, -np.inf, True, False]:
        with pytest.raises(ValueError, match="alpha must be non-negative and finite"):
            SugenoRegressor(model_spec=spec, alpha=bad_alpha).fit(X, y)


def test_model_spec_deep_freezing():
    """SugenoModelSpec deeply freezes nested inputs and rules into immutable tuples."""
    raw_inputs = {
        "x1": [["x1_low", "triangular", [-10.0, 0.0, 10.0]]],
    }
    raw_rules = [
        ["IF x1 IS x1_low THEN y IS c1"],
    ]
    spec = SugenoModelSpec(inputs=raw_inputs, rules=raw_rules, consequents=["c1"])

    # Attempting to mutate nested MF tuple raises TypeError
    with pytest.raises(TypeError):
        spec.inputs["x1"][0][2][0] = 999.0

    # Attempting to mutate rules tuple raises TypeError
    with pytest.raises(TypeError):
        spec.rules[0] = "new rule"

    # Attempting to mutate top-level attributes raises FrozenInstanceError
    with pytest.raises(Exception):
        spec.output_name = "other"


def test_explicit_consequents_validation():
    """model_spec.consequents validates duplicates, missing items, and unused items."""
    spec_base = _make_sample_spec()
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array([1.5, 4.5])

    # 1. Duplicates
    spec_dups = SugenoModelSpec(
        inputs=spec_base.inputs,
        rules=spec_base.rules,
        consequents=["c1", "c1"],
    )
    with pytest.raises(ValueError, match="duplicate names"):
        SugenoRegressor(model_spec=spec_dups).fit(X, y)

    # 2. Missing consequents referenced in rules (c2 missing)
    spec_missing = SugenoModelSpec(
        inputs=spec_base.inputs,
        rules=spec_base.rules,
        consequents=["c1"],
    )
    with pytest.raises(ValueError, match="is missing consequents referenced in rules"):
        SugenoRegressor(model_spec=spec_missing).fit(X, y)

    # 3. Extra unused consequents not in rules (c_extra not in rules)
    spec_unused = SugenoModelSpec(
        inputs=spec_base.inputs,
        rules=spec_base.rules,
        consequents=["c1", "c2", "c_extra"],
    )
    with pytest.raises(ValueError, match="defines consequents not referenced by any rule"):
        SugenoRegressor(model_spec=spec_unused).fit(X, y)

    # 4. Valid explicit consequents matching rules succeeds
    spec_valid = SugenoModelSpec(
        inputs=spec_base.inputs,
        rules=spec_base.rules,
        consequents=["c2", "c1"],  # reversed order
    )
    reg = SugenoRegressor(model_spec=spec_valid).fit(X, y)
    assert reg.consequents_ == ["c2", "c1"]


def test_rule_names_fitted_attribute():
    """SugenoRegressor populates rule_names_ upon fitting."""
    spec = _make_sample_spec()
    reg = SugenoRegressor(model_spec=spec)
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array([1.5, 4.5])
    reg.fit(X, y)

    assert hasattr(reg, "rule_names_")
    assert reg.rule_names_ == ["rule_0", "rule_1"]

