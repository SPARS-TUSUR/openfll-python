from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_native_openfll():
    """Импортировать openfll только при доступном настоящем PyFLL._core."""
    for module_name in ("openfll", "PyFLL", "PyFLL._core", "openfll._core"):
        sys.modules.pop(module_name, None)

    try:
        pyfll_core = importlib.import_module("PyFLL._core")
    except Exception as exc:
        src_dir = Path.cwd() / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
            importlib.invalidate_caches()
        try:
            pyfll_core = importlib.import_module("PyFLL._core")
        except Exception as retry_exc:
            pytest.skip(
                "BLOCKED: настоящий native backend PyFLL._core недоступен. "
                "Нужен wheel/source tree с собранным PyFLL/_core.<SOABI>.pyd "
                f"для текущего Python. Причина: {type(retry_exc).__name__}: "
                f"{retry_exc}"
            )

    openfll = importlib.import_module("openfll")
    assert openfll.SugenoEngine is pyfll_core.SugenoEngine
    assert openfll.SugenoEngine.__module__ == "PyFLL._core"
    return openfll


def _build_full_engine(openfll):
    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")

    e.add_membership_func("x1", "x1_low", "triangular", [-10, 0, 10])
    e.add_membership_func("x1", "x1_mid", "triangular", [0, 5, 10])
    e.add_membership_func("x1", "x1_high", "triangular", [0, 10, 20])
    e.add_membership_func("x1", "x1_vhigh", "triangular", [10, 20, 30])

    e.add_membership_func("x2", "x2_low", "gaussian", [-5, 2])
    e.add_membership_func("x2", "x2_mid", "gaussian", [0, 2])
    e.add_membership_func("x2", "x2_high", "gaussian", [5, 2])

    e.add_membership_func("y", "y_c0", "constant", [10.0])
    e.add_membership_func("y", "y_c1", "constant", [20.0])
    e.add_membership_func("y", "y_c2", "constant", [30.0])
    e.add_membership_func("y", "y_c3", "constant", [40.0])

    x1_names = ["x1_low", "x1_mid", "x1_high", "x1_vhigh"]
    x2_names = ["x2_low", "x2_mid", "x2_high"]
    y_names = ["y_c0", "y_c1", "y_c2", "y_c3"]

    for i, x1_name in enumerate(x1_names):
        for x2_name in x2_names:
            e.add_rule(
                f'IF x1 IS "{x1_name}" '
                f'AND x2 IS "{x2_name}" '
                f'THEN y IS "{y_names[i]}"'
            )

    e.build()
    return e


def _calculate(engine, x1: float, x2: float) -> float:
    engine.set_input("x1", x1)
    engine.set_input("x2", x2)
    engine.calculate()
    return engine.get_output("y")


def _build_tuple_engine(openfll):
    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")
    e.add_membership_func("x1", "x1_low", "triangular", [-10, 0, 10])
    e.add_membership_func("x1", "x1_mid", "triangular", [0, 5, 10])
    e.add_membership_func("x2", "x2_low", "triangular", [-10, 0, 10])
    e.add_membership_func("x2", "x2_mid", "triangular", [0, 5, 10])
    e.add_membership_func("y", "y_low", "constant", [0.0])
    e.add_membership_func("y", "y_high", "constant", [10.0])
    e.add_rule(
        antecedent=[("x1", "x1_low"), ("x2", "x2_low")],
        consequent=[("y", "y_low")],
    )
    e.add_rule(
        antecedent=[("x1", "x1_mid"), ("x2", "x2_mid")],
        consequent=[("y", "y_high")],
        t_norm="prod_and",
    )
    e.build()
    return e


def _build_all_membership_functions_engine(openfll):
    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")

    e.add_membership_func(
        "x1",
        "x1_tri_low",
        "triangular",
        openfll.triangular(-10, 0, 10),
    )
    e.add_membership_func(
        "x1",
        "x1_trap_high",
        "trapezoidal",
        openfll.trapezoidal(0, 5, 10, 15),
    )
    e.add_membership_func(
        "x2",
        "x2_gauss_mid",
        "gaussian",
        openfll.gaussian(center=0, sigma=2),
    )
    e.add_membership_func(
        "x2",
        "x2_trap_high",
        "trapezoidal",
        openfll.trapezoidal(0, 5, 10, 15),
    )
    e.add_membership_func("y", "y_low", "constant", openfll.constant(5.0))
    e.add_membership_func("y", "y_high", "constant", openfll.constant(15.0))

    e.add_rule('IF x1 IS "x1_tri_low" AND x2 IS "x2_gauss_mid" THEN y IS "y_low"')
    e.add_rule(
        'IF x1 IS "x1_trap_high" AND x2 IS "x2_trap_high" THEN y IS "y_high"'
    )
    e.build()
    return e


def test_openfll_uses_real_pyfll_core():
    openfll = _import_native_openfll()

    engine = openfll.SugenoEngine()

    assert engine.__class__.__module__ == "PyFLL._core"
    assert engine.is_built() is False


def test_native_text_rules_match_backend_baseline_numbers():
    openfll = _import_native_openfll()
    engine = _build_full_engine(openfll)

    assert engine.is_built() is True
    assert math.isclose(_calculate(engine, 0.0, 0.0), 10.0, abs_tol=1e-9)
    assert math.isclose(_calculate(engine, 5.0, 0.0), 20.0, abs_tol=1e-9)
    assert math.isclose(_calculate(engine, 5.0, 5.0), 20.0, abs_tol=1e-9)


def test_native_predict_matches_explicit_scalar_workflow():
    openfll = _import_native_openfll()
    explicit_engine = _build_full_engine(openfll)
    predict_engine = _build_full_engine(openfll)

    explicit = _calculate(explicit_engine, 5.0, 0.0)
    predicted = predict_engine.predict({"x1": 5.0, "x2": 0.0})
    predicted_with_output_name = predict_engine.predict(
        {"x1": 5.0, "x2": 0.0},
        output_name="y",
    )

    assert math.isclose(predicted, explicit, abs_tol=1e-9)
    assert math.isclose(predicted_with_output_name, explicit, abs_tol=1e-9)


def test_native_predict_reports_missing_inputs_and_preserves_unknown_input_error():
    openfll = _import_native_openfll()
    engine = _build_full_engine(openfll)

    with pytest.raises(ValueError, match="missing registered input variables: x2"):
        engine.predict({"x1": 5.0})

    with pytest.raises(IndexError):
        engine.predict({"x1": 5.0, "x2": 0.0, "unknown": 1.0})


def test_native_membership_function_types_work_through_openfll():
    openfll = _import_native_openfll()
    engine = _build_all_membership_functions_engine(openfll)

    assert engine.is_built() is True
    assert math.isclose(_calculate(engine, 0.0, 0.0), 5.0, abs_tol=1e-9)
    assert 5.0 <= _calculate(engine, 7.0, 7.0) <= 15.0


def test_native_tuple_rules_default_and_explicit_norms_work_through_openfll():
    openfll = _import_native_openfll()
    engine = _build_tuple_engine(openfll)

    y = _calculate(engine, 0.0, 0.0)

    assert 0.0 <= y <= 10.0


def test_native_backend_rejects_lifecycle_and_invalid_inputs():
    openfll = _import_native_openfll()
    engine = _build_tuple_engine(openfll)

    with pytest.raises(RuntimeError):
        engine.add_input_var("late")

    engine2 = openfll.SugenoEngine()
    engine2.add_input_var("x")
    with pytest.raises(ValueError):
        engine2.add_membership_func("x", "bad", "gaussian", [0.0])


def test_native_backend_rejects_invalid_rule_text():
    openfll = _import_native_openfll()
    engine = openfll.SugenoEngine()
    engine.add_input_var("x1")
    engine.add_input_var("x2")
    engine.add_output_var("y")
    engine.add_membership_func("x1", "x1_low", "triangular", [-10, 0, 10])
    engine.add_membership_func("x2", "x2_low", "triangular", [-10, 0, 10])
    engine.add_membership_func("y", "y_low", "constant", [1.0])

    with pytest.raises(ValueError, match="Rule parse error"):
        engine.add_rule("not a rule")


def test_native_predict_unified_single_input_scalar_and_sequence():
    openfll = _import_native_openfll()
    e = openfll.SugenoEngine()
    e.add_input_var("x")
    e.add_output_var("y")
    e.add_membership_func("x", "low", "triangular", [-10, 0, 10])
    e.add_membership_func("y", "out", "constant", [2.5])
    e.add_rule('IF x IS "low" AND x IS "low" THEN y IS "out"')
    e.build()

    # Scalar input (float and int)
    assert math.isclose(e.predict(0.0), 2.5, abs_tol=1e-9)
    assert math.isclose(e.predict(0), 2.5, abs_tol=1e-9)

    # Mapping
    assert math.isclose(e.predict({"x": 0.0}), 2.5, abs_tol=1e-9)

    # Sequence for single sample
    assert math.isclose(e.predict([0.0]), 2.5, abs_tol=1e-9)
    assert math.isclose(e.predict((0.0,)), 2.5, abs_tol=1e-9)

    # 1D numpy array single sample
    assert math.isclose(e.predict(np.array([0.0])), 2.5, abs_tol=1e-9)

    # 1D numpy array multi-sample batch for 1-input model
    batch_res = e.predict(np.array([0.0, 0.0, 0.0]))
    assert isinstance(batch_res, np.ndarray)
    assert batch_res.shape == (3,)
    assert np.allclose(batch_res, [2.5, 2.5, 2.5], atol=1e-9)

    # 2D numpy array batch
    batch_2d = e.predict(np.array([[0.0], [0.0]]))
    assert isinstance(batch_2d, np.ndarray)
    assert batch_2d.shape == (2,)
    assert np.allclose(batch_2d, [2.5, 2.5], atol=1e-9)


def test_native_predict_unified_multi_input():
    openfll = _import_native_openfll()
    engine = _build_full_engine(openfll)

    # Dict
    y_dict = engine.predict({"x1": 5.0, "x2": 0.0})
    assert math.isclose(y_dict, 20.0, abs_tol=1e-9)

    # List and tuple (single sample)
    assert math.isclose(engine.predict([5.0, 0.0]), 20.0, abs_tol=1e-9)
    assert math.isclose(engine.predict((5.0, 0.0)), 20.0, abs_tol=1e-9)

    # 1D numpy array (single sample)
    assert math.isclose(engine.predict(np.array([5.0, 0.0])), 20.0, abs_tol=1e-9)

    # Scalar on multi-input raises ValueError
    with pytest.raises(ValueError, match="Model requires 2 inputs"):
        engine.predict(5.0)

    # Sequence with wrong number of elements raises ValueError
    with pytest.raises(ValueError, match="Expected 2 inputs"):
        engine.predict([5.0])

    # 2D numpy array without explicit input_names (auto-inferred from registered inputs)
    X = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    y_batch = engine.predict(X)
    assert isinstance(y_batch, np.ndarray)
    assert y_batch.shape == (2,)
    assert math.isclose(y_batch[0], 20.0, abs_tol=1e-9)
    assert math.isclose(y_batch[1], 10.0, abs_tol=1e-9)

    # 2D numpy array with explicit input_names
    y_batch_explicit = engine.predict(X, input_names=["x1", "x2"])
    assert np.allclose(y_batch, y_batch_explicit, atol=1e-9)

    # Nested sequence (list of lists)
    y_nested = engine.predict([[5.0, 0.0], [0.0, 0.0]])
    assert isinstance(y_nested, np.ndarray)
    assert np.allclose(y_nested, y_batch, atol=1e-9)

    # Invalid types
    with pytest.raises(TypeError):
        engine.predict("invalid_input")
    with pytest.raises(TypeError):
        engine.predict(True)
