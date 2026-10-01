from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_native_openfll():
    """Импортировать openfll только при доступном настоящем batch backend."""
    for module_name in ("openfll", "PyFLL", "PyFLL._core", "openfll._core"):
        sys.modules.pop(module_name, None)

    src_dir = Path.cwd() / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        importlib.invalidate_caches()

    try:
        pyfll = importlib.import_module("PyFLL")
        importlib.import_module("PyFLL._core")
    except Exception as exc:
        pytest.fail(
            "BLOCKED: настоящий native backend PyFLL._core недоступен для "
            f"batch tests: {type(exc).__name__}: {exc}"
        )

    openfll = importlib.import_module("openfll")

    assert openfll.SugenoEngine is pyfll.SugenoEngine
    assert openfll.SugenoEngine.__module__ == "PyFLL._core"
    assert hasattr(openfll.SugenoEngine, "predict_batch")
    return openfll


def _build_batch_engine(openfll, *, multi_output: bool = False):
    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")
    if multi_output:
        e.add_output_var("z")

    e.add_membership_func("x1", "x1_low", "triangular", openfll.triangular(-10, 0, 10))
    e.add_membership_func("x1", "x1_high", "triangular", openfll.triangular(0, 10, 20))
    e.add_membership_func("x2", "x2_low", "gaussian", openfll.gaussian(center=0, sigma=2))
    e.add_membership_func(
        "x2",
        "x2_high",
        "trapezoidal",
        openfll.trapezoidal(0, 5, 10, 15),
    )
    e.add_membership_func("y", "y_low", "constant", openfll.constant(5.0))
    e.add_membership_func("y", "y_high", "constant", openfll.constant(15.0))
    if multi_output:
        e.add_membership_func("z", "z_low", "constant", openfll.constant(1.0))

    e.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "y_low"')
    e.add_rule(
        antecedent=[("x1", "x1_high"), ("x2", "x2_high")],
        consequent=[("y", "y_high")],
        t_norm="prod_and",
    )
    if multi_output:
        e.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN z IS "z_low"')

    e.build()
    return e


def _explicit_scalar(engine, x1: float, x2: float) -> float:
    engine.set_input("x1", x1)
    engine.set_input("x2", x2)
    engine.calculate()
    return engine.get_output("y")


def test_native_predict_batch_exposed_through_public_openfll():
    openfll = _import_native_openfll()
    pyfll = importlib.import_module("PyFLL")

    assert openfll.SugenoEngine is pyfll.SugenoEngine
    assert openfll.SugenoEngine.__module__ == "PyFLL._core"
    assert hasattr(openfll.SugenoEngine, "predict_batch")


def test_native_predict_batch_matches_explicit_scalar_workflow_float64():
    openfll = _import_native_openfll()
    batch_engine = _build_batch_engine(openfll)
    scalar_engine = _build_batch_engine(openfll)

    X = np.array(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [7.0, 7.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    y_batch = batch_engine.predict_batch(X, input_names=["x1", "x2"])
    y_scalar = np.array([_explicit_scalar(scalar_engine, *row) for row in X])

    assert y_batch.shape == (4,)
    assert y_batch.dtype == np.float64
    assert np.allclose(y_batch, y_scalar, atol=1e-9)


def test_native_predict_batch_accepts_float32_and_int_inputs():
    openfll = _import_native_openfll()
    engine = _build_batch_engine(openfll)

    y_float32 = engine.predict_batch(
        np.array([[0.0, 0.0], [7.0, 7.0]], dtype=np.float32),
        input_names=["x1", "x2"],
    )
    y_int = engine.predict_batch(
        np.array([[0, 0], [7, 7]], dtype=np.int32),
        input_names=["x1", "x2"],
    )

    assert y_float32.shape == (2,)
    assert y_int.shape == (2,)
    assert y_float32.dtype == np.float64
    assert y_int.dtype == np.float64
    assert np.allclose(y_float32, y_int, atol=1e-9)


def test_native_predict_batch_accepts_non_contiguous_and_empty_arrays():
    openfll = _import_native_openfll()
    engine = _build_batch_engine(openfll)

    base = np.array(
        [
            [0.0, 999.0, 0.0],
            [7.0, 999.0, 7.0],
            [10.0, 999.0, 10.0],
        ],
        dtype=np.float64,
    )
    X = base[:, ::2]
    assert not X.flags.c_contiguous

    y = engine.predict_batch(X, input_names=["x1", "x2"])
    empty = engine.predict_batch(np.empty((0, 2), dtype=np.float64), input_names=["x1", "x2"])

    assert y.shape == (3,)
    assert y.dtype == np.float64
    assert empty.shape == (0,)
    assert empty.dtype == np.float64


def test_native_predict_batch_rejects_bad_shape_names_and_lifecycle():
    openfll = _import_native_openfll()
    unbuilt = openfll.SugenoEngine()
    engine = _build_batch_engine(openfll)

    with pytest.raises(RuntimeError, match="Cannot predict_batch before build"):
        unbuilt.predict_batch(np.zeros((1, 0), dtype=np.float64), input_names=[])
    with pytest.raises(ValueError, match="2-dimensional"):
        engine.predict_batch(np.zeros((2,), dtype=np.float64), input_names=["x1", "x2"])
    with pytest.raises(ValueError, match="2-dimensional"):
        engine.predict_batch(np.zeros((1, 2, 1), dtype=np.float64), input_names=["x1", "x2"])
    with pytest.raises(ValueError, match="len\\(input_names\\)"):
        engine.predict_batch(np.zeros((1, 2), dtype=np.float64), input_names=["x1"])
    with pytest.raises(ValueError, match="duplicate input name"):
        engine.predict_batch(np.zeros((1, 2), dtype=np.float64), input_names=["x1", "x1"])
    with pytest.raises(ValueError, match="every registered input"):
        engine.predict_batch(np.zeros((1, 1), dtype=np.float64), input_names=["x1"])
    with pytest.raises(IndexError, match="Unknown input variable"):
        engine.predict_batch(np.zeros((1, 2), dtype=np.float64), input_names=["x1", "ghost"])
    with pytest.raises(TypeError, match="input_names must be a sequence of strings"):
        engine.predict_batch(np.zeros((1, 2), dtype=np.float64), input_names="x1")
    with pytest.raises(TypeError, match="input_names must be a sequence of strings"):
        engine.predict_batch(np.zeros((1, 2), dtype=np.float64), input_names=["x1", 2])
    with pytest.raises(TypeError):
        engine.predict_batch(
            np.array([["bad", "data"]], dtype=object),
            input_names=["x1", "x2"],
        )


def test_native_predict_batch_rejects_multi_output_v1():
    openfll = _import_native_openfll()
    engine = _build_batch_engine(openfll, multi_output=True)

    with pytest.raises(RuntimeError, match="exactly one output"):
        engine.predict_batch(np.zeros((1, 2), dtype=np.float64), input_names=["x1", "x2"])
