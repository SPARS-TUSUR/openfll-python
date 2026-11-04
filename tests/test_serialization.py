"""Проверки сохранения обеих моделей и расчётов на настоящем нативном модуле."""
from __future__ import annotations

import gc
import json

import numpy as np
import pytest

import openfll


def make_engine(kind="sugeno", defuzzer="wtaver"):
    engine = openfll.MamdaniEngine() if kind == "mamdani" else openfll.SugenoEngine()
    engine.add_input_var("temperature")
    engine.add_output_var("power")
    engine.add_membership_func("temperature", "cold", "triangular", [0, 0, 10])
    engine.add_membership_func("temperature", "hot", "triangular", [0, 10, 10])
    if kind == "mamdani":
        engine.set_defuzzer("centroid")
        engine.set_integrator(-1, 11, 0.01)
        engine.add_membership_func("power", "low", "triangular", [0, 2, 4])
        engine.add_membership_func("power", "high", "triangular", [6, 8, 10])
    else:
        engine.set_defuzzer(defuzzer)
        engine.add_membership_func("power", "low", "constant", [2])
        engine.add_membership_func("power", "high", "constant", [8])
    engine.add_rule('IF temperature IS "cold" THEN power IS "low"')
    engine.add_rule('IF temperature IS "hot" THEN power IS "high"')
    engine.build()
    return engine


@pytest.mark.parametrize("kind,defuzzer", [("sugeno", "wtaver"), ("sugeno", "wtsum"), ("mamdani", "centroid")])
def test_roundtrip_both_paths_and_native_model_view(kind, defuzzer, tmp_path):
    engine = make_engine(kind, defuzzer)
    points = np.array([[1.], [3.], [5.], [7.], [9.]])
    expected = engine.predict(points)
    document = engine.to_dict()
    assert document == json.loads(engine.save_to_string())
    native_document = engine.model_ref().to_dict()
    assert {k: v for k, v in native_document.items() if k != "extra"} == {
        k: v for k, v in document.items() if k != "extra"
    }
    path = tmp_path / "модель с пробелами.json"
    engine.save_to_file(path)
    for restored in (openfll.load_from_string(engine.save_to_string()), openfll.load_from_file(path)):
        assert type(restored) is type(engine)
        assert restored.is_built()
        assert restored.get_input_var_names() == ["temperature"]
        assert restored.get_output_var_names() == ["power"]
        np.testing.assert_allclose(restored.predict(points), expected, rtol=1e-12, atol=1e-12)
        assert restored.predict({"temperature": 3}) == pytest.approx(expected[1])
        assert restored.predict(3) == pytest.approx(expected[1])
        assert restored.predict([3]) == pytest.approx(expected[1])
        np.testing.assert_allclose(restored.predict([1, 3, 5, 7, 9]), expected)
        assert restored.predict_batch(np.empty((0, 1)), input_names=["temperature"]).shape == (0,)
        restored.set_input("temperature", 7)
        restored.model_ref().calculate()
        assert restored.get_output("power") == pytest.approx(expected[3])
        with pytest.raises(RuntimeError):
            restored.add_input_var("new")
        with pytest.raises(RuntimeError):
            restored.set_defuzzer(defuzzer)
        with pytest.raises(ValueError, match="missing"):
            restored.predict({})
        with pytest.raises(IndexError):
            restored.find_belonging("cold", 3)
        # Повторная загрузка и удаление первого движка не должны нарушать владение объектами.
        second = openfll.load_from_string(restored.save_to_string())
        del restored
        gc.collect()
        np.testing.assert_allclose(second.predict(points), expected)


def test_sugeno_defuzzers_have_different_math_and_persist():
    # Два постоянных веса активации по 0.25: wtaver = 5; wtsum = 2.5.
    for name, expected in [("wtaver", 5), ("wtsum", 2.5)]:
        engine = openfll.SugenoEngine()
        engine.add_input_var("x")
        engine.add_output_var("y")
        engine.add_membership_func("x", "active", "constant", [0.25])
        engine.add_membership_func("y", "a", "constant", [2])
        engine.add_membership_func("y", "b", "constant", [8])
        engine.add_rule('IF x IS "active" THEN y IS "a"')
        engine.add_rule('IF x IS "active" THEN y IS "b"')
        engine.set_defuzzer(name)
        engine.build()
        assert engine.predict(0) == pytest.approx(expected)
        assert openfll.load_from_string(engine.save_to_string()).predict(0) == pytest.approx(expected)


def test_mamdani_centroid_matches_symmetric_triangle():
    engine = make_engine("mamdani")
    assert engine.predict(0) == pytest.approx(2, abs=1e-10)
    assert engine.predict(5) == pytest.approx(5, abs=1e-10)
    assert engine.to_dict()["defuzzer"]["step"] == 0.01

    single = openfll.MamdaniEngine()
    single.add_input_var("x")
    single.add_output_var("y")
    single.add_membership_func("x", "on", "constant", [1])
    single.add_membership_func("y", "triangle", "triangular", [6, 8, 10])
    single.add_rule('IF x IS "on" THEN y IS "triangle"')
    single.set_integrator(0, 12, 0.01)
    single.build()
    assert single.predict(0) == pytest.approx(8, abs=1e-10)


def test_membership_degree_and_names_before_build():
    engine = openfll.SugenoEngine()
    engine.add_input_var("z")
    engine.add_input_var("a")
    engine.add_output_var("result")
    engine.add_membership_func("z", "triangle", "triangular", [0, 5, 10])
    engine.add_membership_func("a", "gauss", "gaussian", [0, 1])
    assert engine.get_input_var_names() == ["z", "a"]
    names = engine.get_input_var_names()
    names.clear()
    assert engine.get_input_var_names() == ["z", "a"]
    assert engine.get_output_var_names() == ["result"]
    assert engine.find_belonging("triangle", 2.5) == pytest.approx(0.5)
    assert engine.find_belonging("gauss", 0) == pytest.approx(1)
    with pytest.raises(IndexError):
        engine.find_belonging("unknown", 0)


def test_model_view_keeps_owner_alive(tmp_path):
    engine = make_engine()
    view = engine.model_ref()
    engine.set_input("temperature", 3)
    del engine
    gc.collect()
    view.calculate()
    view.save_to_file(tmp_path / "view.json")
    assert openfll.load_from_file(tmp_path / "view.json").predict(3) == pytest.approx(3.8)


def test_serialization_errors(tmp_path):
    empty = openfll.SugenoEngine()
    for operation in (empty.save_to_string, empty.to_dict, empty.model_ref):
        with pytest.raises(RuntimeError):
            operation()
    with pytest.raises(RuntimeError):
        empty.save_to_file(tmp_path / "empty.json")
    engine = make_engine()
    with pytest.raises(OSError):
        engine.save_to_file(tmp_path / "missing" / "model.json")
    with pytest.raises(OSError):
        openfll.load_from_file(tmp_path / "absent.json")
    with pytest.raises(TypeError):
        openfll.load_from_file(123)
    for invalid in ("not JSON", "{}", "[]"):
        with pytest.raises(ValueError):
            openfll.load_from_string(invalid)
    document = engine.to_dict()
    document["model_type"] = "unsupported"
    with pytest.raises(ValueError):
        openfll.load_from_string(json.dumps(document))


def test_invalid_defuzzer_and_integrator():
    for engine, name in [(openfll.SugenoEngine(), "centroid"), (openfll.MamdaniEngine(), "wtsum")]:
        with pytest.raises(ValueError):
            engine.set_defuzzer(name)
    engine = openfll.MamdaniEngine()
    for args in [(1, 0, .1), (0, 1, 0), (0, 1, 2), (0, 1, float("nan"))]:
        with pytest.raises(ValueError):
            engine.set_integrator(*args)


def test_multi_input_order_and_explicit_norm_survive_loading():
    engine = openfll.SugenoEngine()
    for name in ("z", "a"):
        engine.add_input_var(name)
        engine.add_membership_func(name, name + "_low", "triangular", [0, 0, 10])
        engine.add_membership_func(name, name + "_high", "triangular", [0, 10, 10])
    engine.add_output_var("y")
    engine.add_membership_func("y", "low", "constant", [2])
    engine.add_membership_func("y", "high", "constant", [8])
    engine.add_rule('IF z IS "z_low" AND[prod_and] a IS "a_high" THEN y IS "low"')
    engine.add_rule('IF a IS "a_low" OR[algebraic_sum] z IS "z_high" THEN y IS "high"')
    engine.build()
    restored = openfll.load_from_string(engine.save_to_string())
    assert restored.get_input_var_names() == ["z", "a"]
    values = np.array([[2, 8], [4, 1], [8, 3]])
    np.testing.assert_allclose(restored.predict(values), engine.predict(values))
    assert restored.predict([2, 8]) == pytest.approx(engine.predict([2, 8]))
    np.testing.assert_allclose(restored.predict(values[:, ::-1], input_names=["a", "z"]), engine.predict(values))


@pytest.mark.parametrize("kind", ["sugeno", "mamdani"])
def test_multiple_outputs_and_disabled_consequents_survive_repeated_loading(kind):
    engine = openfll.MamdaniEngine() if kind == "mamdani" else openfll.SugenoEngine()
    engine.add_input_var("x")
    engine.add_membership_func("x", "on", "constant", [1])
    for name, center in [("z", 2), ("a", 8)]:
        engine.add_output_var(name)
        engine.add_membership_func(name, name + "_mf", "triangular" if kind == "mamdani" else "constant",
                                   [center - 2, center, center + 2] if kind == "mamdani" else [center])
        engine.add_rule(f'IF x IS "on" THEN {name} IS "{name}_mf"')
    if kind == "mamdani":
        engine.set_integrator(-1, 11, 0.01)
    engine.build()
    expected = {name: engine.predict(0, output_name=name) for name in ["z", "a"]}
    assert expected == pytest.approx({"z": 2, "a": 8})
    for _ in range(5):
        engine = openfll.load_from_string(engine.save_to_string())
        assert engine.get_output_var_names() == ["z", "a"]
        for name, value in expected.items():
            assert engine.predict({"x": 0}, output_name=name) == pytest.approx(value)


def test_native_json_without_facade_metadata():
    engine = make_engine()
    restored = openfll.load_from_string(engine.model_ref().save_to_string())
    assert restored.predict({"temperature": 3}) == pytest.approx(engine.predict(3))


@pytest.mark.parametrize("field,value", [("Vars", [-1]), ("Mfs", [999]), ("Vars", [])])
def test_malformed_rule_indices_raise_instead_of_crashing(field, value):
    doc = make_engine().to_dict()
    doc["rules"][0][field] = value
    with pytest.raises(ValueError):
        openfll.load_from_string(json.dumps(doc))
