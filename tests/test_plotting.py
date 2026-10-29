from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openfll


def _plotting_engine() -> openfll.SugenoEngine:
    engine = openfll.SugenoEngine()
    engine.add_input_var("x1")
    engine.add_input_var("x2")
    engine.add_output_var("y")
    engine.add_membership_func("x1", "x1_low", "triangular", [-10.0, 0.0, 10.0])
    engine.add_membership_func("x1", "x1_high", "triangular", [0.0, 10.0, 20.0])
    engine.add_membership_func("x2", "x2_center", "gaussian", [0.0, 2.0])
    engine.add_membership_func("x2", "x2_high", "trapezoidal", [0.0, 5.0, 10.0, 15.0])
    engine.add_membership_func("y", "negative", "constant", [-5.0])
    engine.add_membership_func("y", "positive", "constant", [10.0])
    engine.add_rule(
        antecedent=[("x1", "x1_low"), ("x2", "x2_center")],
        consequent=[("y", "negative")],
    )
    engine.add_rule(
        antecedent=[("x1", "x1_high"), ("x2", "x2_high")],
        consequent=[("y", "positive")],
    )
    engine.build()
    return engine


def _close_plots(plots: dict[str, openfll.MembershipPlot]) -> None:
    for result in plots.values():
        plt.close(result.figure)


def test_plot_mf_builds_axes_for_all_inputs_and_sugeno_output(tmp_path: Path):
    engine = _plotting_engine()
    plots = engine.plot_mf(show=False, save_dir=tmp_path)
    try:
        assert list(plots) == ["x1", "x2", "y"]
        assert plots["x1"].variable_type == "input"
        assert plots["y"].variable_type == "output"
        assert plots["x1"].axes.get_xlabel() == "x1"
        assert plots["x1"].axes.get_ylabel() == "Degree of membership"
        assert len(plots["x1"].axes.lines) == 2
        assert max(float(line.get_ydata().max()) for line in plots["x1"].axes.lines) > 0.98

        assert plots["y"].axes.get_xlabel() == "y"
        assert plots["y"].axes.get_ylabel() == "Singleton marker"
        assert "Sugeno singleton" in plots["y"].axes.get_title()
        assert len(plots["y"].axes.collections) == 2
        assert all(result.saved_path is not None for result in plots.values())
        assert all(result.saved_path.is_file() for result in plots.values())
    finally:
        _close_plots(plots)


def test_function_form_selects_variable_and_honors_axis_range():
    engine = _plotting_engine()
    plots = openfll.plot_mf(
        engine,
        "x1",
        ranges={"x1": (-20.0, 30.0)},
        num_points=101,
        show=False,
    )
    try:
        assert list(plots) == ["x1"]
        assert plots["x1"].axes.get_xlim() == pytest.approx((-20.0, 30.0))
        assert len(plots["x1"].axes.lines[0].get_xdata()) == 101
    finally:
        _close_plots(plots)


def test_plot_mf_filters_variable_type_and_validates_arguments():
    engine = _plotting_engine()
    output_plots = engine.plot_mf(variable_type="output", show=False)
    try:
        assert list(output_plots) == ["y"]
    finally:
        _close_plots(output_plots)

    with pytest.raises(KeyError, match="Unknown variables"):
        engine.plot_mf("missing", show=False)
    with pytest.raises(ValueError, match="num_points"):
        engine.plot_mf(num_points=1, show=False)
    with pytest.raises(ValueError, match="finite and increasing"):
        engine.plot_mf("x1", ranges={"x1": (1.0, 1.0)}, show=False)


def test_plot_mf_supports_a_single_input_variable():
    engine = openfll.SugenoEngine()
    engine.add_input_var("x1")
    engine.add_membership_func("x1", "low", "triangular", [-1.0, 0.0, 1.0])

    plots = engine.plot_mf(show=False)
    try:
        assert list(plots) == ["x1"]
        assert len(plots["x1"].axes.lines) == 1
        assert plots["x1"].axes.get_xlabel() == "x1"
    finally:
        _close_plots(plots)


def test_plot_mf_supports_multiple_inputs_and_outputs():
    engine = openfll.SugenoEngine()
    for name in ("x1", "x2", "x3"):
        engine.add_input_var(name)
        engine.add_membership_func(
            name, f"{name}_low", "triangular", [-1.0, 0.0, 1.0]
        )
        engine.add_membership_func(
            name, f"{name}_high", "triangular", [0.0, 1.0, 2.0]
        )
    for name, value in (("y1", 10.0), ("y2", 20.0)):
        engine.add_output_var(name)
        engine.add_membership_func(name, f"{name}_result", "constant", [value])

    plots = engine.plot_mf(show=False)
    try:
        assert list(plots) == ["x1", "x2", "x3", "y1", "y2"]
        assert all(plots[name].variable_type == "input" for name in ("x1", "x2", "x3"))
        assert all(plots[name].variable_type == "output" for name in ("y1", "y2"))
    finally:
        _close_plots(plots)
