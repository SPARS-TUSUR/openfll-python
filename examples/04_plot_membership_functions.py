"""Построение всех входных функций принадлежности и выходов Сугено."""

from pathlib import Path

import openfll


engine = openfll.SugenoEngine()
engine.add_input_var("x1")
engine.add_input_var("x2")
engine.add_output_var("y")

engine.add_membership_func("x1", "x1_low", "triangular", [-10.0, 0.0, 10.0])
engine.add_membership_func("x1", "x1_high", "triangular", [0.0, 10.0, 20.0])
engine.add_membership_func("x2", "x2_low", "gaussian", [-5.0, 2.0])
engine.add_membership_func("x2", "x2_high", "trapezoidal", [0.0, 5.0, 10.0, 15.0])
engine.add_membership_func("y", "small", "constant", [10.0])
engine.add_membership_func("y", "large", "constant", [30.0])

engine.add_rule(
    antecedent=[("x1", "x1_low"), ("x2", "x2_low")],
    consequent=[("y", "small")],
)
engine.add_rule(
    antecedent=[("x1", "x1_high"), ("x2", "x2_high")],
    consequent=[("y", "large")],
)
engine.build()

output_dir = Path(__file__).resolve().parent / "membership_plots"
plots = engine.plot_mf(show=True, save_dir=output_dir)
for variable_name, result in plots.items():
    print(f"{variable_name}: {result.saved_path}")
