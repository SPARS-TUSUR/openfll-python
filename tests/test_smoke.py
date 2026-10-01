"""Smoke-test: openfll импортируется и базовый API работает.

Эти тесты должны проходить на любой платформе с установленным wheel.
"""
from __future__ import annotations


def test_import():
    """Просто `import openfll`."""
    import openfll
    assert openfll.SugenoEngine is not None


def test_helpers_exposed():
    """Python-only helpers доступны через openfll.monte_carlo и т.д."""
    import openfll
    assert hasattr(openfll, "monte_carlo")
    assert hasattr(openfll, "grid_2d")
    assert hasattr(openfll, "check_output_finite")


def test_add_rule_text_form(sugeno_engine_2in_1out):
    """Текстовая форма add_rule работает."""
    e = sugeno_engine_2in_1out
    e.set_input("x1", 0.0)
    e.set_input("x2", 10.0)
    e.calculate()
    y = e.get_output("y")
    assert isinstance(y, float)
    assert y > 0.5  # MF "high" в центре → правило активируется


def test_add_rule_tuple_form(sugeno_engine_empty):
    """Кортежная форма add_rule работает."""
    e = sugeno_engine_empty
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")
    e.add_membership_func("x1", "low", "triangular", [-10, 0, 10])
    e.add_membership_func("x2", "high", "triangular", [0, 10, 20])
    e.add_membership_func("y", "out", "constant", [2.5])
    e.add_rule(
        antecedent=[("x1", "low"), ("x2", "high")],
        consequent=[("y", "out")],
        t_norm="prod_and",
    )
    e.build()
    e.set_input("x1", 0.0)
    e.set_input("x2", 10.0)
    e.calculate()
    assert abs(e.get_output("y") - 2.5) < 1e-9
