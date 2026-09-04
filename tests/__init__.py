# Tests for openfll wheel.

# Этот каталог существует чтобы cibuildwheel нашёл тесты для wheel.
# Реальные тесты — в приватном OpenFLL репо; здесь — только smoke-тест,
# который можно запустить прямо на собранном wheel.

import pytest


def test_import():
    """openfll импортируется и SugenoEngine создаётся."""
    import openfll
    assert hasattr(openfll, "SugenoEngine")
    assert hasattr(openfll, "MfType")
    assert hasattr(openfll, "TNorms")
    assert hasattr(openfll, "SNorms")


def test_sugeno_engine_roundtrip():
    """Минимальный Sugeno-цикл: 2 правила, вычислить, получить выход."""
    import openfll

    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")

    e.add_membership_func("x1", "low",  "triangular", [-10, 0, 10])
    e.add_membership_func("x1", "high", "triangular", [0, 10, 20])
    e.add_membership_func("x2", "low",  "gaussian", [-5, 2])
    e.add_membership_func("x2", "high", "gaussian", [5, 2])
    e.add_membership_func("y",  "lo",  "constant", [10.0])
    e.add_membership_func("y",  "hi",  "constant", [30.0])

    e.add_rule('IF x1 IS "low" AND x2 IS "low" THEN y IS "lo"')
    e.add_rule('IF x1 IS "high" OR x2 IS "high" THEN y IS "hi"')
    e.add_rule('IF x1 IS "low" AND[prod_and] x2 IS "high" THEN y IS "lo"')

    e.build()
    e.set_input("x1", 5.0)
    e.set_input("x2", 0.0)
    e.calculate()
    y = e.get_output("y")
    # y должен быть в разумных пределах [lo, hi] = [10, 30]
    assert 10.0 <= y <= 30.0
