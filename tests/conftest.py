"""Общие фикстуры для pytest.

Важно: эти фикстуры предполагают, что openfll уже установлен в окружение
(`pip install openfll` или `pip install -e .`). CI делает это первым шагом.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def sugeno_engine_2in_1out():
    """Эталонный SugenoEngine с 2 входами, 1 выходом, 1 правилом.

    Используется в большинстве тестов: быстро строится, предсказуемый выход.
    """
    import openfll

    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")
    e.add_membership_func("x1", "low", "triangular", [-10, 0, 10])
    e.add_membership_func("x2", "high", "triangular", [0, 10, 20])
    e.add_membership_func("y", "c", "constant", [1.0])
    e.add_rule('IF x1 IS "low" AND x2 IS "high" THEN y IS "c"')
    e.build()
    return e


@pytest.fixture
def sugeno_engine_empty():
    """Пустой SugenoEngine — для тестов исключений."""
    import openfll
    return openfll.SugenoEngine()
