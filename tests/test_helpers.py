"""Тесты для openfll._helpers (monte_carlo, grid_2d, check_output_finite)."""
from __future__ import annotations

import math

import pytest

import openfll
from openfll import check_output_finite, grid_2d, monte_carlo


@pytest.fixture
def always_firing_engine():
    """SugenoEngine с 2 входами, правило срабатывает на всём пространстве.

    Используется для тестов, где выход должен быть всегда finite:
    оба MF (на x и y) — triangular с mu > 0 на всём диапазоне.
    """
    e = openfll.SugenoEngine()
    e.add_input_var("x")
    e.add_input_var("y")  # dummy 2-й вход — Sugeno требует ≥2 переменных
    e.add_output_var("z")
    # Широкие MF, mu > 0 везде в диапазоне тестов
    e.add_membership_func("x", "anyx", "triangular", [-100, 0, 100])
    e.add_membership_func("y", "anyy", "triangular", [-100, 0, 100])
    e.add_membership_func("z", "out", "constant", [1.0])
    e.add_rule('IF x IS "anyx" AND y IS "anyy" THEN z IS "out"')
    e.build()
    return e


def test_monte_carlo_returns_n_samples(always_firing_engine):
    rows = monte_carlo(
        always_firing_engine,
        var_inputs=[("x", (-10, 10)), ("y", (-10, 10))],
        var_outputs=["z"],
        n_samples=50,
        seed=42,
    )
    assert len(rows) == 50
    assert all("z" in r for r in rows)
    assert all(isinstance(r["z"], float) for r in rows)


def test_monte_carlo_seed_is_deterministic(always_firing_engine):
    rows_a = monte_carlo(
        always_firing_engine, [("x", (-10, 10)), ("y", (-10, 10))],
        ["z"], n_samples=20, seed=123,
    )
    rows_b = monte_carlo(
        always_firing_engine, [("x", (-10, 10)), ("y", (-10, 10))],
        ["z"], n_samples=20, seed=123,
    )
    assert rows_a == rows_b


def test_monte_carlo_all_finite(always_firing_engine):
    """Правило всегда активно в этом диапазоне → выход всегда finite."""
    rows = monte_carlo(
        always_firing_engine,
        var_inputs=[("x", (-10, 10)), ("y", (-10, 10))],
        var_outputs=["z"],
        n_samples=30,
        seed=42,
    )
    assert all(math.isfinite(r["z"]) for r in rows)
    # constant MF → все выходы = 1.0
    assert all(r["z"] == 1.0 for r in rows)


def test_grid_2d_shape(always_firing_engine):
    xs, ys, g = grid_2d(always_firing_engine, "x", "y", (-10, 10), (0, 20), "z", n=5)
    assert len(xs) == 5
    assert len(ys) == 5
    assert len(g) == 5
    assert all(len(row) == 5 for row in g)


def test_grid_2d_endpoints_correct(always_firing_engine):
    """Краевые точки сетки = lo и hi из range."""
    xs, ys, _ = grid_2d(always_firing_engine, "x", "y", (-10, 10), (0, 20), "z", n=5)
    assert xs[0] == -10.0 and xs[-1] == 10.0
    assert ys[0] == 0.0 and ys[-1] == 20.0


def test_grid_2d_all_finite(always_firing_engine):
    """С MF, покрывающим весь диапазон, все точки = constant output."""
    xs, ys, g = grid_2d(always_firing_engine, "x", "y", (-10, 10), (0, 20), "z", n=4)
    for row in g:
        for v in row:
            assert math.isfinite(v)
            assert v == 1.0  # constant MF → 1.0


def test_check_output_finite_runs(always_firing_engine):
    stats = check_output_finite(
        always_firing_engine,
        var_inputs=[("x", (-10, 10)), ("y", (-10, 10))],
        var_outputs=["z"],
        n_samples=50,
    )
    assert stats["total"] == 50
    assert stats["ok"] + stats["bad"] == stats["total"]
    assert stats["bad"] == 0


def test_monte_carlo_with_unknown_var_raises(always_firing_engine):
    """Engine бросает IndexError на незарегистрированное имя переменной."""
    with pytest.raises(IndexError):
        monte_carlo(
            always_firing_engine, [("not_a_var", (0, 1))],
            ["z"], n_samples=2,
        )


def test_grid_2d_n_1_single_point(always_firing_engine):
    """n=1 — крайний случай: одна точка = lo из range."""
    xs, ys, g = grid_2d(always_firing_engine, "x", "y", (0, 5), (0, 5), "z", n=1)
    assert xs == [0.0]
    assert ys == [0.0]
    assert len(g) == 1 and len(g[0]) == 1
