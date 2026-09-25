"""Batch-инференс: прогон множества входов через SugenoEngine.

Полезно для:
  - Monte-Carlo валидации модели (проверить, что выход всегда конечный)
  - Построения таблицы покрытия (coverage grid) для 2D-визуализации
  - Стресстестов: проверить модель на 100k случайных точках за секунды

Не требует расширения C++ API — работает только через `set_input`/`calculate`/`get_output`.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfll import SugenoEngine

__all__ = ["monte_carlo", "grid_2d", "check_output_finite"]


def monte_carlo(
    engine: "SugenoEngine",
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Случайные входы → батч результатов.

    Args:
        engine: собранный SugenoEngine (после `build()`).
        var_inputs: список (var_name, (min, max)) для каждого входа.
        var_outputs: имена выходов, которые нужно собрать.
        n_samples: количество случайных точек.
        seed: зерно random.Random; None = системное.

    Returns:
        Список длины n_samples; каждый элемент — dict {var_name: value}.

    Raises:
        RuntimeError: если `engine` не собран (`build()` не вызван).
        KeyError: если имя переменной не зарегистрировано в engine.

    Example:
        >>> rows = monte_carlo(
        ...     engine,
        ...     var_inputs=[("x1", (-10, 10)), ("x2", (0, 20))],
        ...     var_outputs=["y"],
        ...     n_samples=100, seed=42,
        ... )
        >>> len(rows)
        100
    """
    rng = random.Random(seed)
    results: list[dict[str, float]] = []
    for _ in range(n_samples):
        for var_name, (lo, hi) in var_inputs:
            engine.set_input(var_name, rng.uniform(lo, hi))
        engine.calculate()
        results.append({v: engine.get_output(v) for v in var_outputs})
    return results


def grid_2d(
    engine: "SugenoEngine",
    x_var: str,
    y_var: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_var: str,
    n: int = 20,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Регулярная 2D-сетка (x, y) → значения out_var.

    Используется для heatmap/3D-визуализации выхода нечёткой модели.

    Returns:
        (xs, ys, grid): xs[i] и ys[j] — координаты, grid[i][j] — out_var(xs[i], ys[j]).

    Example:
        >>> xs, ys, g = grid_2d(e, "x1", "x2", (-10, 10), (0, 20), "y", n=20)
        >>> len(xs) == len(ys) == 20
        True
    """
    xs = [_linspace(x_range, n, i) for i in range(n)]
    ys = [_linspace(y_range, n, j) for j in range(n)]
    grid: list[list[float]] = []
    for x in xs:
        row: list[float] = []
        for y in ys:
            engine.set_input(x_var, x)
            engine.set_input(y_var, y)
            engine.calculate()
            row.append(engine.get_output(out_var))
        grid.append(row)
    return xs, ys, grid


def _linspace(rng: tuple[float, float], n: int, i: int) -> float:
    lo, hi = rng
    if n == 1:
        return lo
    return lo + (hi - lo) * i / (n - 1)


def check_output_finite(
    engine: "SugenoEngine",
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, int]:
    """Сэмплирует engine и считает, сколько раз выход был NaN/Inf.

    Args:
        engine: собранный SugenoEngine.
        var_inputs: список (var_name, (min, max)) для каждого входа.
        var_outputs: имена выходов, которые нужно проверять.
        n_samples: количество случайных точек.
        seed: random.Random; None = системное.

    Returns:
        {"total": N, "bad": K, "ok": N - K} — счётчики.
        Используйте для регрессионных тестов: assert stats["bad"] == 0.

    Example:
        >>> stats = check_output_finite(
        ...     e,
        ...     var_inputs=[("x1", (-10, 10)), ("x2", (0, 20))],
        ...     var_outputs=["y"],
        ...     n_samples=100,
        ... )
        >>> assert stats["bad"] == 0
    """
    import math
    bad = 0
    total = 0
    for sample in monte_carlo(engine, var_inputs, var_outputs, n_samples, seed):
        for v in sample.values():
            total += 1
            if not math.isfinite(v):
                bad += 1
    return {"total": total, "bad": bad, "ok": total - bad}
