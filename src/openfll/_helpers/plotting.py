"""Визуализация функций принадлежности, зарегистрированных через API OpenFLL."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from weakref import WeakKeyDictionary

import numpy as np

__all__ = ["MembershipPlot", "plot_mf"]

VariableType = Literal["all", "input", "output"]


@dataclass(frozen=True)
class MembershipPlot:
    """Рисунок и оси, созданные для одной нечёткой переменной."""

    variable_name: str
    variable_type: Literal["input", "output"]
    figure: Any
    axes: Any
    saved_path: Path | None = None


@dataclass(frozen=True)
class _MembershipFunction:
    name: str
    mf_type: str
    params: tuple[float, ...]


_VARIABLE_ROLES: WeakKeyDictionary[object, dict[str, Literal["input", "output"]]] = (
    WeakKeyDictionary()
)
_MEMBERSHIP_FUNCTIONS: WeakKeyDictionary[
    object, dict[str, list[_MembershipFunction]]
] = WeakKeyDictionary()


def _register_variable(
    engine: object,
    name: str,
    variable_type: Literal["input", "output"],
) -> None:
    _VARIABLE_ROLES.setdefault(engine, {})[name] = variable_type


def _register_membership_function(
    engine: object,
    variable_name: str,
    mf_name: str,
    mf_type: str,
    params: Sequence[float],
) -> None:
    definitions = _MEMBERSHIP_FUNCTIONS.setdefault(engine, {})
    definitions.setdefault(variable_name, []).append(
        _MembershipFunction(
            name=mf_name,
            mf_type=mf_type,
            params=tuple(float(value) for value in params),
        )
    )


def _automatic_range(
    definitions: Sequence[_MembershipFunction],
    variable_type: Literal["input", "output"],
) -> tuple[float, float]:
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for definition in definitions:
        params = definition.params
        if definition.mf_type == "triangular":
            lower_bounds.append(params[0])
            upper_bounds.append(params[2])
        elif definition.mf_type == "trapezoidal":
            lower_bounds.append(params[0])
            upper_bounds.append(params[3])
        elif definition.mf_type == "gaussian":
            center, sigma = params
            lower_bounds.append(center - 4.0 * sigma)
            upper_bounds.append(center + 4.0 * sigma)
        elif definition.mf_type == "constant" and variable_type == "output":
            lower_bounds.append(params[0])
            upper_bounds.append(params[0])

    if not lower_bounds:
        return -1.0, 1.0
    lower = min(lower_bounds)
    upper = max(upper_bounds)
    span = upper - lower
    padding = max(span * 0.05, max(abs(lower), abs(upper), 1.0) * 0.05)
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.1, 1.0)
    return lower - padding, upper + padding


def _triangular(x: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    a, b, c = params
    values = np.zeros_like(x)
    if a < b:
        rising = (x >= a) & (x <= b)
        values[rising] = (x[rising] - a) / (b - a)
    else:
        values[x == b] = 1.0
    if b < c:
        falling = (x >= b) & (x <= c)
        values[falling] = np.maximum(values[falling], (c - x[falling]) / (c - b))
    else:
        values[x == b] = 1.0
    return np.clip(values, 0.0, 1.0)


def _trapezoidal(x: np.ndarray, params: tuple[float, ...]) -> np.ndarray:
    a, b, c, d = params
    values = np.zeros_like(x)
    if a < b:
        rising = (x >= a) & (x < b)
        values[rising] = (x[rising] - a) / (b - a)
    plateau = (x >= b) & (x <= c)
    values[plateau] = 1.0
    if c < d:
        falling = (x > c) & (x <= d)
        values[falling] = (d - x[falling]) / (d - c)
    return np.clip(values, 0.0, 1.0)


def _evaluate_membership_function(
    definition: _MembershipFunction,
    x: np.ndarray,
) -> np.ndarray:
    if definition.mf_type == "triangular":
        return _triangular(x, definition.params)
    if definition.mf_type == "trapezoidal":
        return _trapezoidal(x, definition.params)
    if definition.mf_type == "gaussian":
        center, sigma = definition.params
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    if definition.mf_type == "constant":
        return np.full_like(x, definition.params[0], dtype=np.float64)
    raise ValueError(f"Unsupported membership function type: {definition.mf_type!r}")


def _safe_filename(variable_name: str, position: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", variable_name).strip("._")
    return f"membership_{position:02d}_{safe_name or 'variable'}.png"


def plot_mf(
    engine: object,
    variables: str | Sequence[str] | None = None,
    *,
    variable_type: VariableType = "all",
    num_points: int = 181,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, MembershipPlot]:
    """Построить функции принадлежности входных и выходных переменных.

    По умолчанию для каждой переменной создаётся отдельный рисунок. Функции
    входных переменных отображаются как кривые степени принадлежности.
    Константные консеквенты Сугено отображаются singleton-маркерами, поскольку
    являются чёткими выходными значениями, а не нечёткими выходными множествами.

    Параметры:
        engine: Экземпляр ``openfll.SugenoEngine``, настроенный через публичный API.
        variables: Имя одной переменной, последовательность имён или ``None``
            для построения всех переменных.
        variable_type: Фильтр переменных: ``"input"``, ``"output"`` или
            ``"all"``.
        num_points: Количество точек по оси X для каждой непрерывной кривой.
        ranges: Необязательные границы оси X для отдельных переменных.
        show: Вызвать ``matplotlib.pyplot.show()`` после создания рисунков.
        save_dir: Необязательный каталог для сохранения PNG каждой переменной.

    Возвращает:
        Словарь от имени переменной к :class:`MembershipPlot`.

    Исключения:
        RuntimeError: Библиотека matplotlib не установлена.
        ValueError: Аргументы некорректны либо движок не был настроен через
            публичные методы регистрации этого пакета.
    """
    if variable_type not in {"all", "input", "output"}:
        raise ValueError("variable_type must be 'all', 'input', or 'output'")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot_mf requires matplotlib; install it with `pip install openfll_python[plot]`"
        ) from error

    roles = _VARIABLE_ROLES.get(engine, {})
    definitions_by_variable = _MEMBERSHIP_FUNCTIONS.get(engine, {})
    if not roles:
        raise ValueError(
            "No variables were registered through openfll.SugenoEngine; "
            "configure the engine before calling plot_mf"
        )

    if variables is None:
        selected = list(roles)
    elif isinstance(variables, str):
        selected = [variables]
    else:
        selected = list(variables)
        if not selected:
            raise ValueError("variables cannot be empty")

    unknown = [name for name in selected if name not in roles]
    if unknown:
        raise KeyError(
            f"Unknown variables: {', '.join(unknown)}. Registered variables: "
            + ", ".join(roles)
        )
    if variable_type != "all":
        selected = [name for name in selected if roles[name] == variable_type]
    if not selected:
        raise ValueError(f"No {variable_type} variables selected")

    requested_ranges = dict(ranges or {})
    unknown_ranges = [name for name in requested_ranges if name not in roles]
    if unknown_ranges:
        raise KeyError(f"Ranges provided for unknown variables: {', '.join(unknown_ranges)}")
    for name, bounds in requested_ranges.items():
        if len(bounds) != 2:
            raise ValueError(f"Range for {name!r} must contain exactly two values")
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"Range for {name!r} must be finite and increasing")

    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, MembershipPlot] = {}
    for position, variable_name in enumerate(selected, start=1):
        definitions = definitions_by_variable.get(variable_name, [])
        if not definitions:
            raise ValueError(f"Variable {variable_name!r} has no membership functions")
        role = roles[variable_name]
        lower, upper = requested_ranges.get(
            variable_name, _automatic_range(definitions, role)
        )
        x = np.linspace(float(lower), float(upper), num_points, dtype=np.float64)
        figure, axes = plt.subplots(figsize=(8.0, 4.8))

        singleton_output = role == "output" and all(
            definition.mf_type == "constant" for definition in definitions
        )
        if singleton_output:
            for definition in definitions:
                value = definition.params[0]
                axes.vlines(value, 0.0, 1.0, linewidth=2.0)
                axes.plot(
                    [value],
                    [1.0],
                    marker="o",
                    linestyle="none",
                    label=f"{definition.name} = {value:g}",
                )
            axes.set_title(f"Sugeno singleton consequents: {variable_name}")
            axes.set_ylabel("Singleton marker")
        else:
            for definition in definitions:
                axes.plot(
                    x,
                    _evaluate_membership_function(definition, x),
                    linewidth=2.0,
                    label=definition.name,
                )
            axes.set_title(f"Membership functions: {variable_name} ({role})")
            axes.set_ylabel("Degree of membership")

        axes.set_xlabel(variable_name)
        axes.set_xlim(float(lower), float(upper))
        axes.set_ylim(-0.05, 1.05)
        axes.grid(alpha=0.25)
        axes.legend()
        figure.tight_layout()

        saved_path: Path | None = None
        if output_dir is not None:
            saved_path = output_dir / _safe_filename(variable_name, position)
            figure.savefig(saved_path, dpi=160)
        results[variable_name] = MembershipPlot(
            variable_name=variable_name,
            variable_type=role,
            figure=figure,
            axes=axes,
            saved_path=saved_path,
        )

    if show:
        plt.show()
    return results
