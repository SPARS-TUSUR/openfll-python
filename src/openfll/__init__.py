"""Публичный Python API для OpenFLL.

`openfll` — это пакет, предназначенный для непосредственного использования. 
Во время выполнения он делегирует задачи внутреннему пакету-бэкенду `PyFLL`, который содержит модуль расширения на машинном коде `PyFLL._core`.

Ожидаемая цепочка вызовов во время выполнения:

    openfll -> PyFLL -> PyFLL._core -> OFLL::Linear::SugenoEngine

Приведенный ниже класс сохраняется для целей статического анализа и генерации документации. 
При наличии встроенного бэкенда `SugenoEngine` заменяется на `PyFLL.SugenoEngine`.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Literal
from weakref import WeakKeyDictionary

__all__ = [
    "SugenoEngine",
    "MfType",
    "TNorms",
    "SNorms",
    "constant",
    "triangular",
    "trapezoidal",
    "gaussian",
]

_BACKEND_IMPORT_ERROR: BaseException | None = None
_ENGINE_INPUT_NAMES: WeakKeyDictionary[object, list[str]] = WeakKeyDictionary()
_ENGINE_OUTPUT_NAMES: WeakKeyDictionary[object, list[str]] = WeakKeyDictionary()


def _backend_error_message() -> str:
    reason = ""
    if _BACKEND_IMPORT_ERROR is not None:
        reason = (
            f"\nИсходная причина: "
            f"{type(_BACKEND_IMPORT_ERROR).__name__}: {_BACKEND_IMPORT_ERROR}"
        )
    return (
        "openfll не смог загрузить внутренний backend `PyFLL` или его "
        "native module `PyFLL._core`.\n"
        "Проверьте, что пакет установлен из wheel, собранного вместе с "
        "native backend: `pip install openfll`."
        f"{reason}"
    )


# ============================================================================
# Literal типы — экспортируются для IDE/mkdocstrings
# ============================================================================
MfType = Literal[
    "constant", "triangular", "trapezoidal", "gaussian"
]
TNorms = Literal[
    "min", "prod_and", "bounded_diff", "drastic_prod", "einstein_prod", "hamacher_prod"
]
SNorms = Literal[
    "max", "algebraic_sum", "bounded_sum", "drastic_sum", "einstein_sum", "hamacher_sum"
]


def constant(value: float) -> list[float]:
    """Вернуть параметры для constant membership function."""
    return [value]


def triangular(a: float, b: float, c: float) -> list[float]:
    """Вернуть параметры для triangular membership function в порядке a, b, c."""
    return [a, b, c]


def trapezoidal(a: float, b: float, c: float, d: float) -> list[float]:
    """Вернуть параметры для trapezoidal membership function в порядке a, b, c, d."""
    return [a, b, c, d]


def gaussian(*, center: float, sigma: float) -> list[float]:
    """Вернуть параметры для gaussian membership function в порядке center, sigma."""
    return [center, sigma]


def _install_scalar_predict(engine_cls: type) -> None:
    """Добавить Python convenience predict к настоящему native engine class."""
    required_methods = (
        "add_input_var",
        "add_output_var",
        "set_input",
        "calculate",
        "get_output",
    )
    if getattr(engine_cls, "_openfll_predict_installed", False):
        return
    if not all(hasattr(engine_cls, name) for name in required_methods):
        return

    native_add_input_var = engine_cls.add_input_var
    native_add_output_var = engine_cls.add_output_var

    def add_input_var(self, name: str) -> None:
        native_add_input_var(self, name)
        _ENGINE_INPUT_NAMES.setdefault(self, []).append(name)

    def add_output_var(self, name: str) -> None:
        native_add_output_var(self, name)
        _ENGINE_OUTPUT_NAMES.setdefault(self, []).append(name)

    def predict(
        self,
        inputs: Mapping[str, float],
        output_name: str | None = None,
    ) -> float:
        """Выполнить scalar inference через set_input -> calculate -> get_output."""
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping of input name to scalar value")

        input_names = _ENGINE_INPUT_NAMES.get(self, [])
        if input_names:
            missing = [name for name in input_names if name not in inputs]
            if missing:
                raise ValueError(
                    "predict inputs missing registered input variables: "
                    + ", ".join(missing)
                )

        resolved_output_name = output_name
        if resolved_output_name is None:
            output_names = _ENGINE_OUTPUT_NAMES.get(self, [])
            if len(output_names) == 1:
                resolved_output_name = output_names[0]
            else:
                raise ValueError(
                    "predict requires output_name unless exactly one output "
                    "variable was registered through add_output_var"
                )

        for name, value in inputs.items():
            self.set_input(name, value)
        self.calculate()
        return self.get_output(resolved_output_name)

    engine_cls.add_input_var = add_input_var
    engine_cls.add_output_var = add_output_var
    engine_cls.predict = predict
    engine_cls._openfll_predict_installed = True


class SugenoEngine:
    """Placeholder, который заменяется реальным PyFLL.SugenoEngine."""

    def __init__(self) -> None:
        """Создать engine или явно сообщить, что native backend не загружен."""
        raise ImportError(_backend_error_message()) from _BACKEND_IMPORT_ERROR


# ============================================================================
# Время выполнения: загрузка существующего пакета бэкенда. openfll намеренно не
# импортирует и не включает в поставку собственное расширение _core.
# ============================================================================
try:
    _PyFLL = importlib.import_module("PyFLL")
    _RealSugenoEngine = _PyFLL.SugenoEngine
    if getattr(_RealSugenoEngine, "__module__", "") != "PyFLL._core":
        raise ImportError(
            "PyFLL импортирован, но PyFLL.SugenoEngine не загружен из "
            f"PyFLL._core (получен модуль "
            f"{getattr(_RealSugenoEngine, '__module__', '<unknown>')!r})"
        )

    SugenoEngine = _RealSugenoEngine
    _install_scalar_predict(SugenoEngine)
    del _PyFLL
    del _RealSugenoEngine
except ImportError as exc:
    # Для документации и type checking стаб остаётся доступным, но в обычном
    # runtime его нельзя использовать как рабочий engine.
    _BACKEND_IMPORT_ERROR = exc
    pass
