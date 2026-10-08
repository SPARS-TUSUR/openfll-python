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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal
from weakref import WeakKeyDictionary

try:
    import numpy as _np
except ImportError:
    _np = None  # type: ignore[assignment]

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

    def _run_scalar(
        self,
        scalar_inputs: Mapping[str, float],
        output_name: str | None = None,
    ) -> float:
        input_names = _ENGINE_INPUT_NAMES.get(self, [])
        if input_names:
            missing = [name for name in input_names if name not in scalar_inputs]
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

        for name, value in scalar_inputs.items():
            self.set_input(name, value)
        self.calculate()
        return self.get_output(resolved_output_name)

    def predict(
        self,
        inputs: Any,
        output_name: str | None = None,
        *,
        input_names: Sequence[str] | None = None,
    ) -> Any:
        """Выполнить inference для скаляра, словаря, последовательности или NumPy-батча."""
        if isinstance(inputs, Mapping):
            return _run_scalar(self, inputs, output_name)

        if _np is not None and isinstance(inputs, _np.ndarray):
            if inputs.ndim == 2:
                resolved_input_names = input_names
                if resolved_input_names is None:
                    registered = _ENGINE_INPUT_NAMES.get(self, [])
                    if not registered:
                        raise ValueError(
                            "predict with 2D array requires input_names or previously registered input variables"
                        )
                    resolved_input_names = registered
                return self.predict_batch(inputs, input_names=resolved_input_names)
            if inputs.ndim == 1:
                registered = _ENGINE_INPUT_NAMES.get(self, [])
                if len(registered) == 1:
                    if inputs.shape[0] == 1:
                        return _run_scalar(self, {registered[0]: float(inputs[0])}, output_name)
                    resolved_input_names = input_names or registered
                    return self.predict_batch(inputs.reshape(-1, 1), input_names=resolved_input_names)
                if len(registered) > 1 and inputs.shape[0] == len(registered):
                    sample_dict = {name: float(val) for name, val in zip(registered, inputs)}
                    return _run_scalar(self, sample_dict, output_name)
                raise ValueError(
                    f"1D array of shape {inputs.shape} does not match {len(registered)} registered input variables ({', '.join(registered)})"
                )
            if inputs.ndim == 0:
                inputs = float(inputs)

        is_scalar = (
            isinstance(inputs, (int, float))
            or (_np is not None and isinstance(inputs, _np.number))
        ) and not isinstance(inputs, bool)
        if is_scalar:
            registered = _ENGINE_INPUT_NAMES.get(self, [])
            if len(registered) == 1:
                return _run_scalar(self, {registered[0]: float(inputs)}, output_name)
            if len(registered) == 0:
                raise ValueError("No input variables registered to map scalar input")
            raise ValueError(
                f"Model requires {len(registered)} inputs ({', '.join(registered)}), "
                f"but a single scalar was provided. Provide a dict, sequence, or 2D array."
            )

        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 0:
                raise ValueError("inputs sequence cannot be empty")
            if isinstance(inputs[0], (list, tuple)) or (_np is not None and isinstance(inputs[0], _np.ndarray)):
                if _np is None:
                    raise RuntimeError("NumPy is required for batch inference")
                arr = _np.asarray(inputs)
                resolved_input_names = input_names
                if resolved_input_names is None:
                    registered = _ENGINE_INPUT_NAMES.get(self, [])
                    if not registered:
                        raise ValueError(
                            "predict with batch sequence requires input_names or previously registered input variables"
                        )
                    resolved_input_names = registered
                return self.predict_batch(arr, input_names=resolved_input_names)

            registered = _ENGINE_INPUT_NAMES.get(self, [])
            if len(inputs) != len(registered):
                raise ValueError(
                    f"Expected {len(registered)} inputs ({', '.join(registered)}), but got {len(inputs)} values in sequence"
                )
            sample_dict = {name: float(val) for name, val in zip(registered, inputs)}
            return _run_scalar(self, sample_dict, output_name)

        raise TypeError(
            f"inputs must be a mapping, scalar number, sequence, or numpy ndarray, got {type(inputs).__name__}"
        )

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


# ============================================================================
# Python-only helpers (`openfll._helpers.*`).
#
# Эти модули используют ТОЛЬКО публичный Python-API (SugenoEngine) и
# развиваются независимо от C++ ядра. В минорных релизах wheel они могут
# расти свободно, не требуя пересборки .pyd.
#
# Импортируем здесь, чтобы коллеги могли:
#   from openfll import monte_carlo
# вместо
#   from openfll._helpers.sampling import monte_carlo
# ============================================================================
if TYPE_CHECKING:
    # Для IDE / mypy / mkdocstrings — type hints подтягиваются без runtime-импорта.
    from ._helpers.sampling import check_output_finite, grid_2d, monte_carlo
else:
    try:
        from ._helpers.sampling import check_output_finite, grid_2d, monte_carlo
        __all__ = [*__all__, "monte_carlo", "grid_2d", "check_output_finite"]
    except ImportError:
        # Минимальная установка (только wheel, без _helpers) — fallback.
        pass
