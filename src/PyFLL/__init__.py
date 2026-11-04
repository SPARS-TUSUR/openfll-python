"""PyFLL — Python биндинги для OpenFLL Linear Facade.

Стабы классов (для griffe / mypy / IDE) + runtime загрузка .pyd через
стандартный `from ._core import ...` — без importlib-хаков.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from typing import TYPE_CHECKING, Any, List, Literal, Sequence, Tuple, Union

__all__ = [
    "SugenoEngine",
    "MfType",
    "TNorms",
    "SNorms",
    "load_from_string",
    "load_from_file",
]


# ============================================================================
# Literal типы — экспортируются как обычные переменные модуля
# ============================================================================
MfType = Literal[
    "constant", "triangular", "trapezoidal", "gaussian", "polynomial"
]
TNorms = Literal[
    "min", "prod_and", "bounded_diff", "drastic_prod", "einstein_prod", "hamacher_prod"
]
SNorms = Literal[
    "max", "algebraic_sum", "bounded_sum", "drastic_sum", "einstein_sum", "hamacher_sum"
]


# ============================================================================
# Стаб SugenoEngine — type hints и docstring для IDE/mkdocstrings
# ============================================================================
# В runtime этот класс заменяется на реальный из _core.pyd через
# `from ._core import SugenoEngine` ниже. Здесь он нужен только для
# статического анализа (griffe, mypy, IDE).
class SugenoEngine:
    """Linear facade for Sugeno fuzzy logic model.

    Two forms of rules are supported:
      - Textual: add_rule('IF x1 IS "low" AND x2 IS "high" THEN y IS "open"')
        Supports AND/OR with optional [norm_name] for explicit t/s-norms.
      - Structural (tuple): add_rule(
            antecedent=[("x1", "low"), ("x2", "high")],
            consequent=[("y", "open")],
            t_norm="prod_and",
        )
        Preferred for programmatic rule generation; AND-only.

    See full API in PyFLL.pyi (mirrored here for mkdocstrings static parsing).
    """

    def __init__(self) -> None:
        """Create an empty Sugeno engine."""
        ...

    def add_input_var(self, name: str) -> None: ...
    def add_output_var(self, name: str) -> None: ...

    def add_membership_func(
        self,
        var_name: str,
        mf_name: str,
        type: MfType,
        params: Union[List[float], Tuple[float, ...]],
    ) -> None: ...

    def add_rule(self, rule: str) -> None: ...
    def add_rule(
        self,
        antecedent: List[Tuple[str, str]],
        consequent: List[Tuple[str, str]],
        t_norm: Union[TNorms, str, None] = None,
        s_norm: Union[SNorms, str, None] = None,
    ) -> None: ...
    def set_default_t_norm(self, name: TNorms) -> None: ...
    def set_default_s_norm(self, name: SNorms) -> None: ...
    def build(self) -> None: ...
    def set_input(self, name: str, value: float) -> None: ...
    def get_output(self, name: str) -> float: ...
    def calculate(self) -> None: ...
    def predict_batch(self, X: object, *, input_names: Sequence[str]) -> Any: ...
    def is_built(self) -> bool: ...

    # ----- Сериализация -----
    def save_to_string(self) -> str:
        """Serialize the built model to a JSON string."""
        ...

    def save_to_file(self, path: str) -> None:
        """Serialize the built model to a JSON file."""
        ...

    def to_dict(self) -> Any:
        """Return the serialized model as a plain Python dict."""
        ...


# ============================================================================
# Модульные функции загрузки модели.
#
# Статические сигнатуры (для griffe/mypy/IDE) живут в ветке TYPE_CHECKING,
# реальные функции — в runtime-ветке. griffe вычисляет TYPE_CHECKING как True
# и не видит runtime-ветку, поэтому не пытается резолвить алиасы на
# скомпилированный _core (иначе mkdocs build --strict падает с
# AliasResolutionError).
# ============================================================================
if TYPE_CHECKING:
    def load_from_string(json: str) -> "SugenoEngine":
        """Load a model from a JSON string produced by save_to_string().

        Returns a ready-to-use SugenoEngine: it is already built, so set_input(),
        calculate() and get_output() work immediately.

        Note:
          - The loaded engine is inference-only: add_input_var(),
            add_membership_func() and add_rule() raise RuntimeError, exactly as
            after a normal build().
          - Membership-function names are not part of the JSON format, so the
            loaded engine cannot report them. Variable names are preserved.
          - Current input values are not persisted.

        Raises:
            ValueError: malformed JSON or unknown model type.
        """
        ...

    def load_from_file(path: str) -> "SugenoEngine":
        """Load a model from a JSON file written by save_to_file().

        See load_from_string() for the semantics of the returned engine.

        Raises:
            ValueError: malformed JSON or unknown model type.
            OSError: the file could not be read.
        """
        ...
else:
    # Runtime: загрузка SugenoEngine и функций загрузки из _core.pyd.
    # Если .pyd не собран (например, в mkdocs build без wheel) — остаются стабы.
    try:
        from ._core import SugenoEngine as _RealSugenoEngine, MamdaniEngine, IModel
        from ._core import load_from_file, load_from_string

        SugenoEngine = _RealSugenoEngine
        del _RealSugenoEngine
        __all__ = [*__all__, "MamdaniEngine", "IModel"]
    except ImportError:
        def load_from_string(*args: Any, **kwargs: Any) -> Any:
            """Stub: the PyFLL extension module is not built."""
            raise ImportError(
                "PyFLL._core is unavailable: the extension module is not built"
            )

        def load_from_file(*args: Any, **kwargs: Any) -> Any:
            """Stub: the PyFLL extension module is not built."""
            raise ImportError(
                "PyFLL._core is unavailable: the extension module is not built"
            )
