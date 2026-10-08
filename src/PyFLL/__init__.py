"""PyFLL — Python биндинги для OpenFLL Linear Facade.

Стабы классов (для griffe / mypy / IDE) + runtime загрузка .pyd через
стандартный `from ._core import ...` — без importlib-хаков.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from typing import Any, List, Literal, Sequence, Tuple, Union

__all__ = ["SugenoEngine", "MfType", "TNorms", "SNorms"]


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


# ============================================================================
# Runtime: загрузка SugenoEngine из _core.pyd через стандартный import.
# Если .pyd не собран (например, в mkdocs build без wheel) — оставляем стаб.
# ============================================================================
try:
    from ._core import SugenoEngine as _RealSugenoEngine
    SugenoEngine = _RealSugenoEngine
    del _RealSugenoEngine
except ImportError:
    # _core.pyd не найден (исходники без wheel). Стаб остаётся — будет ошибка
    # только при инстанцировании SugenoEngine().
    pass
