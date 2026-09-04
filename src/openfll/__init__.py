"""openfll — Python биндинги для OpenFLL Linear Facade.

Этот модуль загружает .pyd-расширение и подменяет стаб реальным классом.
На уровне статического анализа (griffe/mkdocstrings/mypy) SugenoEngine
определён здесь с type hints; в runtime он заменяется на настоящий.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from typing import List, Literal, Tuple, Union

__all__ = ["SugenoEngine", "MfType", "TNorms", "SNorms"]


# ============================================================================
# Literal типы — экспортируются для IDE/mkdocstrings
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
# Стаб SugenoEngine — для IDE / mkdocstrings / mypy
# ============================================================================
class SugenoEngine:
    """Linear facade for Sugeno fuzzy logic model.

    Build a model by:
      1. Registering input/output variables (add_input_var / add_output_var).
      2. Adding membership functions to those variables
         (add_membership_func: name + type + parameter list).
      3. Adding rules as plain strings
         (add_rule: 'IF x1 IS "low" AND x2 IS "high" THEN y IS "fast"').
      4. Calling build() to compile the model.
      5. Setting input values (set_input), running calculate(), and
         reading outputs (get_output).

    Rules support explicit norms:
      IF x1 IS "low" AND[prod_and] x2 IS "high" OR[algebraic_sum] x3 IS "slow"
      THEN y IS "open"

    See docs at https://spars-tusur.github.io/openfll-python/
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
    def set_default_t_norm(self, name: TNorms) -> None: ...
    def set_default_s_norm(self, name: SNorms) -> None: ...
    def build(self) -> None: ...
    def set_input(self, name: str, value: float) -> None: ...
    def get_output(self, name: str) -> float: ...
    def calculate(self) -> None: ...
    def is_built(self) -> bool: ...


# ============================================================================
# Runtime: загрузить .pyd и подменить стаб реальным классом
# ============================================================================
def _load_native() -> None:
    """Найти .pyd в директории пакета и загрузить через importlib.

    Wheel содержит:
      openfll.cp310-win_amd64.pyd    (для Python 3.10)
      openfll.cp314-...pyd          (для Python 3.14)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # cp310
    candidates = [
        "openfll.cp310-win_amd64.pyd",
        "openfll.cp314-win_amd64.pyd",
        "openfll.cp314-mingw_x86_64_msvcrt_gnu.pyd",
        "openfll.pyd",  # fallback для разработки
    ]
    for name in candidates:
        pyd_path = os.path.join(here, name)
        if os.path.isfile(pyd_path):
            loader = importlib.machinery.ExtensionFileLoader("openfll._native", pyd_path)
            spec = importlib.machinery.ModuleSpec("openfll._native", loader, origin=pyd_path)
            native = importlib.util.module_from_spec(spec)
            sys.modules["openfll._native"] = native
            loader.exec_module(native)
            globals()["SugenoEngine"] = native.SugenoEngine
            return
    # .pyd не найден — оставляем стаб; инстанцирование упадёт с понятной ошибкой.
    # Эта ситуация нормальна только в исходниках без wheel (например, в mkdocs build).
    pass


_load_native()
