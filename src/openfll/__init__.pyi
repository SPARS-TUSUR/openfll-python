"""Type stubs for openfll — для IDE (Pylance, Pyright) и mypy.

Дублирует сигнатуры из R"doc()" в C++ binding. CI проверяет
синхронность через scripts/gen_stubs.py.
"""
from typing import List, Literal, Optional, Tuple, Union

# ============================================================================
# Literal типы
# ============================================================================
MfType = Literal[
    "constant",
    "triangular",
    "trapezoidal",
    "gaussian",
    "polynomial",
]

TNorms = Literal[
    "min",
    "prod_and",
    "bounded_diff",
    "drastic_prod",
    "einstein_prod",
    "hamacher_prod",
]

SNorms = Literal[
    "max",
    "algebraic_sum",
    "bounded_sum",
    "drastic_sum",
    "einstein_sum",
    "hamacher_sum",
]


# ============================================================================
# SugenoEngine — публичный класс
# ============================================================================
class SugenoEngine:
    """Linear facade for Sugeno fuzzy logic model.

    See README.md and docs/ for full usage examples.
    Online docs: https://spars-tusur.github.io/openfll-python/
    """

    def __init__(self) -> None:
        """Create an empty Sugeno engine."""
        ...

    # ----- Configuration -----
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

    # ----- Execution -----
    def set_input(self, name: str, value: float) -> None: ...
    def get_output(self, name: str) -> float: ...
    def calculate(self) -> None: ...

    # ----- Introspection -----
    def is_built(self) -> bool: ...


# ============================================================================
# Python-only helpers (submodule `openfll._helpers`).
# Сигнатуры ниже — ручные type hints; runtime-импорт через `openfll.__init__`.
# ============================================================================
def monte_carlo(
    engine: SugenoEngine,
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> list[dict[str, float]]: ...
def grid_2d(
    engine: SugenoEngine,
    x_var: str,
    y_var: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_var: str,
    n: int = 20,
) -> tuple[list[float], list[float], list[list[float]]]: ...
def check_output_finite(
    engine: SugenoEngine,
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, int]: ...
