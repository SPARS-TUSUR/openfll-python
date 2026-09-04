"""openfll — Python биндинги для OpenFLL Linear Facade.

Этот модуль загружает .pyd-расширение и подменяет стаб реальным классом.
На уровне статического анализа (griffe/mkdocstrings/mypy) SugenoEngine
определён здесь с type hints; в runtime он заменяется на настоящий.
"""
from __future__ import annotations

import os
import sys
from typing import List, Literal, Optional, Tuple, Union

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

    Two forms of rules are supported:
      - Textual: add_rule('IF x1 IS "low" AND x2 IS "high" THEN y IS "open"')
        Supports AND/OR with optional [norm_name] for explicit t/s-norms.
      - Structural (tuple): add_rule(
            antecedent=[("x1", "low"), ("x2", "high")],
            consequent=[("y", "open")],
            t_norm="prod_and",
        )
        Preferred for programmatic rule generation; AND-only.

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
    def is_built(self) -> bool: ...


# ============================================================================
# Runtime: загрузить .pyd и подменить стаб реальным классом
# ============================================================================
def _load_native() -> None:
    """Найти .pyd в директории пакета и загрузить через ctypes.

    Wheel содержит:
      openfll.cp314-...pyd          (для Python 3.14, MinGW)

    Почему ctypes, а не importlib:
      pybind11 экспортирует функцию PyInit_openfll (по имени модуля из
      PYBIND11_MODULE(openfll, m)). Через importlib.util.module_from_spec
      с name="openfll" нельзя — этот модуль уже зарегистрирован в
      sys.modules как сам пакет (этот __init__.py). Через ctypes+PyCapsule
      мы загружаем .pyd, вызываем PyInit_openfll напрямую, получаем
      Module-объект и копируем атрибуты в globals().
    """
    here = os.path.dirname(os.path.abspath(__file__))
    import glob
    candidates = sorted(glob.glob(os.path.join(here, "openfll.cp*.pyd")))
    candidates += [os.path.join(here, "openfll.pyd")]  # fallback для разработки
    for pyd_path in candidates:
        if os.path.isfile(pyd_path):
            # MinGW: добавляем каталог .pyd в DLL search path (для libpython и т.д.).
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(here)
            # pybind11 экспортирует PyInit_<name>, где <name> — аргумент
            # PYBIND11_MODULE(name, m). Пробуем несколько вариантов имён.
            import importlib.machinery
            import importlib.util
            for mod_name_try in ("openfll", "PyFLL"):
                loader = importlib.machinery.ExtensionFileLoader(
                    "_openfll_runtime", pyd_path)
                spec = importlib.machinery.ModuleSpec(
                    "_openfll_runtime", loader, origin=pyd_path)
                spec.name = mod_name_try
                try:
                    native = importlib.util.module_from_spec(spec)
                    sys.modules["_openfll_runtime"] = native
                    loader.exec_module(native)
                    globals()["SugenoEngine"] = native.SugenoEngine
                    return
                except ImportError:
                    continue
            continue
    # .pyd не найден — оставляем стаб; инстанцирование упадёт с понятной ошибкой.
    # Эта ситуация нормальна только в исходниках без wheel (например, в mkdocs build).
    pass


_load_native()
