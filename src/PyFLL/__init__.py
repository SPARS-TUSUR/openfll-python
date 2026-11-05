"""Нативные Python-биндинги общего линейного API OpenFLL."""
from __future__ import annotations
from typing import Literal

MfType = Literal["constant", "triangular", "trapezoidal", "gaussian", "polynomial"]
TNorms = Literal["min", "prod_and", "bounded_diff", "drastic_prod", "einstein_prod", "hamacher_prod"]
SNorms = Literal["max", "algebraic_sum", "bounded_sum", "drastic_sum", "einstein_sum", "hamacher_sum"]

from ._core import BaseLinearEngine, SugenoEngine, MamdaniEngine, ModelView, load_from_file, load_from_string

__all__ = ["BaseLinearEngine", "SugenoEngine", "MamdaniEngine", "ModelView",
           "load_from_file", "load_from_string", "MfType", "TNorms", "SNorms"]
