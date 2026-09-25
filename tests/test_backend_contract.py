from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import get_args

import pytest


def _drop_modules() -> None:
    for name in ("openfll", "PyFLL", "PyFLL._core", "openfll._core"):
        sys.modules.pop(name, None)


def test_openfll_delegates_to_pyfll_backend(tmp_path, monkeypatch):
    """openfll отдаёт наружу PyFLL.SugenoEngine и не создаёт openfll._core."""
    backend = tmp_path / "PyFLL"
    backend.mkdir()
    (backend / "__init__.py").write_text(
        "from ._core import SugenoEngine\n"
        "__all__ = ['SugenoEngine']\n",
        encoding="utf-8",
    )
    (backend / "_core.py").write_text(
        "class SugenoEngine:\n"
        "    def is_built(self):\n"
        "        return False\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop_modules()

    pyfll = importlib.import_module("PyFLL")
    openfll = importlib.import_module("openfll")

    assert openfll.SugenoEngine is pyfll.SugenoEngine
    assert openfll.SugenoEngine().__class__.__module__ == "PyFLL._core"
    assert "openfll._core" not in sys.modules


def test_openfll_keeps_public_symbols_with_backend(tmp_path, monkeypatch):
    backend = tmp_path / "PyFLL"
    backend.mkdir()
    (backend / "__init__.py").write_text(
        "from ._core import SugenoEngine\n",
        encoding="utf-8",
    )
    (backend / "_core.py").write_text(
        "class SugenoEngine:\n"
        "    pass\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop_modules()

    openfll = importlib.import_module("openfll")

    assert hasattr(openfll, "MfType")
    assert hasattr(openfll, "TNorms")
    assert hasattr(openfll, "SNorms")
    assert set(openfll.__all__) == {
        "SugenoEngine",
        "MfType",
        "TNorms",
        "SNorms",
        "constant",
        "triangular",
        "trapezoidal",
        "gaussian",
    }


def test_missing_pyfll_backend_fails_when_engine_is_created(tmp_path, monkeypatch):
    backend = tmp_path / "PyFLL"
    backend.mkdir()
    (backend / "__init__.py").write_text(
        "raise ImportError('native module PyFLL._core is missing')\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop_modules()

    openfll = importlib.import_module("openfll")

    with pytest.raises(ImportError) as exc_info:
        openfll.SugenoEngine()

    message = str(exc_info.value)
    assert "внутренний backend `PyFLL`" in message
    assert "native module `PyFLL._core`" in message
    assert "native module PyFLL._core is missing" in message


def test_pyfll_stub_backend_is_not_treated_as_runtime(tmp_path, monkeypatch):
    backend = tmp_path / "PyFLL"
    backend.mkdir()
    (backend / "__init__.py").write_text(
        "class SugenoEngine:\n"
        "    def is_built(self):\n"
        "        return None\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop_modules()

    openfll = importlib.import_module("openfll")

    with pytest.raises(ImportError) as exc_info:
        openfll.SugenoEngine()

    message = str(exc_info.value)
    assert "PyFLL.SugenoEngine не загружен из PyFLL._core" in message
    assert "получен модуль 'PyFLL'" in message


def test_openfll_import_stays_available_without_backend(tmp_path, monkeypatch):
    backend = tmp_path / "PyFLL"
    backend.mkdir()
    (backend / "__init__.py").write_text(
        "raise ImportError('docs build has no native backend')\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    monkeypatch.syspath_prepend(str(tmp_path))
    _drop_modules()

    openfll = importlib.import_module("openfll")

    assert hasattr(openfll, "SugenoEngine")
    with pytest.raises(ImportError):
        openfll.SugenoEngine()


def test_public_mf_type_lists_only_supported_backend_types(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    _drop_modules()

    openfll = importlib.import_module("openfll")

    assert get_args(openfll.MfType) == (
        "constant",
        "triangular",
        "trapezoidal",
        "gaussian",
    )


def test_membership_function_helpers_return_exact_parameter_lists(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "src"))
    _drop_modules()

    openfll = importlib.import_module("openfll")

    assert openfll.constant(10.0) == [10.0]
    assert openfll.triangular(-10.0, 0.0, 10.0) == [-10.0, 0.0, 10.0]
    assert openfll.trapezoidal(-10.0, -5.0, 5.0, 10.0) == [
        -10.0,
        -5.0,
        5.0,
        10.0,
    ]
    assert openfll.gaussian(center=-5.0, sigma=2.0) == [-5.0, 2.0]


def test_runtime_module_does_not_define_add_rule_overloads():
    source = (Path.cwd() / "src" / "openfll" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "@overload" not in source
    assert "def add_rule" not in source


def test_predict_batch_is_typed_but_not_implemented_as_python_loop():
    runtime_source = (Path.cwd() / "src" / "openfll" / "__init__.py").read_text(
        encoding="utf-8"
    )
    stub_source = (Path.cwd() / "src" / "openfll" / "__init__.pyi").read_text(
        encoding="utf-8"
    )

    assert "def predict_batch" not in runtime_source
    assert "for row in X" not in runtime_source
    assert "def predict_batch" in stub_source
