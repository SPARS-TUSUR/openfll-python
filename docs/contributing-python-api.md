# Contributing: Python API development

Это руководство для коллег, которые расширяют **только Python-часть** openfll —
без правок C++ ядра. Если нужна новая MF, t/s-норма или метод SugenoEngine —
см. [RELEASE_TO_PYTHON.md](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/docs/RELEASE_TO_PYTHON.md)
в upstream-репо и обращайтесь к разработчику C++.

## Что можно делать без C++

Используя **только** публичный Python-API `openfll.SugenoEngine`:

- Сериализация / десериализация моделей (dict ↔ engine)
- Визуализация (matplotlib MF, heatmap выхода)
- Валидация (Monte-Carlo проверка на NaN/Inf, покрытие входов)
- Батч-инференс (массовый `calculate()`)
- Обёртки под sklearn / pandas / asyncio
- Кастомные нормы на уровне Python (post-processing выхода)
- DSL поверх `add_rule()` (свой парсер текстового формата)

Нельзя без C++:
- Добавить новый тип MF (triangular, gaussian и т.д. — закрытый список в C++)
- Добавить новую t/s-норму (см. NormId в `sugeno_engine.cpp`)
- Изменить алгоритм вывода (Sugeno 0/1 порядка)
- Получить список переменных/MF из engine (read-back API отсутствует — пишите на C++)

## Структура репо

```
src/openfll/
├── __init__.py        ← re-export SugenoEngine + helpers
├── __init__.pyi       ← type stubs (Literal типы, сигнатуры helpers)
├── PyFLL.pyi          ← автогенерированный pybind11_stubgen
├── _core.cp314-*.pyd  ← скомпилированное расширение (НЕ править руками)
└── _helpers/          ← ВАШ КОД — Python-only расширения
    ├── __init__.py
    └── sampling.py    ← monte_carlo, grid_2d, check_output_finite

tests/                 ← pytest, без пересборки .pyd
├── conftest.py
├── test_smoke.py
└── test_helpers.py

examples/              ← standalone .py, можно запускать руками
├── 01_basic.py
└── 02_monte_carlo.py

.github/workflows/
├── wheel.yml          ← пересборка .pyd (триггер: tag v*)
└── python-tests.yml   ← pytest (триггер: push main, PR)
```

## Шаг за шагом: добавить новый helper

Допустим, хотите добавить `coverage_report(engine)` — отчёт о покрытии
входного диапазона активными MF.

### 1. Создайте модуль

```python
# src/openfll/_helpers/coverage.py
"""Анализ покрытия входного диапазона MF."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openfll import SugenoEngine

__all__ = ["coverage_report"]


def coverage_report(engine: "SugenoEngine", var: str, n_points: int = 100) -> dict:
    """Проверяет, что для каждой точки входа найдётся MF с принадлежностью > 0.

    Returns:
        {"var": "x1", "n_points": 100, "covered": 87, "uncovered": [...]}
    """
    # ваша логика
    ...
```

### 2. Зарегистрируйте в `__init__.py`

В `src/openfll/__init__.py` в блоке `_helpers`:

```python
try:
    from ._helpers.sampling import check_output_finite, grid_2d, monte_carlo
    from ._helpers.coverage import coverage_report   # ← НОВОЕ
    __all__ = [*__all__, "monte_carlo", "grid_2d",
               "check_output_finite", "coverage_report"]
except ImportError:
    pass
```

### 3. Добавьте type hints в `__init__.pyi`

```python
def coverage_report(engine: SugenoEngine, var: str, n_points: int = 100) -> dict: ...
```

### 4. Напишите тест

```python
# tests/test_helpers_coverage.py
from openfll import coverage_report


def test_coverage_report_full_coverage(sugno_engine_2in_1out):
    rep = coverage_report(sugno_engine_2in_1out, "x1", n_points=50)
    assert rep["var"] == "x1"
    assert rep["covered"] > 0
```

### 5. Локальная проверка

```bash
cd openfll-python
pip install -e .            # редактируемая установка (или `pip install openfll==0.1.0`)
pytest tests/ -v            # все тесты должны быть зелёными
python examples/01_basic.py # sanity-check базового API
```

CI (`.github/workflows/python-tests.yml`) запустит то же самое на
`windows-latest` после вашего PR.

### 6. Откройте PR

```bash
git checkout -b feature/coverage-report
git add src/openfll/_helpers/coverage.py \
        src/openfll/__init__.py \
        src/openfll/__init__.pyi \
        tests/test_helpers_coverage.py
git commit -m "helpers: add coverage_report"
git push origin feature/coverage-report
# → откройте PR через GitHub UI
```

## Правила

1. **Helpers используют ТОЛЬКО публичный API.** Никаких прямых вызовов
   `_core.SugenoEngine._что-то_внутреннее()`. Если нужного метода нет —
   добавьте его в C++ ядро (см. RELEASE_TO_PYTHON.md).

2. **Type hints обязательны.** Каждая функция в `_helpers/` должна иметь
   полную аннотацию параметров и возврата. Затем продублируйте в `__init__.pyi`.

3. **Тесты обязательны.** Новый helper → новый файл в `tests/test_helpers_*.py`.
   Smoke-test в `test_smoke.py` менять не нужно.

4. **Зависимости — никакие новые.** Helpers живут на stdlib + numpy (опционально).
   Если нужна matplotlib / pandas — оформите как optional dependency
   (`extras_require` в `pyproject.toml`) и проверьте `try/except ImportError`.

5. **Не трогайте `PyFLL.pyi`.** Этот файл генерируется `pybind11-stubgen` из
   C++ исходников. Любые ручные правки будут затерты при пересборке wheel.

## Что делать, когда C++ колесо обновилось

Когда maintainer C++ ядра выпустит новый wheel (например, v0.2.0 с новым
методом `SugenoEngine.get_input_var_names()`):

```bash
pip install --upgrade openfll==0.2.0
```

Ваши helpers, использующие новые методы, сразу заработают. Старый код
не сломается (semver гарантирует).

## Контакты

- Ишьюс / PR в этом репо (openfll-python)
- C++ ядро: [SPARS-TUSUR/fuzzy-logic-library](https://github.com/SPARS-TUSUR/fuzzy-logic-library)
- См. также [contributing.md](contributing.md) — общий гайд по проекту
