# OpenFLL

**Open Fuzzy Logic Library** — C++ библиотека нечёткой логики с Python-обвязкой.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Build Status](https://github.com/SPARS-TUSUR/fuzzy-logic-library/actions/workflows/cpp-tests.yml/badge.svg)]()
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)]()

## Что это

OpenFLL — это компактная, статически линкуемая C++20 библиотека для систем нечёткого
вывода (Sugeno, Mamdani), плюс **плоский Python API** через Pybind11.

Проект ориентирован на инженеров, которым нужна **встраиваемая** нечёткая логика без
зависимостей на runtime — никаких внешних интерпретаторов, никакого Python на стороне
вывода. Python используется только для удобного **скриптинга и обучения моделей**.

## Возможности

- **Sugeno 0-го порядка** — рабочий, с WTAver-дефуззификацией (WTSum доступен)
- **Mamdani** — в процессе реализации (дефаззификация через интегрирование)
- **5 типов функций принадлежности**: constant, triangular, trapezoidal, gaussian, polynomial
- **6 t-норм и 6 s-норм** с явным выбором в правиле
- **Linear Facade** — единый плоский API, скрывающий fluent-цепочки ядра
- **Pybind11-биндинги** — `import PyFLL; engine = PyFLL.SugenoEngine()`
- **Type stubs** — `.pyi` с `Literal`-типами для автодополнения в IDE

## 30 секунд до первого расчёта

```python
import PyFLL

e = PyFLL.SugenoEngine()
e.add_input_var("x1")
e.add_input_var("x2")
e.add_output_var("y")

e.add_membership_func("x1", "low",  "triangular", [-10, 0, 10])
e.add_membership_func("x1", "high", "triangular", [0, 10, 20])
e.add_membership_func("x2", "low",  "gaussian",   [-5, 2])
e.add_membership_func("x2", "high", "gaussian",   [5, 2])
e.add_membership_func("y",  "lo",   "constant",   [10.0])
e.add_membership_func("y",  "hi",   "constant",   [30.0])

e.add_rule('IF x1 IS "low" AND x2 IS "low" THEN y IS "lo"')
e.add_rule('IF x1 IS "high" OR x2 IS "high" THEN y IS "hi"')
e.add_rule('IF x1 IS "low" AND[prod_and] x2 IS "high" THEN y IS "lo"')

e.build()
e.set_input("x1", 5.0)
e.set_input("x2", 0.0)
e.calculate()
print(e.get_output("y"))   # → 16.667
```

## Документация

- [Туториал: первый движок](tutorials/01-first-engine.md) — Sugeno от и до
- [Туториал: управление компрессором](tutorials/02-fuzzy-arithmetic.md) — C++ ↔ Python parity
- [API reference](api/sugeno-engine.md) — автогенерированная документация
- [Архитектура Linear Facade](architecture/linear-facade.md) — почему плоский API
- [Сборка](build.md) — MinGW / MSVC / pybind11

## Лицензия

MIT — см. [LICENSE](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/LICENSE).
