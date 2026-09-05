# openfll-python

**Python биндинги для OpenFLL** (C++ библиотека нечёткой логики).

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.14-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

## Что это

`openfll` — Python-обёртка над [OpenFLL](https://github.com/SPARS-TUSUR/fuzzy-logic-library),
предоставляющая плоский API для построения Sugeno-моделей нечёткого вывода.

```python
import openfll

e = openfll.SugenoEngine()
e.add_input_var("x1")
e.add_input_var("x2")
e.add_membership_func("x1", "low", "triangular", [-10, 0, 10])
e.add_membership_func("x2", "high", "gaussian", [5, 2])
e.add_membership_func("y", "lo", "constant", [10.0])

# Две формы правил: текстовая (AND/OR + [norm_name]) и кортежная (для циклов).
e.add_rule(
    antecedent=[("x1", "low"), ("x2", "high")],
    consequent=[("y", "lo")],
    t_norm="prod_and",
)
e.build()
e.set_input("x1", 5.0)
e.set_input("x2", 0.0)
e.calculate()
print(e.get_output("y"))
```

## Установка

```powershell
# Только Windows + Python 3.14 (MinGW-build)
pip install openfll
```

Поддерживаемые платформы (на момент публикации):

| Python | Wheel |
|--------|-------|
| 3.14 | `openfll-0.1.0-cp314-cp314-win_amd64.whl` |

## Документация

- Онлайн: <https://spars-tusur.github.io/openfll-python/>
- Исходники: `docs/` в этом репозитории (MkDocs + Material)
- [API → SugenoEngine](https://spars-tusur.github.io/openfll-python/api/sugeno-engine/)
- [Туториал 01: первый движок](https://spars-tusur.github.io/openfll-python/tutorials/01-first-engine/)
- [Туториал 02: компрессор](https://spars-tusur.github.io/openfll-python/tutorials/02-fuzzy-arithmetic/)

## Что внутри wheel

- `openfll.cp314-...pyd` — скомпилированное расширение
- `openfll.pyi` — type stubs с `Literal` типами для IDE
- `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll`, `libpython3.14.dll` — MinGW/Python runtime
- `__init__.py` — runtime loader + stub для IDE/документации

## Сборка из исходников

Этот репозиторий — **только для дистрибуции**. Исходники C++ (OpenFLL ядро) —
в [отдельном приватном репозитории](https://github.com/SPARS-TUSUR/fuzzy-logic-library).

См. [BUILD.md](BUILD.md) для инструкции по ручной сборке wheels.

## Лицензия

MIT — см. [LICENSE](LICENSE).
