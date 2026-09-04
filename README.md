# openfll-python

**Python биндинги для OpenFLL** (C++ библиотека нечёткой логики).

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.14-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()

## Что это

`openfll` — Python-обёртка над [OpenFLL](https://github.com/SPARS-TUSUR/fuzzy-logic-library),
предоставляющая плоский API для построения Sugeno-моделей нечёткого вывода.

```python
import openfll

e = openfll.SugenoEngine()
e.add_input_var("x1")
e.add_membership_func("x1", "low", "triangular", [-10, 0, 10])
e.add_rule('IF x1 IS "low" AND x2 IS "low" THEN y IS "lo"')
e.build()
e.set_input("x1", 5.0)
e.calculate()
print(e.get_output("y"))
```

## Установка

```powershell
# Только Windows + Python 3.10 или 3.14 (MinGW-build)
pip install openfll
```

Поддерживаемые платформы (на момент публикации):

| Python | Wheel |
|--------|-------|
| 3.10 | `openfll-0.1.0-cp310-cp310-win_amd64.whl` |
| 3.14 | `openfll-0.1.0-cp314-cp314-win_amd64.whl` |

## Документация

- Онлайн: <https://spars-tusur.github.io/openfll-python/>
- Исходники: `docs/` в этом репозитории (MkDocs + Material)

## Что внутри wheel

- `openfll.cp310-...pyd` (или cp314) — скомпилированное расширение
- `openfll.pyi` — type stubs с `Literal` типами для IDE
- `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` — MinGW runtime
- `__init__.py` — runtime loader + stub для IDE/документации

## Сборка из исходников

Этот репозиторий — **только для дистрибуции**. Исходники C++ (OpenFLL ядро) —
в [отдельном приватном репозитории](https://github.com/SPARS-TUSUR/fuzzy-logic-library).

См. [BUILD.md](BUILD.md) для инструкции по ручной сборке wheels.

## Лицензия

MIT — см. [LICENSE](LICENSE).
