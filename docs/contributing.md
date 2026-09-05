# Contributing

`openfll-python` — это тонкий Python-биндинг над [OpenFLL C++ библиотекой](https://github.com/SPARS-TUSUR/fuzzy-logic-library).
Большинство фич (новые MF, t/s-нормы, методы движка) добавляются **в C++-репо**, а в этот пакет они попадают
через пересборку `.pyd` и релиз новой версии wheel.

Здесь — только то, что можно контрибьютить в сам `openfll-python`.

## Сообщить о баге

Создайте [Issue](https://github.com/SPARS-TUSUR/openfll-python/issues) с:

- Версия Python (`python --version`) и платформа (Win/Mac/Linux, `win_amd64` / `manylinux_*`).
- Команда для воспроизведения: `engine.add_rule(...)`, `engine.calculate()` и т.д.
- Полный traceback.
- Минимальный пример (5–15 строк).

Если проблема — в самой fuzzy-логике (неправильный результат `calculate()`), это баг **C++ ядра** →
[SPARS-TUSUR/fuzzy-logic-library/issues](https://github.com/SPARS-TUSUR/fuzzy-logic-library/issues).

## Запустить тесты локально

```bash
pip install openfll                # для smoke-теста API
python -c "import openfll; e = openfll.SugenoEngine(); e.build()"  # проверка загрузки

# Для разработки самого пакета (после git clone):
pip install -e .[dev]              # редактируемая установка
pytest tests/                      # если есть pytest-тесты
```

Wheel под `cp314-cp314-win_amd64` и `cp314-cp314-mingw_x86_64_msvcrt_gnu` собирается
через `cibuildwheel` в GitHub Actions — см. `.github/workflows/wheel.yml`.
Локальная сборка требует MSVC 2022 (Windows) или GCC 14+ (Linux).

## Поправить документацию

Документация — `mkdocs Material`, исходники в `docs/`. Лёгкие правки:

1. Fork → edit `.md` файл → PR в `main`.
2. Локальная проверка: `pip install mkdocs-material mkdocstrings[python]`
   `mkdocs build --strict` — должен пройти без warnings.
3. Туториалы (`docs/tutorials/`) и API reference (`docs/api/`) — самое то для первого PR.

## Добавить пример / tutorial

Положите standalone `.py` скрипт в `examples/` (если появится) или новый `.md`
в `docs/tutorials/`. Без C++ — только Python + `openfll`.

## Стиль кода

- **Python**: PEP 8, type hints обязательны для публичного API.
- **Docstring'и**: Google style (используется `mkdocstrings`).
- **Commits**: один PR — одна фича. Сообщения — на русском или английском,
  с префиксом области: `docs:`, `pyi:`, `ci:`, `tests:`.
- **Тесты**: добавляйте в `tests/python/` рядом с уже существующими
  (smoke / parity / stub-types).

## Что НЕ править в этом репо

| Хочется изменить | Где на самом деле |
|------------------|-------------------|
| Алгоритм Sugeno / Mamdani | [fuzzy-logic-library/src/OpenFLL](https://github.com/SPARS-TUSUR/fuzzy-logic-library/tree/main/src/OpenFLL) |
| Добавить новую MF | [fuzzy-logic-library + mf_factory.cpp](https://github.com/SPARS-TUSUR/fuzzy-logic-library) |
| pybind11 биндинги (`.def(...)`) | [fuzzy-logic-library/src/PyFLL](https://github.com/SPARS-TUSUR/fuzzy-logic-library/tree/main/src/PyFLL) |
| C++-сборка (CMake) | там же |

После принятия C++-фичи в upstream — открывайте issue здесь
с тегом `rebuild-needed`, новая wheel-версия соберётся автоматически.

## Контакты

- GitHub: [SPARS-TUSUR/openfll-python](https://github.com/SPARS-TUSUR/openfll-python)
- C++ ядро: [SPARS-TUSUR/fuzzy-logic-library](https://github.com/SPARS-TUSUR/fuzzy-logic-library)
- Документация: [spars-tusur.github.io/openfll-python](https://spars-tusur.github.io/openfll-python/)
