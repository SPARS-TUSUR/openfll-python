# Сборка OpenFLL

## Поддерживаемые конфигурации

| ОС | Компилятор | Сборка | Тестировано |
|----|-----------|--------|-------------|
| Windows | **MinGW g++ 14.2 (UCRT64)** | primary | ✅ CI |
| Windows | MSVC 2022 | best-effort (constexpr-регресс в `registry.h`) | ⚠️ optional |
| Linux | GCC 11+ | должно работать (не тестировалось в CI) | ❓ |

**Python**: CPython 3.10 (`cp310`). MinGW-built `.pyd` использует MinGW runtime DLL —
MSVC-Python 3.10 тоже подойдёт, но требует `libstdc++-6.dll` рядом.

## Зависимости

- CMake 3.30+
- Git (для submodule `vendor/pybind11`)
- Python 3.10 (для `pip install pybind11-stubgen` и запуска Python-тестов)
- Опционально: `pip install mkdocs mkdocs-material mkdocstrings[python]` (для документации)

## Сборка C++ (Debug + MinGW)

```powershell
# 1. Клонировать рекурсивно (для vendor/pybind11)
git clone --recurse-submodules https://github.com/SPARS-TUSUR/fuzzy-logic-library.git
cd fuzzy-logic-library

# 2. Конфигурировать
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug

# 3. Собрать
cmake --build build -j 4 --target OpenFLL unit_tests compressor_run

# 4. Прогнать тесты
.\build\tests\unit_tests.exe --gtest_brief=1
```

Ожидаемый результат: **52/52 PASS** (SugenoEngine, RuleParser, MfFactory, SugenoHappyPath).

## Сборка Python-обёртки (Release, MinGW)

```powershell
# 1. Конфигурировать Release-вариант (отдельная папка)
cmake -S . -B build_release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release

# 2. Собрать PyFLL.pyd
cmake --build build_release -j 4 --target PyFLL

# 3. Скопировать MinGW runtime DLL рядом с .pyd
Copy-Item D:/msys64/mingw64/bin/libstdc++-6.dll,libgcc_s_seh-1.dll,libwinpthread-1.dll `
          build_release/src/PyFLL/

# 4. Проверить импорт
& C:/Python/Python310_64/python.exe -c "import sys; sys.path.insert(0, 'build_release/src/PyFLL'); import PyFLL; print(PyFLL.SugenoEngine())"
```

`PyFLL.cp310-win_amd64.pyd` + `PyFLL.pyi` будут в `build_release/src/PyFLL/`. `PyFLL.pyi`
автоматически копируется туда из `src/PyFLL/PyFLL.pyi` при каждой сборке (CMake POST_BUILD).

## Запуск Python-тестов

```powershell
# Smoke-тесты SugenoEngine (9 кейсов)
& C:/Python/Python310_64/python.exe tests/python/test_sugeno_smoke.py

# Compressor parity (C++ ↔ Python, 300 шагов)
cmake --build build --target compressor_run
& C:/Python/Python310_64/python.exe tests/python/compressor_parity.py
```

## Сборка документации

### Один раз: установка инструментов

```powershell
# Установить pip-пакеты для документации
scripts\setup_docs.bat
# или вручную:
pip install mkdocs mkdocs-material "mkdocstrings[python]" pybind11-stubgen
```

### Каждый раз: сборка и просмотр

```powershell
# Live preview с hot-reload — http://127.0.0.1:8000
scripts\serve_docs.bat

# Сгенерировать статический HTML в site/ (для GitHub Pages)
scripts\build_docs.bat

# Проверить, что ручной PyFLL.pyi синхронизирован с runtime-сигнатурами .pyd
scripts\gen_stubs.bat
```

### Что генерируется

- **`docs/site/`** — статический HTML-сайт (Material for MkDocs, тёмная/светлая темы, поиск)
- **API reference** — авто-генерируется из `src/PyFLL/__init__.py` (стаб SugenoEngine) + `PyFLL.pyi` (Literal-типы)
- **Структура**:
  - Главная (`docs/index.md`)
  - Туториалы (`docs/tutorials/`)
  - API (`docs/api/`)
  - Архитектура (`docs/architecture/`)
  - Сборка (`docs/build.md`)
  - Contributing (`docs/contributing.md`)

### Pipeline стабов

Источник истины: `R"doc()"` в `src/PyFLL/PyFLL.cpp`.

```
PyFLL.cpp (R"doc()")
  ↓ pybind11 (runtime)
PyFLL.cp310-win_amd64.pyd
  ↓ pybind11-stubgen (через scripts/gen_stubs.py)
build/PyFLL/PyFLL.pyi  (сгенерированный)
  ↓ сравнение сигнатур
src/PyFLL/PyFLL.pyi   (ручной, с Literal-типами)
  ↓ mkdocstrings через __init__.py stub
docs/api/sugeno-engine.md (HTML)
```

CI запускает `gen_stubs.py` при каждом PR — если рассинхрон в **сигнатурах** (имена/число параметров), билд падает. Расхождения в **аннотациях** (Literal vs `str`) — допустимы, это обогащение.

## Частые проблемы

### `Cannot open ... for writing` при запуске `compressor_run.exe`

Запускаете из неправильной директории. Решение:

```powershell
# Из любой директории:
D:/repos/FLL/build/compressor_run.exe
# C++ сам найдёт <repo>/tests/python/data/ по argv[0]
```

### `DLL load failed while importing PyFLL`

`PyFLL.pyd` собран в Debug, а Python — Release (или наоборот). Решение: используйте
**Release**-сборку (Debug-Python требует `python310_d.dll`, которого обычно нет).

### `ModuleNotFoundError: No module named 'PyFLL'`

Python не находит `.pyd`. Добавьте путь:

```powershell
sys.path.insert(0, r'D:/repos/FLL/build_release/src/PyFLL')
```

### MSVC: `error C2131: выражение не определяется константой` в `registry.h`

Известный constexpr-регресс в `utils/registry/registry.h`. Используйте MinGW,
пока не починим ядро. CI настроен на GCC primary, MSVC — best-effort.
