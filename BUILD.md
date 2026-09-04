# Сборка wheel из исходников

Этот репозиторий **только для дистрибуции**. Исходники C++ находятся в
закрытом репозитории [fuzzy-logic-library](https://github.com/SPARS-TUSUR/fuzzy-logic-library).

## Автоматическая сборка (через GitHub Actions)

При каждом push тега `v*` (например, `v0.1.0`) workflow
`.github/workflows/wheel.yml` собирает wheels для Python 3.10 и 3.14
и прикрепляет их к GitHub Release.

Шаги:
1. Сделать `git tag v0.1.0` и `git push origin v0.1.0`
2. CI собирает wheels на `windows-latest` (MinGW GCC + MSYS2 Python)
3. К каждому тегу автоматически создаётся GitHub Release с прикреплёнными `.whl`

## Ручная сборка (на Windows + MinGW)

### Требования

- MinGW-w64 с GCC (через [MSYS2](https://www.msys2.org/))
- CMake 3.30+, Ninja
- Python 3.10 (через [python-build-standalone](https://github.com/indygreg/python-build-standalone))
  или через MSYS2 (`pacman -S mingw-w64-x86_64-python` → 3.14)
- pybind11 (`pacman -S mingw-w64-x86_64-pybind11`)
- cibuildwheel (`pip install cibuildwheel`)

### Шаги

```bash
# 1. Клонировать приватный OpenFLL рядом (или использовать submodule)
git clone https://github.com/SPARS-TUSUR/fuzzy-logic-library.git ../fuzzy-logic-library

# 2. Сконфигурировать и собрать C++ для нужной версии Python
cd ../fuzzy-logic-library
cmake -S . -B build_release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=/path/to/python3.10.exe
cmake --build build_release -j 4 --target PyFLL

# 3. Скопировать .pyd в публичный пакет
PYD=$(ls build_release/src/PyFLL/*.pyd)
cp "$PYD" ../openfll-python/src/openfll/$(basename $PYD | sed 's/^PyFLL/openfll/')

# 4. Собрать wheel
cd ../openfll-python
cibuildwheel --output-dir wheelhouse/ \
    --build "cp310-*" \
    --skip "*-musllinux_* *-manylinux_i686 pp*"
```

## Структура репозитория

```
openfll-python/
├── README.md              ← то, что видит пользователь на GitHub
├── LICENSE                ← MIT
├── BUILD.md               ← этот файл
├── pyproject.toml         ← конфиг wheel + cibuildwheel
├── src/
│   └── openfll/
│       ├── __init__.py     ← runtime loader + стаб SugenoEngine
│       └── __init__.pyi    ← type stubs для IDE
├── tests/
│   └── __init__.py         ← smoke-тест wheel-а
├── mkdocs.yml             ← конфиг документации
├── docs/                  ← исходники документации (mkdocs)
└── .github/
    └── workflows/
        ├── wheel.yml        ← сборка wheels
        └── docs.yml         ← деплой GitHub Pages
```

## Как опубликовать новую версию

1. Обновить `version` в `pyproject.toml`
2. Обновить `src/openfll/__init__.pyi` если менялся API
3. Обновить `docs/`
4. `git tag v0.2.0 && git push origin v0.2.0`
5. CI автоматически:
   - Соберёт wheels (cp310 + cp314)
   - Создаст GitHub Release с wheels + `sdist` tarball
   - Задеплоит docs на https://spars-tusur.github.io/openfll-python/

Пользователи увидят новую версию через `pip install --upgrade openfll`.

## Альтернатива: C++ исходники в публичном репо

Если в будущем C++ код станет публичным (open-source лицензия), wheel-build
упростится: cibuildwheel сможет собрать wheels прямо из исходников в этом же
репо, без клонирования приватного.
