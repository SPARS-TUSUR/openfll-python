# Сборка wheel из исходников

Этот репозиторий **только для дистрибуции**. Исходники C++ находятся в
закрытом репозитории [fuzzy-logic-library](https://github.com/SPARS-TUSUR/fuzzy-logic-library).

## Автоматическая сборка (через GitHub Actions)

При каждом push тега `v*` (например, `v0.1.0`) workflow
`.github/workflows/wheel.yml` собирает wheel для Python 3.14 (MinGW)
и прикрепляет его к GitHub Release.

## Модель поставки backend

`PyFLL` не подключается как внешняя package dependency. Для ближайшего релиза
он поставляется как внутренний backend package внутри того же wheel, что и
`openfll`.

Release-сборка временно собирает `PyFLL._core` из ветки
`fuzzy-logic-library/python/newpybind`, потому что native `predict_batch`
пока находится там до merge обратно в `python/pybind`.
после чего копирует в `openfll-python/src/PyFLL/`:

- `__init__.py` из backend source;
- `PyFLL.pyi`;
- `_core.<SOABI>.pyd`;
- runtime `.dll`.

Пользователь импортирует только `openfll`; цепочка исполнения остаётся
`openfll -> PyFLL -> PyFLL._core`.

Шаги:
1. Сделать `git tag v0.1.0` и `git push origin v0.1.0`
2. CI собирает wheel на `windows-latest` (MinGW GCC + MSYS2 Python 3.14)
3. К каждому тегу автоматически создаётся GitHub Release с прикреплённым `.whl`

## Ручная сборка (на Windows + MinGW)

### Требования

- MinGW-w64 с GCC (через [MSYS2](https://www.msys2.org/))
- CMake 3.30+, Ninja
- Python 3.14 через MSYS2: `pacman -S mingw-w64-x86_64-python`
- pybind11: `pacman -S mingw-w64-x86_64-pybind11`
- NumPy для batch API: `pacman -S mingw-w64-x86_64-python-numpy`
- cibuildwheel: `pip install cibuildwheel`

### Шаги

```bash
# 1. Клонировать приватный OpenFLL рядом (или использовать submodule)
git clone https://github.com/SPARS-TUSUR/fuzzy-logic-library.git ../fuzzy-logic-library
git -C ../fuzzy-logic-library checkout python/newpybind

# 2. Сконфигурировать и собрать C++ для MinGW-Python 3.14
cd ../fuzzy-logic-library
cmake -S . -B build_release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DPython3_EXECUTABLE=/mingw64/bin/python.exe
cmake --build build_release -j 4 --target PyFLL

# 3. Скопировать внутренний backend package PyFLL в публичный wheel source
rm -rf ../openfll-python/src/PyFLL
mkdir -p ../openfll-python/src/PyFLL
cp src/PyFLL/__init__.py ../openfll-python/src/PyFLL/
cp build_release/src/PyFLL/PyFLL.pyi ../openfll-python/src/PyFLL/
cp build_release/src/PyFLL/_core*.pyd ../openfll-python/src/PyFLL/
cp build_release/src/PyFLL/*.dll ../openfll-python/src/PyFLL/ || true

# Скопировать Python runtime DLL рядом с PyFLL._core
cp /mingw64/bin/libpython3.14.dll ../openfll-python/src/PyFLL/

# 4. Собрать wheel
cd ../openfll-python
cibuildwheel --output-dir wheelhouse/ \
    --build "cp314-*" \
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
│   ├── openfll/
│       ├── __init__.py     ← runtime loader + стаб SugenoEngine
│       └── __init__.pyi    ← type stubs для IDE
│   └── PyFLL/              ← внутренний backend package, копируется при release
│       ├── __init__.py
│       ├── _core.*.pyd
│       ├── PyFLL.pyi
│       └── *.dll
├── tests/
│   └── test_backend_contract.py
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
   - Соберёт wheel (cp314)
   - Создаст GitHub Release с wheel + `sdist` tarball
   - Задеплоит docs на https://spars-tusur.github.io/openfll-python/

Пользователи увидят новую версию через `pip install --upgrade openfll`.

## Альтернатива: C++ исходники в публичном репо

Если в будущем C++ код станет публичным (open-source лицензия), wheel-build
упростится: cibuildwheel сможет собрать wheels прямо из исходников в этом же
репо, без клонирования приватного.
