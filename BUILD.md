# Сборка и публикация OpenFLL Python

Дистрибутив публикуется на PyPI под именем `openfll_python`. После установки
пользователь импортирует пакет как `openfll`.

Версия 0.2.0 поддерживает только:

- 64-разрядную Windows;
- официальный CPython 3.14 с python.org;
- wheel `cp314-cp314-win_amd64`.

Исходники C++ находятся в закрытом репозитории
[fuzzy-logic-library](https://github.com/SPARS-TUSUR/fuzzy-logic-library).
В публичный wheel включается скомпилированный native backend `PyFLL._core`, но
не исходный код закрытого репозитория. Source distribution (`sdist`) для этого
релиза не публикуется.

## Автоматическая сборка

Workflow `.github/workflows/wheel.yml` поддерживает три сценария:

1. Ручной запуск с `publish_target=none` — собрать и протестировать wheel, не
   публикуя его.
2. Ручной запуск с `publish_target=testpypi` — собрать, протестировать и
   отправить wheel в TestPyPI.
3. Публикация GitHub Release с тегом `v0.2.0` — собрать и протестировать wheel,
   прикрепить его к GitHub Release и отправить в PyPI.

Production-публикация не запускается от обычного push или pull request.
Workflow сначала собирает wheel, затем устанавливает именно полученный файл в
чистый CPython 3.14 и выполняет тесты. Только успешно проверенный artifact
передаётся publishing-job.

## Доступ к приватному backend

Workflow клонирует `SPARS-TUSUR/fuzzy-logic-library` на зафиксированном commit
`e49109aabdb37cadadb2cee487e9b6f39261f6fa` из ветки `python/pybind`. Это
делает release-сборку воспроизводимой. Если стандартный `GITHUB_TOKEN` не имеет
доступа к соседнему приватному репозиторию, в настройках `openfll-python` нужно
создать Actions secret `OPENFLL_BACKEND_TOKEN_SECRET`. Значением должен быть
fine-grained token или GitHub App token с минимальным правом `Contents: read`
для `fuzzy-logic-library`.

## Trusted Publishing

PyPI-токен в репозитории не хранится. Публикация использует GitHub OIDC и
PyPI Trusted Publishing.

Перед первым запуском необходимо:

1. Создать GitHub Environments `testpypi` и `pypi`.
2. Для environment `pypi` включить обязательное ручное подтверждение.
3. В TestPyPI зарегистрировать pending publisher:
   - project name: `openfll_python`;
   - owner: `SPARS-TUSUR`;
   - repository: `openfll-python`;
   - workflow: `wheel.yml`;
   - environment: `testpypi`.
4. Повторить регистрацию на PyPI с environment `pypi`.

PyPI нормализует подчёркивание в адресе проекта, поэтому страница будет иметь
вид `https://pypi.org/project/openfll-python/`, а установить пакет можно как
`pip install openfll_python` или `pip install openfll-python`.

## Модель поставки backend

`PyFLL` входит во wheel как внутренний пакет:

```text
openfll -> PyFLL -> PyFLL._core -> OFLL::Linear::SugenoEngine
```

Release workflow собирает backend компилятором UCRT64 GCC против официального
CPython 3.14 с python.org. Ожидаемое имя native-модуля —
`_core.cp314-win_amd64.pyd`. Компилятор берётся из MSYS2, но ABI и имя модуля —
от официального CPython, поэтому wheel устанавливается обычным Windows Python.
Файл с MSYS2 Python ABI (`_core.cp314-mingw_*.pyd`) не является заменой.
До сборки workflow применяет локальный `packaging/cpython-windows.patch`: он
добавляет в backend CMake корректное определение `libs` и ABI-суффикса для
официального Windows CPython.

## Ручная сборка на Windows

### Требования

- MSYS2 UCRT64 с GCC, CMake и Ninja;
- официальный 64-разрядный CPython 3.14;
- доступ на чтение к приватному `fuzzy-logic-library`;
- `python -m pip install --upgrade build twine pytest`.

### Сборка backend

```powershell
git clone --branch python/pybind --recurse-submodules `
  https://github.com/SPARS-TUSUR/fuzzy-logic-library.git `
  ..\fuzzy-logic-library
git -C ..\fuzzy-logic-library checkout e49109aabdb37cadadb2cee487e9b6f39261f6fa
$patchFile = (Resolve-Path .\packaging\cpython-windows.patch).Path
git -C ..\fuzzy-logic-library apply $patchFile

$pythonExe = (Get-Command python).Source
$ucrtBin = "C:\msys64\ucrt64\bin"

& "$ucrtBin\cmake.exe" `
  -S ..\fuzzy-logic-library `
  -B ..\fuzzy-logic-library\build_release `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER="$ucrtBin\gcc.exe" `
  -DCMAKE_CXX_COMPILER="$ucrtBin\g++.exe" `
  -DCMAKE_MAKE_PROGRAM="$ucrtBin\ninja.exe" `
  -DPython3_EXECUTABLE="$pythonExe" `
  -DOPENFLL_BUILD_BENCHMARKS=OFF

& "$ucrtBin\cmake.exe" `
  --build ..\fuzzy-logic-library\build_release `
  --target unit_tests PyFLL `
  --parallel 4

$env:Path = "$ucrtBin;$env:Path"
& ..\fuzzy-logic-library\build_release\tests\unit_tests.exe --gtest_brief=1
```

### Сборка wheel

Команды выполняются из корня `openfll-python`:

```powershell
$backendBuild = "..\fuzzy-logic-library\build_release\src\PyFLL"
$core = Get-ChildItem $backendBuild -Recurse `
  -Filter "_core.cp314-win_amd64.pyd" | Select-Object -First 1
$stub = Get-ChildItem $backendBuild -Recurse `
  -Filter "PyFLL.pyi" | Select-Object -First 1

if (-not $core -or -not $stub) {
  throw "CPython 3.14 backend artifacts were not produced"
}

Remove-Item -LiteralPath "src\PyFLL" -Recurse -Force
New-Item -ItemType Directory -Path "src\PyFLL" | Out-Null
Copy-Item "..\fuzzy-logic-library\src\PyFLL\__init__.py" "src\PyFLL\"
Copy-Item $stub.FullName "src\PyFLL\"
Copy-Item $core.FullName "src\PyFLL\"
Get-ChildItem $core.DirectoryName -Filter "*.dll" |
  Copy-Item -Destination "src\PyFLL\"

python -m build --wheel --outdir wheelhouse
$wheel = (Get-ChildItem "wheelhouse\*.whl").FullName
python -m twine check $wheel
python -m pip install --force-reinstall $wheel
python -m pip install "pytest>=7.0" "scikit-learn>=1.3.0" `
  "openpyxl>=3.1,<4" "matplotlib>=3.8" "simpful==2.12.0"
python -m pytest tests -v
```

Ожидаемый файл:

```text
openfll_python-0.2.0-cp314-cp314-win_amd64.whl
```

## Выпуск версии 0.2.0 после проверки

Ветка с подготовкой релиза должна сначала пройти review и быть объединена с
`main`. После этого:

1. Вручную запустить workflow с `publish_target=none`.
2. После успешной сборки запустить его с `publish_target=testpypi`.
3. Проверить установку из TestPyPI на Windows с CPython 3.14.
4. Создать тег `v0.2.0` на проверенном commit из `main`.
5. Создать и опубликовать GitHub Release для `v0.2.0`.
6. Подтвердить deployment в GitHub Environment `pypi`.

Повторно загрузить на PyPI файл с теми же именем и версией нельзя. Если после
публикации потребуется исправление, необходимо выпустить новую версию.
