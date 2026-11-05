# Сериализация и расширения движков

Поддерживаются `openfll.SugenoEngine` и `openfll.MamdaniEngine`. Сериализация
использует нативный JSON OpenFLL через C++ `IModel::pack` и `ser::load`.
Это отдельный механизм от `pickle/joblib` для `SugenoRegressor`.

Оба движка наследуются от `BaseLinearEngine`. Сугено и Мамдани — независимые
виды модели; настройки интегратора принадлежат только `MamdaniEngine`.

## Сохранение и загрузка

```python
from pathlib import Path
import openfll

# engine — настроенный и собранный SugenoEngine или MamdaniEngine.
text = engine.save_to_string()
document = engine.to_dict()  # обычный dict; эквивалент json.loads(text)
engine.save_to_file(Path("model.json"))

restored = openfll.load_from_string(text)
from_file = openfll.load_from_file(Path("model.json"))
assert restored.is_built()
value = restored.predict({"temperature": 3.0})
```

| Метод | Результат |
|---|---|
| `engine.save_to_string()` | `str` с JSON |
| `engine.save_to_file(path)` | запись JSON; существующий файл перезаписывается |
| `engine.to_dict()` | `dict[str, Any]` |
| `openfll.load_from_string(text)` | готовый движок соответствующего типа |
| `openfll.load_from_file(path)` | готовый движок соответствующего типа |

Пути принимают `str` и `pathlib.Path`, включая кириллицу и пробелы.
Родительский каталог файла должен существовать.

Сохраняются переменные, MF, правила, нормы и дефаззификатор. Для Мамдани
также сохраняются диапазон и шаг интегрирования. Необязательный раздел
`extra.linear_api` сохраняет порядок регистрации переменных, имена, типы и
параметры MF, включая неиспользованные переменные и термы. Без этого раздела
используется порядок переменных в нативном JSON. Нативные модели C++ без
метаданных также загружаются. Загрузчик разбирает JSON один раз и использует
одно дерево для восстановления модели и чтения метаданных.

Загруженный движок уже собран и предназначен для вычислений. Его нельзя
изменять через `add_*`, `build()` или `set_defuzzer()`. Текущие значения входов
не сохраняются: перед вычислением задайте все входы. Скаляры, словари,
последовательности и NumPy-батчи работают через обычный `predict()`;
`predict_batch()` поддерживает один выход. При нескольких выходах используйте
`set_input/calculate/get_output` или scalar `predict(..., output_name=...)`.

После загрузки JSON, сохранённого обновлённым фасадом, `find_belonging()`
работает по исходным именам MF, включая неиспользованные термы. У старых JSON
без описаний MF исходные имена восстановить невозможно. Python-метаданные
для `plot_mf()` автоматически не восстанавливаются; график можно построить
по значениям `find_belonging()` на нужной сетке.

Ошибки: сохранение до `build()` — `RuntimeError`; неверный JSON, структура,
неизвестный тип модели/дефаззификатора — `ValueError`; чтение/запись файла —
`OSError`; неверный тип пути — `TypeError`.

## Безопасное представление модели

`view = engine.model_ref()` возвращает `openfll.ModelView` после `build()`.
Это представление фасада, а не сырой C++ `IModel`.
Доступны `view.calculate()`, `view.save_to_string()`, `view.save_to_file(path)`
и `view.to_dict()`. Представление удерживает движок-владелец в памяти.

Сохранение движка и его представления использует один путь, включая все
метаданные `extra.linear_api`. Поэтому `view.to_dict() == engine.to_dict()`;
порядок переменных, неиспользованные переменные и имена MF не теряются.
`IModel::pack(IWriterProvider&)` вызывается внутри C++; сам низкоуровневый
интерфейс не экспортируется в Python.

## Выбор дефаззификатора и получение имён

```python
engine = openfll.SugenoEngine()
engine.set_defuzzer("wtaver")  # по умолчанию: взвешенное среднее
# engine.set_defuzzer("wtsum")  # взвешенная сумма
```

Выбор выполняется до `build()`. Эти режимы дают разные значения, если сумма
активаций правил не равна единице. `SugenoRegressor` использует `wtaver`.
`engine.get_defuzzer()` возвращает фактическое имя (`wtaver`, `wtsum` или
`centroid`), в том числе после загрузки и повторного сохранения модели.

`engine.get_input_var_names()` и `engine.get_output_var_names()` возвращают
независимые списки имён до и после `build()`, а также после загрузки.

`engine.find_belonging("mf_name", 3.0)` вызывает нативный
`IMembershipFunction::find_belonging(3.0)`. Метод доступен до и после `build()`
и после загрузки модели с метаданными именованных MF.
Неизвестное имя вызывает `IndexError`.

Полный пример с автоматическим сравнением результатов находится в
`examples/05_serialization_demo.py`; модель Мамдани описана в
[туториале](../tutorials/03-mamdani.md).
