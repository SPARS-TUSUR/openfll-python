# Сериализация и расширения движков

Поддерживаются `openfll.SugenoEngine` и `openfll.MamdaniEngine`. Сериализация
использует нативный JSON OpenFLL через C++ `IModel::pack` и `ser::load`.
Это отдельный механизм от `pickle/joblib` для `SugenoRegressor`.

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
`extra.linear_api` сохраняет порядок регистрации переменных фасада: без него
используется порядок переменных в нативном JSON. Нативные модели C++ без этого
раздела также загружаются.

Загруженный движок уже собран и предназначен для вычислений. Его нельзя
изменять через `add_*`, `build()` или `set_defuzzer()`. Текущие значения входов
не сохраняются: перед вычислением задайте все входы. Скаляры, словари,
последовательности и NumPy-батчи работают через обычный `predict()`;
`predict_batch()` поддерживает один выход. При нескольких выходах используйте
`set_input/calculate/get_output` или scalar `predict(..., output_name=...)`.

Имена MF и Python-метаданные графиков в нативный формат не входят.
Поэтому `find_belonging()` по прежнему имени MF и `plot_mf()` на загруженном
экземпляре недоступны. Неиспользованные MF не входят в построенную модель.

Ошибки: сохранение до `build()` — `RuntimeError`; неверный JSON, структура,
неизвестный тип модели/дефаззификатора — `ValueError`; чтение/запись файла —
`OSError`; неверный тип пути — `TypeError`.

## Доступ к IModel

`view = engine.model_ref()` возвращает `openfll.IModel` после `build()`.
Доступны `view.calculate()`, `view.save_to_string()`, `view.save_to_file(path)`
и `view.to_dict()`. Сериализаторы вызывают `IModel::pack(IWriterProvider&)`
внутри C++; писать Python-реализацию `IWriterProvider` не требуется.
Ссылка удерживает движок-владелец в памяти.

При сохранении через `IModel` получается стандартный JSON ядра без добавочного
порядка фасада в `extra.linear_api`. Для сохранения порядка колонок Python
предпочитайте `engine.save_to_string()` и `engine.save_to_file()`.

## Выбор дефаззификатора и получение имён

```python
engine = openfll.SugenoEngine()
engine.set_defuzzer("wtaver")  # по умолчанию: взвешенное среднее
# engine.set_defuzzer("wtsum")  # взвешенная сумма
```

Выбор выполняется до `build()`. Эти режимы дают разные значения, если сумма
активаций правил не равна единице. `SugenoRegressor` использует `wtaver`.

`engine.get_input_var_names()` и `engine.get_output_var_names()` возвращают
независимые списки имён до и после `build()`, а также после загрузки.

`engine.find_belonging("mf_name", 3.0)` вызывает нативный
`IMembershipFunction::find_belonging(3.0)`. Метод доступен до и после `build()`
для MF, зарегистрированных вручную. Неизвестное имя вызывает `IndexError`.

Полный пример с автоматическим сравнением результатов находится в
`examples/05_serialization_demo.py`; модель Мамдани описана в
[туториале](../tutorials/03-mamdani.md).
