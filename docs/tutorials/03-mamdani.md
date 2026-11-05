# Туториал 03: Mamdani

`openfll.MamdaniEngine` использует нативные `MamdaniBuilder`, `Mamdani::calculate()`
и `CentroidDefuzzer`. Регистрация переменных, MF и правил совпадает с Sugeno.
Выходные термы здесь являются нечёткими множествами.

```python
import openfll

e = openfll.MamdaniEngine()
e.add_input_var("temperature")
e.add_output_var("power")
e.add_membership_func("temperature", "cold", "triangular", [0, 0, 10])
e.add_membership_func("temperature", "hot", "triangular", [0, 10, 10])
e.add_membership_func("power", "low", "triangular", [0, 2, 4])
e.add_membership_func("power", "high", "triangular", [6, 8, 10])
e.add_rule('IF temperature IS "cold" THEN power IS "low"')
e.add_rule('IF temperature IS "hot" THEN power IS "high"')
e.set_defuzzer("centroid")
e.set_integrator(-1.0, 11.0, 0.01)
e.build()

assert abs(e.predict(5.0) - 5.0) < 1e-10
e.save_to_file("mamdani.json")
restored = openfll.load_from_file("mamdani.json")
assert isinstance(restored, openfll.MamdaniEngine)
assert abs(restored.predict(5.0) - 5.0) < 1e-10
```

Перед `build()` обязательно задайте `set_integrator(start, end, step)`:
конечные числа, `start < end`, `0 < step <= end - start`. Интегратор использует
метод средних прямоугольников. Выбирайте шаг, на который длина интервала
делится без остатка. Один диапазон используется для всех выходов модели;
он должен покрывать их функции принадлежности.

Нативный алгоритм применяет усечение термов весами правил и centroid-расчёт.
Если ни одно правило не активировалось, площадь результирующего множества
равна нулю; ядро возвращает `NaN`. Единственный дефаззификатор Мамдани —
`centroid`; выбор `wtaver/wtsum` отклоняется.

Доступны `predict`, `predict_batch` для одного выхода, `find_belonging`,
`get_input_var_names`, `get_output_var_names` и `model_ref`. Подробности
сохранения и ограничений загруженного экземпляра — в
[описании сериализации](../api/serialization.md).
