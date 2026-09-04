# API Reference

Автогенерированная документация из `src/PyFLL/PyFLL.pyi` (type stubs).

## Доступные классы

- **[`SugenoEngine`](sugeno-engine.md)** — линейная обёртка над Sugeno-моделью OpenFLL

## Соглашения

- **Источник**: `src/PyFLL/PyFLL.pyi`. Этот файл — единственный источник для сигнатур и docstring.
- **Генерация**: `pybind11-stubgen` (см. [Linear Facade](../architecture/linear-facade.md))
  генерирует черновик из `R"doc()"` в C++; ручные `Literal`-типы добавляются в `.pyi`.
- **Синхронизация**: CI сравнивает сгенерированный stub с ручным; рассинхрон ломает build.

## Пример типичного API reference

```python
import PyFLL

e = PyFLL.SugenoEngine()
# IDE подсказывает:
e.add_input_var(name: str) -> None
e.add_membership_func(
    var_name: str,
    mf_name: str,
    type: Literal["constant", "triangular", "trapezoidal", "gaussian", "polynomial"],
    params: list[float],
) -> None
e.add_rule(rule: str) -> None
```

`Literal` типы видны в автодополнении IDE — попробуйте `e.add_membership_func("x1", "low", ` — IDE покажет 5 вариантов.
