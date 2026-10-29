# API Reference

Автогенерированная документация для публичного пакета `openfll`.

## Доступные классы

- **[`SugenoEngine`](sugeno-engine.md)** — линейная обёртка над Sugeno-моделью OpenFLL

## Helper-функции

- `constant(value)` → `[value]`
- `triangular(a, b, c)` → `[a, b, c]`
- `trapezoidal(a, b, c, d)` → `[a, b, c, d]`
- `gaussian(center=..., sigma=...)` → `[center, sigma]`
- `plot_mf(engine, ...)` → отдельные matplotlib Figure/Axes для переменных

Эти функции только делают порядок параметров явным. Они не создают Python
membership-function объекты и не дублируют semantic validation native backend.

## Графики функций принадлежности

`engine.plot_mf()` и `openfll.plot_mf(engine)` автоматически строят графики
для всех функций, зарегистрированных через `add_membership_func`. Входные
переменные отображаются кривыми степени принадлежности. Константные выходы
Сугено отображаются singleton-маркерами.

```python
plots = engine.plot_mf(
    variables=["x1", "x2", "y"],
    ranges={"x1": (-10.0, 20.0)},
    show=False,
    save_dir="plots",
)
axes = plots["x1"].axes
```

Если `ranges` не передан, диапазон каждой оси определяется из параметров её
triangular, trapezoidal и gaussian функций. Matplotlib подключается только при
вызове; установите extra `openfll_python[plot]`.

## Scalar predict

`engine.predict(inputs, output_name=None)` выполняет один scalar inference через
существующий native workflow:

```text
set_input -> calculate -> get_output
```

Это Python convenience, а не native vectorization. Если через `add_output_var`
зарегистрирован ровно один output, `output_name` можно не передавать. Если
outputs несколько, передайте `output_name` явно.

Поведение ошибок:

- если в `inputs` отсутствует зарегистрированная input-переменная,
  `predict` выбрасывает `ValueError`;
- неизвестные input names передаются в native `set_input`, поэтому backend
  сохраняет свой текущий exception contract;
- ошибки lifecycle до `build()` остаются native `RuntimeError`.

## Native batch predict

`engine.predict_batch(X, input_names=[...])` вызывает native C++ batch loop из
`PyFLL._core`. Это не Python loop по samples.

```python
import numpy as np
import openfll

X = np.array(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ],
    dtype=np.float64,
)

y = engine.predict_batch(
    X,
    input_names=["x1", "x2"],
)
```

`input_names` задаёт порядок колонок: `X[:, 0] -> "x1"`,
`X[:, 1] -> "x2"`. В v1 поддерживается single-output engine, поэтому output
shape равен `(n_samples,)`, dtype — `float64`.

## Соглашения

- **Публичный источник**: `src/openfll/__init__.pyi`.
- **Runtime backend**: `PyFLL.SugenoEngine`, который загружает `PyFLL._core`.
- **Важно**: `polynomial`, `fit`, `get_params` и `set_params`
  не входят в текущий публичный API.

## Пример типичного API reference

```python
import openfll

e = openfll.SugenoEngine()
# IDE подсказывает:
e.add_input_var(name: str) -> None
e.add_membership_func(
    var_name: str,
    mf_name: str,
    type: Literal["constant", "triangular", "trapezoidal", "gaussian"],
    params: list[float],
) -> None
openfll.gaussian(center=-5.0, sigma=2.0) -> list[float]
e.add_rule(rule: str) -> None
e.predict({"x1": 1.0, "x2": 2.0}) -> float
e.predict_batch(X, input_names=["x1", "x2"]) -> numpy.ndarray
e.plot_mf(show=True) -> dict[str, MembershipPlot]
```

`Literal` типы видны в автодополнении IDE — попробуйте `e.add_membership_func("x1", "low", ` — IDE покажет 4 поддержанных варианта.
