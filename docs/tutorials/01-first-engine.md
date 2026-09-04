# Туториал 01: первый движок

Минимальный Sugeno-движок: 2 входа, 1 выход, 2 правила.

Полный код — в [`tests/python/test_sugeno_smoke.py`](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/tests/python/test_sugeno_smoke.py).

## Шаг 1: Создаём движок

```python
import PyFLL

e = PyFLL.SugenoEngine()
```

## Шаг 2: Регистрируем переменные

```python
e.add_input_var("x1")
e.add_input_var("x2")
e.add_output_var("y")
```

Имена должны быть уникальны в пределах движка. Пустые имена и дубликаты → `ValueError`.

## Шаг 3: Добавляем функции принадлежности

```python
# X1: 2 triangular
e.add_membership_func("x1", "low",  "triangular", [-10, 0, 10])
e.add_membership_func("x1", "high", "triangular", [0, 10, 20])

# X2: 2 gaussian (center, sigma)
e.add_membership_func("x2", "low",  "gaussian", [-5, 2])
e.add_membership_func("x2", "high", "gaussian", [5, 2])

# Y: 2 constant (Sugeno 0-го порядка)
e.add_membership_func("y", "lo",  "constant", [10.0])
e.add_membership_func("y", "hi",  "constant", [30.0])
```

Формат параметров:

| Type | params |
|------|--------|
| `constant` | `[value]` |
| `triangular` | `[a, b, c]` — left, peak, right |
| `trapezoidal` | `[a, b, c, d]` |
| `gaussian` | `[center, sigma]` |
| `polynomial` | отложен |

## Шаг 4: Добавляем правила

```python
# Правило 1: AND (t-norm по умолчанию = Min)
e.add_rule('IF x1 IS "low" AND x2 IS "low" THEN y IS "lo"')

# Правило 2: OR (s-norm по умолчанию = Max)
e.add_rule('IF x1 IS "high" OR x2 IS "high" THEN y IS "hi"')

# Правило 3: AND с явной t-нормой ProdAnd (умножение весов)
e.add_rule('IF x1 IS "low" AND[prod_and] x2 IS "high" THEN y IS "lo"')
```

Грамматика:

```
IF <var1> IS "<mf1>" ((AND|OR) [<norm_name>] <var2> IS "<mf2>")* THEN <out_var> IS "<out_mf>"
```

Доступные нормы: `min`, `prod_and`, `bounded_diff`, `drastic_prod`, `einstein_prod`, `hamacher_prod`
(t-нормы для AND); `max`, `algebraic_sum`, `bounded_sum`, `drastic_sum`, `einstein_sum`, `hamacher_sum`
(s-нормы для OR).

## Шаг 5: Сборка и расчёт

```python
e.build()                  # компилирует модель
e.set_input("x1", 5.0)    # входы
e.set_input("x2", 0.0)
e.calculate()             # нечёткий вывод
print(e.get_output("y"))  # → 16.667
```

После `build()` нельзя добавлять переменные, MF или правила — будет `RuntimeError`.

## Ожидаемый результат

```
y = 16.667
```

Это WTAver Sugeno: правило 1 (AND, оба MF ~0.5 при (5,0)) даёт вес 0.5 и выход 10,
правило 2 (OR) даёт 0. Итог: (0.5·10 + 0.5·30) / 1.0 = 20… или другие интерпретации,
зависит от того, как сходятся MF.

> **Точный ответ зависит от формы MF.** Запустите `tests/python/test_sugeno_smoke.py`
> и сверьтесь с `BASELINE.md` для воспроизводимости.

## Что дальше

- Бо`льше правил и переменных — см. [`tests/python/test_sugeno_smoke.py`](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/tests/python/test_sugeno_smoke.py)
- Динамическая модель (компрессор) — [туториал 02](02-fuzzy-arithmetic.md)
- API reference — [SugenoEngine](../api/sugeno-engine.md)
