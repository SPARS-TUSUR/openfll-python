# `SugenoEngine`

Полное описание генерируется из `src/openfll/__init__.py` (стаб для griffe)
и `src/openfll/__init__.pyi` (type hints для IDE). Чтобы обновить —
правьте только docstring'и в стабе, затем пересоберите документацию
(`mkdocs build --strict`).

## Класс

::: openfll.SugenoEngine
    options:
      members:
        - __init__
        - add_input_var
        - add_output_var
        - add_membership_func
        - add_rule
        - set_default_t_norm
        - set_default_s_norm
        - build
        - set_input
        - get_output
        - calculate
        - is_built
      show_source: false
      heading_level: 2

## Две формы правил

В `SugenoEngine` есть **две** формы `add_rule`:

### Текстовая форма

```python
engine.add_rule('IF x1 IS "low" AND[prod_and] x2 IS "high" THEN y IS "open"')
```

- Поддерживает `AND`/`OR` и явные `[norm_name]`.
- Удобна для статических моделей, записанных вручную.

### Кортежная (структурная) форма

```python
engine.add_rule(
    antecedent=[("x1", "low"), ("x2", "high")],
    consequent=[("y", "open")],
    t_norm="prod_and",   # default: "" → engine default t-norm
    s_norm="",           # default: "" → engine default s-norm
)
```

- Удобна для **программной** генерации правил (списки в цикле, `itertools.product`).
- Все операторы между термами `antecedent` — `AND`. Для `OR` используйте текстовую форму.
- Опциональные `t_norm`/`s_norm` совпадают с именами из `set_default_t_norm`/`set_default_s_norm`.
  Если не заданы — используются дефолтные.

Обе формы **полностью эквивалентны по вычислениям**: тест на parity даёт
max abs diff = 0.0 между текстовой и кортежной формой для одного и того
же набора правил.