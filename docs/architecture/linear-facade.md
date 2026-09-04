# Linear Facade

## Зачем

Ядро OpenFLL использует **fluent-цепочку** с внутренней state-machine:

```cpp
SugenoBuilder b;
b.add_rule().IF(SM).is(VeryLow).OR(dSM).is(FastDec).then()(dASV).is(EmergencyOpen).end();
```

Это **идиоматично для C++**, но:
- Каждый промежуточный объект (`IMemberAccessor`) должен быть зарегистрирован в Pybind11
- `#define AND`, `#define OR` — макросы пользователя, недоступные в Python
- Длинные namespace-пути вроде `OFLL::Models::Mamdani::Integrator::RectangleMidpointIntegrator` многословны
- Template-параметры для t/s-норм (`t_norm<TN::Min>`) недостижимы из Python

## Решение

**Плоский слой-фасад** скрывает fluent-цепочку и template-API за простыми методами,
принимающими `std::string` и `std::span<const double>`.

## Структура

```
src/OpenFLL/linear/
├── mf_factory.h/.cpp       make_membership_function(type, params)
├── norm_registry.h          parse_norm_name(name, kind) → NormId
├── rule_parser.h/.cpp       parse_rule(text) → ParsedRule
└── sugeno_engine.h/.cpp     SugenoEngine (публичный класс)
```

## Архитектура фасада

### `SugenoEngine` — публичный API

```cpp
namespace OFLL::Linear {
    class SugenoEngine {
    public:
        void add_input_var(std::string_view name);
        void add_output_var(std::string_view name);
        void add_membership_func(std::string_view var, std::string_view mf,
                                 std::string_view type, std::span<const double> params);
        void add_rule(std::string_view rule);              // строковый парсер
        void set_default_t_norm(std::string_view name);
        void set_default_s_norm(std::string_view name);
        void build();
        void set_input(std::string_view name, double value);
        double get_output(std::string_view name) const;
        void calculate();
        bool is_built() const;

    private:
        std::unordered_map<std::string, std::shared_ptr<Var>> _input_vars;
        std::unordered_map<std::string, std::shared_ptr<Var>> _output_vars;
        std::unordered_map<std::string, std::shared_ptr<IMembershipFunction>> _mfs;
        std::vector<std::string> _rule_texts;
        std::unique_ptr<IModel> m_model;
        NormId m_default_t_norm = NormId::Min;
        NormId m_default_s_norm = NormId::Max;
        bool m_built = false;
    };
}
```

### `mf_factory` — по строковому типу создаёт MF

```cpp
auto mf = make_membership_function("triangular", {0.0, 5.0, 10.0});
// → shared_ptr<TriangularMF>(0, 5, 10)

auto mf = make_membership_function("gaussian", {-5.0, 2.0});
// → shared_ptr<GaussianMF>(center=-5, sigma=2)
```

Поддерживаемые типы: `constant`, `triangular`, `trapezoidal`, `gaussian`, `polynomial`
(последний — отложен, требует `PolynomialFactory` с контекстом переменных).

### `norm_registry` — строковое имя нормы → дискриминатор

```cpp
enum class NormId { Min, ProdAnd, BoundedDiff, DrasticProd, EinsteinProd, HamacherProd,
                    Max, AlgebraicSum, BoundedSum, DrasticSum, EinsteinSum, HamacherSum };
```

`t_norm` для AND, `s_norm` для OR. Tag-dispatch в `add_parsed_rule` выбирает
правильный C++-тип через switch.

### `rule_parser` — строковое правило → AST

Грамматика v0.2 (см. [Python API план](python-api-plan.md)):

```
rule        := IF antecedent THEN consequent
antecedent  := term ((AND|OR) [norm_name] term)*
consequent  := term
term        := IDENT IS STRING
```

Пример: `IF x1 IS "low" AND[prod_and] x2 IS "fast" THEN y IS "open"`.

## Почему header-only (не всё)

- **`sugeno_engine.h`/`cpp`** — реализация держит `SugenoBuilder`/`RulesBuilder` инстансы,
  нельзя inline-определить из-за зависимостей.
- **`mf_factory.h`/`cpp`**, **`rule_parser.h`/`cpp`** — то же самое.
- **`norm_registry.h`** — header-only (только `enum` + маппинг).

## Когда расширять

| Хочу добавить | Куда | Пример |
|---------------|------|--------|
| Новый тип MF | `mf_factory.h` + `mf_factory.cpp` | `set_union_mf`, `bell_mf` |
| Новую t/s-норму | `norm_registry.h` | `lambda_norm` |
| Новый синтаксис правил | `rule_parser.cpp` | `NOT` (отрицание), скобки для приоритетов |
| Новый метод `SugenoEngine` | `sugeno_engine.h/.cpp` | `clone()`, `serialize()` |
| Целую новую модель (Mamdani) | `mamdani_engine.h/.cpp` (фаза 4) | — |
