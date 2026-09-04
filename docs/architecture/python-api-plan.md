# Python API для OpenFLL — план работ

**Документ:** `docs/PYTHON_API_PLAN.md`
**Версия:** 0.1 (черновик для обсуждения)
**Статус:** анализ завершён, идея одобрена, детальный план ниже

---

## 1. Резюме

**Проблема.** Текущее публичное C++ API библиотеки OpenFLL построено на fluent-цепочках
(`builder.add_input(...).add_member(...).is(...).AND(...).is(...).then().is(...).end()`),
абстрактных фабриках состояний (`IState`, `RulesBuilderEngine`), макросах, определённых
на стороне пользователя (`#define AND .sign(...)`), и тяжёлых `std::shared_ptr` /
`std::make_shared` конструкциях в публичной поверхности. Всё это идиоматично для C++,
но **непригодно для прямой обвязки через Pybind11** без существенных накладных расходов
и потери строгой типизации.

**Идея.** Создать **плоский (линейный) C++-фасад** над текущим ядром, который принимает
простые типы (`std::string`, `double`, `std::vector<double>`), скрывает внутренние
shared_ptr и билдеры и предоставляет один самодостаточный класс-движок. Pybind11-биндинги
тогда оборачивают **только** этот фасад, а не 15 абстракций ядра.

**Целевой результат для пользователя Python:**

```python
import PyFLL

engine = PyFLL.SugenoEngine()

engine.add_input_var("x1")
engine.add_membership_func("x1", "small", "triangular", [-1.0, 0.0, 10.0])
engine.add_membership_func("x1", "large", "triangular", [0.0, 10.0, 11.0])

engine.add_output_var("y1", constant=True)
engine.add_membership_func("y1", "low",  "constant", [-15.0])
engine.add_membership_func("y1", "hold", "constant", [0.0])

engine.add_rule('IF x1 IS "small" THEN y1 IS "hold"')
engine.add_rule('IF x1 IS "large" AND x1 IS "small" THEN y1 IS "low"')

engine.build()
engine.set_input("x1", 2.5)
engine.calculate()
print(engine.get_output("y1"))
```

---

## 2. Анализ текущей кодовой базы

### 2.1. Что есть в репозитории

| Компонент | Файлы | Состояние |
|-----------|-------|-----------|
| Ядро (стат. библиотека `OpenFLL`) | `src/OpenFLL/**` | Компилируется как `add_library(... STATIC)` |
| Лаунчер | `src/launch.cpp` | Использует «новый» fluent-стиль, демо Sugeno |
| Пример | `examples/compressor.cpp` | Использует «старый» API (`IIncompleteRuleMatrix`, `CompoundRuleBuilder`, `IIntegrator`) — **включает заголовки, которых нет в `src/OpenFLL/`** |
| Тесты GoogleTest | `tests/**` | Сборка + CI через `.github/workflows/cpp-tests.yml` (windows-latest, MSVC) |
| Бенчмарк | `benchmark/fuzzy_bench.cpp` | Подключается через `OPENFLL_BUILD_BENCHMARKS=ON` |
| Заготовка PyFLL | `src/PyFLL/{CMakeLists.txt,PyFLL.cpp}` | `CMakeLists.txt` закомментирован, `PyFLL.cpp` пустой; submodule `vendor/pybind11` подтянут, артефакты сборки остались в `src/PyFLL/build/` |

### 2.2. Архитектура ядра (то, что важно для фасада)

**Типы моделей.** Два наследника `BaseModel` (`src/OpenFLL/models/core/internal/base_model/base_model.h`):
- `Sugeno<is_wtsum>` — `src/OpenFLL/models/sugeno/sugeno.h`. `calculate()` реализован полностью.
- `Mamdani` — `src/OpenFLL/models/mamdani/mamdani.{h,cpp}`. `calculate()` **полностью закомментирован** (см. `mamdani.cpp:23–50`).
  Это означает, что `Mamdani` сегодня — это стаб, а не рабочая модель.

**Билдеры.** `SugenoBuilder` и `MamdaniBuilder` наследуют `BaseBuilder` (`models/core/internal/base_builder/base_builder.h`).
`BaseBuilder::add_rule(Rules::Rule rule)` принимает уже **готовый объект `Rule`**, собранный отдельным `RulesBuilder`.

**RulesBuilder** (`src/OpenFLL/rules/rules_builder/rules_builder.h`) — это fluent-цепочка
с **внутренней машиной состояний** (`IState` + `AddInputVar` → `AddInputMF` →
`AddOperator` → ... → `then` → ... → `end`). Если вызвать метод не в той
последовательности, состояние бросает `std::runtime_error("Incorrect operator call sequence!")`.

**Переменные и MF.** `Var : IVar` (простой `double _value`), MF — `Constant`,
`TriangularMF(a, b, c)`, `TrapezoidalMF(a, b, c, d)`, `GaussianMF(center, sigma)`,
`Polynomial`. Всё передаётся через `std::shared_ptr<...>`.

**t-нормы / s-нормы.** `OFLL::Rules::Core::Operators::TN::Min/ProdAnd/...` и
`SN::Max/AlgebraicSum/...` — это **типы-template-параметры** для `RulesBuilder::t_norm<T>()` /
`s_norm<T>()`. На стороне C++ пользователь пишет `#define AND .t_norm<...>()`.
На стороне Python — это будут строки.

**Storage.** `ModelContext` (`rules/core/internal/model_context.h`) и `RuleContext`
владеют `ResourcePool<shared_ptr<...>>` для входов/выходов/MF. Доступ — через
`model_context_ref()` (внутри `BaseModel`) и `rule_context_ref()` (внутри `RulesBuilder`).
Публичного API для итерации по ним «по имени» **нет**: всё по индексу / shared_ptr.

### 2.3. Что делает текущий API неудобным для Pybind11

1. **Fluent + state machine.** `RulesBuilder::operator()(var) → is(mf) → AND(var) → ...`
   возвращает `RulesBuilder&` на временный объект. Pybind11 не сможет удобно
   выставить такую цепочку, потому что между вызовами требуется хранить
   `RulesBuilder` (а не ссылку) и при этом каждый промежуточный вызов
   потенциально меняет типобезопасность.
2. **Абстракции без владельца.** `IState`, `IOpFactory`, `OpFactoryWithPriority`
   — это внутренние детали state-машины. Если их всех регистрировать в Pybind11,
   это ~30–50 классов в биндингах, ни один из которых Python-пользователю не нужен.
3. **template-параметры для t/s-норм.** `t_norm<OFLL::Rules::Core::Operators::TN::Min>`
   из Python недостижим: Pybind11 не может передавать C++-типы.
4. **shared_ptr повсюду.** `add_input(sm_input, "SM").add_member(std::make_shared<...>, "VeryLow")`
   требует, чтобы Python-код создавал и владел `shared_ptr` — но `shared_ptr`
   на абстрактный `IVar` из Python сделать нельзя без proxy-класса.
5. **Макросы пользователя.** `#define AND ...` — синтаксический сахар на стороне
   .cpp, в Python его нет.
6. **Нет имён во внутренних контейнерах.** `ModelContext` хранит переменные
   безымянно. Из Python «достать Var по строковому имени» невозможно без
   построения своего name→shared_ptr-индекса.

### 2.4. Что уже сделано для PyFLL

- Подключён `vendor/pybind11` (submodule).
- В `src/PyFLL/` лежат **закомментированный** `CMakeLists.txt` и пустой
  `PyFLL.cpp` (только `#include <pybind11/pybind11.h>` + `using namespace pybind11`).
- В `src/PyFLL/build/` остались артефакты прежней попытки сборки (`PyFLL.obj` и пр.).
- Корневой `CMakeLists.txt` уже умеет **условно** подключать `src/PyFLL/`
  при наличии файлов в `vendor/pybind11/*`.

То есть инфраструктура Pybind11 подведена, но **не доведена до рабочего состояния**.
CMakeLists.txt закомментирован, скорее всего, из-за обнаруженной сложности
биндинга — именно той, которую мы и собираемся обойти через фасад.

---

## 3. Оценка идеи «линейный фасад + Pybind11»

### 3.1. Вердикт

**Идея — правильная.** Это классический паттерн Facade, и он точно соответствует
масштабу проблемы. Прямая обвязка fluent-цепочки + state-машины + template-норм
через Pybind11 — это либо тысячи строк boilerplate, либо потеря
type-safety без выигрыша в удобстве.

### 3.2. Плюсы подхода

| Плюс | Обоснование |
|------|-------------|
| Минимальная поверхность биндингов | Один класс-фасад, ~10 методов, простые типы → биндинг в 80–120 строк |
| Управление памятью остаётся в C++ | Python не может сломать shared_ptr-граф, т.к. вся модель инкапсулирована |
| Совместимость со старым API | Facade — **дополнительный** слой, ничего не ломает в существующем коде |
| Тестируемость | Facade тестируется на C++ через `launch.cpp` и существующие Google-тесты |
| Постепенное наращивание | Можно начать с Sugeno + Constant, потом добавлять MF, t/s-нормы, Mamdani |
| Чистая доменная модель | Python-пользователь думает категориями fuzzy logic, а не C++-обёрток |

### 3.3. Минусы и компромиссы

| Минус | Смягчение |
|-------|-----------|
| Потеря compile-time проверки t/s-норм | Валидация на этапе `build()` фасада (бросаем `std::invalid_argument`) |
| Нужен парсер правил | Минимальный лексер строк (IF/AND/OR/THEN/IS) — ~150–200 строк, опционально вторая форма: список кортежей |
| Дублирование документации (C++ fluent + Linear + Python) | Linear API документируется как «рекомендуемый для биндингов» в doxygen-комментариях |
| Facade может стать «скрытым god-классом» | Разбить на `SugenoLinearEngine` / `MamdaniLinearEngine` / общий `LinearEngineBase` (шаблон / CRTP) |

### 3.4. Альтернативы, которые мы **не** выбираем

| Альтернатива | Почему отклонена |
|--------------|-------------------|
| Прямая обвязка fluent `RulesBuilder` | Не выразимо в Pybind11, требует хранить state-машину, теряется type-safety |
| Обвязка каждой `IState` отдельно | Утечка внутренних деталей в Python API, ~30 лишних классов |
| Обвязка только `SugenoBuilder::add_rule(string)` через reflection (refl-cpp) | `SugenoBuilder` не принимает строку — нужно парсить, то же что фасад, но без инкапсуляции |
| DSL на Python-стороне, который строит C++ AST и компилирует в шар | Избыточно для альфа-версии, не вписывается в «простой Python» |

---

## 4. Целевая архитектура

```
┌──────────────────────────────────────────────────────────┐
│                  Python (PyFLL)                          │
│   SugenoEngine().add_input_var("x1").add_rule("...")     │
└─────────────────────────┬────────────────────────────────┘
                          │ pybind11 (≈100 строк .def())
┌─────────────────────────▼────────────────────────────────┐
│       Linear Facade (C++, header-only)                   │
│   OFLL::Linear::SugenoEngine                             │
│     - add_input_var(name)                                │
│     - add_membership_func(var, mf, type, params)         │
│     - add_rule(string | tuple-list)                      │
│     - build()                                            │
│     - set_input(name, value)                             │
│     - calculate()                                        │
│     - get_output(name)                                   │
│   ┌────────────────────────────────────────────┐         │
│   │ Name → shared_ptr<IVar>   (HashMap)        │         │
│   │ Name → shared_ptr<IMF>    (HashMap)        │         │
│   │ String rule parser (IF/AND/OR/THEN/IS)     │         │
│   └────────────────────────────────────────────┘         │
└─────────────────────────┬────────────────────────────────┘
                          │ обычные вызовы ядра
┌─────────────────────────▼────────────────────────────────┐
│      OpenFLL core (уже существующий)                     │
│   SugenoBuilder + RulesBuilder + Var + MFs + Context     │
└──────────────────────────────────────────────────────────┘
```

**Ключевой принцип.** Facade — единственная публичная поверхность для Python.
Ядро (`src/OpenFLL/`) не меняется в этой работе. Facade — отдельный header-only
модуль в `src/Linear/`, собирается в ту же стат. библиотеку или в отдельный
объектный файл, линкуется к `PyFLL.cpp`.

---

## 5. Детальный план по фазам

### Фаза 0 — согласование (1–2 дня)

| # | Задача | Артефакт | Критерий приёмки |
|---|--------|----------|------------------|
| 0.1 | Согласовать строковый формат правил | секция 6.2 этого документа | Утверждённый грамматический формат |
| 0.2 | Согласовать набор MF на фазу 1 | список | Минимум: `constant`, `triangular`, `gaussian` |
| 0.3 | Согласовать имена Python-модулей и классов | секция 7 | Без коллизий с PyPI |

### Фаза 1 — Linear Facade v0.1 (Sugeno only, минимум)

Цель: рабочий header-only класс `OFLL::Linear::SugenoEngine`, который
проходит путь от строкового API до `Sugeno::calculate()` и обратно.

| # | Задача | Файлы | Зависимости |
|---|--------|-------|-------------|
| 1.1 | Скелет `Linear::SugenoEngine` с name→shared_ptr индексами | `src/Linear/sugeno_engine.h` (новый) | — |
| 1.2 | `add_input_var(name)`, `add_output_var(name)` | `sugeno_engine.h` | 1.1 |
| 1.3 | `add_membership_func(var, mf_name, type, params)` для `constant`, `triangular`, `gaussian` | `sugeno_engine.h` + `src/Linear/mf_factory.h` (новый) | 1.2 |
| 1.4 | `build()` — собирает `SugenoBuilder` + `RulesBuilder` и сохраняет `unique_ptr<IModel>` | `sugeno_engine.h` | 1.3, 1.5 |
| 1.5 | Сборка `SugenoBuilder` + `RulesBuilder` из name-индексов | внутри `sugeno_engine.h` | 1.3 |
| 1.6 | `set_input(name, value)`, `get_output(name)`, `calculate()` | `sugeno_engine.h` | 1.4 |
| 1.7 | **Строковый парсер правил v0.1** (только `IF ... IS ... AND ... IS ... THEN ... IS ...`) | `src/Linear/rule_parser.h` (новый) | 1.4 |
| 1.8 | C++-тест фасада: повторяет сценарий из `examples/compressor.cpp` | `tests/OpenFLL/linear/sugeno_engine_test.cpp` (новый) | 1.7 |
| 1.9 | Включение `src/Linear/` в CMake | `src/OpenFLL/CMakeLists.txt` (дополнить) | 1.1–1.7 |
| 1.10 | CI green: build + gtest | `.github/workflows/cpp-tests.yml` уже подходит | 1.8, 1.9 |

**Критерий завершения фазы 1:** C++ тест
`linear_sugeno_engine_test.cpp` даёт **побайтово те же выходы** для того же
входного сценария, что и сценарий из `examples/compressor.cpp` через
прямой `SugenoBuilder`. Это baseline-инвариант, который мы зафиксируем
в `tests/OpenFLL/linear/BASELINE.md`.

### Фаза 2 — Pybind11 биндинги v0.1

Цель: модуль `PyFLL._core` импортируется в Python и позволяет повторить
тот же сценарий, что и фаза 1, через Python-скрипт.

| # | Задача | Файлы | Зависимости |
|---|--------|-------|-------------|
| 2.1 | Раскомментировать и привести в порядок `src/PyFLL/CMakeLists.txt` | `src/PyFLL/CMakeLists.txt` | — |
| 2.2 | Подключить `pybind11_add_module(PyFLL ...)` | `src/PyFLL/CMakeLists.txt` | 2.1 |
| 2.3 | Зарегистрировать `SugenoEngine` в `pybind11` | `src/PyFLL/PyFLL.cpp` | фаза 1, 2.1 |
| 2.4 | Stubs для исключений (`std::invalid_argument`, `std::runtime_error` → `ValueError`/`RuntimeError`) | `PyFLL.cpp` | 2.3 |
| 2.5 | Сборка `PyFLL.pyd` (`_core` на Windows) | `src/PyFLL/CMakeLists.txt` | 2.2, 2.3 |
| 2.6 | Минимальный smoke-тест на Python: повторяет C++ тест из фазы 1 | `tests/python/test_sugeno_smoke.py` (новый) | 2.5 |
| 2.7 | Сборка Python wheel через `pyproject.toml` (опционально, можно отложить) | `pyproject.toml` (новый) | 2.5 |

**Критерий завершения фазы 2:** запуск `python tests/python/test_sugeno_smoke.py`
выдаёт идентичные числа тем, что даёт C++ тест из фазы 1.

### Фаза 3 — расширение фасада

| # | Задача | Файлы | Зависимости |
|---|--------|-------|-------------|
| 3.1 | Добавить `trapezoidal`, `polynomial` в `mf_factory` | `src/Linear/mf_factory.h` | 1.3 |
| 3.2 | Расширить парсер правил: `OR`, явный выбор t/s-нормы (`AND min`, `OR max`, `OR algebraic_sum`) | `src/Linear/rule_parser.h` | 1.7 |
| 3.3 | Поддержка альтернативной формы правил — список кортежей `("x1 small", "x2 low") → ("y1 high")` | `rule_parser.h` | 3.2 |
| 3.4 | `set_default_t_norm(string)`, `set_default_s_norm(string)` | `sugeno_engine.h` | 3.2 |
| 3.5 | `save(path)` / `load(path)` через существующую сериализацию (если дозрела) | `sugeno_engine.h` | базовая сериализация ядра |
| 3.6 | Обновить C++ тесты: новые MF, OR, нормы | `sugeno_engine_test.cpp` | 3.1–3.4 |
| 3.7 | Обновить Python smoke-тесты | `tests/python/` | 3.6 |

### Фаза 4 — `MamdaniLinearEngine`

| # | Задача | Файлы | Зависимости |
|---|--------|-------|-------------|
| 4.1 | Дождаться работающей `Mamdani::calculate()` (см. `mamdani.cpp` — сейчас закомментировано) | вне нашего скоупа | чужой WIP |
| 4.2 | Выделить общий `LinearEngineBase` (CRTP или template) | `src/Linear/linear_engine_base.h` (новый) | фаза 3 |
| 4.3 | `OFLL::Linear::MamdaniEngine` с теми же методами + `set_integrator(name, params)` | `src/Linear/mamdani_engine.h` | 4.1, 4.2 |
| 4.4 | Обвязка в `PyFLL.cpp` | `PyFLL.cpp` | 4.3 |
| 4.5 | C++ и Python тесты | `tests/...` | 4.4 |

**Важно.** Фаза 4 блокируется внешним WIP по `Mamdani::calculate()`. До тех пор
фасад Mamdani можно сделать как **стаб с `throw not_implemented`**, чтобы
зафиксировать API, но фактически не использовать.

### Фаза 5 — полировка и публикация

| # | Задача | Файлы | Зависимости |
|---|--------|-------|-------------|
| 5.1 | Документация Python API (docstrings → авто-генерация) | `docs/python_api.md` | 3.x |
| 5.2 | Примеры в `examples/python/` (`compressor.py`, `simple_demo.py`) | новые | 3.x |
| 5.3 | CI: отдельный workflow для Python-сборки | `.github/workflows/python-tests.yml` | 2.6 |
| 5.4 | Публикация wheel на PyPI (опционально) | `pyproject.toml` + `twine` | 5.1–5.3 |
| 5.5 | README секция «Python API» | `README.md` | 5.1 |

---

## 6. Дизайн Linear Facade (C++)

### 6.1. Сигнатуры

```cpp
namespace OFLL::Linear {

    class SugenoEngine {
    public:
        SugenoEngine() = default;
        ~SugenoEngine() = default;

        SugenoEngine(const SugenoEngine&) = delete;
        SugenoEngine& operator=(const SugenoEngine&) = delete;

        // --- Конфигурация ---
        void add_input_var(std::string_view name);
        void add_output_var(std::string_view name);

        // type ∈ {"constant", "triangular", "trapezoidal", "gaussian", "polynomial"}
        // params — список параметров функции (зависит от type).
        //   constant     : [value]
        //   triangular   : [a, b, c]
        //   trapezoidal  : [a, b, c, d]
        //   gaussian     : [center, sigma]
        //   polynomial   : [a0, a1, ..., an]
        // Бросает std::invalid_argument при неизвестном type или неверном числе параметров.
        void add_membership_func(std::string_view var_name,
                                 std::string_view mf_name,
                                 std::string_view type,
                                 std::span<const double> params);

        // Строка формата:
        //   IF <var1> IS "<mf1>" (AND|OR <var2> IS "<mf2">)* THEN <var_out> IS "<mf_out>"
        // Бросает std::invalid_argument при синтаксической ошибке.
        // Бросает std::out_of_range при ссылке на незарегистрированное имя.
        void add_rule(std::string_view rule);

        // Альтернативная форма (для программной генерации правил):
        //   inputs  = [{"x1", "small"}, {"x2", "low"}]   -- {{var_name, mf_name}, ...}
        //   outputs = [{"y1", "high"}]
        void add_rule_tuple(std::initializer_list<std::pair<std::string_view, std::string_view>> inputs,
                            std::initializer_list<std::pair<std::string_view, std::string_view>> outputs);

        // Нормы по умолчанию: "min" | "prod_and" | "hamacher_prod" | ... (t-нормы)
        //                   "max" | "algebraic_sum" | ... (s-нормы)
        void set_default_t_norm(std::string_view name);
        void set_default_s_norm(std::string_view name);

        // Сборка модели. После build() нельзя add_* — будет runtime_error.
        // Бросает std::runtime_error при конфликтах (пустые входы, неизвестные имена).
        void build();

        // --- Исполнение ---
        void set_input(std::string_view name, double value);
        double get_output(std::string_view name) const;

        void calculate();

    private:
        // name → shared_ptr — для повторного использования в RulesBuilder
        std::unordered_map<std::string, std::shared_ptr<OFLL::Var::Var>, StringHash, StringEq> _vars;
        std::unordered_map<std::string, std::shared_ptr<OFLL::Abstract::IMembershipFunction>, ...> _mfs;

        // Построенная модель
        std::unique_ptr<OFLL::Abstract::IModel> _model;
        std::string _default_t_norm = "min";
        std::string _default_s_norm = "max";
        bool _built = false;
    };

} // namespace OFLL::Linear
```

### 6.2. Грамматика строкового правила (v0.1)

```
rule        := antecedent THEN consequent
antecedent  := term ( (AND|OR) term )*
consequent  := term
term        := IDENT IS STRING
IDENT       := name of a registered input/output variable
STRING      := double-quoted name of a membership function ("low", "high", ...)
AND         := "AND" | "and"
OR          := "OR" | "or"
THEN        := "THEN" | "then"
IS          := "IS" | "is"
```

**Whitespace игнорируется.** Кавычки обязательны. Регистр операторов не важен.
Имена переменных и MF — **регистрозависимые** (как в C++).

**Примеры:**
```
IF x1 IS "small" THEN y1 IS "low"
IF x1 IS "small" AND x2 IS "low" THEN y1 IS "low"
IF x1 IS "small" AND x2 IS "low" OR x1 IS "large" THEN y1 IS "high"
```

**Известное ограничение v0.1:** приоритет `AND` над `OR` **не** учитывается,
правило вычисляется слева направо в лексическом порядке. Для сложных правил
— использовать `add_rule_tuple` или дождаться v0.2.

### 6.3. Внутреннее устройство `build()`

```
SugenoEngine::build()
   ↓
для каждого правила:
   создать RulesBuilder
   для каждого терма в antecedent:
     RulesBuilder::operator()(var_ref)
     RulesBuilder::is(mf_ref)
     если следующий оператор — AND: RulesBuilder::AND(var_ref)
     если OR:                            RulesBuilder::OR(var_ref)
   RulesBuilder::then()
   для выходного терма:
     RulesBuilder::operator()(out_var)
     RulesBuilder::is(out_mf)
   SugenoBuilder->add_rule( RulesBuilder::end() )
   ↓
SugenoBuilder::build() → unique_ptr<IModel> → сохранить в _model
```

### 6.4. Поведение при ошибках

| Ситуация | Исключение | Где |
|----------|-----------|-----|
| Неизвестный type MF | `std::invalid_argument` | `add_membership_func` |
| Неверное число параметров | `std::invalid_argument` | `add_membership_func` |
| Незарегистрированное имя в правиле | `std::out_of_range` | `add_rule` |
| Синтаксическая ошибка правила | `std::invalid_argument` | `add_rule` |
| Конфликт: одно и то же имя дважды | `std::invalid_argument` | `add_input_var`/`add_output_var` |
| `add_*` после `build()` | `std::logic_error` | все `add_*` |
| `calculate()` до `build()` | `std::logic_error` | `calculate`/`set_input` |
| `set_input` неизвестного имени | `std::out_of_range` | `set_input`/`get_output` |

Исключения пробрасываются в Python как `ValueError` / `RuntimeError` / `KeyError`
(см. маппинг в `PyFLL.cpp`).

---

## 7. Дизайн Python API (PyFLL)

### 7.1. Имена

- Модуль: `PyFLL` (как ожидается по существующему `CMakeLists.txt`).
- Классы: `PyFLL.SugenoEngine`, `PyFLL.MamdaniEngine` (фаза 4).
- Исключения: пробрасываются стандартные (`ValueError`, `RuntimeError`).

### 7.2. Соответствие C++ ↔ Python

| C++ | Python | Заметка |
|-----|--------|---------|
| `add_input_var(name)` | `add_input_var(name)` | str |
| `add_output_var(name)` | `add_output_var(name)` | str |
| `add_membership_func(var, mf, type, params)` | `add_membership_func(var, mf, type, params)` | `type: str`, `params: list[float]` |
| `add_rule(text)` | `add_rule(text)` | str |
| `add_rule_tuple(...)` | `add_rule(antecedent, consequent)` с распаковкой dict | см. 7.3 |
| `set_default_t_norm(name)` | `set_default_t_norm(name)` | str |
| `set_default_s_norm(name)` | `set_default_s_norm(name)` | str |
| `build()` | `build()` | — |
| `set_input(name, value)` | `set_input(name, value)` | — |
| `get_output(name)` | `get_output(name)` | возвращает `float` |
| `calculate()` | `calculate()` | — |

### 7.3. Альтернативная форма правила через `dict`/`list`

Чтобы не заставлять пользователя конкатенировать строки, Python-биндинг
**дополнительно** принимает форму dict-list:

```python
engine.add_rule(
    inputs=[("x1", "small"), ("x2", "low")],
    outputs=[("y1", "high")],
    # опционально:
    # operator="AND" / "OR" / список операторов
)
```

Это обёртка над `add_rule_tuple` на стороне `PyFLL.cpp`, ничего не добавляет
к C++-фасаду.

### 7.4. Минимальный пример итогового Python-кода

```python
import PyFLL

engine = PyFLL.SugenoEngine()
engine.add_input_var("x1")
engine.add_membership_func("x1", "small", "triangular", [-1.0, 0.0, 10.0])
engine.add_membership_func("x1", "large", "triangular", [0.0, 10.0, 11.0])

engine.add_output_var("y1")
engine.add_membership_func("y1", "low",  "constant", [-15.0])
engine.add_membership_func("y1", "hold", "constant", [0.0])

engine.add_rule('IF x1 IS "small" THEN y1 IS "hold"')
engine.add_rule('IF x1 IS "large" THEN y1 IS "low"')

engine.build()
engine.set_input("x1", 2.5)
engine.calculate()
print(engine.get_output("y1"))
```

---

## 8. Парсер правил — отдельные заметки

### 8.1. Почему свой парсер, а не regex

- IF/THEN/AND/OR — токены, не символы.
- Имена переменных и MF — литералы в кавычках, нужна корректная обработка пробелов.
- На расширение (приоритеты, скобки, NOT) regex плохо масштабируется.

### 8.2. Структура

`src/Linear/rule_parser.h`:

```cpp
namespace OFLL::Linear::internal {

struct ParsedRule {
    struct Term { std::string var; std::string mf; };
    enum class Op { And, Or };
    std::vector<Term> antecedent;
    std::vector<Op>   antecedent_ops;   // size = antecedent.size() - 1
    std::vector<Term> consequent;
};

[[nodiscard]] ParsedRule parse_rule(std::string_view text);
// Бросает std::invalid_argument с позицией и описанием ошибки.

} // namespace
```

Реализация — простой рекурсивный спуск на ~150–200 строк, без зависимостей.
Тесты — табличные (`tests/OpenFLL/linear/rule_parser_test.cpp`).

### 8.3. Расширения (отложенные)

- v0.2: приоритеты AND/OR + скобки.
- v0.3: NOT (`IF x1 IS NOT "small"`).
- v0.4: вес правила (`IF ... THEN ... WEIGHT 0.7`).

---

## 9. Тестирование

### 9.1. Уровни

| Уровень | Что проверяет | Где |
|---------|---------------|-----|
| Unit (C++) | Каждый метод фасада, парсер, MF-factory | `tests/OpenFLL/linear/` |
| Integration (C++) | `SugenoEngine` end-to-end на сценарии из `compressor.cpp` | `tests/OpenFLL/linear/sugeno_engine_test.cpp` |
| Numerical baseline | Побайтовое совпадение с прямым `SugenoBuilder` | `tests/OpenFLL/linear/BASELINE.md` (зафиксированные числа) |
| Python smoke | `import PyFLL; ...` повторяет C++ сценарий | `tests/python/test_sugeno_smoke.py` |
| Cross-stack parity | Те же входные данные → те же выходные в C++ и Python | сравнение в CI |

### 9.2. Baseline-сценарий

Берётся **сценарий из `examples/compressor.cpp`** (Sugeno с 4 MF на `SM`, 3 MF на `dSM`,
4 MF на `dASV`, 12 правил). Прогоняется двумя способами:
1. **Baseline:** через прямой `SugenoBuilder` + `RulesBuilder` (как в `launch.cpp`).
2. **Through facade:** через `OFLL::Linear::SugenoEngine` со строковыми правилами.

Все выходы `dASV(t)` сравниваются с допуском `1e-9`. При совпадении —
числа фиксируются в `tests/OpenFLL/linear/BASELINE.md` и используются как
golden в обоих CI.

### 9.3. Числа для Python-теста

Тот же baseline-сценарий воспроизводится в Python, числа сравниваются с
теми же значениями. Это гарантирует, что Pybind11-обвязка не вносит
побочных эффектов (aliasing, копирование shared_ptr, лишние аллокации).

---

## 10. Сборка и CMake

### 10.1. Изменения в `src/OpenFLL/CMakeLists.txt`

Дополнить список `MFL_SOURCE` новыми файлами:

```cmake
linear/linear_engine_base.h
linear/mamdani_engine.h
linear/mf_factory.h
linear/rule_parser.h
linear/sugeno_engine.h
linear/rule_parser.cpp   # если выносим реализацию из header
```

`linear/` — header-only, отдельной библиотеки не нужно. Просто добавить
заголовки в список, они попадут в стат. библиотеку `OpenFLL`.

### 10.2. Изменения в `src/PyFLL/CMakeLists.txt`

```cmake
pybind11_add_module(PyFLL SHARED PyFLL.cpp)
target_link_libraries(PyFLL PRIVATE OpenFLL)
target_include_directories(PyFLL PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
set_target_properties(PyFLL PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CXX_STANDARD 20
)
```

Опционально: `install(TARGETS PyFLL ...)` и `find_package(Python3 ...)` — отложить до фазы 5.

### 10.3. Тесты Python

Запуск через `pytest` из `tests/python/`. В CI (`python-tests.yml`)
конфигурируется как:
1. собрать `PyFLL.pyd` через CMake,
2. скопировать в `tests/python/`,
3. `pytest tests/python/`.

---

## 11. Документация

- `docs/PYTHON_API_PLAN.md` — этот документ.
- `docs/python_api.md` (фаза 5) — пользовательская документация Python API.
- `src/Linear/*.h` — doxygen-комментарии, описывающие каждое решение.
- `src/PyFLL/PyFLL.cpp` — docstring'и через `R"pbdoc(...)pbdoc"` для автогенерации.
- `examples/python/` (фаза 5) — рабочие примеры: `simple_demo.py`, `compressor.py`.

---

## 12. Риски и контрмеры

| Риск | Вероятность | Импакт | Контрмера |
|------|-------------|--------|-----------|
| `Mamdani::calculate()` не будет доделан в обозримый срок | высокая | средняя | Фаза 4 — стаб; фазы 1–3 не зависят от Mamdani |
| Парсер правил упирается в неоднозначность грамматики | средняя | низкая | Версионировать формат: v0.1 (минимальный) → v0.2 (приоритеты) |
| Сборка wheel под Windows + MSVC ломается в CI | средняя | средняя | Использовать `cibuildwheel` или `scikit-build-core`; зафиксировать версию MSVC |
| `vendor/pybind11` submodule устарел | низкая | низкая | Обновить submodule до актуальной версии (2.13+) |
| Facade «протекает» — Python-код хочет доступ к shared_ptr/builder | низкая | средняя | Не добавлять «хаки» в facade; вместо этого расширять API |
| Compile-time t/s-нормы выдают разные результаты в Sugeno и в фасаде | низкая | высокая | Baseline-тест (9.2) ловит это с первого запуска |
| Производительность PyFLL ниже нативного C++ | средняя | низкая | Документировать, что facade — для удобства, а не для скорости; для скорости есть `AutoincrementMapper` (когда дозреет) |

---

## 13. Метрики приёмки

| Метрика | Целевое значение |
|---------|------------------|
| Покрытие C++ кода фасада gtest'ом | ≥ 90 % строк |
| Покрытие Python API smoke-тестами | 100 % публичных методов |
| Время одного `calculate()` (Sugeno, 12 правил) через Python | ≤ 2× от C++ baseline |
| Размер `PyFLL.pyd` (Debug, MSVC) | ≤ 3 МБ |
| Размер строкового парсера правил | ≤ 300 строк кода |
| Размер `PyFLL.cpp` | ≤ 200 строк кода |
| Время сборки инкрементальной (только `PyFLL.cpp`) | ≤ 5 сек |

---

## 14. Контрольный чек-лист «Definition of Done» для каждой фазы

**Фаза 1:**
- [ ] `SugenoEngine` собирается, тест из 9.2 зелёный, числа совпадают с `compressor.cpp`.
- [ ] Парсер правил покрыт табличными тестами (валидные + 5 категорий невалидных).
- [ ] `src/OpenFLL/CMakeLists.txt` дополнен, CI на windows-latest зелёный.
- [ ] В `tests/OpenFLL/linear/BASELINE.md` зафиксированы golden-числа.

**Фаза 2:**
- [ ] `PyFLL.pyd` собирается через `cmake --build build --config Release --target PyFLL`.
- [ ] `tests/python/test_sugeno_smoke.py` зелёный, числа совпадают с C++ baseline.
- [ ] Документирована команда сборки и путь к wheel/pyd.

**Фаза 3:**
- [ ] Поддержаны `trapezoidal`, `polynomial`, `OR`, выбор t/s-норм.
- [ ] Расширен Python smoke-тест.
- [ ] Никаких регрессий в фазах 1–2.

**Фаза 4:**
- [ ] `MamdaniEngine` компилируется; `calculate()` бросает `not_implemented` до тех пор, пока ядро не дозреет (явный warning в выводе).
- [ ] Обвязка в `PyFLL.cpp` повторяет структуру Sugeno.

**Фаза 5:**
- [ ] `docs/python_api.md` опубликован.
- [ ] `examples/python/simple_demo.py` запускается из README одной командой.
- [ ] (Опционально) wheel опубликован на тестовом PyPI.

---

## 15. Открытые вопросы для обсуждения

1. **Формат правил.** Устраивает ли версия v0.1 (без приоритетов, без скобок)? Если
   есть пилотные сценарии, где это неудобно — лучше сразу расширить.
2. **Имя Python-модуля.** Оставляем `PyFLL` (как в существующем `CMakeLists.txt`)?
   Альтернативы: `openfll`, `fll`, `pyfll` (lowercase). Текущее имя
   `PyFLL` чувствительно к регистру на Linux — стоит обсудить.
3. **Mamdani.** Действительно ли он нужен в Python-обвязке в первом релизе, или
   достаточно `SugenoEngine` (который работает), а Mamdani добавим позже?
4. **Совместимость с будущим `AutoincrementMapper`.** Сейчас его нет в ядре.
   Стоит ли заложить в `SugenoEngine` опцию `use_fast_mapper()` (no-op сейчас,
   задел на будущее)?
5. **Сериализация.** Ядро частично умеет `pack()`/`unpack()` для модели (см.
   `sugeno.h:46` — пока `throw "";`). Стоит ли на фазах 1–3 закладывать
   `save(path)` / `load(path)` в facade как заглушки, или отложить полностью
   до стабилизации сериализации?

---

**Согласование:** ждём ответа по открытым вопросам и ставим фазу 1 в работу.

---

## Статус фаз (обновлено 2026-08-28)

| Фаза | Статус | Артефакты |
|------|--------|-----------|
| **Фаза 0** — согласование | ✅ завершена | Этот документ |
| **Фаза 1** — Linear Facade (Sugeno) | ✅ завершена | `src/OpenFLL/linear/`, `tests/OpenFLL/linear/`, `docs/PHASE1_REPORT.md` |
| **Фаза 2** — Pybind11 биндинги | ✅ завершена | `src/PyFLL/`, `PyFLL.cp310-win_amd64.pyd`, `tests/python/test_sugeno_smoke.py` |
| **Фаза 3** — расширения (явные нормы, defaults, trapezoidal) | ✅ завершена | `src/OpenFLL/linear/norm_registry.h`, 49/49 C++ + 9/9 Python |
| Фаза 4 — `MamdaniLinearEngine` | ⏳ заблокирована (ядро) | ждёт реализации `Mamdani::calculate()` |
| Фаза 5 — документация, wheel | ⏳ не начата | |

## Что сделано в фазе 3

**Грамматика правил расширена до v0.2:**
```
IF <var1> IS "<mf1>" ((AND|OR) [<norm_name>] <var2> IS "<mf2>")* THEN <out_var> IS "<out_mf>"
```

`<norm_name>` опционально после `AND`/`OR`:
- t-нормы: `min`, `prod_and`, `bounded_diff`, `drastic_prod`, `einstein_prod`, `hamacher_prod`
- s-нормы: `max`, `algebraic_sum`, `bounded_sum`, `drastic_sum`, `einstein_sum`, `hamacher_sum`

Если норма не указана — используется default (задаётся через `set_default_t_norm` / `set_default_s_norm`).

**Новые C++ файлы:**
- `src/OpenFLL/linear/norm_registry.h` — `enum NormId`, `parse_norm_name()`, валидация (t/s-норма не подходит к AND/OR).

**Изменённые файлы:**
- `src/OpenFLL/linear/rule_parser.{h,cpp}` — `Op` стал struct с `kind + optional norm`, добавлен парсинг `[name]`.
- `src/OpenFLL/linear/sugeno_engine.{h,cpp}` — `set_default_t_norm` / `set_default_s_norm`, поля `m_default_t_norm/s_norm`, tag-dispatch через switch в `add_parsed_rule`.
- `src/OpenFLL/CMakeLists.txt` — добавлен `norm_registry.h` в `MFL_SOURCE`.
- `src/PyFLL/PyFLL.cpp` — обвязка `set_default_t_norm` / `set_default_s_norm`.

**Новые тесты (фаза 3):**
- `RuleParserTest`: `AndWithExplicitTNorm`, `OrWithExplicitSNorm`, `AndWithUnknownNorm`, `AndWithSnormRejected`, `OrWithTnormRejected`, `UnterminatedBracketRejected` (6 новых).
- `SugenoEngineTest`: `SetDefaultTNormAndSNorm`, `SetDefaultNormsAfterBuildRejected`, `ExplicitTNormInRule`, `ExplicitSNormInRule`, `TrapezoidalMf` (5 новых).
- `tests/python/test_sugeno_smoke.py`: `test_explicit_t_norm`, `test_explicit_s_norm`, `test_default_norms`, + 1 кейс в `test_errors` (4 новых).

**Итоги тестов после фазы 3:**
- C++ (gtest): **49/49 PASS** (12 mf_factory + 22 rule_parser + 15 sugeno_engine)
- Python (smoke): **9/9 PASS** (3 full + 1 OR + 1 explicit-t + 1 explicit-s + 1 default-norms + 3 errors)

**Не реализовано в фазе 3 (вне scope):**
- `polynomial` MF через фабрику (требует рефакторинга ядра — `PolynomialCls` живёт в `polynomial_impl`).
- `save(path) / load(path)` (сериализация ядра не готова — `pack()` бросает `""`).
- Кортежная форма правил через `add_rule(inputs, outputs)` — частично возможно (C++ уже есть `add_rule_tuple` идея), но Python обвязка не добавлена (отложено).

**End-to-end подтверждение:** Python (`PyFLL.SugenoEngine`) даёт **идентичные** числа с C++ `OFLL::Linear::SugenoEngine`, который в свою очередь даёт **идентичные** числа с прямым `SugenoBuilder` + `RulesBuilder` (с точностью `1e-9`).

См. детали:
- `docs/PHASE1_REPORT.md` — отчёт о фазе 1
- `tests/OpenFLL/linear/BASELINE.md` — golden-числа C++ baseline
- `tests/python/test_sugeno_smoke.py` — Python cross-check (6/6 PASS)
