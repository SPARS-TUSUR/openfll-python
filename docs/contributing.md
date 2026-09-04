# Contributing

Как добавить новую фичу в OpenFLL (MF, t/s-норму, метод SugenoEngine, и т.д.).

## Добавить новую функцию принадлежности

1. **Реализуйте класс MF** в `src/OpenFLL/membership_functions/<name>.{h,cpp}`,
   наследуя от `Abstract::IMembershipFunction`. Должен быть метод `find_belonging(double) → double`.

2. **Добавьте в реестр** `src/OpenFLL/linear/mf_factory.cpp`:
   ```cpp
   if (type_lc == "my_mf") {
       expect_params(params, N, type);
       return std::make_shared<OFLL::MembershipFunc::MyMF>(...);
   }
   ```

3. **Добавьте type literal** в `src/PyFLL/PyFLL.pyi`:
   ```python
   MfType = Literal["constant", "triangular", "trapezoidal", "gaussian", "polynomial", "my_mf"]
   ```

4. **Тесты** в `tests/OpenFLL/linear/mf_factory_test.cpp`:
   ```cpp
   TEST(MfFactoryTest, MyMfRoundtrip) {
       auto mf = make_membership_function("my_mf", {1.0, 2.0});
       EXPECT_NEAR(mf->find_belonging(0.5), ..., 1e-9);
       EXPECT_THROW(make_membership_function("my_mf", {}), std::invalid_argument);
   }
   ```

5. **CI**: запустите `cmake --build build --target unit_tests`,
   `tests/python/test_stub_types_good.py` (mypy) — должны пройти.

## Добавить новую t/s-норму

1. **Реализуйте структуру** в `src/OpenFLL/rules/core/operators/{t,s}_norm/<name>.h`:
   ```cpp
   namespace OFLL::Rules::Core::Operators::TN {
       struct MyNorm {
           double constexpr operator()(const double a, const double b) const {
               return /* ... */;
           }
       };
   }
   ```

2. **Добавьте в NormId** `src/OpenFLL/linear/norm_registry.h`:
   ```cpp
   enum class NormId {
       Min, ProdAnd, ..., MyNorm,
       Max, ..., MySNorm,
   };
   ```
   И в `parse_norm_name` и `norm_id_to_name` — case для новой нормы.

3. **Tag dispatch** в `src/OpenFLL/linear/sugeno_engine.cpp` — switch по `NormId`:
   ```cpp
   case NormId::MyNorm: rb.t_norm<TN::MyNorm>(v).is(m); break;
   ```

4. **Type stub** в `src/PyFLL/PyFLL.pyi`:
   ```python
   TNorms = Literal["min", ..., "my_norm"]
   ```

5. **Тесты** в `tests/OpenFLL/linear/rule_parser_test.cpp` и `sugeno_engine_test.cpp`.

## Добавить новый метод SugenoEngine

1. **Объявите в `src/OpenFLL/linear/sugeno_engine.h`** (public-секция).
2. **Реализуйте в `src/OpenFLL/linear/sugeno_engine.cpp`**.
3. **Зарегистрируйте в `src/PyFLL/PyFLL.cpp`** через `.def(...)` с docstring.
4. **Добавьте в `src/PyFLL/PyFLL.pyi`** (public-метод) с docstring.
5. **Тесты** в `tests/OpenFLL/linear/sugeno_engine_test.cpp`.
6. **Документация**: если метод публичный — добавьте секцию в `docs/api/sugeno-engine.md`
   (через `:::` директиву с новым методом в `members`).

## Процесс code review

1. Убедитесь, что:
   - C++ тесты зелёные: `cmake --build build --target unit_tests && build/tests/unit_tests.exe`
   - Python тесты зелёные: `python tests/python/test_sugeno_smoke.py`
   - mypy strict: `mypy --config-file tests/python/mypy.ini tests/python/test_stub_types_good.py`
   - Parity сохранён: `python tests/python/compressor_parity.py` (max diff < 1e-9)
2. Обновите `docs/architecture/phase-reports.md` если это меняет публичный API.
3. Если меняли `R"doc()"` в C++ — регенерируйте `PyFLL.pyi` (см. [API Reference](api/index.md)).

## Стиль кода

- **C++**: Google C++ Style Guide, где возможно. Namespace-ы `OFLL::*`. `using` — минимально.
- **Python**: PEP 8, type hints обязательны для публичного API. Docstring'и — Google style.
- **Commits**: осмысленные сообщения. Один PR — одна фича.
- **Документация**: каждое изменение публичного API → обновить `.pyi` **в том же коммите**.

## Контакты

- GitHub: [SPARS-TUSUR/fuzzy-logic-library](https://github.com/SPARS-TUSUR/fuzzy-logic-library)
- Issues: создавайте в GitHub
- Документация: [spars-tusur.github.io/fuzzy-logic-library](https://spars-tusur.github.io/fuzzy-logic-library/)
