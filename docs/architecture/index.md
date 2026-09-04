# Архитектура

## Слои

```
┌─────────────────────────────────────────────────────────────┐
│ Python (PyFLL)                                               │
│   engine.add_input_var("x1")                                │
│   engine.add_membership_func("x1", "low", "triangular", ...)│
│   engine.add_rule('IF x1 IS "low" AND x2 IS "fast" THEN ...') │
│   engine.build(); engine.calculate(); engine.get_output("y") │
└─────────────────────────┬───────────────────────────────────┘
                          │ pybind11
┌─────────────────────────▼───────────────────────────────────┐
│ Linear Facade (C++, header-only)                            │
│   OFLL::Linear::SugenoEngine                                │
│   - add_input_var / add_output_var / add_membership_func     │
│   - add_rule (строковый парсер)                              │
│   - set_default_t_norm / set_default_s_norm                  │
│   - build / set_input / calculate / get_output               │
│   - name→shared_ptr индексы (внутри)                         │
│   - NormId → tag-dispatch в switch (Min/Max/ProdAnd/...)     │
└─────────────────────────┬───────────────────────────────────┘
                          │ прямые вызовы
┌─────────────────────────▼───────────────────────────────────┐
│ Ядро OpenFLL (стат. библиотека)                              │
│   SugenoBuilder / MamdaniBuilder / RulesBuilder              │
│   Sugeno / Mamdani (наследники BaseModel)                    │
│   IDefuzzer (WtaverDefuzzer / WtsumDefuzzer для Sugeno)      │
│   Var / IMembershipFunction / t-norm / s-norm операторы      │
│   ResourcePool (internal storage)                           │
└─────────────────────────────────────────────────────────────┘
```

## Документы

- **[Linear Facade](linear-facade.md)** — почему плоский API, что он скрывает
- **[Python API план](python-api-plan.md)** — оригинальный дизайн (фазы 0-5)
- **[Отчёты по фазам](phase-reports.md)** — сводка реализованного и TODO
