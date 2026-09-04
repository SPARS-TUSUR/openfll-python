# Туториал 03: Mamdani (планируется)

!!! warning "В разработке"
    `MamdaniEngine` появится в **фазе 4** плана (`docs/architecture/python-api-plan.md`).
    Ядро: `src/OpenFLL/models/mamdani/mamdani.cpp::Mamdani::calculate()` сейчас закомментирован,
    фасад ожидает его реализации.

## Что будет

- **MamdaniEngine** — обёртка над Mamdani-моделью OpenFLL
- **Integrator** — выбор метода дефаззификации (`RectangleMidpointIntegrator`)
- **Совместимый API** с `SugenoEngine` (те же `add_input_var`, `add_membership_func`, `add_rule`)

## Примерный код (когда будет готов)

```python
import PyFLL

e = PyFLL.MamdaniEngine()  # ← появится в фазе 4
e.add_input_var("x1")
e.add_membership_func("x1", "low",  "triangular", [-10, 0, 10])
e.add_membership_func("x1", "high", "triangular", [0, 10, 20])
# ... тот же API что и у SugenoEngine
e.build()
```

## Следить за статусом

- [Python API план](../architecture/python-api-plan.md) — фаза 4
- [PHASE1_REPORT.md](../architecture/phase-reports.md) — статус реализации

## Помочь с реализацией

Если у вас есть опыт с дефаззификацией через численное интегрирование — присоединяйтесь.
Текущая точка входа: `src/OpenFLL/models/mamdani/mamdani.cpp::Mamdani::calculate()`.
