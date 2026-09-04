# Туториалы

Пошаговые примеры использования PyFLL.

- **[01 — Первый движок](01-first-engine.md)**: минимальный Sugeno, 2 входа, 1 выход, 2 правила
- **[02 — Управление компрессором](02-fuzzy-arithmetic.md)**: реальный сценарий с динамической моделью, C++ ↔ Python parity 3.55e-15
- **03 — Mamdani**: появится, когда `Mamdani::calculate()` будет реализован в ядре

Все примеры рабочие — взяты из `tests/python/`, где они уже валидируются
как тесты.

## Запуск

```powershell
# После сборки PyFLL.pyd
& C:/Python/Python310_64/python.exe tests/python/<example>.py
```
