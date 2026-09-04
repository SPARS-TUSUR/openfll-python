# Туториал 02: управление компрессором

Реальный сценарий: Sugeno-контроллер давления в компрессоре. Динамическая модель
компрессора + нечёткий регулятор. C++ и Python выдают **bit-exact** результаты
(max abs diff = 3.55e-15 на 300 шагах).

Полный код — в [`tests/python/compressor_parity.py`](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/tests/python/compressor_parity.py) и [`examples/compressor_run.cpp`](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/examples/compressor_run.cpp).

## Сценарий

Давление SM в компрессоре регулируется через клапан. Sugeno-FIS на основе
`SM` (текущее давление) и `dSM` (скорость изменения) выдаёт `dASV` —
управляющее воздействие на клапан.

```
            +---------+     +--------+     +---------+
  входной  | Sugeno  | dASV| Compre-| dSM |   SM     |
  сигнал → |   FIS  +--→+ | ssorMo |--+ |(объект) |
            +---------+     |  del   |   +---------+
                             +--------+        ↑
                                                |
                              +-------------+  |
                              | возмущение  |——+
                              +-------------+
```

## Sugeno-контроллер

12 правил: для каждой пары `(SM_mf, dSM_mf)` — одно правило AND.
Sugeno 0-го порядка: выход правила = константа (EmergencyOpen / FastOpen / Hold / SmoothClose).

### Функции принадлежности

| Переменная | MF | Параметры |
|------------|-----|-----------|
| SM (VeryLow) | triangular | (-10, 0, 10) |
| SM (Low) | triangular | (5, 10, 15) |
| SM (Normal) | triangular | (10, 15, 25) |
| SM (High) | triangular | (20, 30, 40) |
| dSM (FastDec) | gaussian | center=-5, sigma=2 |
| dSM (Stable) | gaussian | center=0, sigma=2 |
| dSM (FastInc) | gaussian | center=5, sigma=2 |
| dASV (EmergencyOpen) | constant | 50 |
| dASV (FastOpen) | constant | 20 |
| dASV (Hold) | constant | 0 |
| dASV (SmoothClose) | constant | -15 |

### Правила (12 штук, логика «реактивного управления»)

```python
# Предиктивное управление: при падении давления — открывать клапан
e.add_rule('IF x1 IS "sm_vlow" AND x2 IS "dsm_dec"  THEN y IS "emergency_open"')
e.add_rule('IF x1 IS "sm_low"  AND x2 IS "dsm_dec"  THEN y IS "emergency_open"')
e.add_rule('IF x1 IS "sm_norm" AND x2 IS "dsm_dec"  THEN y IS "fast_open"')
e.add_rule('IF x1 IS "sm_high" AND x2 IS "dsm_dec"  THEN y IS "hold"')   # нормальное падение при высоком SM

# Стабильное состояние
e.add_rule('IF x1 IS "sm_vlow" AND x2 IS "dsm_stab" THEN y IS "fast_open"')
e.add_rule('IF x1 IS "sm_low"  AND x2 IS "dsm_stab" THEN y IS "fast_open"')
e.add_rule('IF x1 IS "sm_norm" AND x2 IS "dsm_stab" THEN y IS "hold"')
e.add_rule('IF x1 IS "sm_high" AND x2 IS "dsm_stab" THEN y IS "smooth_close"')

# Восстановление после удара
e.add_rule('IF x1 IS "sm_vlow" AND x2 IS "dsm_inc"  THEN y IS "hold"')
e.add_rule('IF x1 IS "sm_low"  AND x2 IS "dsm_inc"  THEN y IS "hold"')
e.add_rule('IF x1 IS "sm_norm" AND x2 IS "dsm_inc"  THEN y IS "smooth_close"')
e.add_rule('IF x1 IS "sm_high" AND x2 IS "dsm_inc"  THEN y IS "smooth_close"')
```

## Динамическая модель (ODE)

```python
class CompressorModel:
    def __init__(self, sm=25.0, dsm=0.0, valve=0.0, tau=0.5):
        self.SM = sm
        self.dSM = dsm
        self.valve_pos = valve
        self.tau = tau

    def step(self, dASV, disturbance, dt):
        self.valve_pos = max(0.0, min(100.0, self.valve_pos + dASV * dt))
        accel = (2.0 * self.valve_pos - self.dSM - disturbance) / self.tau
        self.dSM += accel * dt
        self.SM += self.dSM * dt
        return self.SM
```

Физика: клапан открывается пропорционально `dASV · dt`, ускорение зависит от
`valve_pos` (вход) и `dSM` (собственная скорость), затухание через `tau = 0.5`.

## Цикл симуляции

```python
T_END = 15.0
DT = 0.05
N_STEPS = int(T_END / DT)   # 300 шагов
comp = CompressorModel()

for i in range(N_STEPS):
    t = i * DT
    dist = 50.0 if (t > 2.0 and t < 4.0) else 10.0  # возмущение в [2,4]
    in_sm  = max(0.0, min(30.0, comp.SM))
    in_dsm = max(-10.0, min(10.0, comp.dSM))
    engine.set_input("x1", in_sm)
    engine.set_input("x2", in_dsm)
    engine.calculate()
    dASV = engine.get_output("y")
    comp.step(dASV, dist, DT)
```

## Результат parity-теста

| Variable | MSE | MAE | Max abs diff |
|----------|-----|-----|--------------|
| `sm` | 0 | 0 | 0 |
| `valve_pos` | 0 | 0 | 0 |
| `in_sm`, `in_dsm` | 0 | 0 | 0 |
| `dASV` | 9.5e-32 | 3.1e-17 | **3.55e-15** |

`dASV` показывает крошечное расхождение (16 ULP) из-за некоммутативности
IEEE 754 при суммировании весов — норма для cross-language численного parity.

## График

`tests/python/compressor_parity.py` строит 6-панельный PNG:

- `sm(t)` — давление
- `valve_pos(t)` — позиция клапана
- `in_sm(t)`, `in_dsm(t)` — нормализованные входы
- `dASV(t)` — выход Sugeno
- `C++sm - PySm` — разность

Видно, как `dist = 50` в окне `[2, 4]` вызывает резкое падение SM, а Sugeno-контроллер
реагирует, увеличивая `dASV` (EmergencyOpen), что возвращает SM к норме.

## Что дальше

- Настоящий C++ туториал с тем же сценарием — [`examples/compressor_run.cpp`](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/examples/compressor_run.cpp)
- Сравнение Python и C++ результатов на графике — [`compressor_parity.png`](https://github.com/SPARS-TUSUR/fuzzy-logic-library/blob/main/tests/python/data/compressor_parity.png)
- API reference — [SugenoEngine](../api/sugeno-engine.md)
