"""Monte-Carlo инференс через Python-хелпер openfll.monte_carlo.

Полезно для регрессионного тестирования модели: после изменений в C++
ядре прогон 1000 случайных точек и проверка, что выход всегда конечный
и попадает в разумные пределы.
"""
import openfll
from openfll import check_output_finite, monte_carlo

# Собираем модель: 2 входа (x1, x2) → 1 выход (y), одно правило.
e = openfll.SugenoEngine()
e.add_input_var("x1")
e.add_input_var("x2")
e.add_output_var("y")
e.add_membership_func("x1", "low", "triangular", [-10, 0, 10])
e.add_membership_func("x2", "high", "triangular", [0, 10, 20])
e.add_membership_func("y", "c", "constant", [1.0])
e.add_rule('IF x1 IS "low" AND x2 IS "high" THEN y IS "c"')
e.build()

# 1000 случайных точек в заданных диапазонах.
rows = monte_carlo(
    e,
    var_inputs=[("x1", (-10, 10)), ("x2", (0, 20))],
    var_outputs=["y"],
    n_samples=1000,
    seed=42,
)

# Проверяем, что все выходы — конечные float.
stats = check_output_finite(
    e,
    var_inputs=[("x1", (-10, 10)), ("x2", (0, 20))],
    var_outputs=["y"],
    n_samples=1000,
    seed=42,
)
print(f"Monte-Carlo: {len(rows)} samples, {stats['bad']} bad (NaN/Inf)")
print(f"First sample: {rows[0]}")
print(f"Min y: {min(r['y'] for r in rows):.4f}")
print(f"Max y: {max(r['y'] for r in rows):.4f}")
