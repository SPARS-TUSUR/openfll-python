"""Базовый пример openfll: 1 вход → 1 выход, одно правило.

Запуск:
    python examples/01_basic.py
Ожидаемый вывод: y = 0.5 (для x = 5.0 в MF triangular [-10, 0, 10]).
"""
import openfll

e = openfll.SugenoEngine()
e.add_input_var("x1")
e.add_input_var("x2")  # Sugeno требует ≥2 переменных в правиле
e.add_output_var("y")
e.add_membership_func("x1", "low", "triangular", [-10, 0, 10])
e.add_membership_func("x2", "any", "triangular", [-100, 0, 100])  # всегда активно
e.add_membership_func("y", "out", "constant", [1.0])

# Текстовая форма
e.add_rule('IF x1 IS "low" AND x2 IS "any" THEN y IS "out"')

e.build()

e.set_input("x1", 5.0)
e.set_input("x2", 0.0)
e.calculate()
print(f"y = {e.get_output('y')}")  # 0.5 (5.0 в центре triangular → 1.0)
