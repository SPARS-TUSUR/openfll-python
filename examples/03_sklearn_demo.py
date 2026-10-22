"""Примеры практических вычислений на базе OpenFLL и SugenoRegressor.

Демонстрирует:
1. Точное восстановление аналитической зависимости (машинная точность ~1e-14).
2. Аппроксимацию нелинейной поверхности f(x1, x2) = sin(x1) * cos(x2).
3. Обучение на зашумлённых данных с подбором Ridge-регуляризации через GridSearchCV.
4. Проверку скорости нативного батча на 50 000 точек и надёжность сериализации (pickle).
"""
import time
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from openfll.sklearn import SugenoModelSpec, SugenoRegressor


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def demo_1_exact_parameter_recovery():
    print_header("ПРИМЕР 1. Точное восстановление линейной комбинации правил (МНК)")

    # 2 входных переменных, каждая разбита на 'low' и 'high'
    spec = SugenoModelSpec(
        inputs={
            "x1": [("x1_low", "triangular", [0.0, 0.0, 10.0]), ("x1_high", "triangular", [0.0, 10.0, 10.0])],
            "x2": [("x2_low", "triangular", [0.0, 0.0, 10.0]), ("x2_high", "triangular", [0.0, 10.0, 10.0])],
        },
        rules=[
            'IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c_ll"',
            'IF x1 IS "x1_low" AND x2 IS "x2_high" THEN y IS "c_lh"',
            'IF x1 IS "x1_high" AND x2 IS "x2_low" THEN y IS "c_hl"',
            'IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c_hh"',
        ],
        output_name="y",
    )

    # Задаём известные истинные константы в вершинах сетки:
    # (0, 0) -> 10.0, (0, 10) -> 25.0, (10, 0) -> -5.0, (10, 10) -> 40.0
    true_c = {"c_ll": 10.0, "c_lh": 25.0, "c_hl": -5.0, "c_hh": 40.0}

    # Генерируем обучающую выборку
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.0, 10.0, size=(100, 2))

    # Считаем точные значения через модель с истинными весами
    reg_true = SugenoRegressor(model_spec=spec)
    # Ручной инференс матрицы базиса
    basis_engine = reg_true._build_engine(spec, true_c, list(true_c.keys()))
    y_train = basis_engine.predict_batch(X_train, input_names=["x1", "x2"])

    # Теперь обучаем SugenoRegressor с нуля методом МНК
    reg_fit = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)
    reg_fit.fit(X_train, y_train)

    print(f"Число обучающих сэмплов: {len(X_train)}")
    print(f"Ранг матрицы базиса (rank_): {reg_fit.rank_} / {len(spec.rules)}")
    print("\nСравнение истинных и восстановленных констант заключений:")
    print(f"{'Консеквент':<12} | {'Истинное значение':<18} | {'Восстановленное (МНК)':<22} | {'Ошибка':<12}")
    print("-" * 72)
    for name, c_est in zip(reg_fit.consequents_, reg_fit.coef_):
        c_true = true_c[name]
        err = abs(c_est - c_true)
        print(f"{name:<12} | {c_true:<18.4f} | {c_est:<22.6f} | {err:<12.2e}")

    X_test = rng.uniform(0.0, 10.0, size=(50, 2))
    y_test_true = basis_engine.predict_batch(X_test, input_names=["x1", "x2"])
    y_test_pred = reg_fit.predict(X_test)
    max_err = float(np.max(np.abs(y_test_pred - y_test_true)))
    print(f"\nМаксимальная ошибка на независимой тестовой выборке: {max_err:.2e}")
    assert max_err < 1e-12, "Восстановление должно быть точным до машинной погрешности"


def demo_2_nonlinear_surface_approximation():
    print_header("ПРИМЕР 2. Аппроксимация нелинейной поверхности: f(x1, x2) = sin(x1) * cos(x2)")

    # Нелинейная целевая функция
    def target_func(x1, x2):
        return np.sin(x1) * np.cos(x2)

    # Сетка разбиения области [0, pi] x [0, pi] на 3 терма по каждой координате (всего 9 правил)
    pi = np.pi
    spec = SugenoModelSpec(
        inputs={
            "x1": [
                ("x1_s", "triangular", [-pi / 2, 0.0, pi / 2]),
                ("x1_m", "triangular", [0.0, pi / 2, pi]),
                ("x1_l", "triangular", [pi / 2, pi, 3 * pi / 2]),
            ],
            "x2": [
                ("x2_s", "triangular", [-pi / 2, 0.0, pi / 2]),
                ("x2_m", "triangular", [0.0, pi / 2, pi]),
                ("x2_l", "triangular", [pi / 2, pi, 3 * pi / 2]),
            ],
        },
        rules=[
            f'IF x1 IS "x1_{t1}" AND x2 IS "x2_{t2}" THEN y IS "c_{t1}_{t2}"'
            for t1 in ["s", "m", "l"]
            for t2 in ["s", "m", "l"]
        ],
        output_name="y",
    )

    # 400 случайных обучающих точек
    rng = np.random.default_rng(123)
    X_train = rng.uniform(0.0, pi, size=(400, 2))
    y_train = target_func(X_train[:, 0], X_train[:, 1])

    # Обучение модели
    reg = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)
    reg.fit(X_train, y_train)

    # Тестирование на регулярной сетке 100 точек
    X_test = rng.uniform(0.0, pi, size=(100, 2))
    y_test = target_func(X_test[:, 0], X_test[:, 1])
    y_pred = reg.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Число нечётких правил: {len(spec.rules)}")
    print(f"Метрики качества на отложенной выборке:")
    print(f"  Коэффициент детерминации R^2: {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"  Среднеквадратичная ошибка MSE: {mse:.6f}")
    print(f"  Средняя абсолютная ошибка MAE: {mae:.6f}")

    print("\nПримеры 5 контрольных точек (X1, X2 -> Y_true vs Y_pred):")
    print(f"{'x1':<8} | {'x2':<8} | {'Y_true':<10} | {'Y_pred':<10} | {'Погрешность':<12}")
    print("-" * 56)
    for i in range(5):
        diff = abs(y_test[i] - y_pred[i])
        print(f"{X_test[i, 0]:<8.3f} | {X_test[i, 1]:<8.3f} | {y_test[i]:<10.4f} | {y_pred[i]:<10.4f} | {diff:<12.4f}")


def demo_3_noisy_data_ridge_grid_search():
    print_header("ПРИМЕР 3. Обучение на зашумлённых данных и подбор Ridge-регуляризации через GridSearchCV")

    spec = SugenoModelSpec(
        inputs={
            "x1": [("x1_low", "triangular", [-5, 0, 5]), ("x1_high", "triangular", [0, 5, 10])],
            "x2": [("x2_low", "triangular", [-5, 0, 5]), ("x2_high", "triangular", [0, 5, 10])],
        },
        rules=[
            'IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"',
            'IF x1 IS "x1_low" AND x2 IS "x2_high" THEN y IS "c2"',
            'IF x1 IS "x1_high" AND x2 IS "x2_low" THEN y IS "c3"',
            'IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c4"',
        ],
        output_name="y",
    )

    rng = np.random.default_rng(2026)
    X = rng.uniform(0.0, 5.0, size=(120, 2))
    # Истинный сигнал + случайный шум N(0, 0.4^2)
    y_clean = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 3.0
    noise = rng.normal(0.0, 0.4, size=len(X))
    y_noisy = y_clean + noise

    # Условная сетка гиперпараметров (строгий solver-контракт)
    param_grid = [
        {"solver": ["lstsq"], "alpha": [0.0]},
        {"solver": ["ridge"], "alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]},
    ]

    base_reg = SugenoRegressor(model_spec=spec)
    grid = GridSearchCV(base_reg, param_grid, cv=5, scoring="neg_mean_squared_error", error_score="raise")
    grid.fit(X, y_noisy)

    best_p = grid.best_params_
    best_score = -grid.best_score_
    print(f"Размер выборки с шумом: {len(X)} точек (сигма шума = 0.4)")
    print(f"Число исследованных комбинаций гиперпараметров: {len(grid.cv_results_['params'])}")
    print(f"Лучший найденный солвер и alpha: {best_p}")
    print(f"Лучший средний MSE на 5-фолдовой кросс-валидации: {best_score:.4f}")

    best_reg = grid.best_estimator_
    print(f"Подобранные константы c_r лучшей модели: {np.round(best_reg.coef_, 4)}")


def demo_4_performance_and_serialization():
    print_header("ПРИМЕР 4. Производительность C++ батча и сериализация модели (pickle)")

    spec = SugenoModelSpec(
        inputs={
            "x1": [("x1_low", "triangular", [-10, 0, 10]), ("x1_high", "triangular", [0, 10, 20])],
            "x2": [("x2_low", "triangular", [-10, 0, 10]), ("x2_high", "triangular", [0, 10, 20])],
        },
        rules=[
            'IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "c1"',
            'IF x1 IS "x1_high" AND x2 IS "x2_high" THEN y IS "c2"',
        ],
        output_name="y",
    )

    reg = SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)
    X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
    y_train = np.array([2.0, 8.0])
    reg.fit(X_train, y_train)

    # Генерируем 50 000 точек для теста скорости
    rng = np.random.default_rng(999)
    X_large = rng.uniform(0.0, 10.0, size=(50_000, 2))

    t0 = time.perf_counter()
    y_pred_original = reg.predict(X_large)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    throughput = len(X_large) / (t1 - t0) / 1e6

    print(f"Инференс на 50 000 точках через C++ ядро:")
    print(f"  Время расчёта: {elapsed_ms:.2f} мс")
    print(f"  Пропускная способность: {throughput:.2f} млн сэмплов/сек")

    # Сериализация модели в байты pickle
    data = pickle.dumps(reg)
    print(f"\nСериализация через pickle.dumps():")
    print(f"  Размер сериализованного состояния: {len(data)} байт")

    # Десериализация
    restored_reg = pickle.loads(data)
    y_pred_restored = restored_reg.predict(X_large)

    max_diff = float(np.max(np.abs(y_pred_original - y_pred_restored)))
    print(f"  Максимальное расхождение предсказаний до и после pickle: {max_diff}")
    assert max_diff == 0.0, "Предсказания должны совпадать абсолютно точно"
    print("  Проверка эквивалентности: УСПЕШНО (расхождение = 0.0)")


if __name__ == "__main__":
    demo_1_exact_parameter_recovery()
    demo_2_nonlinear_surface_approximation()
    demo_3_noisy_data_ridge_grid_search()
    demo_4_performance_and_serialization()
    print("\n" + "=" * 78)
    print(" ВСЕ 4 ДЕМОНСТРАЦИОННЫХ СЦЕНАРИЯ УСПЕШНО ЗАВЕРШЕНЫ")
    print("=" * 78 + "\n")
