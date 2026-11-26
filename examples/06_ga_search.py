"""Сравнение МНК (SugenoRegressor) и генетической оптимизации (SugenoGARegressor)
с подбором гиперпараметров через GridSearchCV и RandomizedSearchCV.

Демонстрирует три сценария:
1. Вырожденные стартовые функции принадлежности (1D, y = tanh(3x)):
   обе входные MF стартуют из одинакового гауссиана [0.0, 0.6].
   МНК с фиксированными MF физически может выучить только константу (rank = 1),
   тогда как GA сдвигает центры и ширины MF и снижает ошибку на порядки.
2. Грубые/смещённые стартовые MF на 2D нелинейной поверхности:
   y = 1.5 * tanh(3 * (x1 - 0.35)) + 1.0 * tanh(3 * (x2 + 0.30)) + eps.
   Стартовые MF заданы грубо около 0, а зоны перехода смещены. GA подбирает
   формы антецедентов и консеквенты, а GridSearchCV/RandomizedSearchCV выбирают
   t-норму и параметры мутации.
3. Реальные данные scikit-learn load_diabetes (bmi, bp) при разумных стартовых MF:
   контрольный пример границ применимости при небольшом бюджете GA (24x25).

Дополнительно можно сохранить графики MF до и после GA:
    python examples/06_ga_search.py --save-plots build/ga_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)

from openfll.sklearn import SugenoGARegressor, SugenoModelSpec, SugenoRegressor


def print_header(title: str) -> None:
    print("\n" + "=" * 84)
    print(f" {title}")
    print("=" * 84)


def format_mf_params(engine: Any) -> list[str]:
    terms = engine.to_dict().get("extra", {}).get("linear_api", {}).get("membership_functions", [])
    lines: list[str] = []
    for term in terms:
        params = ", ".join(f"{p:.3f}" for p in term["parameters"])
        lines.append(f"{term['variable']}.{term['name']} ({term['type']}): [{params}]")
    return lines


def save_mf_plots(engine: Any, out_dir: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    target_dir = out_dir / tag
    plots = engine.plot_mf(show=False, save_dir=target_dir)
    for plot in plots.values():
        plt.close(plot.figure)


def compare(
    *,
    title: str,
    description: str,
    X: np.ndarray,
    y: np.ndarray,
    spec: SugenoModelSpec,
    param_grid: Mapping[str, Sequence[Any]],
    param_distributions: Mapping[str, Sequence[Any]],
    mf_bounds: Mapping[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
    population_size: int = 24,
    generations: int = 25,
    seeds: tuple[int, ...] = (7, 19, 42),
    plot_prefix: str | None = None,
    plots_dir: Path | None = None,
) -> dict[str, Any]:
    print_header(title)
    print(description)
    print(f"Выборка: {len(X)} точек (75% train, 25% holdout test), 3-fold KFold CV, бюджет GA: {population_size}x{generations}.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=42).split(X_train))

    baseline = SugenoRegressor(spec).fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_r2 = float(r2_score(y_test, baseline_pred))
    print(
        f"\nБазовый МНК (фиксированные MF): "
        f"test RMSE = {baseline_rmse:.4f} | test R2 = {baseline_r2:.4f} | rank = {baseline.rank_}"
    )

    if plots_dir is not None and plot_prefix is not None:
        save_mf_plots(baseline.engine_, plots_dir, f"{plot_prefix}_before_ga")

    n_candidates = len(param_distributions) and 8
    common = {"cv": folds, "scoring": "neg_root_mean_squared_error", "error_score": "raise"}
    print(f"Поиск ({n_candidates} кандидатов x 3 фолда + refit на train):")
    print(f"{'Метод':<20} | {'seed':>4} | {'CV RMSE':>8} | {'test RMSE':>9} | {'test R2':>7} | {'Время':>6} | Лучшие параметры")
    print("-" * 105)

    results_by_method: dict[str, list[tuple[float, float, Any]]] = {
        "GridSearchCV": [],
        "RandomizedSearchCV": [],
    }

    for seed in seeds:
        estimator = SugenoGARegressor(
            spec,
            mf_bounds=mf_bounds,
            population_size=population_size,
            generations=generations,
            random_state=seed,
        )
        searches = [
            GridSearchCV(estimator, param_grid, **common),
            RandomizedSearchCV(
                estimator, param_distributions, n_iter=n_candidates, random_state=seed, **common
            ),
        ]
        for search in searches:
            method = type(search).__name__
            start = perf_counter()
            search.fit(X_train, y_train)
            seconds = perf_counter() - start
            pred = search.predict(X_test)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            test_r2 = float(r2_score(y_test, pred))
            results_by_method[method].append((test_rmse, test_r2, search.best_estimator_))
            print(
                f"{method:<20} | {seed:>4d} | {-search.best_score_:>8.4f} | "
                f"{test_rmse:>9.4f} | {test_r2:>7.4f} | {seconds:>5.2f}s | {search.best_params_}"
            )

    summary: dict[str, Any] = {
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
    }
    best_overall_rmse = float("inf")
    best_overall_estimator = None

    print("\nМедиана по seed:")
    for method, rows in results_by_method.items():
        rmses = [r[0] for r in rows]
        r2s = [r[1] for r in rows]
        med_rmse = float(np.median(rmses))
        med_r2 = float(np.median(r2s))
        summary[f"{method}_med_rmse"] = med_rmse
        summary[f"{method}_med_r2"] = med_r2
        print(f"  {method:<20}: median test RMSE = {med_rmse:.4f}, median test R2 = {med_r2:.4f}")
        for rmse, _, est in rows:
            if rmse < best_overall_rmse:
                best_overall_rmse = rmse
                best_overall_estimator = est

    if best_overall_estimator is not None:
        if plots_dir is not None and plot_prefix is not None:
            save_mf_plots(best_overall_estimator.engine_, plots_dir, f"{plot_prefix}_after_ga")
        print("\nПараметры MF (исходные -> после лучшего GA):")
        for before, after in zip(format_mf_params(baseline.engine_), format_mf_params(best_overall_estimator.engine_)):
            print(f"  {before}  -->  {after}")

    return summary


def gaussian_2d_spec(centers: tuple[tuple[float, float, float], ...]) -> SugenoModelSpec:
    return SugenoModelSpec(
        inputs={
            f"x{i}": [("low", "gaussian", [low, sigma]), ("high", "gaussian", [high, sigma])]
            for i, (low, high, sigma) in enumerate(centers, start=1)
        },
        rules=[
            f'IF x1 IS "{a}" AND x2 IS "{b}" THEN y IS "c_{a}_{b}"'
            for a in ("low", "high")
            for b in ("low", "high")
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение SugenoRegressor и SugenoGARegressor с GridSearchCV / RandomizedSearchCV.")
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Необязательный каталог для сохранения графиков MF до и после GA.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Сценарий 1: Вырожденные стартовые MF (одинаковые функции на входе)
    # ------------------------------------------------------------------
    X_1d = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y_tanh = np.tanh(3.0 * X_1d[:, 0])
    identical_spec = SugenoModelSpec(
        inputs={"x": [("low", "gaussian", [0.0, 0.6]), ("high", "gaussian", [0.0, 0.6])]},
        rules=['IF x IS "low" THEN y IS "left"', 'IF x IS "high" THEN y IS "right"'],
    )
    compare(
        title="СЦЕНАРИЙ 1. Вырожденные одинаковые стартовые MF (1D: y = tanh(3x))",
        description=(
            "Обе входные гауссианы стартуют из идентичной точки [0.0, 0.6]. "
            "Линейная матрица А(X) имеет ранг 1 — МНК не может отличить x < 0 от x > 0. "
            "GA разводит центры MF и находит крутизну переходов."
        ),
        X=X_1d,
        y=y_tanh,
        spec=identical_spec,
        mf_bounds={
            ("x", "low"): [(-1.0, 1.0), (0.35, 1.0)],
            ("x", "high"): [(-1.0, 1.0), (0.35, 1.0)],
        },
        param_grid={
            "crossover_rate": [0.75, 0.9],
            "mutation_rate": [0.15, 0.35],
            "mutation_sigma": [0.1, 0.2],
        },
        param_distributions={
            "crossover_rate": [0.65, 0.75, 0.85, 0.95],
            "mutation_rate": [0.1, 0.15, 0.25, 0.35, 0.45],
            "mutation_sigma": [0.05, 0.1, 0.15, 0.2, 0.3],
        },
        population_size=32,
        generations=40,
        plot_prefix="01_identical_tanh",
        plots_dir=args.save_plots,
    )

    # ------------------------------------------------------------------
    # Сценарий 2: Грубые / смещённые стартовые MF на 2D нелинейной поверхности
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    X_2d = rng.uniform(-1.0, 1.0, size=(180, 2))
    x1, x2 = X_2d.T
    y_2d = (
        1.5 * np.tanh(3.0 * (x1 - 0.35))
        + 1.0 * np.tanh(3.0 * (x2 + 0.30))
        + rng.normal(0.0, 0.05, len(X_2d))
    )
    bad_spec_2d = gaussian_2d_spec(((-0.1, 0.1, 0.5), (-0.1, 0.1, 0.5)))
    bounds_2d = {
        ("x1", "low"): [(-1.0, 1.0), (0.25, 1.2)],
        ("x1", "high"): [(-1.0, 1.0), (0.25, 1.2)],
        ("x2", "low"): [(-1.0, 1.0), (0.25, 1.2)],
        ("x2", "high"): [(-1.0, 1.0), (0.25, 1.2)],
    }
    grid_2d = {
        "default_t_norm": ["min", "prod_and"],
        "mutation_rate": [0.15, 0.35],
        "mutation_sigma": [0.1, 0.2],
    }
    dist_2d = {
        "default_t_norm": ["min", "prod_and"],
        "mutation_rate": [0.1, 0.15, 0.25, 0.35, 0.45],
        "mutation_sigma": [0.05, 0.1, 0.15, 0.2, 0.3],
    }
    compare(
        title="СЦЕНАРИЙ 2. Грубые стартовые MF на 2D поверхности со смещёнными переходами",
        description=(
            "Зоны перехода поверхности сдвинуты в (x1=0.35, x2=-0.30), но начальные MF заданы "
            "грубо около нуля ([-0.1, 0.5] и [0.1, 0.5]). Подбираем также default_t_norm (min vs prod_and)."
        ),
        X=X_2d,
        y=y_2d,
        spec=bad_spec_2d,
        mf_bounds=bounds_2d,
        param_grid=grid_2d,
        param_distributions=dist_2d,
        population_size=36,
        generations=50,
        plot_prefix="02_shifted_2d",
        plots_dir=args.save_plots,
    )

    # ------------------------------------------------------------------
    # Сценарий 3: Контрольный пример на реальных данных (удачные экспертные MF)
    # ------------------------------------------------------------------
    diabetes = load_diabetes(scaled=False)
    columns = [diabetes.feature_names.index(name) for name in ("bmi", "bp")]
    compare(
        title="СЦЕНАРИЙ 3. Реальные данные load_diabetes (bmi, bp) при разумных стартовых MF",
        description=(
            "Начальные MF уже разумно расставлены по диапазонам признаков ((21, 32, 6) и (80, 110, 15)). "
            "При малом бюджете GA (24x25) аналитический МНК остаётся сильным базовым методом."
        ),
        X=diabetes.data[:, columns],
        y=diabetes.target,
        spec=gaussian_2d_spec(((21, 32, 6), (80, 110, 15))),
        param_grid=grid_2d,
        param_distributions=dist_2d,
        population_size=24,
        generations=25,
    )

    print_header("ИТОГ")
    print("1. Когда стартовые MF вырождены или смещены, GA (SugenoGARegressor) кратно превосходит МНК.")
    print("2. GridSearchCV и RandomizedSearchCV подбирают t-норму и параметры мутации без утечки между фолдами.")
    print("3. Когда экспертные MF уже удачно заданы, SugenoRegressor (МНК) находит оптимум консеквентов мгновенно.")


if __name__ == "__main__":
    main()
