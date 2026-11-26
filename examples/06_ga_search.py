"""Сравнить GridSearchCV и RandomizedSearchCV поверх нативного обучения GA."""
from time import perf_counter

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)

from openfll.sklearn import SugenoGARegressor, SugenoModelSpec, SugenoRegressor


def compare(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    spec: SugenoModelSpec,
    *,
    mf_bounds: dict[tuple[str, str], list[tuple[float, float]]] | None = None,
    population_size: int = 24,
    generations: int = 25,
) -> None:
    print(f"\n{dataset} ({len(X)} samples)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    folds = list(KFold(n_splits=3, shuffle=True, random_state=42).split(X_train))

    baseline = SugenoRegressor(spec).fit(X_train, y_train)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline.predict(X_test)))
    print(f"Fixed MF (lstsq), test RMSE: {baseline_rmse:.4f}")

    grid = {
        "default_t_norm": ["min", "prod_and"],
        "mutation_rate": [0.15, 0.35],
        "mutation_sigma": [0.1, 0.2],
    }
    distributions = {
        "default_t_norm": grid["default_t_norm"],
        "mutation_rate": [0.1, 0.15, 0.25, 0.35, 0.45],
        "mutation_sigma": [0.05, 0.1, 0.15, 0.2, 0.3],
    }
    common = {"cv": folds, "scoring": "neg_root_mean_squared_error", "error_score": "raise"}
    print(
        f"Each search: 8 candidates x 3 folds, then refit. "
        f"GA: {population_size} individuals, {generations} generations."
    )
    for seed in (7, 19, 42):
        estimator = SugenoGARegressor(
            spec,
            mf_bounds=mf_bounds,
            population_size=population_size,
            generations=generations,
            random_state=seed,
        )
        searches = [
            GridSearchCV(estimator, grid, **common),
            RandomizedSearchCV(estimator, distributions, n_iter=8, random_state=seed, **common),
        ]
        for search in searches:
            start = perf_counter()
            search.fit(X_train, y_train)
            seconds = perf_counter() - start
            test_rmse = np.sqrt(mean_squared_error(y_test, search.predict(X_test)))
            print(
                f"{type(search).__name__:20s} seed={seed:2d} "
                f"CV RMSE={-search.best_score_:.4f} test RMSE={test_rmse:.4f} "
                f"fit={seconds:.2f}s {search.best_params_}"
            )


def model_spec(centers: tuple[tuple[float, float, float], ...]) -> SugenoModelSpec:
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
    rng = np.random.default_rng(42)

    X_1d = np.linspace(-1.0, 1.0, 100).reshape(-1, 1)
    y_tanh = np.tanh(3.0 * X_1d[:, 0])
    identical_spec = SugenoModelSpec(
        inputs={"x": [("low", "gaussian", [0.0, 0.6]), ("high", "gaussian", [0.0, 0.6])]},
        rules=['IF x IS "low" THEN y IS "left"', 'IF x IS "high" THEN y IS "right"'],
    )
    compare(
        "Controlled recovery from identical initial MFs (tanh)",
        X_1d,
        y_tanh,
        identical_spec,
        mf_bounds={
            ("x", "low"): [(-1.0, 1.0), (0.35, 1.0)],
            ("x", "high"): [(-1.0, 1.0), (0.35, 1.0)],
        },
        population_size=32,
        generations=40,
    )

    X = rng.uniform(-1.0, 1.0, size=(180, 2))
    x1, x2 = X.T
    y = 1.5 * np.sin(1.6 * x1) + 0.7 * x2 + 0.4 * x1 * x2 + rng.normal(0, 0.05, len(X))
    compare("Synthetic nonlinear surface", X, y, model_spec(((-0.5, 0.5, 0.6), (-0.5, 0.5, 0.6))))

    diabetes = load_diabetes(scaled=False)
    columns = [diabetes.feature_names.index(name) for name in ("bmi", "bp")]
    compare(
        "scikit-learn diabetes regression (BMI and blood pressure)",
        diabetes.data[:, columns],
        diabetes.target,
        model_spec(((21, 32, 6), (80, 110, 15))),
    )


if __name__ == "__main__":
    main()
