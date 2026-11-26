"""Nested validation of native GA on the 26-record fracture-toughness dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .run_experiment import (
    INPUT_NAMES,
    OUTPUT_NAME,
    Observation,
    build_model_definition,
    build_openfll_engine,
    create_surface_points,
    load_niihara_workbooks,
    load_observations,
    metric_summary,
    observations_to_arrays,
    predict_numpy,
    predict_openfll,
    stratified_holdout_indices,
)


class FractureGARegressor(RegressorMixin, BaseEstimator):
    """Build one rule per training observation, then optionally optimize native MFs.

    The structure, parameter bounds and initial consequents are derived exclusively
    from the current fit data. This is an experimental estimator for three inputs
    in the order P, a, c. The original reproduction experiment stays separate.
    """

    def __init__(
        self,
        *,
        optimize: bool = True,
        train_consequents: bool = False,
        center_radius: float = 0.25,
        mutation_rate: float = 0.1,
        population_size: int = 24,
        generations: int = 40,
        random_state: int = 42,
    ) -> None:
        self.optimize = optimize
        self.train_consequents = train_consequents
        self.center_radius = center_radius
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state

    def fit(self, X: object, y: object) -> FractureGARegressor:
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=False)
        if X.shape[1] != 3:
            raise ValueError("FractureGARegressor expects columns P, a, c")
        if not np.isfinite(self.center_radius) or self.center_radius <= 0:
            raise ValueError("center_radius must be finite and positive")
        if np.any(y <= 0):
            raise ValueError("Experimental KIC values must be positive")
        records = [Observation(i, 0.0, *row, value) for i, (row, value) in enumerate(zip(X, y))]
        definition = build_model_definition(records)
        engine = build_openfll_engine(definition)
        initial_mse = float(np.mean((predict_openfll(engine, X) - y) ** 2))
        bounds_by_term = {}
        if self.optimize:
            for variable, terms in zip(INPUT_NAMES, (definition.p_terms, definition.a_terms, definition.c_terms)):
                for name, center, sigma in terms:
                    bounds = [
                        (center - self.center_radius * sigma, center + self.center_radius * sigma),
                        (0.5 * sigma, 2.0 * sigma),
                    ]
                    engine.set_mf_bounds(variable, name, bounds)
                    bounds_by_term[(variable, name)] = bounds
            for rule in definition.rules:
                value = rule.output_value
                if self.train_consequents:
                    bounds = [(0.5 * float(y.min()), 1.5 * float(y.max()))]
                else:
                    # The native API requires lower < upper. Retain values to one ULP.
                    bounds = [(float(np.nextafter(value, -np.inf)), float(np.nextafter(value, np.inf)))]
                engine.set_mf_bounds(OUTPUT_NAME, rule.output_term, bounds)
                bounds_by_term[(OUTPUT_NAME, rule.output_term)] = bounds
            history = engine.fit_ga(
                X, y, input_names=INPUT_NAMES, population_size=self.population_size,
                generations=self.generations, mutation_rate=self.mutation_rate, seed=self.random_state,
            )
        else:
            history = []

        terms = engine.to_dict()["extra"]["linear_api"]["membership_functions"]
        actual_consequents = {
            term["name"]: term["parameters"][0] for term in terms if term["variable"] == OUTPUT_NAME
        }
        delta = max(abs(actual_consequents[rule.output_term] - rule.output_value) for rule in definition.rules)
        if self.optimize and not self.train_consequents:
            tolerance = max(float(np.spacing(abs(rule.output_value))) for rule in definition.rules)
            if delta > tolerance:
                raise RuntimeError("Native GA changed a restricted consequent beyond one ULP")
        self.engine_ = engine
        self.definition_ = definition
        self.training_X_ = X.copy()
        self.training_y_ = y.copy()
        self.bounds_ = bounds_by_term
        self.n_features_in_ = 3
        self.loss_history_ = history
        self.initial_train_mse_ = initial_mse
        self.consequent_max_abs_change_ = delta
        return self

    def predict(self, X: object) -> np.ndarray:
        check_is_fitted(self, "engine_")
        return predict_openfll(self.engine_, check_array(X, dtype=np.float64))


def cv_splits(X: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.unique(X[:, 0], return_inverse=True)[1]
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, labels))


def make_search(method: str, *, train_consequents: bool, seed: int, folds: list,
                population_size: int, generations: int):
    estimator = FractureGARegressor(
        train_consequents=train_consequents, random_state=seed,
        population_size=population_size, generations=generations,
    )
    options = dict(cv=folds, scoring="neg_root_mean_squared_error", error_score="raise", n_jobs=1)
    if method == "grid":
        return GridSearchCV(estimator, {"center_radius": [0.25, 0.75], "mutation_rate": [0.1, 0.3]}, **options)
    return RandomizedSearchCV(
        estimator, {"center_radius": [0.25, 0.375, 0.5, 0.625, 0.75],
                    "mutation_rate": [0.1, 0.15, 0.2, 0.25, 0.3]},
        n_iter=4, random_state=seed, **options,
    )


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(metrics: list[dict]) -> list[dict]:
    groups = {}
    baseline = {}
    for row in metrics:
        groups.setdefault((row["protocol"], row["method"], row["mode"]), []).append(row)
        if row["method"] == "baseline":
            baseline[row["protocol"]] = row["rmse"]
    summary = []
    for (protocol, method, mode), rows in groups.items():
        rmse = [row["rmse"] for row in rows]
        median = float(np.median(rmse))
        summary.append(dict(protocol=protocol, method=method, mode=mode, n_seeds=len(rows),
                            median_rmse=median, min_rmse=min(rmse), max_rmse=max(rmse),
                            median_mae=float(np.median([row["mae"] for row in rows])),
                            median_r2=float(np.median([row["r2"] for row in rows])),
                            rmse_change_percent=100 * (median / baseline[protocol] - 1)))
    return summary


def run(output: Path, *, seeds: list[int], outer_folds: int, population_size: int,
        generations: int, predictors: Path | None = None, response: Path | None = None) -> dict:
    import openfll
    import PyFLL._core as core
    import sklearn

    if not hasattr(openfll.SugenoEngine, "fit_ga"):
        raise RuntimeError("This experiment requires a GA-enabled PyFLL._core")
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(__file__).with_name("article_clean_26.csv")
    records = load_observations(data_path)
    X, y = observations_to_arrays(records)
    train, test = stratified_holdout_indices(records)
    outer = cv_splits(X, outer_folds, 42)
    metrics, predictions, candidates, histories, splits = [], [], [], [], []
    summaries = []
    baseline_predictions = np.empty(len(y))
    for fold, (fit_indices, evaluate_indices) in enumerate(outer, start=1):
        baseline_predictions[evaluate_indices] = FractureGARegressor(optimize=False).fit(
            X[fit_indices], y[fit_indices]
        ).predict(X[evaluate_indices])
        for role, indices in (("train", fit_indices), ("test", evaluate_indices)):
            splits.extend(dict(protocol="nested_cv", fold=fold, role=role,
                               record_id=records[i].record_id) for i in indices)
    baseline = FractureGARegressor(optimize=False).fit(X[train], y[train])
    for protocol, target, actual in (
        ("holdout", y[test], baseline.predict(X[test])),
        ("nested_cv", y, baseline_predictions),
    ):
        metrics.append(dict(protocol=protocol, method="baseline", mode="original", seed=42,
                            seconds=0.0, **metric_summary(target, actual)))
    full_baseline = FractureGARegressor(optimize=False).fit(X, y)
    metrics.append(dict(protocol="full_train", method="baseline", mode="original", seed=42,
                        seconds=0.0, **metric_summary(y, full_baseline.predict(X))))
    external = load_niihara_workbooks(predictors, response) if predictors else None
    if external:
        external_X, external_y = external
        metrics.append(dict(protocol="external_theory", method="baseline", mode="original", seed=42,
                            seconds=0.0, **metric_summary(external_y, full_baseline.predict(external_X))))
    surface_X, _, _, _ = create_surface_points(records, grid_size=40)
    parity_error = float(np.max(np.abs(full_baseline.predict(surface_X) - predict_numpy(full_baseline.definition_, surface_X))))
    plot_predictions = {"baseline": baseline_predictions}
    for seed in seeds:
        for method in ("grid", "random"):
            for joint in (False, True):
                mode = "joint" if joint else "consequents_one_ulp"
                key = f"{method}_{mode}_{seed}"
                outer_predictions = np.empty(len(y))
                started = perf_counter()
                for fold, (fit_indices, evaluate_indices) in enumerate(outer, start=1):
                    search = make_search(
                        method, train_consequents=joint, seed=seed,
                        folds=cv_splits(X[fit_indices], 3, 42), population_size=population_size,
                        generations=generations,
                    ).fit(X[fit_indices], y[fit_indices])
                    outer_predictions[evaluate_indices] = search.predict(X[evaluate_indices])
                    summaries.append(dict(protocol="nested_cv", method=method, mode=mode, seed=seed,
                                          fold=fold, inner_cv_rmse=-search.best_score_,
                                          best_params=search.best_params_,
                                          consequent_max_abs_change=search.best_estimator_.consequent_max_abs_change_))
                    print(f"{key} outer fold {fold}/{outer_folds}", flush=True)
                metrics.append(dict(protocol="nested_cv", method=method, mode=mode, seed=seed,
                                    seconds=perf_counter() - started, **metric_summary(y, outer_predictions)))
                plot_predictions[key] = outer_predictions
                for i, predicted in enumerate(outer_predictions):
                    predictions.append(dict(protocol="nested_cv", method=method, mode=mode, seed=seed,
                                            record_id=records[i].record_id, target=y[i], predicted=predicted))

                started = perf_counter()
                holdout_search = make_search(
                    method, train_consequents=joint, seed=seed, folds=cv_splits(X[train], 3, 42),
                    population_size=population_size, generations=generations,
                ).fit(X[train], y[train])
                metrics.append(dict(protocol="holdout", method=method, mode=mode, seed=seed,
                                    seconds=perf_counter() - started,
                                    **metric_summary(y[test], holdout_search.predict(X[test]))))

                # External theory is evaluated only after selecting settings on physical data.
                started = perf_counter()
                full_search = make_search(
                    method, train_consequents=joint, seed=seed, folds=cv_splits(X, 3, 42),
                    population_size=population_size, generations=generations,
                ).fit(X, y)
                full = full_search.best_estimator_
                for protocol, current_search in (("holdout", holdout_search), ("full_physical_cv", full_search)):
                    summaries.append(dict(protocol=protocol, method=method, mode=mode, seed=seed,
                                          fold=0, inner_cv_rmse=-current_search.best_score_,
                                          best_params=current_search.best_params_,
                                          consequent_max_abs_change=current_search.best_estimator_.consequent_max_abs_change_))
                    for candidate_index, parameters in enumerate(current_search.cv_results_["params"]):
                        candidates.append(dict(protocol=protocol, method=method, mode=mode, seed=seed,
                                               candidate=candidate_index, params=json.dumps(parameters),
                                               mean_cv_rmse=-current_search.cv_results_["mean_test_score"][candidate_index]))
                if external:
                    metrics.append(dict(protocol="external_theory", method=method, mode=mode, seed=seed,
                                        seconds=perf_counter() - started,
                                        **metric_summary(external_y, full.predict(external_X))))
                metrics.append(dict(protocol="full_train", method=method, mode=mode, seed=seed,
                                    seconds=0.0, **metric_summary(y, full.predict(X))))
                full.engine_.save_to_file(output / f"{key}_model.json")
                for generation, loss in enumerate(full.loss_history_, start=1):
                    histories.append(dict(method=method, mode=mode, seed=seed, generation=generation, mse=loss))
    for role, indices in (("train", train), ("test", test)):
        splits.extend(dict(protocol="holdout", fold=0, role=role,
                           record_id=records[i].record_id) for i in indices)
    for i, predicted in enumerate(baseline_predictions):
        predictions.append(dict(protocol="nested_cv", method="baseline", mode="original", seed=42,
                                record_id=records[i].record_id, target=y[i], predicted=predicted))
    for name, rows in (("metrics", metrics), ("predictions", predictions), ("candidates", candidates),
                       ("loss_history", histories), ("splits", splits)):
        _write_rows(output / f"{name}.csv", rows)
    _write_rows(output / "metrics_summary.csv", aggregate_metrics(metrics))
    summary = dict(
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=np.__version__, sklearn=sklearn.__version__, native_module=core.__file__,
                         native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                         backend_build=json.loads((Path(core.__file__).parent / "backend_build.json").read_text(encoding="utf-8-sig"))),
        dataset=dict(n=26, sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                     holdout_train_ids=[records[i].record_id for i in train],
                     holdout_test_ids=[records[i].record_id for i in test],
                     theoretical_n=len(external_y) if external else 0,
                     theory_files={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (predictors, response) if path is not None}),
        protocol=dict(seeds=seeds, outer_folds=outer_folds, inner_folds=3, candidates_per_search=4,
                      population_size=population_size, generations=generations, score="RMSE",
                      grid={"center_radius": [0.25, 0.75], "mutation_rate": [0.1, 0.3]},
                      random={"center_radius": [0.25, 0.375, 0.5, 0.625, 0.75],
                              "mutation_rate": [0.1, 0.15, 0.2, 0.25, 0.3]},
                      sigma_bounds="[0.5 * initial_sigma, 2 * initial_sigma]",
                      center_bounds="initial_center +/- center_radius * initial_sigma",
                      joint_output_bounds="[0.5 * training_min_KIC, 1.5 * training_max_KIC]",
                      restricted_output_bounds="adjacent float64 values, at most one ULP change",
                      theory_used_for_selection=False),
        baseline_numpy_parity_max_abs=parity_error,
        metrics=metrics, searches=summaries,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(output, y, plot_predictions, histories, metrics)
    return summary


def _plot(output: Path, y: np.ndarray, predictions: dict, histories: list[dict], metrics: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(y, predictions["baseline"], label="Original", s=30, color="black")
    for method in ("grid", "random"):
        for mode in ("consequents_one_ulp", "joint"):
            blocks = [values for key, values in predictions.items() if key.startswith(f"{method}_{mode}_")]
            median_predictions = np.median(blocks, axis=0)
            label = f"{method}, {'MF only (1 ULP)' if mode == 'consequents_one_ulp' else 'MF + consequents'}"
            axes[0].scatter(y, median_predictions, label=label, s=24, alpha=0.7)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "--", color="gray")
    axes[0].set(xlabel="Experimental KIC", ylabel="Outer-CV predicted KIC", title="Nested CV, median prediction over seeds")
    axes[0].legend(fontsize=7)
    for mode in ("consequents_one_ulp", "joint"):
        for method in ("grid", "random"):
            groups = {}
            for row in histories:
                if row["mode"] == mode and row["method"] == method:
                    groups.setdefault(row["seed"], []).append(row["mse"])
            axes[1].plot(np.median(list(groups.values()), axis=0), label=f"{method}, {mode}")
    axes[1].set(xlabel="Generation (zero-based)", ylabel="Training MSE", title="Full-data refit, median over seeds")
    axes[1].legend(fontsize=7)
    figure.savefig(output / "validation.png", dpi=160)
    plt.close(figure)

    protocols = ["nested_cv", "holdout"]
    if any(row["protocol"] == "external_theory" for row in metrics):
        protocols.append("external_theory")
    figure, axes = plt.subplots(1, len(protocols), figsize=(5 * len(protocols), 4.8), constrained_layout=True)
    labels = ["Grid\nMF", "Random\nMF", "Grid\nMF + KIC", "Random\nMF + KIC"]
    for axis, protocol in zip(axes, protocols):
        baseline = next(row["rmse"] for row in metrics if row["protocol"] == protocol and row["method"] == "baseline")
        axis.axhline(baseline, linestyle="--", color="black", label=f"Original RMSE {baseline:.4f}")
        for i, (mode, method) in enumerate((
            ("consequents_one_ulp", "grid"), ("consequents_one_ulp", "random"),
            ("joint", "grid"), ("joint", "random"),
        )):
            values = [row["rmse"] for row in metrics if row["protocol"] == protocol
                      and row["mode"] == mode and row["method"] == method]
            center = float(np.median(values))
            axis.bar(i, center, color="#427c9a" if mode == "consequents_one_ulp" else "#bf8b45", alpha=0.85)
            axis.errorbar(i, center, yerr=[[center - min(values)], [max(values) - center]],
                          color="black", capsize=4)
        axis.set_xticks(range(4), labels)
        axis.set(ylabel="RMSE (lower is better)", title=protocol.replace("_", " "))
        axis.legend(fontsize=8)
    figure.suptitle("Fracture toughness: median and min/max over GA seeds\nMF: centers and widths; MF + KIC: also train consequents", fontsize=11)
    figure.savefig(output / "comparison.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("ga_results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--niihara-predictors-xlsx", type=Path)
    parser.add_argument("--niihara-response-xlsx", type=Path)
    args = parser.parse_args()
    if bool(args.niihara_predictors_xlsx) != bool(args.niihara_response_xlsx):
        parser.error("Supply both Niihara workbooks")
    summary = run(args.output_dir, seeds=args.seeds, outer_folds=args.outer_folds,
                  population_size=args.population_size, generations=args.generations,
                  predictors=args.niihara_predictors_xlsx, response=args.niihara_response_xlsx)
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
