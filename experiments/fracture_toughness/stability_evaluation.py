"""Restricted native GA, regularization, repeated validation and learning curves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import GridSearchCV

from .ga_evaluation import FractureGARegressor, _write_rows, cv_splits
from .run_experiment import (
    _xlsx_records, load_observations, observations_to_arrays, load_niihara_workbooks, metric_summary,
)

CASES = ("original", "wide_ga", "three_widths", "three_widths_large", "regularized_widths")


def fit_case(case: str, X: np.ndarray, y: np.ndarray, seed: int):
    if case == "original":
        return FractureGARegressor(optimize=False).fit(X, y), {}
    if case == "wide_ga":
        return FractureGARegressor(random_state=seed).fit(X, y), {}
    options = dict(width_variables=("P",), width_radius=0.2, random_state=seed)
    if case == "three_widths_large":
        return FractureGARegressor(**options, population_size=48, generations=120).fit(X, y), {}
    estimator = FractureGARegressor(**options)
    if case == "three_widths":
        return estimator.fit(X, y), {}
    search = GridSearchCV(
        estimator,
        [{"deviation_penalty": [0.0, 0.3, 3.0], "narrow_sigma_penalty": [0.0]},
         {"deviation_penalty": [0.0], "narrow_sigma_penalty": [0.3, 3.0]}],
        cv=cv_splits(X, 3, 42), scoring="neg_root_mean_squared_error", error_score="raise", n_jobs=1,
    ).fit(X, y)
    return search.best_estimator_, dict(search.best_params_, inner_cv_rmse=-search.best_score_)


def training_subset(X: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Use nested stratified prefixes, preserving all porosity strata."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(X[:, 0] == value)) for value in np.unique(X[:, 0])]
    return np.sort(np.concatenate([group[:max(2, int(round(len(group) * fraction)))] for group in groups]))


def audit_raw_table(path: Path, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    header, rows = _xlsx_records(path)
    columns = {name: i for i, name in enumerate(header)}
    data = np.asarray([[row[columns[name]] for name in ("P", "a", "c", "Kc")] for row in rows], dtype=float)
    if data.shape[1] != 4 or not np.isfinite(data).all() or np.any(data[:, 3] <= 0):
        raise ValueError("Raw physical table requires finite P, a, c and positive Kc")
    clean = {tuple(row) for row in np.column_stack([X, y])}
    raw = {tuple(row) for row in data}
    if not clean.issubset(raw) or len(raw) != len(data):
        raise ValueError("Raw table must contain the cleaned data and have unique observations")
    audit = [dict(excel_row=i + 2, P=row[0], a=row[1], c=row[2], KIC=row[3],
                  status="in_clean26" if tuple(row) in clean else "raw_only_unconfirmed")
             for i, row in enumerate(data)]
    return data[:, :3], data[:, 3], audit


def _drift(model: FractureGARegressor) -> float:
    initial = {name: sigma for name, _, sigma in model.definition_.p_terms}
    terms = model.engine_.to_dict()["extra"]["linear_api"]["membership_functions"]
    changes = [np.log(term["parameters"][1] / initial[term["name"]]) ** 2
               for term in terms if term["variable"] == "P"]
    return float(np.mean(changes))


def diagnose_inner_cv(X: np.ndarray, y: np.ndarray) -> list[dict]:
    failures = []
    for outer, (train, _) in enumerate(cv_splits(X, 5, 42)):
        for inner, (fit, evaluate) in enumerate(cv_splits(X[train], 3, 42)):
            baseline = FractureGARegressor(optimize=False).fit(X[train][fit], y[train][fit])
            values = baseline.predict(X[train][evaluate])
            for index in np.flatnonzero(~np.isfinite(values)):
                row = int(train[evaluate[index]])
                failures.append(dict(outer_fold=outer, inner_fold=inner, row_index=row,
                                     P=X[row, 0], a=X[row, 1], c=X[row, 2], target=y[row]))
    return failures


def run(output: Path, seeds: list[int], repeats: int, raw: Path | None,
        predictors: Path | None, response: Path | None) -> dict:
    import PyFLL._core as core

    if "deviation_penalty" not in core.BaseLinearEngine.fit_ga.__doc__:
        raise RuntimeError("Build the regularized GA backend before running this stage")
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(__file__).with_name("article_clean_26.csv")
    X, y = observations_to_arrays(load_observations(data_path))
    datasets = {"clean26": (X, y)}
    if raw:
        raw_X, raw_y, audit = audit_raw_table(raw, X, y)
        datasets["raw35_sensitivity"] = (raw_X, raw_y)
        _write_rows(output / "raw_data_audit.csv", audit)
        numerical_failures = diagnose_inner_cv(raw_X, raw_y)
        if numerical_failures:
            _write_rows(output / "raw_cv_nonfinite.csv", numerical_failures)
    metric_rows, prediction_rows, curve_rows, selection_rows, split_rows = [], [], [], [], []
    started = perf_counter()
    for dataset, (X, y) in datasets.items():
        current_repeats = repeats if dataset == "clean26" else 1
        cases = CASES if dataset == "clean26" else CASES[:-1]
        for repeat in range(current_repeats):
            folds = cv_splits(X, 5, 42 + repeat)
            for fold, (train, test) in enumerate(folds):
                for role, indices in (("train", train), ("test", test)):
                    split_rows.extend(dict(dataset=dataset, repeat=repeat, fold=fold,
                                           role=role, row_index=int(index)) for index in indices)
            for case in cases:
                for seed in ([-1] if case == "original" else seeds):
                    predictions = np.empty(len(y))
                    train_losses, drifts, initial_losses = [], [], []
                    case_start = perf_counter()
                    for fold, (train, test) in enumerate(folds):
                        model, selected = fit_case(case, X[train], y[train], seed)
                        predictions[test] = model.predict(X[test])
                        train_losses.append(float(np.mean((model.predict(X[train]) - y[train]) ** 2)))
                        initial_losses.append(model.initial_train_mse_)
                        drifts.append(_drift(model))
                        if selected:
                            selection_rows.append(dict(dataset=dataset, repeat=repeat, fold=fold,
                                                       seed=seed, **selected))
                        if dataset == "clean26" and case in ("original", "wide_ga", "three_widths") and (
                            case == "original" or seed in seeds[:3]
                        ):
                            for fraction in (0.5, 0.75, 1.0):
                                indices = training_subset(X[train], fraction, 1000 + 10 * repeat + fold)
                                smaller = model if fraction == 1.0 else fit_case(case, X[train][indices], y[train][indices], seed)[0]
                                train_rmse = np.sqrt(np.mean((smaller.predict(X[train][indices]) - y[train][indices]) ** 2))
                                test_rmse = np.sqrt(np.mean((smaller.predict(X[test]) - y[test]) ** 2))
                                curve_rows.append(dict(repeat=repeat, fold=fold, seed=seed, case=case,
                                                       fraction=fraction, n_train=len(indices),
                                                       train_rmse=float(train_rmse), test_rmse=float(test_rmse)))
                    if not np.isfinite(predictions).all():
                        raise RuntimeError(f"Non-finite outer predictions for {dataset}/{case}/{seed}")
                    metric_rows.append(dict(dataset=dataset, repeat=repeat, seed=seed, case=case,
                                            seconds=perf_counter() - case_start,
                                            mean_train_mse=float(np.mean(train_losses)),
                                            mean_initial_train_mse=float(np.mean(initial_losses)),
                                            mean_log_sigma_drift=float(np.mean(drifts)),
                                            **metric_summary(y, predictions)))
                    prediction_rows.extend(dict(dataset=dataset, repeat=repeat, seed=seed, case=case,
                                                row_index=i, target=float(y[i]), predicted=float(value))
                                           for i, value in enumerate(predictions))
                    print(f"{dataset} repeat={repeat + 1}/{current_repeats} {case} seed={seed}", flush=True)
                    # Save each completed case, including the data needed for the plots.
                    for name, rows in (("metrics", metric_rows), ("predictions", prediction_rows),
                                       ("learning_curves", curve_rows), ("selections", selection_rows),
                                       ("splits", split_rows)):
                        if rows:
                            _write_rows(output / f"{name}.csv", rows)

    theory_rows = []
    if predictors:
        theory_X, theory_y = load_niihara_workbooks(predictors, response)
        for dataset, (X, y) in datasets.items():
            for case in (CASES if dataset == "clean26" else CASES[:-1]):
                for seed in ([-1] if case == "original" else seeds):
                    model, selected = fit_case(case, X, y, seed)
                    theory_rows.append(dict(dataset=dataset, case=case, seed=seed,
                                            **metric_summary(theory_y, model.predict(theory_X))))
                    if selected:
                        selection_rows.append(dict(dataset=dataset, repeat=-1, fold=-1, seed=seed, **selected))
    for name, rows in (("learning_curves", curve_rows), ("selections", selection_rows),
                       ("splits", split_rows), ("external_theory", theory_rows)):
        if rows:
            _write_rows(output / f"{name}.csv", rows)
    comparisons = []
    for dataset in datasets:
        baseline = {row["repeat"]: row["rmse"] for row in metric_rows
                    if row["dataset"] == dataset and row["case"] == "original"}
        for case in CASES:
            rows = [row for row in metric_rows if row["dataset"] == dataset and row["case"] == case]
            if not rows:
                continue
            rmse = [row["rmse"] for row in rows]
            paired = [100 * (row["rmse"] / baseline[row["repeat"]] - 1) for row in rows]
            comparisons.append(dict(dataset=dataset, case=case, runs=len(rows),
                                    median_rmse=float(np.median(rmse)), min_rmse=min(rmse), max_rmse=max(rmse),
                                    median_paired_change_percent=float(np.median(paired)),
                                    better_than_original_fraction=float(np.mean(np.asarray(paired) < 0))))
    _write_rows(output / "comparison.csv", comparisons)
    summary = dict(
        protocol=dict(seeds=seeds, repeats=repeats, outer_folds=5, penalty_inner_folds=3,
                      meaningful_width_parameters=3, width_bounds="0.8..1.2 * initial sigma",
                      penalties="training_variance * mean squared log(sigma/sigma_initial)",
                      penalty_candidates=[(0, 0), (0.3, 0), (3, 0), (0, 0.3), (0, 3)],
                      small_budget=dict(population_size=24, generations=40),
                      large_budget=dict(population_size=48, generations=120),
                      raw35_eligibility="sensitivity only, nine excluded records need scientific review"),
        raw35_penalty_search="excluded: inner CV extrapolation can return non-finite predictions",
        data_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (data_path, raw, predictors, response) if path is not None},
        native=dict(path=core.__file__, sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()),
        seconds=perf_counter() - started, comparisons=comparisons,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(output, metric_rows, curve_rows)
    return summary


def plot(output: Path, metrics: list[dict], curves: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = ["Original", "Wide GA", "3 widths", "3 widths\nlarge budget", "3 widths\nregularized"]
    for i, case in enumerate(CASES):
        values = [row["rmse"] for row in metrics if row["dataset"] == "clean26" and row["case"] == case]
        axes[0].scatter(np.full(len(values), i), values, alpha=0.6, s=18)
        axes[0].plot([i - 0.2, i + 0.2], [np.median(values)] * 2, color="black", linewidth=2)
    axes[0].set_xticks(range(len(CASES)), labels, fontsize=8)
    axes[0].set(ylabel="Outer RMSE", title="Repeated outer CV: splits and GA seeds")
    for case in ("original", "wide_ga", "three_widths"):
        x, train, test = [], [], []
        for fraction in (0.5, 0.75, 1.0):
            rows = [row for row in curves if row["case"] == case and row["fraction"] == fraction]
            x.append(np.mean([row["n_train"] for row in rows]))
            train.append(np.mean([row["train_rmse"] for row in rows]))
            test.append(np.mean([row["test_rmse"] for row in rows]))
        line, = axes[1].plot(x, test, "o-", label=f"{case}: validation")
        axes[1].plot(x, train, "--", color=line.get_color(), label=f"{case}: train")
    axes[1].set(xlabel="Mean physical training observations", ylabel="Mean fold RMSE", title="Learning curves, fixed outer test folds")
    axes[1].legend(fontsize=7)
    figure.savefig(output / "stability.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("stability_results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 42, 73, 101])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--raw-experimental-xlsx", type=Path)
    parser.add_argument("--niihara-predictors-xlsx", type=Path)
    parser.add_argument("--niihara-response-xlsx", type=Path)
    args = parser.parse_args()
    if bool(args.niihara_predictors_xlsx) != bool(args.niihara_response_xlsx):
        parser.error("Supply both Niihara workbooks")
    if not args.seeds or args.repeats < 1:
        parser.error("Supply seeds and at least one repeat")
    result = run(args.output_dir, args.seeds, args.repeats, args.raw_experimental_xlsx,
                 args.niihara_predictors_xlsx, args.niihara_response_xlsx)
    print(json.dumps(result["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
