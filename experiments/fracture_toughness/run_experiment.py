"""Reproduce and validate the fracture-toughness fuzzy surface.

The model is a zero-order Takagi-Sugeno system with one rule per cleaned
experimental observation. OpenFLL predictions are compared with Simpful and an
independent vectorized NumPy implementation of the same weighted-average
formula.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


INPUT_NAMES = ["P", "a", "c"]
OUTPUT_NAME = "KIC"
DEFAULT_GRID_SIZE = 100
DEFAULT_SIMPFUL_SURFACE_SAMPLE = 300
ARTICLE_A_RANGE = (14.0, 39.0)
ARTICLE_C_RANGE = (48.0, 109.0)


@dataclass(frozen=True)
class Observation:
    record_id: int
    concentration_mass_pct: float
    P: float
    a: float
    c: float
    KIC: float


@dataclass(frozen=True)
class RuleDefinition:
    index: int
    p_term: str
    a_term: str
    c_term: str
    output_term: str
    output_value: float


@dataclass(frozen=True)
class ModelDefinition:
    p_terms: tuple[tuple[str, float, float], ...]
    a_terms: tuple[tuple[str, float, float], ...]
    c_terms: tuple[tuple[str, float, float], ...]
    rules: tuple[RuleDefinition, ...]


def load_observations(path: Path) -> list[Observation]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = [
            Observation(
                record_id=int(row["record_id"]),
                concentration_mass_pct=float(row["concentration_mass_pct"]),
                P=float(row["P"]),
                a=float(row["a"]),
                c=float(row["c"]),
                KIC=float(row["KIC"]),
            )
            for row in reader
        ]
    if len(records) != 26:
        raise ValueError(f"Expected the article-cleaned 26 observations, got {len(records)}")
    if len({item.record_id for item in records}) != len(records):
        raise ValueError("record_id values must be unique")
    return records


def _term_name(prefix: str, index: int) -> str:
    return f"{prefix}_{index:02d}"


def neighbor_sigmas(values: Sequence[float], *, minimum_span_fraction: float = 0.05) -> dict[float, float]:
    """Return deterministic Gaussian widths from sorted unique centers.

    For an isolated adjacent gap d, sigma=d/(2*sqrt(2*ln(2))) makes equal
    Gaussians reach membership 0.5 at their midpoint. For irregularly spaced
    centers the mean adjacent gap is used, with a lower bound proportional to
    the complete domain span to prevent numerical holes in the response surface.
    """
    centers = np.asarray(sorted(set(float(v) for v in values)), dtype=np.float64)
    if centers.size < 2:
        raise ValueError("At least two distinct centers are required")
    domain_span = float(centers[-1] - centers[0])
    min_sigma = max(domain_span * minimum_span_fraction, np.finfo(np.float64).eps)
    half_height_factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    result: dict[float, float] = {}
    for index, center in enumerate(centers):
        gaps: list[float] = []
        if index > 0:
            gaps.append(float(center - centers[index - 1]))
        if index + 1 < centers.size:
            gaps.append(float(centers[index + 1] - center))
        local_gap = float(np.mean(gaps))
        result[float(center)] = max(local_gap / half_height_factor, min_sigma)
    return result


def build_model_definition(records: Sequence[Observation]) -> ModelDefinition:
    p_values = [item.P for item in records]
    a_values = [item.a for item in records]
    c_values = [item.c for item in records]
    p_sigmas = neighbor_sigmas(p_values)
    a_sigmas = neighbor_sigmas(a_values)
    c_sigmas = neighbor_sigmas(c_values)

    sorted_p = sorted(set(p_values))
    p_name_by_value = {value: _term_name("p", index) for index, value in enumerate(sorted_p)}
    p_terms = tuple((p_name_by_value[value], value, p_sigmas[value]) for value in sorted_p)
    a_terms = tuple((_term_name("a", index), item.a, a_sigmas[item.a]) for index, item in enumerate(records))
    c_terms = tuple((_term_name("c", index), item.c, c_sigmas[item.c]) for index, item in enumerate(records))
    rules = tuple(
        RuleDefinition(
            index=index,
            p_term=p_name_by_value[item.P],
            a_term=_term_name("a", index),
            c_term=_term_name("c", index),
            output_term=_term_name("kic", index),
            output_value=item.KIC,
        )
        for index, item in enumerate(records)
    )
    return ModelDefinition(p_terms=p_terms, a_terms=a_terms, c_terms=c_terms, rules=rules)


def observations_to_arrays(records: Sequence[Observation]) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray([[item.P, item.a, item.c] for item in records], dtype=np.float64)
    y = np.asarray([item.KIC for item in records], dtype=np.float64)
    return X, y


def _gaussian(values: np.ndarray, centers: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((values[:, None] - centers[None, :]) / sigmas[None, :]) ** 2)


def predict_numpy(model: ModelDefinition, X: np.ndarray, *, consequents: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"X must have shape (n_samples, 3), got {X.shape}")

    p_term_by_name = {name: (center, sigma) for name, center, sigma in model.p_terms}
    p_centers = np.asarray([p_term_by_name[rule.p_term][0] for rule in model.rules])
    p_sigmas = np.asarray([p_term_by_name[rule.p_term][1] for rule in model.rules])
    a_centers = np.asarray([term[1] for term in model.a_terms])
    a_sigmas = np.asarray([term[2] for term in model.a_terms])
    c_centers = np.asarray([term[1] for term in model.c_terms])
    c_sigmas = np.asarray([term[2] for term in model.c_terms])
    if consequents is None:
        consequents = np.asarray([rule.output_value for rule in model.rules], dtype=np.float64)
    else:
        consequents = np.asarray(consequents, dtype=np.float64)

    weights = (
        _gaussian(X[:, 0], p_centers, p_sigmas)
        * _gaussian(X[:, 1], a_centers, a_sigmas)
        * _gaussian(X[:, 2], c_centers, c_sigmas)
    )
    totals = weights.sum(axis=1)
    result = np.full(X.shape[0], np.nan, dtype=np.float64)
    active = totals > 0.0
    result[active] = (weights[active] @ consequents) / totals[active]
    return result


def build_openfll_engine(model: ModelDefinition) -> Any:
    import openfll

    engine = openfll.SugenoEngine()
    engine.set_default_t_norm("prod_and")
    engine.add_input_var("P")
    engine.add_input_var("a")
    engine.add_input_var("c")
    engine.add_output_var(OUTPUT_NAME)
    for name, center, sigma in model.p_terms:
        engine.add_membership_func("P", name, "gaussian", [center, sigma])
    for name, center, sigma in model.a_terms:
        engine.add_membership_func("a", name, "gaussian", [center, sigma])
    for name, center, sigma in model.c_terms:
        engine.add_membership_func("c", name, "gaussian", [center, sigma])
    for rule in model.rules:
        engine.add_membership_func(OUTPUT_NAME, rule.output_term, "constant", [rule.output_value])
        engine.add_rule(
            antecedent=[("P", rule.p_term), ("a", rule.a_term), ("c", rule.c_term)],
            consequent=[(OUTPUT_NAME, rule.output_term)],
            t_norm="prod_and",
        )
    engine.build()
    return engine


def predict_openfll(engine: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(engine.predict_batch(np.asarray(X, dtype=np.float64), input_names=INPUT_NAMES), dtype=np.float64)


def build_simpful_system(model: ModelDefinition) -> Any:
    from simpful import FuzzySet, FuzzySystem, Gaussian_MF, LinguisticVariable

    system = FuzzySystem(operators=["AND_PRODUCT"], show_banner=False, verbose=False)

    def linguistic_variable(terms: Iterable[tuple[str, float, float]]) -> Any:
        terms = tuple(terms)
        fuzzy_sets = [
            FuzzySet(function=Gaussian_MF(mu=center, sigma=sigma), term=name)
            for name, center, sigma in terms
        ]
        centers = [term[1] for term in terms]
        sigmas = [term[2] for term in terms]
        universe = [min(c - 4.0 * s for c, s in zip(centers, sigmas)), max(c + 4.0 * s for c, s in zip(centers, sigmas))]
        return LinguisticVariable(fuzzy_sets, universe_of_discourse=universe)

    system.add_linguistic_variable("P", linguistic_variable(model.p_terms), verbose=False)
    system.add_linguistic_variable("a", linguistic_variable(model.a_terms), verbose=False)
    system.add_linguistic_variable("c", linguistic_variable(model.c_terms), verbose=False)
    for rule in model.rules:
        system.set_crisp_output_value(rule.output_term, rule.output_value, verbose=False)
    system.add_rules(
        [
            f"IF (P IS {rule.p_term}) AND (a IS {rule.a_term}) AND (c IS {rule.c_term}) "
            f"THEN ({OUTPUT_NAME} IS {rule.output_term})"
            for rule in model.rules
        ],
        verbose=False,
    )
    return system


def predict_simpful_from_strengths(system: Any, model: ModelDefinition, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    strengths = np.asarray(
        system.get_firing_strengths(
            {"P": X[:, 0].tolist(), "a": X[:, 1].tolist(), "c": X[:, 2].tolist()}
        ),
        dtype=np.float64,
    )
    expected_shape = (len(X), len(model.rules))
    if strengths.shape != expected_shape and strengths.shape == (len(model.rules), len(X)):
        strengths = strengths.T
    if strengths.shape != expected_shape:
        raise RuntimeError(f"Unexpected Simpful firing-strength shape {strengths.shape}; expected {expected_shape}")
    totals = strengths.sum(axis=1)
    consequents = np.asarray([rule.output_value for rule in model.rules], dtype=np.float64)
    result = np.full(len(X), np.nan, dtype=np.float64)
    active = totals > 0.0
    result[active] = (strengths[active] @ consequents) / totals[active]
    return result


def predict_simpful_scalar(system: Any, X: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for P, a, c in np.asarray(X, dtype=np.float64):
        system.set_variable("P", float(P), verbose=False)
        system.set_variable("a", float(a), verbose=False)
        system.set_variable("c", float(c), verbose=False)
        values.append(float(system.Sugeno_inference([OUTPUT_NAME], verbose=False)[OUTPUT_NAME]))
    return np.asarray(values, dtype=np.float64)


def build_regressor(model: ModelDefinition) -> Any:
    from openfll.sklearn import SugenoModelSpec, SugenoRegressor

    spec = SugenoModelSpec(
        inputs={
            "P": [(name, "gaussian", [center, sigma]) for name, center, sigma in model.p_terms],
            "a": [(name, "gaussian", [center, sigma]) for name, center, sigma in model.a_terms],
            "c": [(name, "gaussian", [center, sigma]) for name, center, sigma in model.c_terms],
        },
        rules=[
            {
                "antecedent": [("P", rule.p_term), ("a", rule.a_term), ("c", rule.c_term)],
                "consequent": [(OUTPUT_NAME, rule.output_term)],
                "t_norm": "prod_and",
            }
            for rule in model.rules
        ],
        output_name=OUTPUT_NAME,
        consequents=[rule.output_term for rule in model.rules],
        default_t_norm="prod_and",
    )
    return SugenoRegressor(model_spec=spec, solver="lstsq", alpha=0.0)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_pred)
    if not finite.all():
        return {"n": int(len(y_true)), "finite": int(finite.sum()), "nan_or_inf": int((~finite).sum())}
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(len(y_true)),
        "finite": int(finite.sum()),
        "nan_or_inf": 0,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / denominator),
    }


def leave_one_out_fixed_predictions(records: Sequence[Observation]) -> np.ndarray:
    """Predict every observation from a model rebuilt without its own rule."""
    predictions: list[float] = []
    for index, item in enumerate(records):
        retained = list(records[:index]) + list(records[index + 1 :])
        model = build_model_definition(retained)
        engine = build_openfll_engine(model)
        point = np.asarray([[item.P, item.a, item.c]], dtype=np.float64)
        predictions.append(float(predict_openfll(engine, point)[0]))
    return np.asarray(predictions, dtype=np.float64)


def parity_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    difference = np.abs(left[finite] - right[finite])
    return {
        "n": int(len(left)),
        "finite_pairs": int(finite.sum()),
        "nonfinite_pairs": int((~finite).sum()),
        "max_abs_diff": float(np.max(difference)) if difference.size else math.nan,
        "mean_abs_diff": float(np.mean(difference)) if difference.size else math.nan,
        "rmse_diff": float(np.sqrt(np.mean(difference**2))) if difference.size else math.nan,
    }


def create_surface_points(records: Sequence[Observation], grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_axis = np.linspace(*ARTICLE_A_RANGE, grid_size, dtype=np.float64)
    c_axis = np.linspace(*ARTICLE_C_RANGE, grid_size, dtype=np.float64)
    a_grid, c_grid = np.meshgrid(a_axis, c_axis, indexing="xy")
    blocks = []
    concentrations = []
    for P in sorted({item.P for item in records}, reverse=True):
        concentration = next(item.concentration_mass_pct for item in records if item.P == P)
        blocks.append(np.column_stack([np.full(a_grid.size, P), a_grid.ravel(), c_grid.ravel()]))
        concentrations.append(np.full(a_grid.size, concentration))
    return np.vstack(blocks), np.concatenate(concentrations), np.asarray([a_axis, c_axis])


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_plots(
    results_dir: Path,
    surface_X: np.ndarray,
    concentrations: np.ndarray,
    openfll_values: np.ndarray,
    numpy_values: np.ndarray,
    grid_size: int,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    written: list[str] = []
    unique_concentrations = sorted(set(float(v) for v in concentrations))
    for concentration in unique_concentrations:
        mask = concentrations == concentration
        a_grid = surface_X[mask, 1].reshape(grid_size, grid_size)
        c_grid = surface_X[mask, 2].reshape(grid_size, grid_size)
        z_openfll = openfll_values[mask].reshape(grid_size, grid_size)
        z_diff = np.abs(openfll_values[mask] - numpy_values[mask]).reshape(grid_size, grid_size)

        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        surface = axis.plot_surface(a_grid, c_grid, z_openfll, cmap="viridis", linewidth=0, antialiased=True)
        axis.set_xlabel("a (micrometers)")
        axis.set_ylabel("c (micrometers)")
        axis.set_zlabel("KIC (MPa*m^0.5)")
        axis.set_title(f"OpenFLL fuzzy surface, MWCNT={concentration:.1f} mass.%")
        fig.colorbar(surface, ax=axis, shrink=0.65, pad=0.12)
        fig.tight_layout()
        surface_name = f"surface_openfll_{concentration:.1f}.png"
        fig.savefig(results_dir / surface_name, dpi=160)
        plt.close(fig)
        written.append(surface_name)

        fig, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(
            z_diff,
            origin="lower",
            extent=[ARTICLE_A_RANGE[0], ARTICLE_A_RANGE[1], ARTICLE_C_RANGE[0], ARTICLE_C_RANGE[1]],
            aspect="auto",
            cmap="magma",
        )
        axis.set_xlabel("a (micrometers)")
        axis.set_ylabel("c (micrometers)")
        axis.set_title(f"|OpenFLL - NumPy oracle|, MWCNT={concentration:.1f} mass.%")
        fig.colorbar(image, ax=axis, label="absolute difference")
        fig.tight_layout()
        difference_name = f"difference_openfll_numpy_{concentration:.1f}.png"
        fig.savefig(results_dir / difference_name, dpi=160)
        plt.close(fig)
        written.append(difference_name)
    return written


def model_to_json(model: ModelDefinition) -> dict[str, Any]:
    return {
        "inputs": {
            "P": [{"name": n, "center": c, "sigma": s} for n, c, s in model.p_terms],
            "a": [{"name": n, "center": c, "sigma": s} for n, c, s in model.a_terms],
            "c": [{"name": n, "center": c, "sigma": s} for n, c, s in model.c_terms],
        },
        "rules": [asdict(rule) for rule in model.rules],
        "t_norm": "product",
        "defuzzification": "normalized_weighted_average",
    }


def stratified_surface_sample_indices(concentrations: np.ndarray, max_points: int) -> np.ndarray:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    groups = sorted(set(float(value) for value in concentrations))
    base = max_points // len(groups)
    remainder = max_points % len(groups)
    selected: list[np.ndarray] = []
    for position, group in enumerate(groups):
        candidates = np.flatnonzero(concentrations == group)
        count = min(len(candidates), base + (1 if position < remainder else 0))
        offsets = np.linspace(0, len(candidates) - 1, count, dtype=int)
        selected.append(candidates[offsets])
    return np.sort(np.concatenate(selected))


def run_experiment(
    data_path: Path,
    results_dir: Path,
    grid_size: int = DEFAULT_GRID_SIZE,
    simpful_surface_sample: int = DEFAULT_SIMPFUL_SURFACE_SAMPLE,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    records = load_observations(data_path)
    model = build_model_definition(records)
    X_train, y_train = observations_to_arrays(records)
    surface_X, concentrations, _ = create_surface_points(records, grid_size)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    openfll_engine = build_openfll_engine(model)
    openfll_train = predict_openfll(openfll_engine, X_train)
    openfll_surface = predict_openfll(openfll_engine, surface_X)
    timings["openfll_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    numpy_train = predict_numpy(model, X_train)
    numpy_surface = predict_numpy(model, surface_X)
    timings["numpy_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    simpful_system = build_simpful_system(model)
    simpful_train = predict_simpful_from_strengths(simpful_system, model, X_train)
    simpful_indices = stratified_surface_sample_indices(concentrations, simpful_surface_sample)
    simpful_surface = predict_simpful_from_strengths(simpful_system, model, surface_X[simpful_indices])
    simpful_scalar_check = predict_simpful_scalar(simpful_system, X_train[:5])
    timings["simpful_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    regressor = build_regressor(model)
    regressor.fit(X_train, y_train)
    regressor_train = np.asarray(regressor.predict(X_train), dtype=np.float64)
    regressor_surface = np.asarray(regressor.predict(surface_X), dtype=np.float64)
    timings["sugeno_regressor_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    leave_one_out_predictions = leave_one_out_fixed_predictions(records)
    timings["fixed_model_leave_one_out_seconds"] = time.perf_counter() - t0

    write_csv(
        results_dir / "training_predictions.csv",
        [
            "record_id", "concentration_mass_pct", "P", "a", "c", "KIC_observed",
            "KIC_openfll_fixed", "KIC_simpful_fixed", "KIC_numpy_fixed", "KIC_openfll_fitted",
            "abs_openfll_simpful", "abs_openfll_numpy",
        ],
        (
            [
                item.record_id, item.concentration_mass_pct, item.P, item.a, item.c, item.KIC,
                openfll_train[index], simpful_train[index], numpy_train[index], regressor_train[index],
                abs(openfll_train[index] - simpful_train[index]),
                abs(openfll_train[index] - numpy_train[index]),
            ]
            for index, item in enumerate(records)
        ),
    )
    write_csv(
        results_dir / "surface_predictions.csv",
        [
            "concentration_mass_pct", "P", "a", "c", "KIC_openfll_fixed",
            "KIC_numpy_fixed", "KIC_openfll_fitted", "abs_openfll_numpy",
        ],
        (
            [
                concentrations[index], *surface_X[index], openfll_surface[index],
                numpy_surface[index], regressor_surface[index],
                abs(openfll_surface[index] - numpy_surface[index]),
            ]
            for index in range(len(surface_X))
        ),
    )
    write_csv(
        results_dir / "simpful_surface_sample.csv",
        ["surface_index", "concentration_mass_pct", "P", "a", "c", "KIC_openfll", "KIC_simpful", "abs_diff"],
        (
            [
                int(index), concentrations[index], *surface_X[index], openfll_surface[index],
                simpful_surface[position], abs(openfll_surface[index] - simpful_surface[position]),
            ]
            for position, index in enumerate(simpful_indices)
        ),
    )
    write_csv(
        results_dir / "leave_one_out_predictions.csv",
        ["record_id", "concentration_mass_pct", "P", "a", "c", "KIC_observed", "KIC_predicted", "abs_error"],
        (
            [
                item.record_id, item.concentration_mass_pct, item.P, item.a, item.c, item.KIC,
                leave_one_out_predictions[index], abs(item.KIC - leave_one_out_predictions[index]),
            ]
            for index, item in enumerate(records)
        ),
    )

    plots = save_plots(results_dir, surface_X, concentrations, openfll_surface, numpy_surface, grid_size)
    surface_statistics = {
        f"{concentration:.1f}": {
            "mean": float(np.mean(openfll_surface[concentrations == concentration])),
            "std": float(np.std(openfll_surface[concentrations == concentration])),
            "min": float(np.min(openfll_surface[concentrations == concentration])),
            "max": float(np.max(openfll_surface[concentrations == concentration])),
        }
        for concentration in sorted(set(float(value) for value in concentrations))
    }
    summary: dict[str, Any] = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "openfll_engine_module": type(openfll_engine).__module__,
            "simpful": "2.12.0",
        },
        "dataset": {
            "records": len(records),
            "porosity_values": sorted({item.P for item in records}),
            "concentrations_mass_pct": sorted({item.concentration_mass_pct for item in records}),
        },
        "model": {
            "rules": len(model.rules),
            "p_terms": len(model.p_terms),
            "a_terms": len(model.a_terms),
            "c_terms": len(model.c_terms),
            "grid_size_per_surface": grid_size,
            "surface_points_total": len(surface_X),
            "simpful_surface_sample_points": int(len(simpful_indices)),
        },
        "fixed_model_training_metrics": metric_summary(y_train, openfll_train),
        "fixed_model_leave_one_out_metrics": metric_summary(y_train, leave_one_out_predictions),
        "fitted_regressor_training_metrics": metric_summary(y_train, regressor_train),
        "fitted_regressor": {
            "rank": int(regressor.rank_),
            "coefficient_count": int(len(regressor.coef_)),
            "coefficient_min": float(np.min(regressor.coef_)),
            "coefficient_max": float(np.max(regressor.coef_)),
        },
        "parity": {
            "training_openfll_vs_simpful": parity_summary(openfll_train, simpful_train),
            "training_openfll_vs_numpy": parity_summary(openfll_train, numpy_train),
            "surface_sample_openfll_vs_simpful": parity_summary(openfll_surface[simpful_indices], simpful_surface),
            "surface_openfll_vs_numpy": parity_summary(openfll_surface, numpy_surface),
            "simpful_scalar_vs_strengths_first_5": parity_summary(simpful_scalar_check, simpful_train[:5]),
        },
        "surface_ranges": {
            "openfll_min": float(np.nanmin(openfll_surface)),
            "openfll_max": float(np.nanmax(openfll_surface)),
            "openfll_nonfinite": int((~np.isfinite(openfll_surface)).sum()),
            "simpful_sample_nonfinite": int((~np.isfinite(simpful_surface)).sum()),
            "regressor_min": float(np.nanmin(regressor_surface)),
            "regressor_max": float(np.nanmax(regressor_surface)),
            "regressor_nonfinite": int((~np.isfinite(regressor_surface)).sum()),
        },
        "fixed_model_surface_statistics_by_concentration": surface_statistics,
        "timings": timings,
        "plots": plots,
        "limitations": [
            "The supplied article does not include the optimized Gaussian centers and sigmas.",
            "This run uses deterministic neighbor-based initial widths, not the article's genetic optimization.",
            "Training metrics are in-sample reproduction metrics and do not measure unseen physical concentrations.",
            "Leave-one-out validation holds out observations but still evaluates only the three experimentally measured porosity levels.",
            "Simpful surface parity uses a deterministic stratified sample because its rule evaluation is scalar and much slower than OpenFLL batch inference.",
        ],
    }
    (results_dir / "model_definition.json").write_text(
        json.dumps(model_to_json(model), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=experiment_dir / "article_clean_26.csv")
    parser.add_argument("--results-dir", type=Path, default=experiment_dir / "results")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--simpful-surface-sample", type=int, default=DEFAULT_SIMPFUL_SURFACE_SAMPLE)
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")
    if args.simpful_surface_sample < 1:
        parser.error("--simpful-surface-sample must be positive")
    summary = run_experiment(args.data, args.results_dir, args.grid_size, args.simpful_surface_sample)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
