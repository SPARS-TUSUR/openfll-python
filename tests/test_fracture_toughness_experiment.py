from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.fracture_toughness.run_experiment import (
    build_model_definition,
    build_openfll_engine,
    build_simpful_system,
    concentration_to_porosity,
    create_surface_points,
    error_breakdown_rows,
    error_diagnostics,
    evaluate_experimental_cross_validation,
    evaluate_experimental_holdout,
    load_observations,
    load_niihara_reference,
    load_niihara_workbooks,
    metric_summary_by_porosity,
    observations_to_arrays,
    porosity_axis,
    predict_numpy,
    predict_openfll,
    predict_simpful_from_strengths,
    predict_simpful_scalar,
    repeated_stratified_folds,
    stratified_holdout_indices,
)


DATA_PATH = Path(__file__).parents[1] / "experiments" / "fracture_toughness" / "article_clean_26.csv"


def _model_data():
    records = load_observations(DATA_PATH)
    return records, build_model_definition(records)


def test_article_cleaned_dataset_and_rule_topology():
    records, model = _model_data()
    assert len(records) == 26
    assert sorted({record.P for record in records}) == [7.96274, 23.01, 27.3715]
    assert len(model.p_terms) == 3
    assert len(model.a_terms) == 26
    assert len(model.c_terms) == 26
    assert len(model.rules) == 26
    assert all(sigma > 0.0 for _, _, sigma in (*model.p_terms, *model.a_terms, *model.c_terms))
    assert all(rule.output_value > 0.0 for rule in model.rules)
    assert [rule.output_value for rule in model.rules] == [record.KIC for record in records]


def test_openfll_matches_independent_numpy_oracle_on_training_and_surface():
    records, model = _model_data()
    X_train, _ = observations_to_arrays(records)
    surface_X, _, _, _ = create_surface_points(records, grid_size=20)
    X = np.vstack([X_train, surface_X])
    engine = build_openfll_engine(model)
    actual = predict_openfll(engine, X)
    expected = predict_numpy(model, X)
    assert np.all(np.isfinite(actual))
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fixed_article_model_reproduces_reference_training_mse():
    records, model = _model_data()
    X_train, y_train = observations_to_arrays(records)
    engine = build_openfll_engine(model)
    predictions = predict_openfll(engine, X_train)
    mse = float(np.mean((predictions - y_train) ** 2))
    assert mse == pytest.approx(0.0034064026864190082, rel=1e-12, abs=1e-15)


def test_intermediate_concentrations_map_inside_measured_porosity_range():
    records, _ = _model_data()
    concentrations = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    porosity = concentration_to_porosity(records, concentrations)
    assert porosity[[0, 1, 5]].tolist() == pytest.approx([27.3715, 23.01, 7.96274])
    assert np.all(np.diff(porosity) < 0.0)
    sweep = porosity_axis(records)
    assert sweep[0] == pytest.approx(7.96274)
    assert sweep[-1] == pytest.approx(27.3715)


def test_external_niihara_reference_loader(tmp_path):
    reference_path = tmp_path / "niihara.csv"
    reference_path.write_text(
        "P,a,c,KIC\n27.3715,14.0,48.0,0.75\n7.96274,39.0,109.0,1.64\n",
        encoding="utf-8",
    )
    X, y = load_niihara_reference(reference_path)
    assert X.shape == (2, 3)
    assert y.tolist() == pytest.approx([0.75, 1.64])
    assert np.all(np.isfinite(X))


def test_external_niihara_workbook_pair_loader(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    predictors_path = tmp_path / "predictors.xlsx"
    response_path = tmp_path / "response.xlsx"

    predictors = openpyxl.Workbook()
    predictors.active.append(["a", "c", "P", "c/a"])
    predictors.active.append([14.0, 48.0, 27.3715, 48.0 / 14.0])
    predictors.active.append([39.0, 109.0, 7.96274, 109.0 / 39.0])
    predictors.save(predictors_path)

    response = openpyxl.Workbook()
    response.active.append(["Kc"])
    response.active.append([0.75])
    response.active.append([1.64])
    response.save(response_path)

    X, y = load_niihara_workbooks(predictors_path, response_path)
    assert np.allclose(X, [[27.3715, 14.0, 48.0], [7.96274, 39.0, 109.0]])
    assert y.tolist() == pytest.approx([0.75, 1.64])


def test_external_niihara_workbook_pair_rejects_misaligned_rows(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    predictors_path = tmp_path / "predictors.xlsx"
    response_path = tmp_path / "response.xlsx"

    predictors = openpyxl.Workbook()
    predictors.active.append(["a", "c", "P", "c/a"])
    predictors.active.append([14.0, 48.0, 27.3715, 48.0 / 14.0])
    predictors.active.append([39.0, 109.0, 7.96274, 109.0 / 39.0])
    predictors.save(predictors_path)

    response = openpyxl.Workbook()
    response.active.append(["Kc"])
    response.active.append([0.75])
    response.save(response_path)

    with pytest.raises(ValueError, match="different row counts"):
        load_niihara_workbooks(predictors_path, response_path)


def test_theoretical_metrics_are_grouped_by_porosity():
    X = np.asarray(
        [
            [7.0, 14.0, 48.0],
            [7.0, 15.0, 49.0],
            [23.0, 14.0, 48.0],
            [23.0, 15.0, 49.0],
        ]
    )
    y_true = np.asarray([1.0, 2.0, 1.0, 3.0])
    y_pred = np.asarray([1.0, 2.0, 2.0, 3.0])
    metrics = metric_summary_by_porosity(X, y_true, y_pred)
    assert [row["P"] for row in metrics] == [7.0, 23.0]
    assert metrics[0]["mse"] == pytest.approx(0.0)
    assert metrics[1]["mse"] == pytest.approx(0.5)


def test_error_diagnostics_and_breakdown_identify_direction_and_worst_point():
    X = np.asarray(
        [
            [7.0, 10.0, 40.0],
            [7.0, 20.0, 50.0],
            [23.0, 30.0, 60.0],
            [23.0, 40.0, 70.0],
        ]
    )
    y_true = np.asarray([1.0, 1.0, 1.0, 1.0])
    y_pred = np.asarray([0.9, 1.0, 1.2, 1.5])
    diagnostics = error_diagnostics(X, y_true, y_pred)
    assert diagnostics["mean_residual_bias"] == pytest.approx(0.15)
    assert diagnostics["share_abs_error_le_0_10"] == pytest.approx(0.5)
    assert diagnostics["worst_point"] == pytest.approx(
        {
            "P": 23.0,
            "a": 40.0,
            "c": 70.0,
            "KIC_reference": 1.0,
            "KIC_predicted": 1.5,
            "residual": 0.5,
            "abs_error": 0.5,
        }
    )
    rows = error_breakdown_rows(X, y_true, y_pred, bins=2)
    assert len(rows) == 6
    assert [row["grouping"] for row in rows[:2]] == ["P", "P"]
    assert sum(row["n"] for row in rows if row["grouping"] == "a_bin") == 4


def test_experimental_holdout_is_stratified_reproducible_and_leak_free():
    records, _ = _model_data()
    train_indices, test_indices = stratified_holdout_indices(records)
    train_again, test_again = stratified_holdout_indices(records)
    assert train_indices.tolist() == train_again.tolist()
    assert test_indices.tolist() == test_again.tolist()
    assert len(train_indices) == 20
    assert len(test_indices) == 6
    assert not np.intersect1d(train_indices, test_indices).size
    assert sorted(records[index].P for index in test_indices) == pytest.approx(
        [7.96274, 7.96274, 23.01, 23.01, 27.3715, 27.3715]
    )

    summary, rows = evaluate_experimental_holdout(records)
    assert summary["train_records"] == 20
    assert summary["test_records"] == 6
    assert summary["test_counts_by_porosity"] == {"7.96274": 2, "23.01": 2, "27.3715": 2}
    assert len(rows) == 6
    assert summary["metrics"]["finite"] == 6
    assert summary["implementation_parity_openfll_vs_numpy"]["max_abs_diff"] < 1e-12


def test_repeated_stratified_cross_validation_covers_every_record_without_leakage():
    records, _ = _model_data()
    folds = repeated_stratified_folds(records, n_splits=5, n_repeats=3)
    assert len(folds) == 15
    for repeat in range(1, 4):
        repeat_tests = []
        for current_repeat, _, train_indices, test_indices in folds:
            if current_repeat != repeat:
                continue
            assert not np.intersect1d(train_indices, test_indices).size
            assert set(records[index].P for index in test_indices) == {7.96274, 23.01, 27.3715}
            repeat_tests.extend(test_indices.tolist())
        assert sorted(repeat_tests) == list(range(26))

    summary, prediction_rows, fold_rows, repeat_rows = evaluate_experimental_cross_validation(
        records, n_splits=5, n_repeats=3
    )
    assert summary["fitted_models"] == 15
    assert summary["held_out_predictions"] == 78
    assert set(summary["predictions_per_record"].values()) == {3}
    assert summary["aggregate_metrics"]["finite"] == 78
    assert summary["max_openfll_numpy_abs_diff"] < 1e-12
    assert len(prediction_rows) == 78
    assert len(fold_rows) == 15
    assert len(repeat_rows) == 3


def test_simpful_matches_openfll_and_its_scalar_sugeno_inference():
    pytest.importorskip("simpful")
    records, model = _model_data()
    X_train, _ = observations_to_arrays(records)
    surface_X, _, _, _ = create_surface_points(records, grid_size=6)
    X = np.vstack([X_train, surface_X])
    engine = build_openfll_engine(model)
    system = build_simpful_system(model)
    openfll_values = predict_openfll(engine, X)
    simpful_values = predict_simpful_from_strengths(system, model, X)
    assert np.all(np.isfinite(simpful_values))
    assert np.allclose(openfll_values, simpful_values, rtol=1e-11, atol=1e-11)
    scalar_values = predict_simpful_scalar(system, X_train[:5])
    assert np.allclose(scalar_values, simpful_values[:5], rtol=1e-12, atol=1e-12)
