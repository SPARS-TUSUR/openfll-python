from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.fracture_toughness.run_experiment import (
    build_model_definition,
    build_openfll_engine,
    build_simpful_system,
    create_surface_points,
    leave_one_out_fixed_predictions,
    load_observations,
    observations_to_arrays,
    predict_numpy,
    predict_openfll,
    predict_simpful_from_strengths,
    predict_simpful_scalar,
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


def test_openfll_matches_independent_numpy_oracle_on_training_and_surface():
    records, model = _model_data()
    X_train, _ = observations_to_arrays(records)
    surface_X, _, _ = create_surface_points(records, grid_size=20)
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


def test_leave_one_out_predictions_are_finite_and_nontrivial():
    records, _ = _model_data()
    _, y_train = observations_to_arrays(records)
    predictions = leave_one_out_fixed_predictions(records)
    mse = float(np.mean((predictions - y_train) ** 2))
    assert predictions.shape == y_train.shape
    assert np.all(np.isfinite(predictions))
    assert mse == pytest.approx(0.01665686202785139, rel=1e-12, abs=1e-15)


def test_simpful_matches_openfll_and_its_scalar_sugeno_inference():
    pytest.importorskip("simpful")
    records, model = _model_data()
    X_train, _ = observations_to_arrays(records)
    surface_X, _, _ = create_surface_points(records, grid_size=6)
    X = np.vstack([X_train, surface_X])
    engine = build_openfll_engine(model)
    system = build_simpful_system(model)
    openfll_values = predict_openfll(engine, X)
    simpful_values = predict_simpful_from_strengths(system, model, X)
    assert np.all(np.isfinite(simpful_values))
    assert np.allclose(openfll_values, simpful_values, rtol=1e-11, atol=1e-11)
    scalar_values = predict_simpful_scalar(system, X_train[:5])
    assert np.allclose(scalar_values, simpful_values[:5], rtol=1e-12, atol=1e-12)
