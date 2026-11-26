from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from experiments.fracture_toughness.ga_evaluation import FractureGARegressor, cv_splits, make_search
from experiments.fracture_toughness.run_experiment import (
    INPUT_NAMES, OUTPUT_NAME, ModelDefinition, RuleDefinition,
    load_observations, observations_to_arrays, predict_numpy, stratified_holdout_indices,
)


@pytest.fixture
def fracture_data():
    import openfll
    if not hasattr(openfll.SugenoEngine, "fit_ga"):
        pytest.skip("Requires a GA-enabled native backend")
    path = Path(__file__).parents[1] / "experiments/fracture_toughness/article_clean_26.csv"
    records = load_observations(path)
    return records, *observations_to_arrays(records)


@pytest.mark.parametrize("joint", [False, True])
def test_fracture_ga_respects_bounds_and_preserves_physical_initialization(fracture_data, joint):
    records, X, y = fracture_data
    train, test = stratified_holdout_indices(records)
    model = FractureGARegressor(train_consequents=joint, population_size=12, generations=8).fit(X[train], y[train])
    np.testing.assert_array_equal(model.training_X_, X[train])
    np.testing.assert_array_equal(model.training_y_, y[train])
    assert len(model.definition_.rules) == len(train)
    assert [rule.output_value for rule in model.definition_.rules] == list(y[train])
    assert model.loss_history_[-1] <= model.initial_train_mse_ + 1e-12
    assert np.all(np.diff(model.loss_history_) <= 1e-12)
    assert np.all(np.isfinite(model.predict(X[test])))
    terms = model.engine_.to_dict()["extra"]["linear_api"]["membership_functions"]
    for term in terms:
        for value, (lower, upper) in zip(term["parameters"], model.bounds_[(term["variable"], term["name"])]):
            assert lower <= value <= upper
    if not joint:
        assert model.consequent_max_abs_change_ <= max(np.spacing(y[train]))


def test_fracture_ga_matches_numpy_after_training_and_json_loading(fracture_data):
    import openfll
    _, X, y = fracture_data
    estimator = FractureGARegressor(train_consequents=True, population_size=12, generations=8).fit(X, y)
    metadata = estimator.engine_.to_dict()["extra"]["linear_api"]["membership_functions"]
    terms = {name: [] for name in INPUT_NAMES}
    consequents = {}
    for term in metadata:
        if term["variable"] == OUTPUT_NAME:
            consequents[term["name"]] = term["parameters"][0]
        else:
            terms[term["variable"]].append((term["name"], *term["parameters"]))
    rules = tuple(RuleDefinition(rule.index, rule.p_term, rule.a_term, rule.c_term,
                                 rule.output_term, consequents[rule.output_term])
                  for rule in estimator.definition_.rules)
    trained = ModelDefinition(*(tuple(terms[name]) for name in INPUT_NAMES), rules)
    np.testing.assert_allclose(estimator.predict(X), predict_numpy(trained, X), rtol=1e-12, atol=1e-12)
    restored = openfll.load_from_string(estimator.engine_.save_to_string())
    np.testing.assert_allclose(estimator.predict(X), restored.predict_batch(X, input_names=INPUT_NAMES),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("method", ["grid", "random"])
def test_fracture_search_builds_rules_only_from_current_fit_data(fracture_data, method):
    records, X, y = fracture_data
    train, test = stratified_holdout_indices(records)
    for inner_train, inner_test in cv_splits(X[train], 3, 42):
        assert not np.intersect1d(inner_train, inner_test).size
        assert len(np.unique(X[train][inner_train, 0])) == 3
    search = make_search(method, train_consequents=False, seed=7,
                         folds=cv_splits(X[train], 3, 42), population_size=8, generations=3)
    search.fit(X[train], y[train])
    fitted = search.best_estimator_
    assert len(search.cv_results_["params"]) == 4
    assert len(fitted.definition_.rules) == len(train)
    assert not hasattr(clone(fitted), "engine_")
    assert not np.intersect1d(train, test).size
    np.testing.assert_array_equal(fitted.training_X_, X[train])
    assert np.isfinite(search.best_score_)


def test_restricted_model_changes_only_three_porosity_widths(fracture_data):
    _, X, y = fracture_data
    model = FractureGARegressor(width_variables=("P",), population_size=12, generations=8).fit(X, y)
    initial = {}
    for variable, terms in zip(INPUT_NAMES, (model.definition_.p_terms, model.definition_.a_terms, model.definition_.c_terms)):
        for name, center, sigma in terms:
            initial[(variable, name)] = [center, sigma]
    for rule in model.definition_.rules:
        initial[(OUTPUT_NAME, rule.output_term)] = [rule.output_value]
    free = 0
    terms = model.engine_.to_dict()["extra"]["linear_api"]["membership_functions"]
    for term in terms:
        key = (term["variable"], term["name"])
        for index, (current, previous) in enumerate(zip(term["parameters"], initial[key])):
            if term["variable"] == "P" and index == 1:
                free += 1
                assert 0.8 * previous <= current <= 1.2 * previous
            else:
                assert np.nextafter(previous, -np.inf) <= current <= np.nextafter(previous, np.inf)
    assert free == 3


def test_training_curve_subsets_preserve_strata_and_are_nested(fracture_data):
    from experiments.fracture_toughness.stability_evaluation import training_subset
    _, X, _ = fracture_data
    small = training_subset(X, 0.5, 17)
    middle = training_subset(X, 0.75, 17)
    full = training_subset(X, 1.0, 17)
    assert set(small).issubset(middle)
    assert set(middle).issubset(full)
    assert len(full) == len(X)
    assert len(np.unique(X[small, 0])) == 3


def test_restricted_penalty_history_matches_native_objective(fracture_data):
    import openfll
    if "deviation_penalty" not in openfll.SugenoEngine.fit_ga.__doc__:
        pytest.skip("Requires regularized native GA")
    _, X, y = fracture_data
    model = FractureGARegressor(
        width_variables=("P",), deviation_penalty=0.3, narrow_sigma_penalty=0.7,
        population_size=12, generations=8,
    ).fit(X, y)
    original = {name: sigma for name, _, sigma in model.definition_.p_terms}
    terms = model.engine_.to_dict()["extra"]["linear_api"]["membership_functions"]
    changes = np.array([np.log(term["parameters"][1] / original[term["name"]])
                        for term in terms if term["variable"] == "P"])
    mse = np.mean((model.predict(X) - y) ** 2)
    expected = mse + np.var(y) * (
        0.3 * np.mean(changes**2) + 0.7 * np.mean(np.minimum(changes, 0)**2)
    )
    assert model.loss_history_[-1] == pytest.approx(expected, abs=1e-12)


def test_raw_sensitivity_audit_identifies_the_nonfinite_inner_cv_point(fracture_data):
    import csv
    from experiments.fracture_toughness.stability_evaluation import diagnose_inner_cv

    path = Path(__file__).parents[1] / "experiments/fracture_toughness/stability_results/raw_data_audit.csv"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    X = np.array([[float(row[name]) for name in INPUT_NAMES] for row in rows])
    y = np.array([float(row["KIC"]) for row in rows])
    assert sum(row["status"] == "raw_only_unconfirmed" for row in rows) == 9
    failures = diagnose_inner_cv(X, y)
    assert len(failures) == 1
    assert failures[0]["c"] == pytest.approx(172.37)
    assert failures[0]["row_index"] == 12
