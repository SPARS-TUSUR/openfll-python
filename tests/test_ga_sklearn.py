from __future__ import annotations

import pickle

import numpy as np
import pytest

import openfll
import PyFLL
from openfll.sklearn import SugenoGARegressor, SugenoModelSpec, SugenoRegressor


@pytest.fixture
def ga_backend():
    if not hasattr(openfll.SugenoEngine, "fit_ga"):
        pytest.skip("The legacy bundled backend does not have native GA bindings")


@pytest.fixture
def ga_model(ga_backend):
    X = np.linspace(0.0, 10.0, 32).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    spec = SugenoModelSpec(
        inputs={"x": [("low", "triangular", [-5.0, 0.0, 10.0]),
                      ("high", "triangular", [0.0, 10.0, 15.0])]},
        rules=['IF x IS "low" THEN y IS "low_value"',
               'IF x IS "high" THEN y IS "high_value"'],
    )
    return X, y, spec


@pytest.fixture
def constant_regressor(ga_backend):
    spec = SugenoModelSpec(
        inputs={"x": [("active", "constant", [1.0])]},
        rules=['IF x IS "active" THEN y IS "value"'],
    )
    return SugenoGARegressor(
        spec, mf_bounds={("x", "active"): [(0.5, 1.5)]},
        population_size=32, generations=60, random_state=42,
    )


def test_public_native_ga_methods(ga_model, record_property):
    X, y, spec = ga_model
    assert openfll.SugenoEngine is PyFLL.SugenoEngine
    assert openfll.SugenoEngine.__module__ == "PyFLL._core"

    engine = openfll.SugenoEngine()
    engine.add_input_var("x")
    engine.add_output_var("y")
    for name, mf_type, params in spec.inputs["x"]:
        engine.add_membership_func("x", name, mf_type, params)
    for name in ("low_value", "high_value"):
        engine.add_membership_func("y", name, "constant", [0.0])
    for rule in spec.rules:
        engine.add_rule(rule)
    engine.build()
    engine.set_mf_bounds("y", "low_value", [(0.0, 5.0)])
    start = np.mean((engine.predict_batch(X, input_names=["x"]) - y) ** 2)
    history = engine.fit_ga(X, y, input_names=["x"], population_size=32, generations=40, seed=42)
    final = np.mean((engine.predict_batch(X, input_names=["x"]) - y) ** 2)
    record_property("initial_model_mse", start)
    record_property("first_generation_mse", history[0])
    record_property("last_generation_mse", final)
    assert len(history) == 40
    assert np.all(np.diff(history) <= 0)
    assert final == pytest.approx(history[-1])
    assert final < start / 10


def test_ga_regressor_reproducible_refit_clone_and_pickle(ga_model):
    sklearn = pytest.importorskip("sklearn.base")
    X, y, spec = ga_model
    reg = SugenoGARegressor(
        spec, generations=20, population_size=20, random_state=17,
        mf_bounds={("y", "low_value"): [(0.0, 5.0)]},
    )
    assert reg.fit(X, y) is reg
    first_engine = reg.engine_
    first_predictions = reg.predict(X)
    first_history = reg.loss_history_.copy()

    cloned = sklearn.clone(reg)
    assert not hasattr(cloned, "engine_")
    assert cloned.mf_bounds == reg.mf_bounds
    assert cloned.random_state == 17

    reg.fit(X, y)
    assert reg.engine_ is not first_engine
    np.testing.assert_array_equal(reg.loss_history_, first_history)
    np.testing.assert_array_equal(reg.predict(X), first_predictions)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(reg)).predict(X), first_predictions)


def test_plot_shows_optimized_consequents(ga_model):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    X, y, spec = ga_model
    engine = SugenoGARegressor(spec, population_size=32, generations=40).fit(X, y).engine_
    terms = engine.to_dict()["extra"]["linear_api"]["membership_functions"]
    values = [term["parameters"][0] for term in terms if term["variable"] == "y"]
    assert any(value != 0.0 for value in values)
    restored = openfll.load_from_string(engine.save_to_string())
    for model in (engine, restored):
        plot = model.plot_mf("y", show=False)["y"]
        try:
            assert [line.get_xdata()[0] for line in plot.axes.lines] == pytest.approx(values)
        finally:
            plt.close(plot.figure)


@pytest.mark.parametrize("randomized", [False, True])
def test_ga_regressor_hyperparameter_search(ga_backend, randomized):
    model_selection = pytest.importorskip("sklearn.model_selection")
    X = np.linspace(0.0, 10.0, 32).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    # Gaussian MFs have global support so holdout edge points never drop to 0 activation (NaN).
    spec = SugenoModelSpec(
        inputs={"x": [("low", "gaussian", [0.0, 4.0]), ("high", "gaussian", [10.0, 4.0])]},
        rules=['IF x IS "low" THEN y IS "low_value"', 'IF x IS "high" THEN y IS "high_value"'],
    )
    estimator = SugenoGARegressor(spec, generations=6, population_size=12, random_state=12)
    candidates = {"mutation_rate": [0.15, 0.3], "mutation_sigma": [0.1, 0.2]}
    cv = model_selection.KFold(n_splits=3, shuffle=True, random_state=42)
    options = {"scoring": "neg_mean_squared_error", "cv": cv, "error_score": "raise"}
    if randomized:
        search = model_selection.RandomizedSearchCV(
            estimator, candidates, n_iter=2, random_state=42, **options,
        )
    else:
        search = model_selection.GridSearchCV(estimator, candidates, **options)
    search.fit(X, y)

    assert np.isfinite(search.best_score_)
    assert len(search.best_estimator_.loss_history_) == 6
    assert search.predict(X).shape == y.shape
    assert not hasattr(estimator, "engine_")


@pytest.mark.parametrize(
    ("loss", "optimal_prediction", "optimal_loss"),
    [("mse", 2.0, 17.2), ("rmse", 2.0, np.sqrt(17.2)), ("mae", 1.0, 3.4)],
)
def test_ga_loss_matches_analytic_optimum(
    constant_regressor, loss, optimal_prediction, optimal_loss, record_property,
):
    # A constant MSE/RMSE fit targets the mean. MAE targets the median.
    y = np.array([-2.0, -2.0, 1.0, 4.0, 9.0])
    X = np.arange(len(y)).reshape(-1, 1)
    constant_regressor.loss = loss
    constant_regressor.fit(X, y)
    predictions = constant_regressor.predict(X)
    record_property("prediction", predictions[0])
    record_property("loss", constant_regressor.loss_history_[-1])

    np.testing.assert_allclose(predictions, optimal_prediction, atol=0.02, rtol=0)
    assert constant_regressor.loss_history_[-1] == pytest.approx(optimal_loss, abs=0.01)


@pytest.mark.parametrize(("bounds", "expected"), [((1.0, 2.0), 1.0), ((-2.0, -1.0), -1.0)])
def test_ga_uses_bounds_when_the_optimum_is_outside(
    constant_regressor, bounds, expected, record_property,
):
    X = np.arange(8).reshape(-1, 1)
    y = np.zeros(len(X))
    constant_regressor.mf_bounds = {**constant_regressor.mf_bounds, ("y", "value"): [bounds]}
    constant_regressor.fit(X, y)
    predictions = constant_regressor.predict(X)
    record_property("prediction", predictions[0])
    record_property("mse", constant_regressor.loss_history_[-1])

    np.testing.assert_allclose(predictions, expected, atol=1e-12, rtol=0)
    assert constant_regressor.loss_history_[-1] == pytest.approx(1.0)


def test_ga_cross_validation_fits_each_training_fold(constant_regressor, record_property):
    model_selection = pytest.importorskip("sklearn.model_selection")
    X = np.arange(15).reshape(-1, 1)
    y = np.repeat([-4.0, 0.0, 8.0], 5)
    folds = list(model_selection.KFold(n_splits=3).split(X))
    result = model_selection.cross_validate(
        constant_regressor, X, y, cv=folds,
        scoring="neg_mean_squared_error", return_estimator=True, error_score="raise",
    )
    assert not hasattr(constant_regressor, "engine_")
    estimators = result["estimator"]
    assert len({id(model.engine_) for model in estimators}) == len(folds)

    for index, (model, (train, test)) in enumerate(zip(estimators, folds), start=1):
        expected = y[train].mean()
        predictions = model.predict(X[test])
        record_property(f"fold_{index}_prediction", predictions[0])
        np.testing.assert_allclose(predictions, expected, atol=0.03, rtol=0)


@pytest.mark.parametrize("seed", [7, 19, 42])
def test_ga_can_recover_from_identical_input_memberships(ga_backend, seed, record_property):
    # With identical input MFs, fitting the consequents alone can only learn a constant.
    spec = SugenoModelSpec(
        inputs={"x": [("low", "gaussian", [0.0, 0.6]), ("high", "gaussian", [0.0, 0.6])]},
        rules=['IF x IS "low" THEN y IS "left"', 'IF x IS "high" THEN y IS "right"'],
    )
    X = np.linspace(-1.0, 1.0, 101).reshape(-1, 1)
    y = np.tanh(3.0 * X[:, 0])
    X_train, X_test = X[::2], X[1::2]
    y_train, y_test = y[::2], y[1::2]

    baseline = SugenoRegressor(spec).fit(X_train, y_train)
    ga = SugenoGARegressor(
        spec, population_size=40, generations=80, random_state=seed,
        mf_bounds={
            ("x", "low"): [(-1.0, 1.0), (0.35, 1.0)],
            ("x", "high"): [(-1.0, 1.0), (0.35, 1.0)],
        },
    ).fit(X_train, y_train)
    baseline_rmse = np.sqrt(np.mean((baseline.predict(X_test) - y_test) ** 2))
    ga_rmse = np.sqrt(np.mean((ga.predict(X_test) - y_test) ** 2))
    record_property("baseline_rmse", baseline_rmse)
    record_property("ga_rmse", ga_rmse)

    assert baseline.rank_ == 1
    assert ga_rmse < 0.35 * baseline_rmse


def test_failed_native_refit_preserves_the_trained_model(ga_model):
    X, y, spec = ga_model
    estimator = SugenoGARegressor(spec, generations=12, population_size=16).fit(X, y)
    engine = estimator.engine_
    predictions = estimator.predict(X)
    history = estimator.loss_history_.copy()

    estimator.generations = 0
    with pytest.raises(ValueError, match="generations"):
        estimator.fit(X, y)

    assert estimator.engine_ is engine
    assert estimator.loss_history_ == history
    np.testing.assert_array_equal(estimator.predict(X), predictions)


def test_ga_offset_target_converges_tightly(ga_model):
    X, _, spec = ga_model
    y_offset = 1000.0 + 2.0 * X[:, 0]
    reg = SugenoGARegressor(spec, population_size=40, generations=60, random_state=42).fit(X, y_offset)
    rmse = float(np.sqrt(np.mean((reg.predict(X) - y_offset) ** 2)))
    assert reg.loss_history_[0] < 100.0
    assert rmse < 0.5


def test_ga_regressor_parameter_validation(ga_model):
    X, y, spec = ga_model
    bad_configs = [
        {"random_state": True},
        {"random_state": -1},
        {"crossover_rate": True},
        {"mutation_rate": True},
        {"mutation_sigma": True},
        {"population_size": True},
        {"population_size": -1},
        {"population_size": 1},
        {"generations": True},
        {"generations": 0},
        {"tournament_size": 0},
        {"elitism_count": -1},
        {"elitism_count": 10, "population_size": 10},
        {"defuzzer": None},
        {"defuzzer": "bad"},
        {"loss": "unknown"},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"mutation_sigma": 0.0},
        {"mf_bounds": {"xy": [(0.0, 1.0)]}},
    ]
    for kwargs in bad_configs:
        with pytest.raises(ValueError):
            SugenoGARegressor(spec, **kwargs).fit(X, y)

    r_rs1 = SugenoGARegressor(spec, population_size=12, generations=5, random_state=np.random.RandomState(42)).fit(X, y)
    r_rs2 = SugenoGARegressor(spec, population_size=12, generations=5, random_state=np.random.RandomState(42)).fit(X, y)
    assert r_rs1.loss_history_ == r_rs2.loss_history_

    r_gen1 = SugenoGARegressor(spec, population_size=12, generations=5, random_state=np.random.default_rng(42)).fit(X, y)
    r_gen2 = SugenoGARegressor(spec, population_size=12, generations=5, random_state=np.random.default_rng(42)).fit(X, y)
    assert r_gen1.loss_history_ == r_gen2.loss_history_


def test_ga_regressor_wtsum_and_pickle(ga_model):
    X, y, spec = ga_model
    reg = SugenoGARegressor(spec, defuzzer="wtsum", population_size=20, generations=15, random_state=42).fit(X, y)
    preds = reg.predict(X)
    restored = pickle.loads(pickle.dumps(reg))
    assert restored.engine_.get_defuzzer() == "wtsum"
    np.testing.assert_allclose(restored.predict(X), preds, atol=1e-12, rtol=0)


def test_ga_regressor_parallel_grid_search(ga_backend):
    joblib = pytest.importorskip("joblib")
    model_selection = pytest.importorskip("sklearn.model_selection")
    X = np.linspace(0.0, 10.0, 32).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    spec = SugenoModelSpec(
        inputs={"x": [("low", "gaussian", [0.0, 4.0]), ("high", "gaussian", [10.0, 4.0])]},
        rules=['IF x IS "low" THEN y IS "low_value"', 'IF x IS "high" THEN y IS "high_value"'],
    )
    cv = model_selection.KFold(n_splits=3, shuffle=True, random_state=42)
    grid = {"mutation_rate": [0.1, 0.3]}

    gs_loky = model_selection.GridSearchCV(
        SugenoGARegressor(spec, population_size=16, generations=10, random_state=42),
        grid, cv=cv, n_jobs=2, error_score="raise",
    ).fit(X, y)

    gs_thread = model_selection.GridSearchCV(
        SugenoGARegressor(spec, population_size=16, generations=10, random_state=42),
        grid, cv=cv, n_jobs=2, error_score="raise",
    )
    with joblib.parallel_backend("threading", n_jobs=2):
        gs_thread.fit(X, y)

    assert np.isfinite(gs_loky.best_score_)
    assert gs_thread.best_score_ == pytest.approx(gs_loky.best_score_)
    np.testing.assert_allclose(gs_thread.predict(X), gs_loky.predict(X), atol=1e-12, rtol=0)


def test_public_mamdani_native_fit_ga(ga_backend):
    engine = openfll.MamdaniEngine()
    engine.add_input_var("x")
    engine.add_output_var("y")
    engine.add_membership_func("x", "low", "triangular", [-2.0, 0.0, 6.0])
    engine.add_membership_func("x", "high", "triangular", [4.0, 10.0, 12.0])
    engine.add_membership_func("y", "y_low", "triangular", [0.0, 2.0, 4.0])
    engine.add_membership_func("y", "y_high", "triangular", [6.0, 8.0, 10.0])
    engine.set_integrator(0.0, 10.0, 0.1)
    engine.add_rule('IF x IS "low" THEN y IS "y_low"')
    engine.add_rule('IF x IS "high" THEN y IS "y_high"')
    engine.build()

    X = np.array([[0.0], [5.0], [10.0]], dtype=np.float64)
    y = np.array([1.5, 5.0, 8.5], dtype=np.float64)
    history = engine.fit_ga(X, y, input_names=["x"], population_size=12, generations=10, seed=42)
    assert len(history) == 10
    assert np.all(np.diff(history) <= 0)
    assert history[-1] < history[0]


def test_plot_mf_handles_uppercase_mf_type_after_ga(ga_backend):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    engine = openfll.SugenoEngine()
    engine.add_input_var("x")
    engine.add_output_var("y")
    engine.add_membership_func("x", "g", "Gaussian", [0.0, 2.0])
    engine.add_membership_func("y", "c", "Constant", [0.0])
    engine.add_rule('IF x IS "g" THEN y IS "c"')
    engine.build()
    engine.fit_ga(np.array([[0.0], [2.0]]), np.array([3.0, 3.0]), input_names=["x"], population_size=8, generations=5, seed=42)
    plots = engine.plot_mf(show=False)
    try:
        assert set(plots) == {"x", "y"}
    finally:
        for plot in plots.values():
            plt.close(plot.figure)
