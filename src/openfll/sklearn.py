"""scikit-learn совместимый регрессор на базе нечёткой модели Sugeno нулевого порядка.

Предоставляет `SugenoRegressor` — эстимейтор для обучения числовых констант
заключений правил методом наименьших квадратов (МНК / Ridge) и выполнения
пакетного инференса через C++ ядро OpenFLL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import openfll

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
except ImportError:
    # Заглушки для сред без установленного scikit-learn
    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return {
                "model_spec": getattr(self, "model_spec", None),
                "default_t_norm": getattr(self, "default_t_norm", None),
                "default_s_norm": getattr(self, "default_s_norm", None),
                "solver": getattr(self, "solver", "lstsq"),
                "alpha": getattr(self, "alpha", 0.0),
            }

        def set_params(self, **params: Any) -> Any:
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class RegressorMixin:  # type: ignore[no-redef]
        def score(self, X: Any, y: Any) -> float:
            y_pred = getattr(self, "predict")(X)
            y_true = np.asarray(y, dtype=np.float64)
            u = float(np.sum((y_true - y_pred) ** 2))
            v = float(np.sum((y_true - np.mean(y_true)) ** 2))
            if v == 0.0:
                return 0.0
            return 1.0 - (u / v)

    class NotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
        pass

    def check_is_fitted(estimator: Any, attributes: Sequence[str] | str) -> None:  # type: ignore[misc]
        attrs = [attributes] if isinstance(attributes, str) else attributes
        if not all(hasattr(estimator, a) for a in attrs):
            raise NotFittedError(f"This {type(estimator).__name__} instance is not fitted yet.")

    def check_array(X: Any, **kwargs: Any) -> np.ndarray:  # type: ignore[misc]
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.ndim}D array instead")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64')")
        return arr

    def check_X_y(X: Any, y: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[misc]
        X_arr = check_array(X)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise ValueError(f"Expected 1D target array, got ndim={y_arr.ndim}")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64')")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: [{X_arr.shape[0]}, {y_arr.shape[0]}]"
            )
        return X_arr, y_arr


__all__ = ["FrozenDict", "SugenoModelSpec", "SugenoRegressor"]

_THEN_RE = re.compile(
    r"\bTHEN\b\s+(?:\w+)\s+\bIS\b\s+[\"']?([^\"'\s,;]+)[\"']?",
    re.IGNORECASE,
)


class FrozenDict(Mapping[str, Any]):
    """Неизменяемый словарь (immutable mapping) с поддержкой pickle, hash и unpack (**)."""

    def __init__(
        self,
        data: Mapping[Any, Any] | Sequence[tuple[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        raw = dict(data or {}, **kwargs)
        self._data: dict[Any, Any] = {k: _deep_freeze_structure(v) for k, v in raw.items()}
        self._hash: int | None = None

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Any:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(f"'{type(self).__name__}' object does not support item assignment")

    def __delitem__(self, key: Any) -> None:
        raise TypeError(f"'{type(self).__name__}' object does not support item deletion")

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (self._data,))

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def copy(self) -> dict[Any, Any]:
        return self._data.copy()


def _deep_freeze_structure(obj: Any) -> Any:
    """Рекурсивно преобразует коллекции в неизменяемые структуры (tuple, frozenset и FrozenDict)."""
    if isinstance(obj, FrozenDict):
        return obj
    if isinstance(obj, Mapping):
        return FrozenDict(obj)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return tuple(_deep_freeze_structure(x) for x in obj.tolist())
    if isinstance(obj, (set, frozenset)):
        return frozenset(_deep_freeze_structure(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze_structure(x) for x in obj)
    return obj


@dataclass(frozen=True)
class SugenoModelSpec:
    """Декларативная полностью неизменяемая спецификация структуры модели Sugeno.

    Корневой класс заморожен (@dataclass(frozen=True)), а все вложенные словари,
    списки, NumPy-массивы и множества при инициализации глубоко замораживаются
    в неизменяемые структуры (FrozenDict, tuple, frozenset), гарантируя полную
    иммутабельность и защиту от рассинхронизации инференса обученного регрессора.

    Attributes:
        inputs: неизменяемый словарь (FrozenDict) соответствия имени входа кортежу MF:
            (mf_name, mf_type, params).
        rules: кортеж текстовых правил, словарей (FrozenDict) или кортежей правил.
        output_name: имя целевой выходной переменной.
        consequents: явный кортеж имён констант заключений (если None, извлекаются из rules).
        default_t_norm: t-норма по умолчанию для правил без явного указания.
        default_s_norm: s-норма по умолчанию.
    """

    inputs: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    rules: Sequence[Any] = field(default_factory=tuple)
    output_name: str = "y"
    consequents: Sequence[str] | None = None
    default_t_norm: str | None = None
    default_s_norm: str | None = None

    def __post_init__(self) -> None:
        frozen_inputs = FrozenDict(
            {
                str(var): tuple(_deep_freeze_structure(mf) for mf in mfs)
                for var, mfs in self.inputs.items()
            }
        )
        object.__setattr__(self, "inputs", frozen_inputs)
        object.__setattr__(
            self,
            "rules",
            tuple(_deep_freeze_structure(r) for r in self.rules),
        )
        if self.consequents is not None:
            object.__setattr__(self, "consequents", tuple(str(c) for c in self.consequents))


class SugenoRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn совместимый регрессор на базе нечёткой модели Sugeno нулевого порядка.

    Обучает числовые константы заключений правил фиксированной нечёткой структуры
    методом наименьших квадратов (МНК или Ridge) с использованием нативного
    C++ ядра OpenFLL.

    Параметры:
        model_spec: Декларативная спецификация модели (SugenoModelSpec или dict).
        default_t_norm: t-норма по умолчанию для нечётких правил ("min", "prod_and", ...).
        default_s_norm: s-норма по умолчанию ("max", "algebraic_sum", ...).
        solver: Алгоритм решения задачи регрессии ("lstsq" или "ridge").
        alpha: Параметр регуляризации (L2) для Ridge (требуется alpha > 0 при solver='ridge').

    Атрибуты после обучения:
        engine_: Обученный нативный экземпляр openfll.SugenoEngine.
        coef_: 1D массив подобранных констант заключений правил (np.float64).
        consequents_: Список имён констант заключений, соответствующих coef_.
        rule_names_: Список имён правил ('rule_0', 'rule_1', ...).
        n_features_in_: Число входных признаков обучающей выборки.
        feature_names_in_: Имена признаков (если передавались через DataFrame).
        rank_: Численный ранг матрицы влияния правил (design matrix).
    """

    def __init__(
        self,
        model_spec: Mapping[str, Any] | SugenoModelSpec | None = None,
        *,
        default_t_norm: str | None = None,
        default_s_norm: str | None = None,
        solver: str = "lstsq",
        alpha: float = 0.0,
    ) -> None:
        self.model_spec = model_spec
        self.default_t_norm = default_t_norm
        self.default_s_norm = default_s_norm
        self.solver = solver
        self.alpha = alpha

    def __getstate__(self) -> dict[str, Any]:
        """Возвращает состояние для сериализации через pickle/joblib без нативного engine_."""
        state = self.__dict__.copy()
        state.pop("engine_", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Восстанавливает состояние и пересоздаёт C++ engine_ при десериализации."""
        self.__dict__.update(state)
        if hasattr(self, "coef_") and hasattr(self, "consequents_") and self.model_spec is not None:
            spec = self._normalize_spec()
            fitted_vals = {name: float(self.coef_[m]) for m, name in enumerate(self.consequents_)}
            self.engine_ = self._build_engine(spec, fitted_vals, self.consequents_)

    def _normalize_spec(self) -> SugenoModelSpec:
        """Привести model_spec к нормализованному экземпляру SugenoModelSpec."""
        if self.model_spec is None:
            raise ValueError("model_spec must be specified before calling fit().")
        if isinstance(self.model_spec, SugenoModelSpec):
            return self.model_spec
        if isinstance(self.model_spec, Mapping):
            return SugenoModelSpec(
                inputs=self.model_spec.get("inputs", {}),
                rules=self.model_spec.get("rules", []),
                output_name=self.model_spec.get("output_name", "y"),
                consequents=self.model_spec.get("consequents", None),
                default_t_norm=self.model_spec.get("default_t_norm", None),
                default_s_norm=self.model_spec.get("default_s_norm", None),
            )
        raise TypeError(
            f"model_spec must be SugenoModelSpec or Mapping, got {type(self.model_spec).__name__}"
        )

    def _extract_rule_consequents(self, rules: Sequence[Any]) -> list[str]:
        """Извлечь уникальные имена констант заключений из списка правил в порядке появления."""
        consequents: list[str] = []
        for rule in rules:
            if isinstance(rule, str):
                match = _THEN_RE.search(rule)
                if not match:
                    raise ValueError(
                        f"Could not parse consequent term from rule text: {rule!r}. "
                        f"Expected format: 'IF ... THEN <output_var> IS <consequent_term>'."
                    )
                c_name = match.group(1)
                if c_name not in consequents:
                    consequents.append(c_name)
            elif isinstance(rule, Mapping):
                consequent = rule.get("consequent")
                if not consequent:
                    raise ValueError(f"Rule mapping missing 'consequent': {rule!r}")
                for item in consequent:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        c_name = str(item[1])
                    elif isinstance(item, str):
                        c_name = item
                    else:
                        raise ValueError(f"Invalid consequent item in rule: {item!r}")
                    if c_name not in consequents:
                        consequents.append(c_name)
            elif isinstance(rule, (tuple, list)):
                if len(rule) < 2:
                    raise ValueError(f"Tuple rule must have at least (antecedent, consequent): {rule!r}")
                consequent = rule[1]
                if isinstance(consequent, (tuple, list)):
                    for item in consequent:
                        if isinstance(item, (tuple, list)) and len(item) == 2:
                            c_name = str(item[1])
                        elif isinstance(item, str):
                            c_name = item
                        else:
                            raise ValueError(f"Invalid consequent item in tuple rule: {item!r}")
                        if c_name not in consequents:
                            consequents.append(c_name)
                elif isinstance(consequent, str):
                    if consequent not in consequents:
                        consequents.append(consequent)
            else:
                raise TypeError(f"Unsupported rule format: {type(rule).__name__}")

        return consequents

    def _resolve_consequents(self, spec: SugenoModelSpec) -> list[str]:
        """Извлечь и провалидировать список уникальных имён констант заключений правил."""
        rule_consequents = self._extract_rule_consequents(spec.rules)
        if not rule_consequents:
            raise ValueError("No consequent membership functions found in model_spec rules.")

        if spec.consequents is not None:
            consequents_list = list(spec.consequents)
            if not consequents_list:
                raise ValueError("model_spec.consequents cannot be empty.")

            # Проверка на дубликаты
            if len(set(consequents_list)) != len(consequents_list):
                seen: set[str] = set()
                dups: list[str] = []
                for c in consequents_list:
                    if c in seen and c not in dups:
                        dups.append(c)
                    seen.add(c)
                raise ValueError(f"model_spec.consequents contains duplicate names: {dups}")

            # Проверка соответствия именам в правилах
            rule_set = set(rule_consequents)
            spec_set = set(consequents_list)

            missing_in_spec = rule_set - spec_set
            if missing_in_spec:
                raise ValueError(
                    f"model_spec.consequents is missing consequents referenced in rules: {sorted(missing_in_spec)}"
                )

            unused_in_rules = spec_set - rule_set
            if unused_in_rules:
                raise ValueError(
                    f"model_spec.consequents defines consequents not referenced by any rule: {sorted(unused_in_rules)}"
                )

            return consequents_list

        return rule_consequents

    def _build_engine(
        self,
        spec: SugenoModelSpec,
        consequent_values: Mapping[str, float],
        consequent_names: Sequence[str],
    ) -> openfll.SugenoEngine:
        """Построить и скомпилировать экземпляр SugenoEngine с заданными константами."""
        engine = openfll.SugenoEngine()

        t_norm = self.default_t_norm or spec.default_t_norm
        s_norm = self.default_s_norm or spec.default_s_norm
        if t_norm:
            engine.set_default_t_norm(t_norm)
        if s_norm:
            engine.set_default_s_norm(s_norm)

        for var_name, mfs in spec.inputs.items():
            engine.add_input_var(var_name)
            for mf in mfs:
                if isinstance(mf, Mapping):
                    mf_name = mf["name"]
                    mf_type = mf["type"]
                    mf_params = mf["params"]
                elif isinstance(mf, (tuple, list)) and len(mf) == 3:
                    mf_name, mf_type, mf_params = mf
                else:
                    raise ValueError(
                        f"Invalid membership function definition for variable {var_name!r}: {mf!r}"
                    )
                engine.add_membership_func(var_name, str(mf_name), str(mf_type), list(mf_params))

        engine.add_output_var(spec.output_name)

        for name in consequent_names:
            val = float(consequent_values[name])
            engine.add_membership_func(spec.output_name, name, "constant", [val])

        for rule in spec.rules:
            if isinstance(rule, str):
                engine.add_rule(rule)
            elif isinstance(rule, Mapping):
                engine.add_rule(**rule)
            elif isinstance(rule, (tuple, list)):
                engine.add_rule(*rule)
            else:
                raise TypeError(f"Unsupported rule format: {type(rule).__name__}")

        engine.build()
        return engine

    def fit(self, X: Any, y: Any) -> "SugenoRegressor":
        """Обучить константы заключений правил модели Sugeno нулевого порядка методом МНК."""
        if self.model_spec is None:
            raise ValueError("model_spec must be specified before calling fit().")

        if isinstance(self.alpha, bool) or not np.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError(
                f"Regularization parameter alpha must be non-negative and finite, got {self.alpha}"
            )

        spec = self._normalize_spec()
        if not spec.inputs:
            raise ValueError("model_spec.inputs must define at least one input variable.")
        if not spec.rules:
            raise ValueError("model_spec.rules must define at least one fuzzy rule.")

        feature_names = None
        if hasattr(X, "columns"):
            feature_names = [str(c) for c in X.columns]

        X_arr, y_arr = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            y_numeric=True,
            multi_output=False,
        )

        input_names = list(spec.inputs.keys())
        if X_arr.shape[1] != len(input_names):
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but model_spec expects {len(input_names)} "
                f"features ({input_names})."
            )

        if feature_names is not None and feature_names != input_names:
            raise ValueError(
                f"Feature names in X {feature_names} do not match model_spec inputs {input_names}."
            )

        consequent_names = self._resolve_consequents(spec)
        if not consequent_names:
            raise ValueError("No consequent membership functions found in model_spec rules.")

        M = len(consequent_names)
        N = X_arr.shape[0]

        # Строим матрицу влияния consequents A(X) через нативный batch (метод базисной декомпозиции)
        A = np.empty((N, M), dtype=np.float64)
        for m, c_name in enumerate(consequent_names):
            basis_vals = {name: (1.0 if name == c_name else 0.0) for name in consequent_names}
            basis_engine = self._build_engine(spec, basis_vals, consequent_names)
            col = basis_engine.predict_batch(X_arr, input_names=input_names)
            A[:, m] = col

        # Проверка на нулевую сумму активаций (NaN в строках матрицы A)
        nan_mask = np.isnan(A).any(axis=1)
        if np.any(nan_mask):
            num_bad = int(np.sum(nan_mask))
            raise ValueError(
                f"Training data contains {num_bad} sample(s) outside the support of the fuzzy rule base "
                f"(total rule activation is zero, producing NaN). Ensure membership functions and rules "
                f"cover the domain of X."
            )

        # Решение задачи наименьших квадратов
        if self.solver == "lstsq":
            if self.alpha != 0.0:
                raise ValueError(
                    f"solver='lstsq' does not support L2 regularization (got alpha={self.alpha}). "
                    f"Set solver='ridge' to use L2 regularization, or set alpha=0.0."
                )
            coef, _, rank, _ = np.linalg.lstsq(A, y_arr, rcond=None)
        elif self.solver == "ridge":
            if self.alpha <= 0.0:
                raise ValueError(
                    f"solver='ridge' requires alpha > 0.0, got {self.alpha}. "
                    f"Use solver='lstsq' for unregularized least squares."
                )
            reg_mat = A.T @ A + self.alpha * np.eye(M, dtype=np.float64)
            rhs = A.T @ y_arr
            coef = np.linalg.solve(reg_mat, rhs)
            rank = int(np.linalg.matrix_rank(A))
        else:
            raise ValueError(f"Unknown solver: {self.solver!r}. Supported solvers are 'lstsq' and 'ridge'.")

        # Создаём и компилируем итоговый обученный engine с найденными константами
        fitted_vals = {name: float(coef[m]) for m, name in enumerate(consequent_names)}
        fitted_engine = self._build_engine(spec, fitted_vals, consequent_names)

        # Сохраняем обученные атрибуты (в соответствии с конвенцией sklearn)
        self.engine_ = fitted_engine
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.consequents_ = list(consequent_names)
        self.rule_names_ = [f"rule_{i}" for i in range(len(spec.rules))]
        self.rank_ = int(rank)
        self.n_features_in_ = X_arr.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            delattr(self, "feature_names_in_")

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Выполнить batch inference на обученной модели."""
        check_is_fitted(self, attributes=["engine_", "coef_"])

        feature_names = None
        if hasattr(X, "columns"):
            feature_names = [str(c) for c in X.columns]

        X_arr = check_array(X, accept_sparse=False, dtype=np.float64)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but {type(self).__name__} is fitted with {self.n_features_in_} features"
            )

        if feature_names is not None and hasattr(self, "feature_names_in_"):
            expected = list(self.feature_names_in_)
            if feature_names != expected:
                raise ValueError(
                    f"The feature names should match those that were passed during fit.\n"
                    f"Feature names seen at fit time: {expected}\n"
                    f"Feature names seen at predict time: {feature_names}"
                )

        spec = self._normalize_spec()
        input_names = list(spec.inputs.keys())
        return self.engine_.predict_batch(X_arr, input_names=input_names)
