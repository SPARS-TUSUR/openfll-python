"""Type stubs для публичного API openfll.

openfll — пользовательский пакет. Во время выполнения он делегирует работу
внутреннему backend-пакету PyFLL, который загружает PyFLL._core.

Эти stubs описывают только существующий API. Они намеренно не объявляют
будущие методы вроде fit, get_params или set_params.
"""
from typing import Any, List, Literal, Mapping, Sequence, Tuple, Union, overload

MfType = Literal[
    "constant",
    "triangular",
    "trapezoidal",
    "gaussian",
]

TNorms = Literal[
    "min",
    "prod_and",
    "bounded_diff",
    "drastic_prod",
    "einstein_prod",
    "hamacher_prod",
]

SNorms = Literal[
    "max",
    "algebraic_sum",
    "bounded_sum",
    "drastic_sum",
    "einstein_sum",
    "hamacher_sum",
]


def constant(value: float) -> List[float]:
    """Вернуть params для constant membership function: [value]."""
    ...


def triangular(a: float, b: float, c: float) -> List[float]:
    """Вернуть params для triangular membership function: [a, b, c]."""
    ...


def trapezoidal(a: float, b: float, c: float, d: float) -> List[float]:
    """Вернуть params для trapezoidal membership function: [a, b, c, d]."""
    ...


def gaussian(*, center: float, sigma: float) -> List[float]:
    """Вернуть params для gaussian membership function: [center, sigma]."""
    ...


class SugenoEngine:
    """Плоский native-backed facade для модели нечеткой логики Сугено.

    Жизненный цикл:
        До build() настраиваются переменные, membership-функции, правила и
        default-нормы. После build() используются set_input(), calculate() и
        get_output(). Нарушения lifecycle backend сообщает как RuntimeError.

    Примечание:
        Неизвестные native-имена сейчас приходят из std::out_of_range и в
        текущем backend отображаются pybind11 как IndexError.
    """

    def __init__(self) -> None:
        """Создать пустой engine."""
        ...

    def add_input_var(self, name: str) -> None:
        """Зарегистрировать input-переменную до build()."""
        ...

    def add_output_var(self, name: str) -> None:
        """Зарегистрировать output-переменную до build()."""
        ...

    def add_membership_func(
        self,
        var_name: str,
        mf_name: str,
        type: MfType,
        params: Union[List[float], Tuple[float, ...]],
    ) -> None:
        """Добавить поддержанную membership-функцию к зарегистрированной переменной.

        Поддержанные значения type:
            constant: params = [value]
            triangular: params = [a, b, c]
            trapezoidal: params = [a, b, c, d]
            gaussian: params = [center, sigma]

        polynomial не входит в публичный API openfll v1.
        """
        ...

    @overload
    def add_rule(self, rule: str) -> None:
        """Добавить текстовое правило.

        Текстовые правила поддерживают AND/OR и явные имена норм, например
        AND[prod_and] or OR[algebraic_sum].
        """
        ...

    @overload
    def add_rule(
        self,
        antecedent: List[Tuple[str, str]],
        consequent: List[Tuple[str, str]],
        t_norm: Union[TNorms, str, None] = None,
        s_norm: Union[SNorms, str, None] = None,
    ) -> None:
        """Добавить кортежное/структурное правило.

        Tuple rules задают AND-цепочку по всем antecedent-термам. Если нужен
        OR, используйте текстовую форму правила.
        """
        ...

    def set_default_t_norm(self, name: TNorms) -> None:
        """Задать default t-норму для AND в правилах без явной нормы."""
        ...

    def set_default_s_norm(self, name: SNorms) -> None:
        """Задать default s-норму для OR в правилах без явной нормы."""
        ...

    def build(self) -> None:
        """Скомпилировать настроенную модель и перейти к execution lifecycle."""
        ...

    def set_input(self, name: str, value: float) -> None:
        """Задать значение input-переменной после build()."""
        ...

    def calculate(self) -> None:
        """Запустить fuzzy inference на текущих input-значениях."""
        ...

    def get_output(self, name: str) -> float:
        """Прочитать output-значение после calculate()."""
        ...

    def predict(
        self,
        inputs: Mapping[str, float],
        output_name: str | None = None,
    ) -> float:
        """Выполнить scalar inference через set_input, calculate и get_output.

        Если у engine ровно один output, зарегистрированный через add_output_var,
        output_name можно не передавать. Для multi-output или неизвестного
        Python-side registry передайте output_name явно.
        """
        ...

    def predict_batch(self, X: object, *, input_names: Sequence[str]) -> Any:
        """Выполнить native batch inference.

        Args:
            X: 2D NumPy-compatible numeric array shape (n_samples, n_features).
                Non-contiguous arrays accepted by the native backend.
            input_names: обязательное соответствие колонок input-переменным.
                Колонка i соответствует input_names[i], все input-переменные
                должны присутствовать ровно один раз.

        Returns:
            NumPy ndarray dtype float64, shape (n_samples,) для v1 single-output.
        """
        ...

    def is_built(self) -> bool:
        """Вернуть True после успешного build()."""
        ...


# ============================================================================
# Python-only helpers (submodule `openfll._helpers`).
# Сигнатуры ниже — ручные type hints; runtime-импорт через `openfll.__init__`.
# ============================================================================
def monte_carlo(
    engine: SugenoEngine,
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> list[dict[str, float]]: ...
def grid_2d(
    engine: SugenoEngine,
    x_var: str,
    y_var: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_var: str,
    n: int = 20,
) -> tuple[list[float], list[float], list[list[float]]]: ...
def check_output_finite(
    engine: SugenoEngine,
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, int]: ...
