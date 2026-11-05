"""Type stubs для публичного API openfll.

openfll — пользовательский пакет. Во время выполнения он делегирует работу
внутреннему backend-пакету PyFLL, который загружает PyFLL._core.
"""
from pathlib import Path
from typing import Generic, TypeVar, Any, List, Literal, Mapping, Sequence, Tuple, Union, overload
import numpy as np

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


class MembershipPlot:
    """Рисунок и оси, созданные для одной нечёткой переменной."""

    variable_name: str
    variable_type: Literal["input", "output"]
    figure: Any
    axes: Any
    saved_path: Path | None


class ModelView:
    def calculate(self) -> None: ...
    def save_to_string(self) -> str: ...
    def save_to_file(self, path: str | Path) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

def load_from_string(json: str) -> SugenoEngine | MamdaniEngine: ...
def load_from_file(path: str | Path) -> SugenoEngine | MamdaniEngine: ...

_DefuzzerName = TypeVar("_DefuzzerName", bound=str, contravariant=True)


class BaseLinearEngine(Generic[_DefuzzerName]):
    """Общий линейный API моделей нечеткой логики Сугено и Мамдани.

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

    def save_to_string(self) -> str: ...
    def save_to_file(self, path: str | Path) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
    def model_ref(self) -> ModelView: ...
    def set_defuzzer(self, name: _DefuzzerName) -> None: ...
    def get_defuzzer(self) -> str: ...
    def find_belonging(self, mf_name: str, value: float) -> float: ...
    def get_input_var_names(self) -> list[str]: ...
    def get_output_var_names(self) -> list[str]: ...

    def set_input(self, name: str, value: float) -> None:
        """Задать значение input-переменной после build()."""
        ...

    def calculate(self) -> None:
        """Запустить fuzzy inference на текущих input-значениях."""
        ...

    def get_output(self, name: str) -> float:
        """Прочитать output-значение после calculate()."""
        ...

    @overload
    def predict(
        self,
        inputs: Mapping[str, float] | float | int,
        output_name: str | None = None,
    ) -> float: ...

    @overload
    def predict(
        self,
        inputs: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        output_name: str | None = None,
        *,
        input_names: Sequence[str] | None = None,
    ) -> float | np.ndarray:
        """Выполнить универсальный inference для скаляра, словаря, последовательности или NumPy-батча.

        Args:
            inputs: входные данные:
                - Mapping[str, float]: словарь входов вида {"x1": 1.0, "x2": 2.0} -> float;
                - float | int: скалярное значение для модели с ровно одним входом -> float;
                - Sequence[float]: список значений для одного сэмпла (многовходовая модель) или батч сэмплов (одновходовая модель) -> float | ndarray;
                - 1D ndarray / 2D ndarray | Sequence[Sequence]: батч сэмплов -> numpy ndarray.
            output_name: имя целевой выходной переменной (если не передано, используется
                единственный зарегистрированный output).
            input_names: соответствие колонок именам входных переменных для батч-массива.
                Если не указано, берутся имена в порядке регистрации в модели.

        Returns:
            float для единичного предсказания или ndarray для батча.
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

    def plot_mf(
        self,
        variables: str | Sequence[str] | None = None,
        *,
        variable_type: Literal["all", "input", "output"] = "all",
        num_points: int = 181,
        ranges: Mapping[str, tuple[float, float]] | None = None,
        show: bool = True,
        save_dir: str | Path | None = None,
    ) -> dict[str, MembershipPlot]:
        """Построить графики MF для зарегистрированных переменных."""
        ...

    def is_built(self) -> bool:
        """Вернуть True после успешного build()."""
        ...


# ============================================================================
class SugenoEngine(BaseLinearEngine[Literal["wtaver", "wtsum"]]):
    def __init__(self) -> None: ...
    def set_defuzzer(self, name: Literal["wtaver", "wtsum"]) -> None: ...
    def get_defuzzer(self) -> Literal["wtaver", "wtsum"]: ...


class MamdaniEngine(BaseLinearEngine[Literal["centroid"]]):
    def __init__(self) -> None: ...
    def set_defuzzer(self, name: Literal["centroid"]) -> None: ...
    def get_defuzzer(self) -> Literal["centroid"]: ...
    def set_integrator(self, start: float, end: float, step: float) -> None: ...

# Python-only helpers (submodule `openfll._helpers`).
# Сигнатуры ниже — ручные type hints; runtime-импорт через `openfll.__init__`.
# ============================================================================
def plot_mf(
    engine: BaseLinearEngine,
    variables: str | Sequence[str] | None = None,
    *,
    variable_type: Literal["all", "input", "output"] = "all",
    num_points: int = 181,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    show: bool = True,
    save_dir: str | Path | None = None,
) -> dict[str, MembershipPlot]: ...


def monte_carlo(
    engine: BaseLinearEngine,
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> list[dict[str, float]]: ...
def grid_2d(
    engine: BaseLinearEngine,
    x_var: str,
    y_var: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    out_var: str,
    n: int = 20,
) -> tuple[list[float], list[float], list[list[float]]]: ...
def check_output_finite(
    engine: BaseLinearEngine,
    var_inputs: list[tuple[str, tuple[float, float]]],
    var_outputs: list[str],
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, int]: ...
