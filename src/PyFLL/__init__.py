"""Python-биндинги OpenFLL и определения API для сборки документации.

Нативные классы используются при наличии PyFLL._core. Без собранного расширения
импорт пакета остаётся доступным для MkDocs/griffe и статического анализа.
Вычисления через определения для документации недоступны: создание движка и
загрузка модели явно завершаются ImportError.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, NoReturn, Sequence, TypeVar

MfType = Literal["constant", "triangular", "trapezoidal", "gaussian", "polynomial"]
TNorms = Literal["min", "prod_and", "bounded_diff", "drastic_prod", "einstein_prod", "hamacher_prod"]
SNorms = Literal["max", "algebraic_sum", "bounded_sum", "drastic_sum", "einstein_sum", "hamacher_sum"]

__all__ = ["BaseLinearEngine", "SugenoEngine", "MamdaniEngine", "ModelView",
           "load_from_file", "load_from_string", "MfType", "TNorms", "SNorms"]

_BACKEND_IMPORT_ERROR: ImportError | None = None

if not TYPE_CHECKING:
    try:
        from ._core import (
            BaseLinearEngine, SugenoEngine, MamdaniEngine, ModelView,
            load_from_file, load_from_string,
        )
    except ImportError as exc:
        # Чистая сборка документации не требует компиляции C++ и установки wheel.
        # Сохраняем исходную причину, чтобы попытка вычислений не скрывала ошибку.
        _BACKEND_IMPORT_ERROR = exc

if TYPE_CHECKING or _BACKEND_IMPORT_ERROR is not None:
    # Эти определения нужны griffe/mkdocstrings и инструментам статического анализа.
    # Они остаются доступны без .pyd; при наличии расширения экспортируются его
    # реальные классы. Подробные сигнатуры IDE дополнительно заданы в .pyi.
    _DefuzzerName = TypeVar("_DefuzzerName", bound=str, contravariant=True)

    def _native_unavailable() -> NoReturn:
        raise ImportError(
            "PyFLL._core недоступен: установите wheel с нативным модулем. "
            "Определения для документации не выполняют вычисления."
        ) from _BACKEND_IMPORT_ERROR

    class ModelView:
        """Представление построенной модели, которое удерживает движок-владелец.

        Получается через engine.model_ref(). Сохранение использует полный
        набор метаданных фасада, включая порядок переменных и имена MF.
        """

        def __init__(self) -> None:
            raise TypeError("ModelView получается через engine.model_ref()")

        def calculate(self) -> None:
            """Рассчитать модель на текущих значениях входов."""
            _native_unavailable()

        def save_to_string(self) -> str:
            """Сохранить модель и метаданные фасада в строку JSON."""
            _native_unavailable()

        def save_to_file(self, path: str | Path) -> None:
            """Сохранить модель и метаданные фасада в файл JSON."""
            _native_unavailable()

        def to_dict(self) -> dict[str, Any]:
            """Вернуть сериализованную модель как словарь Python."""
            _native_unavailable()

    class BaseLinearEngine(Generic[_DefuzzerName]):
        """Общие регистрация переменных, правила, вычисления и сериализация.

        Сугено и Мамдани независимо наследуют этот интерфейс. Настройка
        интегратора доступна только у MamdaniEngine. Экземпляр движка
        предназначен для использования в одном потоке одновременно.
        """

        def __init__(self) -> None:
            _native_unavailable()

        def add_input_var(self, name: str) -> None:
            """Зарегистрировать входную переменную до build()."""
            _native_unavailable()

        def add_output_var(self, name: str) -> None:
            """Зарегистрировать выходную переменную до build()."""
            _native_unavailable()

        def add_membership_func(self, var_name: str, mf_name: str, type: MfType,
                                params: Sequence[float]) -> None:
            """Добавить именованную функцию принадлежности к переменной."""
            _native_unavailable()

        def add_rule(self, rule: str | None = None, *,
                     antecedent: Sequence[tuple[str, str]] | None = None,
                     consequent: Sequence[tuple[str, str]] | None = None,
                     t_norm: str = "", s_norm: str = "") -> None:
            """Добавить правило текстом либо списками пар (переменная, MF).

            Текст поддерживает AND/OR и явные нормы, например AND[prod_and].
            В структурной форме термы антецедента соединяются через AND.
            """
            _native_unavailable()

        def set_default_t_norm(self, name: TNorms) -> None:
            """Выбрать норму AND по умолчанию до сборки модели."""
            _native_unavailable()

        def set_default_s_norm(self, name: SNorms) -> None:
            """Выбрать норму OR по умолчанию до сборки модели."""
            _native_unavailable()

        def set_defuzzer(self, name: _DefuzzerName) -> None:
            """Выбрать дефаззификатор, поддерживаемый конкретным видом модели."""
            _native_unavailable()

        def get_defuzzer(self) -> str:
            """Получить имя фактического дефаззификатора."""
            _native_unavailable()

        def build(self) -> None:
            """Собрать модель; последующее изменение конфигурации запрещено."""
            _native_unavailable()

        def set_input(self, name: str, value: float) -> None:
            """Установить значение входа собранной модели."""
            _native_unavailable()

        def calculate(self) -> None:
            """Вычислить выходы модели по текущим значениям входов."""
            _native_unavailable()

        def get_output(self, name: str) -> float:
            """Прочитать рассчитанное значение выхода."""
            _native_unavailable()

        def predict_batch(self, X: object, *, input_names: Sequence[str]) -> Any:
            """Выполнить пакетный расчёт модели с одним выходом на массиве NumPy."""
            _native_unavailable()

        def is_built(self) -> bool:
            """Проверить, собрана ли модель."""
            _native_unavailable()

        def find_belonging(self, mf_name: str, value: float) -> float:
            """Вычислить степень принадлежности именованной MF.

            Имена сохраняются при загрузке JSON с метаданными линейного API.
            """
            _native_unavailable()

        def get_input_var_names(self) -> list[str]:
            """Вернуть имена входов в порядке регистрации."""
            _native_unavailable()

        def get_output_var_names(self) -> list[str]:
            """Вернуть имена выходов в порядке регистрации."""
            _native_unavailable()

        def model_ref(self) -> ModelView:
            """Получить безопасное представление собранной модели."""
            _native_unavailable()

        def save_to_string(self) -> str:
            """Сохранить модель и метаданные фасада в строку JSON."""
            _native_unavailable()

        def save_to_file(self, path: str | Path) -> None:
            """Сохранить модель и метаданные фасада в файл JSON."""
            _native_unavailable()

        def to_dict(self) -> dict[str, Any]:
            """Вернуть сериализованную модель как словарь Python."""
            _native_unavailable()

    class SugenoEngine(BaseLinearEngine[Literal["wtaver", "wtsum"]]):
        """Линейный фасад Сугено; по умолчанию используется wtaver."""

        def set_defuzzer(self, name: Literal["wtaver", "wtsum"]) -> None:
            """Выбрать взвешенное среднее wtaver или взвешенную сумму wtsum."""
            _native_unavailable()

        def get_defuzzer(self) -> Literal["wtaver", "wtsum"]:
            """Получить фактический режим Сугено."""
            _native_unavailable()

    class MamdaniEngine(BaseLinearEngine[Literal["centroid"]]):
        """Линейный фасад Мамдани с центроидным дефаззификатором."""

        def set_defuzzer(self, name: Literal["centroid"]) -> None:
            """Выбрать поддерживаемый дефаззификатор centroid."""
            _native_unavailable()

        def get_defuzzer(self) -> Literal["centroid"]:
            """Получить имя дефаззификатора Мамдани."""
            _native_unavailable()

        def set_integrator(self, start: float, end: float, step: float) -> None:
            """Задать диапазон и шаг интегрирования до build()."""
            _native_unavailable()

    def load_from_string(json: str) -> SugenoEngine | MamdaniEngine:
        """Восстановить собранный движок Сугено или Мамдани из JSON."""
        _native_unavailable()

    def load_from_file(path: str | Path) -> SugenoEngine | MamdaniEngine:
        """Загрузить движок из JSON-файла, в том числе по пути pathlib.Path."""
        _native_unavailable()
