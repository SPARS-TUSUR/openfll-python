"""Type stubs for PyFLL.

These stubs provide IDE autocompletion and mypy type-checking for the
PyFLL Python module, even though the actual implementation is a compiled
.pyd (C++ extension via pybind11).

Source of truth:
  - C++ binding:  src/PyFLL/PyFLL.cpp
  - C++ API:     src/OpenFLL/linear/sugeno_engine.h

When you change the C++ binding, please update this file to match.
"""
from typing import Any, List, Literal, Sequence, Tuple, Union
from pathlib import Path

# ============================================================================
# Literal type aliases — for IDE autocomplete on string-enum parameters
# ============================================================================

# Membership function type names
MfType = Literal[
    "constant",
    "triangular",
    "trapezoidal",
    "gaussian",
    "polynomial",
]

# t-norm names (used for AND or set_default_t_norm)
TNorms = Literal[
    "min",
    "prod_and",
    "bounded_diff",
    "drastic_prod",
    "einstein_prod",
    "hamacher_prod",
]

# s-norm names (used for OR or set_default_s_norm)
SNorms = Literal[
    "max",
    "algebraic_sum",
    "bounded_sum",
    "drastic_sum",
    "einstein_sum",
    "hamacher_sum",
]


# ============================================================================
# SugenoEngine — the main public class
# ============================================================================

class ModelView:
    def calculate(self) -> None: ...
    def save_to_string(self) -> str: ...
    def save_to_file(self, path: str | Path) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class BaseLinearEngine:
    """Общий линейный API моделей Сугено и Мамдани.

    Provides a flat, name-based API on top of OpenFLL's fluent
    SugenoBuilder / RulesBuilder. Build the model by:

      1. Registering input/output variables (add_input_var / add_output_var).
      2. Adding membership functions to those variables
         (add_membership_func: name + type + parameter list).
      3. Adding rules as plain strings
         (add_rule: 'IF x1 IS "low" AND x2 IS "high" THEN y IS "fast"').
      4. Calling build() to compile the model.
      5. Setting input values (set_input), running calculate(),
         and reading outputs (get_output).

    Threading: instances are NOT thread-safe. Use one engine per thread.
    """

    def model_ref(self) -> ModelView: ...
    def set_defuzzer(self, name: Literal["wtaver", "wtsum", "centroid"]) -> None: ...
    def get_defuzzer(self) -> str: ...
    def find_belonging(self, mf_name: str, value: float) -> float: ...
    def get_input_var_names(self) -> list[str]: ...
    def get_output_var_names(self) -> list[str]: ...

    def __init__(self) -> None:
        """Create an empty Sugeno engine."""
        ...

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    def add_input_var(self, name: str) -> None:
        """Register an input variable.

        Args:
            name: variable name. Must be non-empty and not already used.

        Raises:
            ValueError: name is empty or already registered.
            RuntimeError: called after build().
        """
        ...

    def add_output_var(self, name: str) -> None:
        """Register an output variable.

        Args:
            name: variable name. Must be non-empty and not already used.

        Raises:
            ValueError: name is empty or already registered.
            RuntimeError: called after build().
        """
        ...

    def add_membership_func(
        self,
        var_name: str,
        mf_name: str,
        type: MfType,
        params: Union[List[float], Tuple[float, ...]],
    ) -> None:
        """Add a membership function to a registered variable.

        Args:
            var_name: name of a previously registered input or output variable.
            mf_name: unique name of the membership function (e.g. "low", "high").
            type: one of:
                "constant"    — params = [value]
                "triangular"  — params = [a, b, c]
                "trapezoidal" — params = [a, b, c, d]
                "gaussian"    — params = [center, sigma]
                "polynomial"  — currently raises ValueError (not yet supported)
            params: list/tuple of doubles; size depends on type.

        Raises:
            ValueError: bad type or wrong parameter count, duplicate mf_name.
            IndexError: var_name is not registered.
            RuntimeError: called after build().
        """
        ...

    def add_rule(self, rule: str) -> None:
        """Add a fuzzy rule in textual form.

        Format:
            IF <var1> IS "<mf1>" ((AND|OR) [<norm_name>] <var2> IS "<mf2>")*
                THEN <out_var> IS "<out_mf>"

        Operators (IF/AND/OR/THEN/IS) are case-insensitive. Variable and MF
        names are case-sensitive. Quoted MF names may contain spaces.

        Optional [<norm_name>] after AND/OR sets an explicit t/s-norm
        (e.g. "AND[prod_and]", "OR[algebraic_sum]"). If omitted, the engine's
        default norm (set_default_t_norm / set_default_s_norm) is used.

        Args:
            rule: the rule as a string.

        Raises:
            ValueError: syntax error or unknown norm name.
            IndexError: a variable or MF name is unknown.
            RuntimeError: called after build().
        """
        ...

    def add_rule(
        self,
        antecedent: List[Tuple[str, str]],
        consequent: List[Tuple[str, str]],
        t_norm: Union[TNorms, str, None] = None,
        s_norm: Union[SNorms, str, None] = None,
    ) -> None:
        """Add a fuzzy rule in structural (tuple) form.

        Equivalent to the textual form, but expressed as Python data structures
        (lists of (var_name, mf_name) tuples). This is the preferred form for
        programmatic rule generation: easier to build dynamically and faster
        to parse than the textual form.

        Args:
            antecedent: list of (var_name, mf_name) pairs. Non-empty.
                All operators between terms are AND (use the textual
                add_rule(str) to express OR).
            consequent: list with exactly one (var_name, mf_name) pair.
            t_norm: optional t-norm name applied to all AND-operators in
                the antecedent. If None or empty string, the engine's
                default t-norm (set_default_t_norm) is used.
                One of: "min", "prod_and", "bounded_diff", "drastic_prod",
                "einstein_prod", "hamacher_prod".
            s_norm: optional s-norm name. (Used only if you later extend
                the engine to support OR in tuple form; currently unused
                for AND-only rules.) If None or empty string, the engine's
                default s-norm (set_default_s_norm) is used.

        Example:
            >>> engine.add_rule(
            ...     antecedent=[("x1", "low"), ("x2", "high")],
            ...     consequent=[("y", "open")],
            ...     t_norm="prod_and",
            ... )

        Raises:
            ValueError: empty antecedent, wrong consequent size, unknown
                variable/MF, or unknown norm.
            IndexError: a variable or MF is unknown.
            RuntimeError: called after build().
        """
        ...

    def set_default_t_norm(self, name: TNorms) -> None:
        """Set default t-norm (for AND) used when a rule has no [<norm>].

        Args:
            name: one of "min", "prod_and", "bounded_diff", "drastic_prod",
                  "einstein_prod", "hamacher_prod".

        Raises:
            ValueError: unknown name or not a t-norm.
            RuntimeError: called after build().
        """
        ...

    def set_default_s_norm(self, name: SNorms) -> None:
        """Set default s-norm (for OR) used when a rule has no [<norm>].

        Args:
            name: one of "max", "algebraic_sum", "bounded_sum", "drastic_sum",
                  "einstein_sum", "hamacher_sum".

        Raises:
            ValueError: unknown name or not an s-norm.
            RuntimeError: called after build().
        """
        ...

    def build(self) -> None:
        """Compile the model.

        After build() you cannot add more variables, MFs, or rules. Must be
        called before calculate() / set_input() / get_output().

        Raises:
            RuntimeError: configuration conflict (no inputs, no outputs,
                          no rules, duplicate names, etc.).
        """
        ...

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    def set_input(self, name: str, value: float) -> None:
        """Set value of an input variable.

        Args:
            name: name of a registered input variable.
            value: new crisp value.

        Raises:
            IndexError: name is not a registered input variable.
            RuntimeError: called before build().
        """
        ...

    def get_output(self, name: str) -> float:
        """Read value of an output variable.

        Must be called after calculate() to reflect the latest result.

        Returns:
            Crisp output value (defuzzified for Mamdani, weighted sum for Sugeno).

        Raises:
            IndexError: name is not a registered output variable.
            RuntimeError: called before build().
        """
        ...

    def calculate(self) -> None:
        """Run fuzzy inference.

        Reads current input values (set via set_input) and writes defuzzified
        results to registered output variables (read via get_output).

        Raises:
            RuntimeError: called before build().
        """
        ...

    def predict_batch(self, X: object, *, input_names: Sequence[str]) -> Any:
        """Run native batch inference.

        Args:
            X: 2D NumPy-compatible numeric array with shape
                (n_samples, n_features). Non-contiguous arrays are accepted.
            input_names: required sequence of unique strings. Column i maps to
                input_names[i] and every registered input must appear once.

        Returns:
            NumPy ndarray with dtype float64 and shape (n_samples,).

        Raises:
            TypeError: X cannot be converted to a numeric array or input_names
                is not a sequence of strings.
            ValueError: wrong shape, name/feature count mismatch, duplicate
                names, or incomplete input_names.
            IndexError: an input name is unknown.
            RuntimeError: called before build() or unsupported output config.
        """
        ...

    # ------------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------------

    def is_built(self) -> bool:
        """Return True if build() has been called successfully."""
        ...

    # ------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------

    def save_to_string(self) -> str:
        """Serialize the built model to a JSON string.

        The JSON keeps the model structure — variables, membership functions
        and rules — but not the current input values.

        Raises:
            RuntimeError: called before build().
        """
        ...

    def save_to_file(self, path: str | Path) -> None:
        """Serialize the built model to a JSON file.

        Args:
            path: destination file. An existing file is overwritten.

        Raises:
            RuntimeError: called before build().
            OSError: the file could not be written.
        """
        ...

    def to_dict(self) -> Any:
        """Return the serialized model as a plain Python dict.

        Equivalent to json.loads(engine.save_to_string()).

        Raises:
            RuntimeError: called before build().
        """
        ...


# ============================================================================
# Module-level loading functions
# ============================================================================

def load_from_string(json: str) -> SugenoEngine | MamdaniEngine:
    """Восстановить движок из JSON, созданного save_to_string().

    Returns a ready-to-use SugenoEngine: it is already built, so set_input(),
    calculate() and get_output() work immediately.

    Note:
      - The loaded engine is inference-only: add_input_var(),
        add_membership_func() and add_rule() raise RuntimeError, exactly as
        after a normal build().
      - Метаданные фасада сохраняют имена MF и порядок переменных.
        Старые JSON без этих метаданных также поддерживаются.
      - Current input values are not persisted.

    Raises:
        ValueError: malformed JSON or unknown model type.
    """
    ...

class SugenoEngine(BaseLinearEngine):
    def __init__(self) -> None: ...


class MamdaniEngine(BaseLinearEngine):
    def __init__(self) -> None: ...
    def set_integrator(self, start: float, end: float, step: float) -> None: ...

def load_from_file(path: str | Path) -> SugenoEngine | MamdaniEngine:
    """Load a model from a JSON file written by save_to_file().

    See load_from_string() for the semantics of the returned engine.

    Raises:
        ValueError: malformed JSON or unknown model type.
        OSError: the file could not be read.
    """
    ...
