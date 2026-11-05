"""Сохранение и загрузка моделей Сугено и Мамдани в нативном JSON OpenFLL.

Запуск из репозитория: python examples/05_serialization_demo.py --output-dir demo_models
Запуск из архива: python 05_serialization_demo.py --output-dir demo_models
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openfll


def create_engine(kind: str):
    engine = openfll.MamdaniEngine() if kind == "mamdani" else openfll.SugenoEngine()
    engine.add_input_var("temperature")
    engine.add_output_var("power")
    engine.add_membership_func("temperature", "cold", "triangular", [0, 0, 10])
    engine.add_membership_func("temperature", "hot", "triangular", [0, 10, 10])
    if kind == "mamdani":
        engine.set_defuzzer("centroid")
        engine.set_integrator(-1, 11, 0.01)
        engine.add_membership_func("power", "low", "triangular", [0, 2, 4])
        engine.add_membership_func("power", "high", "triangular", [6, 8, 10])
    else:
        engine.set_defuzzer(kind)
        engine.add_membership_func("power", "low", "constant", [2])
        engine.add_membership_func("power", "high", "constant", [8])
    engine.add_rule('IF temperature IS "cold" THEN power IS "low"')
    engine.add_rule('IF temperature IS "hot" THEN power IS "high"')
    engine.build()
    return engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_models"),
                        help="Каталог для сохранения JSON-моделей")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = np.array([[1.], [3.], [5.], [7.], [9.]])
    for kind in ("wtaver", "wtsum", "mamdani"):
        engine = create_engine(kind)
        expected = engine.predict(points)
        text = engine.save_to_string()
        path = args.output_dir / f"{kind}.json"
        engine.save_to_file(path)
        restored_text = openfll.load_from_string(text)
        restored_file = openfll.load_from_file(path)
        np.testing.assert_allclose(restored_text.predict(points), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(restored_file.predict(points), expected, rtol=1e-12, atol=1e-12)
        print(f"{type(engine).__name__} ({kind}): {expected.tolist()}")
        print(f"  Inputs={restored_file.get_input_var_names()}, outputs={restored_file.get_output_var_names()}")
        print(f"  cold(3)={engine.find_belonging('cold', 3):.6f}; JSON={path.resolve()}")
        print("  Serialization and deserialization via string and file: success")
    print("All native serialization checks passed successfully.")


if __name__ == "__main__":
    main()
