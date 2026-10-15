"""Benchmark inference paths for OpenFLL Python API.

Compares 3 inference paths:
1. Explicit workflow in Python loop: set_input -> calculate -> get_output
2. Scalar predict in Python loop: engine.predict({"x1": ..., "x2": ...})
3. Native predict_batch: engine.predict(X) via C++ PyFLL._core

Measures:
- Wall time (ms)
- Throughput (samples/sec)
- Latency / overhead per sample (us)
- Speedup factor of native batch vs loops
"""
from __future__ import annotations

import gc
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openfll


def get_git_info() -> tuple[str, str]:
    """Retrieve current branch and short commit hash if git is available."""
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return branch, commit
    except Exception:
        return "unknown", "unknown"


def build_benchmark_engine() -> openfll.SugenoEngine:
    """Build a deterministic standard Sugeno model for benchmarking."""
    e = openfll.SugenoEngine()
    e.add_input_var("x1")
    e.add_input_var("x2")
    e.add_output_var("y")

    e.add_membership_func("x1", "x1_low", "triangular", openfll.triangular(-10, 0, 10))
    e.add_membership_func("x1", "x1_high", "triangular", openfll.triangular(0, 10, 20))
    e.add_membership_func("x2", "x2_low", "gaussian", openfll.gaussian(center=0, sigma=2))
    e.add_membership_func("x2", "x2_high", "trapezoidal", openfll.trapezoidal(0, 5, 10, 15))

    e.add_membership_func("y", "y_low", "constant", openfll.constant(5.0))
    e.add_membership_func("y", "y_high", "constant", openfll.constant(15.0))

    e.add_rule('IF x1 IS "x1_low" AND x2 IS "x2_low" THEN y IS "y_low"')
    e.add_rule(
        antecedent=[("x1", "x1_high"), ("x2", "x2_high")],
        consequent=[("y", "y_high")],
        t_norm="prod_and",
    )
    e.build()
    return e


def generate_benchmark_data(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic inputs in the active model domain."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-5.0, 15.0, size=n_samples)
    x2 = rng.uniform(-2.0, 12.0, size=n_samples)
    return np.column_stack([x1, x2]).astype(np.float64)


def run_explicit_loop(engine: openfll.SugenoEngine, X: np.ndarray) -> np.ndarray:
    n = len(X)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        engine.set_input("x1", float(X[i, 0]))
        engine.set_input("x2", float(X[i, 1]))
        engine.calculate()
        out[i] = engine.get_output("y")
    return out


def run_scalar_predict_dict_loop(engine: openfll.SugenoEngine, X: np.ndarray) -> np.ndarray:
    n = len(X)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = engine.predict({"x1": float(X[i, 0]), "x2": float(X[i, 1])})
    return out


def run_scalar_predict_seq_loop(engine: openfll.SugenoEngine, X: np.ndarray) -> np.ndarray:
    n = len(X)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = engine.predict([float(X[i, 0]), float(X[i, 1])])
    return out


def run_native_batch(engine: openfll.SugenoEngine, X: np.ndarray) -> np.ndarray:
    return engine.predict(X)


def time_callable(func, engine, X, repeats: int = 5, warmup: int = 2) -> float:
    """Measure median execution time (in seconds) over repeated runs with warm-up."""
    # Warm-up runs
    for _ in range(warmup):
        _ = func(engine, X)

    timings = []
    for _ in range(repeats):
        gc.disable()
        try:
            start = time.perf_counter()
            _ = func(engine, X)
            end = time.perf_counter()
            timings.append(end - start)
        finally:
            gc.enable()

    return statistics.median(timings)


def run_benchmarks(batch_sizes: list[int] | None = None) -> list[dict]:
    if batch_sizes is None:
        batch_sizes = [10, 100, 1_000, 10_000, 50_000, 100_000]

    engine = build_benchmark_engine()
    results = []

    print("=" * 80)
    print("OpenFLL Benchmark Inference Paths")
    print("=" * 80)

    for n in batch_sizes:
        print(f"\nEvaluating batch size: {n:,} samples...")
        X = generate_benchmark_data(n)

        # 1. Native batch
        t_batch = time_callable(run_native_batch, engine, X, repeats=5, warmup=2)
        batch_samples_sec = n / t_batch if t_batch > 0 else float("inf")
        batch_overhead_us = (t_batch / n) * 1e6

        # 2. Explicit loop
        t_explicit = time_callable(run_explicit_loop, engine, X, repeats=5, warmup=2)
        explicit_samples_sec = n / t_explicit if t_explicit > 0 else float("inf")
        explicit_overhead_us = (t_explicit / n) * 1e6

        # 3. Scalar predict dict loop
        t_predict_dict = time_callable(run_scalar_predict_dict_loop, engine, X, repeats=5, warmup=2)
        dict_samples_sec = n / t_predict_dict if t_predict_dict > 0 else float("inf")
        dict_overhead_us = (t_predict_dict / n) * 1e6

        # 4. Scalar predict seq loop
        t_predict_seq = time_callable(run_scalar_predict_seq_loop, engine, X, repeats=5, warmup=2)
        seq_samples_sec = n / t_predict_seq if t_predict_seq > 0 else float("inf")
        seq_overhead_us = (t_predict_seq / n) * 1e6

        speedup_vs_explicit = t_explicit / t_batch if t_batch > 0 else 0
        speedup_vs_dict = t_predict_dict / t_batch if t_batch > 0 else 0
        speedup_vs_seq = t_predict_seq / t_batch if t_batch > 0 else 0

        res = {
            "n_samples": n,
            "native_batch": {
                "wall_time_ms": t_batch * 1000,
                "samples_sec": batch_samples_sec,
                "overhead_us": batch_overhead_us,
            },
            "explicit_loop": {
                "wall_time_ms": t_explicit * 1000,
                "samples_sec": explicit_samples_sec,
                "overhead_us": explicit_overhead_us,
                "speedup": speedup_vs_explicit,
            },
            "predict_dict_loop": {
                "wall_time_ms": t_predict_dict * 1000,
                "samples_sec": dict_samples_sec,
                "overhead_us": dict_overhead_us,
                "speedup": speedup_vs_dict,
            },
            "predict_seq_loop": {
                "wall_time_ms": t_predict_seq * 1000,
                "samples_sec": seq_samples_sec,
                "overhead_us": seq_overhead_us,
                "speedup": speedup_vs_seq,
            },
        }
        results.append(res)

        print(f"  Native Batch:        {t_batch*1000:9.2f} ms | {batch_samples_sec:12,.0f} samples/s | {batch_overhead_us:6.2f} us/sample")
        print(f"  Explicit Loop:       {t_explicit*1000:9.2f} ms | {explicit_samples_sec:12,.0f} samples/s | {explicit_overhead_us:6.2f} us/sample | Speedup: {speedup_vs_explicit:5.1f}x")
        print(f"  Predict (Dict) Loop: {t_predict_dict*1000:9.2f} ms | {dict_samples_sec:12,.0f} samples/s | {dict_overhead_us:6.2f} us/sample | Speedup: {speedup_vs_dict:5.1f}x")
        print(f"  Predict (List) Loop: {t_predict_seq*1000:9.2f} ms | {seq_samples_sec:12,.0f} samples/s | {seq_overhead_us:6.2f} us/sample | Speedup: {speedup_vs_seq:5.1f}x")

    return results


def format_markdown_report(results: list[dict], branch: str, commit: str) -> str:
    lines = []
    lines.append("# Benchmark Inference Paths Report\n")
    lines.append(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Repository**: openfll-python")
    lines.append(f"- **Branch**: `{branch}`")
    lines.append(f"- **Commit**: `{commit}`")
    lines.append(f"- **Python**: {platform.python_version()} ({platform.python_implementation()})")
    lines.append(f"- **NumPy**: {np.__version__}")
    lines.append(f"- **OS**: {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"- **Processor**: {platform.processor() or platform.machine()}\n")

    lines.append("## 1. Сводная таблица времени выполнения (Wall time, ms)\n")
    lines.append("| Batch Size | Native Batch | Explicit Loop | Predict (Dict) Loop | Predict (List) Loop |")
    lines.append("|-----------:|-------------:|--------------:|--------------------:|--------------------:|")
    for r in results:
        lines.append(
            f"| {r['n_samples']:,} | {r['native_batch']['wall_time_ms']:.2f} ms "
            f"| {r['explicit_loop']['wall_time_ms']:.2f} ms "
            f"| {r['predict_dict_loop']['wall_time_ms']:.2f} ms "
            f"| {r['predict_seq_loop']['wall_time_ms']:.2f} ms |"
        )

    lines.append("\n## 2. Пропускная способность (Throughput, samples/sec)\n")
    lines.append("| Batch Size | Native Batch | Explicit Loop | Predict (Dict) Loop | Predict (List) Loop |")
    lines.append("|-----------:|-------------:|--------------:|--------------------:|--------------------:|")
    for r in results:
        lines.append(
            f"| {r['n_samples']:,} | {r['native_batch']['samples_sec']:,.0f} "
            f"| {r['explicit_loop']['samples_sec']:,.0f} "
            f"| {r['predict_dict_loop']['samples_sec']:,.0f} "
            f"| {r['predict_seq_loop']['samples_sec']:,.0f} |"
        )

    lines.append("\n## 3. Накладные расходы на один сэмпл (Latency, us / sample)\n")
    lines.append("| Batch Size | Native Batch | Explicit Loop | Predict (Dict) Loop | Predict (List) Loop |")
    lines.append("|-----------:|-------------:|--------------:|--------------------:|--------------------:|")
    for r in results:
        lines.append(
            f"| {r['n_samples']:,} | {r['native_batch']['overhead_us']:.2f} µs "
            f"| {r['explicit_loop']['overhead_us']:.2f} µs "
            f"| {r['predict_dict_loop']['overhead_us']:.2f} µs "
            f"| {r['predict_seq_loop']['overhead_us']:.2f} µs |"
        )

    lines.append("\n## 4. Ускорение Native Batch (Speedup factor)\n")
    lines.append("| Batch Size | vs Explicit Loop | vs Predict (Dict) Loop | vs Predict (List) Loop |")
    lines.append("|-----------:|-----------------:|-----------------------:|-----------------------:|")
    for r in results:
        lines.append(
            f"| {r['n_samples']:,} | **{r['explicit_loop']['speedup']:.1f}x** "
            f"| **{r['predict_dict_loop']['speedup']:.1f}x** "
            f"| **{r['predict_seq_loop']['speedup']:.1f}x** |"
        )

    lines.append("\n## 5. Выводы и архитектурные рекомендации\n")
    lines.append("1. **Преимущество Native Batch**: на батчах от 1,000 сэмплов native batch работает в десятки раз быстрее циклов на чистом Python благодаря исключению накладных расходов перехода Python <-> C++ на каждом сэмпле.")
    lines.append("2. **Интеграция с `SugenoRegressor`**: метод `predict(X)` в Scikit-learn адаптере строго делегирует в native batch (`engine.predict(X)`), чтобы обеспечить высокую производительность при валидации и кросс-валидации.")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    branch, commit = get_git_info()
    results = run_benchmarks()
    report = format_markdown_report(results, branch, commit)

    output_path = Path(__file__).parent / "RESULTS.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] Benchmark report saved to: {output_path}")
