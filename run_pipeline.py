from __future__ import annotations

import time
from typing import Callable

from src.data import (
    dataset_audit,
    clean_interactions,
    clean_recipes,
    build_modelling_datasets,
    make_chronological_splits,
    build_features,
)
from src.models import (
    train_popularity,
    train_cf,
    train_svd,
    train_hybrid,
    train_bpr,
)
from src.evaluation import evaluate_models 
from src.evaluation import bias_coverage


PipelineStep = tuple[str, Callable[[], None]]


PIPELINE_STEPS: list[PipelineStep] = [
    ("Dataset Audit", dataset_audit.main),
    ("Clean Interactions", clean_interactions.main),
    ("Clean Recipes", clean_recipes.main),
    ("Build Modelling Datasets", build_modelling_datasets.main),
    ("Make Chronological Splits", make_chronological_splits.main),
    ("Build Features", build_features.main),
    ("Train Popularity", train_popularity.main),
    ("Train CF", train_cf.main),
    ("Train SVD", train_svd.main),
    ("Train Hybrid", train_hybrid.main),
    ("Train BPR", train_bpr.main),
    ("Evaluation", evaluate_models.main), 
    ("Bias and Coverage", bias_coverage.main)
]


def run_step(step_number: int, total_steps: int, step_name: str, step_func: Callable[[], None]) -> None:
    print(f"\n[{step_number}/{total_steps}] {step_name}...")
    start = time.perf_counter()

    try:
        step_func()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"[FAILED] {step_name} ({elapsed:.2f}s)")
        raise RuntimeError(f"Pipeline failed at step {step_number}/{total_steps}: {step_name}") from exc

    elapsed = time.perf_counter() - start
    print(f"[DONE] {step_name} ({elapsed:.2f}s)")


def main() -> None:
    total_steps = len(PIPELINE_STEPS)
    pipeline_start = time.perf_counter()

    print("=" * 80)
    print(f"Starting full pipeline ({total_steps} steps)")
    print("=" * 80)

    for step_number, (step_name, step_func) in enumerate(PIPELINE_STEPS, start=1):
        run_step(
            step_number=step_number,
            total_steps=total_steps,
            step_name=step_name,
            step_func=step_func,
        )

    total_elapsed = time.perf_counter() - pipeline_start

    print("\n" + "=" * 80)
    print(f"Pipeline completed successfully in {total_elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()