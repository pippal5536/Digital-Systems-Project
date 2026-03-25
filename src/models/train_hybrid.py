"""
src/models/train_hybrid.py

Purpose:
Train and evaluate a hybrid recommender on the chronological implicit splits.

This module combines two recommendation strategies:

1. truncated SVD matrix factorisation for mapped warm-start users
2. popularity fallback for users not covered by the train-fitted mapping

For mapped users, recommendation scores are blended using:
- a normalised SVD relevance score
- a train-only popularity prior

Responsibilities:
- load implicit chronological train, validation, and test splits
- quantify mapping coverage and routing eligibility
- fit truncated SVD on the train interaction matrix only
- tune the hybrid blend weight on validation data
- generate Top-N hybrid recommendations
- evaluate recommendation quality on mapped users
- summarise routing coverage for full validation and test users
- save dashboard-friendly and academic-report-friendly tables
- save accessible figures designed for readability and long-term reuse
- save reusable model artifacts for dashboard inference
- save a compact structured JSON run log

Design notes:
- all user and item mappings come from the train split to avoid leakage
- validation is used to tune alpha
- test is used only for final evaluation
- already-seen training items are excluded from recommendation lists
- popularity fallback is retained for practical coverage
- mapped-user metric evaluation mirrors the SVD evaluation setting
- outputs follow the existing project naming pattern with 10_* filenames
- model artifacts are saved to output/saved_models for dashboard loading
- figures are styled for readability, accessibility, and long-term reuse
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)


# Project imports

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.paths import (
    FIGURES_DIR,
    LOGS_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    ensure_directories,
)


# Configuration

MODEL_NAME = "hybrid"
MODEL_LABEL = "Hybrid (SVD + Popularity)"

TOP_K_VALUES = [5, 10, 20]
EXPORT_TOP_N = 10

N_COMPONENTS = 64
N_ITER = 10
RANDOM_STATE = 42
ALPHA_CANDIDATES = [0.60, 0.70, 0.80, 0.90, 1.00]

IMPLICIT_TRAIN_PATH = SPLITS_DIR / "implicit_train.parquet"
IMPLICIT_VALID_PATH = SPLITS_DIR / "implicit_valid.parquet"
IMPLICIT_TEST_PATH = SPLITS_DIR / "implicit_test.parquet"

TABLES_SUBDIR = TABLES_DIR / MODEL_NAME
FIGURES_SUBDIR = FIGURES_DIR / MODEL_NAME
LOGS_SUBDIR = LOGS_DIR / MODEL_NAME

SPLIT_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_split_summary.csv"
MAPPING_COVERAGE_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_mapping_coverage_summary.csv"
MATRIX_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_matrix_summary.csv"
COMPONENT_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_component_summary.csv"
ITEM_POPULARITY_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_item_popularity_table.csv"
ALPHA_TUNING_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_alpha_tuning.csv"
ALPHA_TUNING_ACADEMIC_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_alpha_tuning_academic.csv"
)

METRICS_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_metrics.csv"
USER_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_user_metrics.csv"
ROUTING_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_routing_summary.csv"

VALID_RECOMMENDATIONS_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_valid_recommendations.csv"
)
TEST_RECOMMENDATIONS_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_test_recommendations.csv"
VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_valid_recommendation_popularity.csv"
)
TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_test_recommendation_popularity.csv"
)

VALID_CONCENTRATION_CURVE_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_valid_recommendation_concentration_curve.csv"
)
TEST_CONCENTRATION_CURVE_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_test_recommendation_concentration_curve.csv"
)

DASHBOARD_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_metrics_dashboard.csv"
ACADEMIC_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_metrics_academic.csv"
ACADEMIC_COMPONENT_OUTPUT_PATH = (
    TABLES_SUBDIR / "10_hybrid_component_summary_academic.csv"
)
ACADEMIC_ROUTING_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_routing_summary_academic.csv"
DASHBOARD_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "10_hybrid_dashboard_summary.csv"

EXPLAINED_VARIANCE_FIGURE_PNG = (
    FIGURES_SUBDIR / "10_hybrid_cumulative_explained_variance.png"
)
EXPLAINED_VARIANCE_FIGURE_SVG = (
    FIGURES_SUBDIR / "10_hybrid_cumulative_explained_variance.svg"
)

ALPHA_TUNING_FIGURE_PNG = FIGURES_SUBDIR / "10_hybrid_alpha_tuning_ndcg_at_10.png"
ALPHA_TUNING_FIGURE_SVG = FIGURES_SUBDIR / "10_hybrid_alpha_tuning_ndcg_at_10.svg"

PRECISION_FIGURE_PNG = FIGURES_SUBDIR / "10_hybrid_precision_by_k.png"
PRECISION_FIGURE_SVG = FIGURES_SUBDIR / "10_hybrid_precision_by_k.svg"

RECALL_FIGURE_PNG = FIGURES_SUBDIR / "10_hybrid_recall_by_k.png"
RECALL_FIGURE_SVG = FIGURES_SUBDIR / "10_hybrid_recall_by_k.svg"

NDCG_FIGURE_PNG = FIGURES_SUBDIR / "10_hybrid_ndcg_by_k.png"
NDCG_FIGURE_SVG = FIGURES_SUBDIR / "10_hybrid_ndcg_by_k.svg"

VALID_CONCENTRATION_FIGURE_PNG = (
    FIGURES_SUBDIR / "10_hybrid_valid_recommendation_concentration_curve.png"
)
VALID_CONCENTRATION_FIGURE_SVG = (
    FIGURES_SUBDIR / "10_hybrid_valid_recommendation_concentration_curve.svg"
)

TEST_CONCENTRATION_FIGURE_PNG = (
    FIGURES_SUBDIR / "10_hybrid_test_recommendation_concentration_curve.png"
)
TEST_CONCENTRATION_FIGURE_SVG = (
    FIGURES_SUBDIR / "10_hybrid_test_recommendation_concentration_curve.svg"
)

USER_ACTIVITY_FIGURE_PNG = (
    FIGURES_SUBDIR / "10_hybrid_train_user_interaction_distribution.png"
)
USER_ACTIVITY_FIGURE_SVG = (
    FIGURES_SUBDIR / "10_hybrid_train_user_interaction_distribution.svg"
)

ITEM_ACTIVITY_FIGURE_PNG = (
    FIGURES_SUBDIR / "10_hybrid_train_item_interaction_distribution.png"
)
ITEM_ACTIVITY_FIGURE_SVG = (
    FIGURES_SUBDIR / "10_hybrid_train_item_interaction_distribution.svg"
)

LOG_OUTPUT_PATH = LOGS_SUBDIR / "10_hybrid_run_log.json"

MODEL_DIR = PROJECT_ROOT / "outputs" / "saved_models"
MODEL_OUTPUT_PATH = MODEL_DIR / "10_hybrid_model.joblib"
MODEL_METADATA_OUTPUT_PATH = MODEL_DIR / "10_hybrid_model_metadata.json"


# Helper functions


def ensure_output_directories() -> None:
    """
    Ensure output directories exist.
    """
    ensure_directories()
    TABLES_SUBDIR.mkdir(parents=True, exist_ok=True)
    FIGURES_SUBDIR.mkdir(parents=True, exist_ok=True)
    LOGS_SUBDIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_files() -> None:
    """
    Validate that all required input files exist.
    """
    required_paths = [
        IMPLICIT_TRAIN_PATH,
        IMPLICIT_VALID_PATH,
        IMPLICIT_TEST_PATH,
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Missing required hybrid input files:\n" + "\n".join(missing_paths)
        )


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """
    Load one parquet split file.
    """
    df = pd.read_parquet(path).copy()
    print(f"{split_name} shape: {df.shape}")
    return df


def validate_required_columns(
    df: pd.DataFrame,
    split_name: str,
    required_cols: list[str],
) -> None:
    """
    Validate required columns in a dataframe.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split is missing required columns: {missing}")


def build_split_summary(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary table for the three implicit splits.
    """
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": int(len(implicit_train)),
                "users": int(implicit_train["user_id"].nunique()),
                "recipes": int(implicit_train["recipe_id"].nunique()),
                "min_date": implicit_train["date"].min(),
                "max_date": implicit_train["date"].max(),
            },
            {
                "split": "valid",
                "rows": int(len(implicit_valid)),
                "users": int(implicit_valid["user_id"].nunique()),
                "recipes": int(implicit_valid["recipe_id"].nunique()),
                "min_date": implicit_valid["date"].min(),
                "max_date": implicit_valid["date"].max(),
            },
            {
                "split": "test",
                "rows": int(len(implicit_test)),
                "users": int(implicit_test["user_id"].nunique()),
                "recipes": int(implicit_test["recipe_id"].nunique()),
                "min_date": implicit_test["date"].min(),
                "max_date": implicit_test["date"].max(),
            },
        ]
    )


def summarise_hybrid_coverage(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Summarise mapping coverage and routing eligibility for one split.
    """
    rows = int(len(df))
    rows_with_both = int((df["user_idx"].notna() & df["item_idx"].notna()).sum())
    rows_with_user_only = int((df["user_idx"].notna() & df["item_idx"].isna()).sum())

    return pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": rows,
                "missing_user_idx": int(df["user_idx"].isna().sum()),
                "missing_item_idx": int(df["item_idx"].isna().sum()),
                "rows_with_both_indices": rows_with_both,
                "rows_with_user_idx_only": rows_with_user_only,
                "rows_evaluable_by_svd_pct": (
                    round((rows_with_both / rows) * 100, 2) if rows else 0.0
                ),
                "rows_with_known_user_pct": (
                    round(((rows_with_both + rows_with_user_only) / rows) * 100, 2)
                    if rows
                    else 0.0
                ),
            }
        ]
    )


def filter_evaluable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with both user_idx and item_idx available.
    """
    out = df[df["user_idx"].notna() & df["item_idx"].notna()].copy()
    out["user_idx"] = out["user_idx"].astype(int)
    out["item_idx"] = out["item_idx"].astype(int)
    return out


def build_train_matrix(train_df: pd.DataFrame) -> sparse.csr_matrix:
    """
    Build a sparse user-item interaction matrix from training interactions.
    """
    n_users = int(train_df["user_idx"].max()) + 1
    n_items = int(train_df["item_idx"].max()) + 1

    return sparse.csr_matrix(
        (
            train_df["implicit_feedback"].astype(float).to_numpy(),
            (train_df["user_idx"].to_numpy(), train_df["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )


def build_matrix_summary(
    train_matrix: sparse.csr_matrix,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact matrix summary table.
    """
    n_users, n_items = train_matrix.shape
    density = (
        train_matrix.nnz / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0
    )

    return pd.DataFrame(
        [
            {
                "n_users": int(n_users),
                "n_items": int(n_items),
                "nnz": int(train_matrix.nnz),
                "density": float(density),
                "train_rows": int(len(train_df)),
                "unique_user_ids": int(train_df["user_id"].nunique()),
                "unique_recipe_ids": int(train_df["recipe_id"].nunique()),
            }
        ]
    )


def build_item_popularity_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an item popularity table from training interactions only.
    """
    item_popularity = (
        train_df.groupby(["item_idx", "recipe_id"], as_index=False)
        .agg(
            train_interaction_count=("implicit_feedback", "sum"),
            train_user_count=("user_id", "nunique"),
            first_seen_date=("date", "min"),
            last_seen_date=("date", "max"),
        )
        .sort_values(
            by=["train_interaction_count", "train_user_count", "item_idx"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    item_popularity["item_popularity_rank"] = np.arange(1, len(item_popularity) + 1)
    item_popularity["popularity_score_raw"] = np.log1p(
        item_popularity["train_interaction_count"]
    )

    pop_scaler = MinMaxScaler()
    item_popularity["popularity_score"] = pop_scaler.fit_transform(
        item_popularity[["popularity_score_raw"]]
    )

    total_interactions = item_popularity["train_interaction_count"].sum()
    item_popularity["interaction_share"] = (
        item_popularity["train_interaction_count"] / total_interactions
    )
    item_popularity["cumulative_interaction_share"] = item_popularity[
        "interaction_share"
    ].cumsum()
    item_popularity["cumulative_item_share"] = np.arange(
        1, len(item_popularity) + 1
    ) / len(item_popularity)

    return item_popularity


def build_seen_items_by_user(train_df: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a mapping of user_idx to seen item_idx values from training.
    """
    return (
        train_df.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(values.astype(int).tolist()))
        .to_dict()
    )


def build_truth_lookup(holdout_df: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a mapping of user_idx to relevant holdout item_idx values.
    """
    return (
        holdout_df.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(values.astype(int).tolist()))
        .to_dict()
    )


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Return Precision@K.
    """
    if k <= 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Return Recall@K.
    """
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Return Hit Rate@K.
    """
    return float(len(set(recommended[:k]) & relevant) > 0)


def dcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Return DCG@K.
    """
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Return nDCG@K.
    """
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / ideal_dcg


def novelty_at_k(
    recommended: list[int],
    popularity_lookup: dict[int, int],
    total_events: int,
    k: int,
) -> float:
    """
    Return novelty@K using train interaction frequency.
    """
    values: list[float] = []
    for item in recommended[:k]:
        pop = popularity_lookup.get(item, 1)
        values.append(-math.log2(pop / total_events))
    return float(np.mean(values)) if values else 0.0


def score_user_all_items(
    user_idx: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> np.ndarray:
    """
    Return raw SVD scores for all items for one mapped user.
    """
    return user_factors[user_idx] @ item_factors.T


def recommend_top_n_popularity(
    user_idx: int,
    seen_items_by_user: dict[int, set[int]],
    ranked_popular_items: list[int],
    top_n: int,
) -> list[int]:
    """
    Return popularity fallback recommendations for one user.
    """
    seen = seen_items_by_user.get(user_idx, set())
    recommendations: list[int] = []

    for item in ranked_popular_items:
        if item not in seen:
            recommendations.append(int(item))
        if len(recommendations) == top_n:
            break

    return recommendations


def recommend_top_n_hybrid_mapped_user(
    user_idx: int,
    alpha: float,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items_by_user: dict[int, set[int]],
    popularity_score_lookup: dict[int, float],
    n_items: int,
    top_n: int,
) -> tuple[list[int], list[float]]:
    """
    Return blended Top-N recommendations for one mapped user.

    hybrid_score = alpha * normalised_svd_score + (1 - alpha) * popularity_score
    """
    raw_scores = score_user_all_items(user_idx, user_factors, item_factors).copy()

    seen = seen_items_by_user.get(user_idx, set())
    if seen:
        raw_scores[list(seen)] = -np.inf

    finite_mask = np.isfinite(raw_scores)
    scaled_svd_scores = np.full_like(raw_scores, fill_value=-np.inf, dtype=float)

    if finite_mask.sum() > 0:
        finite_values = raw_scores[finite_mask]
        min_value = finite_values.min()
        max_value = finite_values.max()

        if max_value > min_value:
            scaled_svd_scores[finite_mask] = (finite_values - min_value) / (
                max_value - min_value
            )
        else:
            scaled_svd_scores[finite_mask] = 0.0

    popularity_scores = np.zeros(n_items, dtype=float)
    for item_idx, score in popularity_score_lookup.items():
        popularity_scores[item_idx] = score

    hybrid_scores = alpha * scaled_svd_scores + (1.0 - alpha) * popularity_scores
    hybrid_scores[~finite_mask] = -np.inf

    top_items = np.argpartition(hybrid_scores, -top_n)[-top_n:]
    top_items = top_items[np.argsort(hybrid_scores[top_items])[::-1]]
    top_scores = hybrid_scores[top_items]

    return top_items.tolist(), top_scores.astype(float).tolist()


def evaluate_hybrid_on_validation(
    alpha: float,
    valid_truth: dict[int, set[int]],
    top_k_values: list[int],
    train_event_count: int,
    popularity_lookup: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items_by_user: dict[int, set[int]],
    popularity_score_lookup: dict[int, float],
    n_items: int,
) -> pd.DataFrame:
    """
    Evaluate one alpha candidate on mapped validation users.
    """
    rows: list[dict[str, Any]] = []
    all_recommended_items = {k: set() for k in top_k_values}

    for user_idx, relevant_items in valid_truth.items():
        user_recommendations, _ = recommend_top_n_hybrid_mapped_user(
            user_idx=user_idx,
            alpha=alpha,
            user_factors=user_factors,
            item_factors=item_factors,
            seen_items_by_user=seen_items_by_user,
            popularity_score_lookup=popularity_score_lookup,
            n_items=n_items,
            top_n=max(top_k_values),
        )

        for k in top_k_values:
            rec_k = user_recommendations[:k]
            all_recommended_items[k].update(rec_k)

            rows.append(
                {
                    "alpha": float(alpha),
                    "user_idx": int(user_idx),
                    "k": int(k),
                    "precision_at_k": precision_at_k(rec_k, relevant_items, k),
                    "recall_at_k": recall_at_k(rec_k, relevant_items, k),
                    "hit_rate_at_k": hit_rate_at_k(rec_k, relevant_items, k),
                    "ndcg_at_k": ndcg_at_k(rec_k, relevant_items, k),
                    "novelty_at_k": novelty_at_k(
                        rec_k,
                        popularity_lookup,
                        train_event_count,
                        k,
                    ),
                    "catalog_coverage_at_k": float(
                        len(all_recommended_items[k]) / n_items
                    ),
                }
            )

    return pd.DataFrame(rows)


def evaluate_top_n_hybrid(
    truth_lookup: dict[int, set[int]],
    split_name: str,
    top_k_values: list[int],
    alpha: float,
    export_top_n: int,
    train_event_count: int,
    popularity_lookup: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items_by_user: dict[int, set[int]],
    popularity_score_lookup: dict[int, float],
    n_items: int,
    recipe_lookup: dict[int, int],
    item_popularity_lookup: dict[int, dict[str, float | int]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate mapped-user hybrid Top-N performance and export recommendations.
    """
    metric_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    user_summary_rows: list[dict[str, Any]] = []
    all_recommended_items = {k: set() for k in top_k_values}

    for user_idx, relevant_items in truth_lookup.items():
        user_recommendations, user_scores = recommend_top_n_hybrid_mapped_user(
            user_idx=user_idx,
            alpha=alpha,
            user_factors=user_factors,
            item_factors=item_factors,
            seen_items_by_user=seen_items_by_user,
            popularity_score_lookup=popularity_score_lookup,
            n_items=n_items,
            top_n=max(max(top_k_values), export_top_n),
        )

        for rank, (item, score) in enumerate(
            zip(user_recommendations[:export_top_n], user_scores[:export_top_n]),
            start=1,
        ):
            popularity_row = item_popularity_lookup.get(item, {})

            export_rows.append(
                {
                    "split": split_name,
                    "route": "svd_hybrid",
                    "alpha": float(alpha),
                    "user_idx": int(user_idx),
                    "recipe_id": int(recipe_lookup.get(item, -1)),
                    "item_idx": int(item),
                    "recommendation_rank": int(rank),
                    "score": float(score),
                    "global_item_popularity_rank": popularity_row.get(
                        "item_popularity_rank"
                    ),
                    "train_interaction_count": popularity_row.get(
                        "train_interaction_count"
                    ),
                    "popularity_score": popularity_row.get("popularity_score"),
                }
            )

        user_summary: dict[str, Any] = {
            "split": split_name,
            "user_idx": int(user_idx),
            "alpha": float(alpha),
            "relevant_count": int(len(relevant_items)),
        }

        for k in top_k_values:
            rec_k = user_recommendations[:k]
            all_recommended_items[k].update(rec_k)

            precision_value = precision_at_k(rec_k, relevant_items, k)
            recall_value = recall_at_k(rec_k, relevant_items, k)
            hit_rate_value = hit_rate_at_k(rec_k, relevant_items, k)
            ndcg_value = ndcg_at_k(rec_k, relevant_items, k)
            novelty_value = novelty_at_k(
                rec_k,
                popularity_lookup,
                train_event_count,
                k,
            )

            user_summary[f"precision_at_{k}"] = precision_value
            user_summary[f"recall_at_{k}"] = recall_value
            user_summary[f"hit_rate_at_{k}"] = hit_rate_value
            user_summary[f"ndcg_at_{k}"] = ndcg_value
            user_summary[f"novelty_at_{k}"] = novelty_value

            metric_rows.append(
                {
                    "split": split_name,
                    "user_idx": int(user_idx),
                    "k": int(k),
                    "alpha": float(alpha),
                    "precision_at_k": precision_value,
                    "recall_at_k": recall_value,
                    "hit_rate_at_k": hit_rate_value,
                    "ndcg_at_k": ndcg_value,
                    "novelty_at_k": novelty_value,
                    "relevant_count": int(len(relevant_items)),
                }
            )

        user_summary_rows.append(user_summary)

    user_metric_rows = pd.DataFrame(metric_rows)
    recommendation_export = pd.DataFrame(export_rows)
    user_summary_df = pd.DataFrame(user_summary_rows)

    summary_rows: list[dict[str, Any]] = []
    for k in top_k_values:
        subset = user_metric_rows[user_metric_rows["k"] == k].copy()
        user_count = int(subset["user_idx"].nunique())

        summary_rows.append(
            {
                "split": split_name,
                "k": int(k),
                "alpha": float(alpha),
                "users_evaluated": user_count,
                "precision_at_k": float(subset["precision_at_k"].mean()),
                "recall_at_k": float(subset["recall_at_k"].mean()),
                "hit_rate_at_k": float(subset["hit_rate_at_k"].mean()),
                "ndcg_at_k": float(subset["ndcg_at_k"].mean()),
                "novelty_at_k": float(subset["novelty_at_k"].mean()),
                "catalog_coverage_at_k": float(len(all_recommended_items[k]) / n_items),
                "recommendation_count": int(user_count * k),
            }
        )

    metrics_summary = pd.DataFrame(summary_rows)
    return metrics_summary, recommendation_export, user_summary_df


def summarise_user_routing(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Summarise how many full-split users are routed to hybrid vs fallback.
    """
    user_level = (
        df.groupby("user_id")["user_idx"]
        .apply(lambda series: series.notna().any())
        .reset_index(name="has_known_user_idx")
    )

    user_level["route"] = np.where(
        user_level["has_known_user_idx"],
        "svd_hybrid",
        "popularity_fallback",
    )
    user_level["split"] = split_name

    summary = user_level.groupby(["split", "route"]).size().reset_index(name="users")
    summary["user_share_pct"] = round(
        summary["users"] / summary["users"].sum() * 100, 2
    )
    return summary


def build_recommendation_popularity_summary(
    recommendations_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Summarise how concentrated the recommendation pool is.
    """
    if recommendations_df.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "item_idx",
                "recipe_id",
                "times_recommended",
                "train_interaction_count",
                "item_popularity_rank",
                "popularity_score",
            ]
        )

    summary = (
        recommendations_df.groupby(
            [
                "item_idx",
                "recipe_id",
                "train_interaction_count",
                "global_item_popularity_rank",
                "popularity_score",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "times_recommended",
                "global_item_popularity_rank": "item_popularity_rank",
            }
        )
        .sort_values(["times_recommended", "item_idx"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["split"] = split_name

    return summary[
        [
            "split",
            "item_idx",
            "recipe_id",
            "times_recommended",
            "train_interaction_count",
            "item_popularity_rank",
            "popularity_score",
        ]
    ].copy()


def build_recommendation_concentration_curve(
    recommendation_popularity_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a cumulative recommendation concentration curve table.
    """
    if recommendation_popularity_summary.empty:
        return pd.DataFrame(
            columns=[
                "times_recommended",
                "recommendation_share",
                "cumulative_recommendation_share",
                "cumulative_item_share",
            ]
        )

    curve_df = recommendation_popularity_summary.copy()
    total_recommendations = curve_df["times_recommended"].sum()

    curve_df["recommendation_share"] = (
        curve_df["times_recommended"] / total_recommendations
    )
    curve_df["cumulative_recommendation_share"] = curve_df[
        "recommendation_share"
    ].cumsum()
    curve_df["cumulative_item_share"] = np.arange(1, len(curve_df) + 1) / len(curve_df)

    return curve_df[
        [
            "times_recommended",
            "recommendation_share",
            "cumulative_recommendation_share",
            "cumulative_item_share",
        ]
    ].copy()


def build_dashboard_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dashboard-friendly model comparison table.
    """
    dashboard_df = metrics_df.copy()

    dashboard_df["model"] = MODEL_LABEL
    dashboard_df["split_k"] = (
        dashboard_df["split"].str.upper() + " @ " + dashboard_df["k"].astype(str)
    )

    dashboard_df = dashboard_df[
        [
            "model",
            "split",
            "k",
            "alpha",
            "split_k",
            "users_evaluated",
            "precision_at_k",
            "recall_at_k",
            "hit_rate_at_k",
            "ndcg_at_k",
            "novelty_at_k",
            "catalog_coverage_at_k",
            "recommendation_count",
        ]
    ].copy()

    numeric_cols = [
        "alpha",
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "ndcg_at_k",
        "novelty_at_k",
        "catalog_coverage_at_k",
    ]
    dashboard_df[numeric_cols] = dashboard_df[numeric_cols].round(4)

    return dashboard_df


def build_academic_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an academic-report-friendly metric summary table.
    """
    academic_df = metrics_df.copy()

    academic_df["Model"] = MODEL_LABEL
    academic_df["Split"] = academic_df["split"].str.capitalize()
    academic_df["K"] = academic_df["k"]
    academic_df["Alpha"] = academic_df["alpha"].round(2)

    academic_df["Users Evaluated"] = academic_df["users_evaluated"].astype(int)
    academic_df["Precision@K"] = academic_df["precision_at_k"].round(4)
    academic_df["Recall@K"] = academic_df["recall_at_k"].round(4)
    academic_df["Hit Rate@K"] = academic_df["hit_rate_at_k"].round(4)
    academic_df["nDCG@K"] = academic_df["ndcg_at_k"].round(4)
    academic_df["Novelty@K"] = academic_df["novelty_at_k"].round(4)
    academic_df["Catalogue Coverage@K"] = academic_df["catalog_coverage_at_k"].round(4)

    return academic_df[
        [
            "Model",
            "Split",
            "K",
            "Alpha",
            "Users Evaluated",
            "Precision@K",
            "Recall@K",
            "Hit Rate@K",
            "nDCG@K",
            "Novelty@K",
            "Catalogue Coverage@K",
        ]
    ].copy()


def build_academic_component_table(component_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build an academic-report-friendly component summary table.
    """
    academic_df = component_summary.copy()

    academic_df["Component"] = academic_df["component"].astype(int)
    academic_df["Explained Variance Ratio"] = academic_df[
        "explained_variance_ratio"
    ].round(6)
    academic_df["Cumulative Explained Variance Ratio"] = academic_df[
        "cumulative_explained_variance_ratio"
    ].round(6)

    return academic_df[
        [
            "Component",
            "Explained Variance Ratio",
            "Cumulative Explained Variance Ratio",
        ]
    ].copy()


def build_academic_routing_table(routing_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build an academic-report-friendly routing summary table.
    """
    academic_df = routing_summary.copy()
    academic_df["Split"] = academic_df["split"].str.capitalize()
    academic_df["Route"] = academic_df["route"].str.replace("_", " ").str.title()
    academic_df["Users"] = academic_df["users"].astype(int)
    academic_df["User Share (%)"] = academic_df["user_share_pct"].round(2)

    return academic_df[["Split", "Route", "Users", "User Share (%)"]].copy()


def build_academic_alpha_tuning_table(
    alpha_tuning_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build an academic-report-friendly alpha tuning table.
    """
    academic_df = alpha_tuning_summary.copy()
    academic_df["Alpha"] = academic_df["alpha"].round(2)
    academic_df["K"] = academic_df["k"].astype(int)
    academic_df["Users Evaluated"] = academic_df["users_evaluated"].astype(int)
    academic_df["Precision@K"] = academic_df["precision_at_k"].round(4)
    academic_df["Recall@K"] = academic_df["recall_at_k"].round(4)
    academic_df["Hit Rate@K"] = academic_df["hit_rate_at_k"].round(4)
    academic_df["nDCG@K"] = academic_df["ndcg_at_k"].round(4)
    academic_df["Novelty@K"] = academic_df["novelty_at_k"].round(4)
    academic_df["Catalogue Coverage@K"] = academic_df["catalog_coverage_at_k"].round(4)

    return academic_df[
        [
            "Alpha",
            "K",
            "Users Evaluated",
            "Precision@K",
            "Recall@K",
            "Hit Rate@K",
            "nDCG@K",
            "Novelty@K",
            "Catalogue Coverage@K",
        ]
    ].copy()


def build_dashboard_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact dashboard summary table.
    """
    summary_df = metrics_df.copy()
    summary_df["model"] = MODEL_NAME
    summary_df["label"] = (
        summary_df["split"].str.upper() + "_AT_" + summary_df["k"].astype(str)
    )

    summary_df["precision"] = summary_df["precision_at_k"].round(4)
    summary_df["recall"] = summary_df["recall_at_k"].round(4)
    summary_df["hit_rate"] = summary_df["hit_rate_at_k"].round(4)
    summary_df["ndcg"] = summary_df["ndcg_at_k"].round(4)
    summary_df["novelty"] = summary_df["novelty_at_k"].round(4)
    summary_df["coverage"] = summary_df["catalog_coverage_at_k"].round(4)
    summary_df["alpha"] = summary_df["alpha"].round(2)

    return summary_df[
        [
            "model",
            "label",
            "split",
            "k",
            "alpha",
            "precision",
            "recall",
            "hit_rate",
            "ndcg",
            "novelty",
            "coverage",
            "users_evaluated",
        ]
    ].copy()


def plot_accessible_histogram(
    values: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible histogram figure.
    """
    plt.figure(figsize=(9.5, 6.5))
    plt.hist(values, bins=50, edgecolor="black", linewidth=0.6)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def plot_accessible_metric_lines(
    metrics_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible metric-by-K figure for validation and test.
    """
    plt.figure(figsize=(9.5, 6.5))

    line_styles = {
        "valid": {"marker": "o", "linestyle": "-", "label": "Validation"},
        "test": {"marker": "s", "linestyle": "--", "label": "Test"},
    }

    for split_name in ["valid", "test"]:
        subset = metrics_df[metrics_df["split"] == split_name].copy()
        if subset.empty:
            continue

        style = line_styles[split_name]

        plt.plot(
            subset["k"],
            subset[metric_col],
            linewidth=2.2,
            markersize=6,
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=style["label"],
        )

    plt.xlabel("K", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def plot_accessible_concentration_curve(
    curve_df: pd.DataFrame,
    title: str,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible cumulative recommendation concentration figure.
    """
    if curve_df.empty:
        return

    plt.figure(figsize=(9.5, 6.5))

    plt.plot(
        curve_df["cumulative_item_share"],
        curve_df["cumulative_recommendation_share"],
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        markevery=max(1, len(curve_df) // 20),
        label="Observed recommendation concentration",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Equality reference",
    )

    plt.xlabel("Cumulative share of recommended items", fontsize=12)
    plt.ylabel("Cumulative share of recommendation volume", fontsize=12)
    plt.title(title, fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def plot_accessible_explained_variance_curve(
    component_summary: pd.DataFrame,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible cumulative explained variance figure.
    """
    plt.figure(figsize=(9.5, 6.5))

    plt.plot(
        component_summary["component"],
        component_summary["cumulative_explained_variance_ratio"],
        linewidth=2.4,
        marker="o",
        markersize=4,
        markevery=max(1, len(component_summary) // 20),
        label="Cumulative explained variance",
    )

    plt.xlabel("Number of latent components", fontsize=12)
    plt.ylabel("Cumulative explained variance ratio", fontsize=12)
    plt.title("Hybrid SVD component explained variance", fontsize=13)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def plot_accessible_alpha_tuning_curve(
    alpha_tuning_summary: pd.DataFrame,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible validation alpha tuning figure.
    """
    subset = alpha_tuning_summary[alpha_tuning_summary["k"] == 10].copy()

    plt.figure(figsize=(9.5, 6.5))
    plt.plot(
        subset["alpha"],
        subset["ndcg_at_k"],
        linewidth=2.4,
        marker="o",
        markersize=6,
        label="Validation nDCG@10",
    )

    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("nDCG@10", fontsize=12)
    plt.title("Hybrid alpha tuning on validation data", fontsize=13)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def save_model_artifact(
    svd_model: TruncatedSVD,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    recipe_lookup: dict[int, int],
    seen_items_by_user: dict[int, set[int]],
    ranked_popular_items: list[int],
    popularity_score_lookup: dict[int, float],
    item_popularity: pd.DataFrame,
    best_alpha: float,
) -> None:
    """
    Save the hybrid artifact for dashboard reuse.
    """
    model_artifact = {
        "model_name": MODEL_NAME,
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_label": MODEL_LABEL,
        "n_components": N_COMPONENTS,
        "n_iter": N_ITER,
        "random_state": RANDOM_STATE,
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "alpha_candidates": ALPHA_CANDIDATES,
        "best_alpha": best_alpha,
        "svd_model": svd_model,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "recipe_lookup": recipe_lookup,
        "seen_items_by_user": seen_items_by_user,
        "ranked_popular_items": ranked_popular_items,
        "popularity_score_lookup": popularity_score_lookup,
        "item_popularity_table": item_popularity,
    }

    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_output_path": str(MODEL_OUTPUT_PATH),
        "n_components": N_COMPONENTS,
        "n_iter": N_ITER,
        "random_state": RANDOM_STATE,
        "best_alpha": best_alpha,
        "alpha_candidates": ALPHA_CANDIDATES,
        "user_factor_shape": [int(user_factors.shape[0]), int(user_factors.shape[1])],
        "item_factor_shape": [int(item_factors.shape[0]), int(item_factors.shape[1])],
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "fields_available": [
            "svd_model",
            "user_factors",
            "item_factors",
            "recipe_lookup",
            "seen_items_by_user",
            "ranked_popular_items",
            "popularity_score_lookup",
            "item_popularity_table",
            "best_alpha",
        ],
    }

    with open(MODEL_METADATA_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_run_log(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
    train_hybrid: pd.DataFrame,
    valid_hybrid: pd.DataFrame,
    test_hybrid: pd.DataFrame,
    train_matrix: sparse.csr_matrix,
    coverage_summary: pd.DataFrame,
    routing_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    alpha_tuning_summary: pd.DataFrame,
    hybrid_metrics: pd.DataFrame,
    best_alpha: float,
) -> None:
    """
    Save a structured JSON log for monitoring and dashboard use.
    """
    best_row = hybrid_metrics.sort_values(
        by=["split", "ndcg_at_k", "recall_at_k"],
        ascending=[True, False, False],
    ).iloc[0]

    n_users, n_items = train_matrix.shape
    density = (
        train_matrix.nnz / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0
    )

    run_log = {
        "model": MODEL_NAME,
        "model_label": MODEL_LABEL,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "train_rows_raw": int(len(implicit_train)),
            "valid_rows_raw": int(len(implicit_valid)),
            "test_rows_raw": int(len(implicit_test)),
            "train_rows_evaluable": int(len(train_hybrid)),
            "valid_rows_evaluable": int(len(valid_hybrid)),
            "test_rows_evaluable": int(len(test_hybrid)),
            "train_users_raw": int(implicit_train["user_id"].nunique()),
            "valid_users_raw": int(implicit_valid["user_id"].nunique()),
            "test_users_raw": int(implicit_test["user_id"].nunique()),
            "train_recipes_raw": int(implicit_train["recipe_id"].nunique()),
            "valid_recipes_raw": int(implicit_valid["recipe_id"].nunique()),
            "test_recipes_raw": int(implicit_test["recipe_id"].nunique()),
        },
        "mapping_coverage": coverage_summary.to_dict(orient="records"),
        "routing_summary": routing_summary.to_dict(orient="records"),
        "matrix_summary": {
            "shape": [int(n_users), int(n_items)],
            "nnz": int(train_matrix.nnz),
            "density": float(density),
        },
        "config": {
            "n_components": N_COMPONENTS,
            "n_iter": N_ITER,
            "random_state": RANDOM_STATE,
            "alpha_candidates": ALPHA_CANDIDATES,
            "best_alpha": best_alpha,
            "top_k_values": TOP_K_VALUES,
            "export_top_n": EXPORT_TOP_N,
        },
        "explained_variance": {
            "explained_variance_ratio_sum": float(
                component_summary["explained_variance_ratio"].sum()
            )
        },
        "alpha_tuning_summary": alpha_tuning_summary.to_dict(orient="records"),
        "artifacts": {
            "split_summary_csv": str(SPLIT_SUMMARY_OUTPUT_PATH),
            "mapping_coverage_csv": str(MAPPING_COVERAGE_OUTPUT_PATH),
            "matrix_summary_csv": str(MATRIX_SUMMARY_OUTPUT_PATH),
            "component_summary_csv": str(COMPONENT_SUMMARY_OUTPUT_PATH),
            "alpha_tuning_csv": str(ALPHA_TUNING_OUTPUT_PATH),
            "metrics_csv": str(METRICS_OUTPUT_PATH),
            "dashboard_metrics_csv": str(DASHBOARD_METRICS_OUTPUT_PATH),
            "academic_metrics_csv": str(ACADEMIC_METRICS_OUTPUT_PATH),
            "dashboard_summary_csv": str(DASHBOARD_SUMMARY_OUTPUT_PATH),
            "model_joblib": str(MODEL_OUTPUT_PATH),
            "model_metadata_json": str(MODEL_METADATA_OUTPUT_PATH),
        },
        "best_metric_row": {
            "split": str(best_row["split"]),
            "k": int(best_row["k"]),
            "alpha": float(best_row["alpha"]),
            "precision_at_k": float(best_row["precision_at_k"]),
            "recall_at_k": float(best_row["recall_at_k"]),
            "hit_rate_at_k": float(best_row["hit_rate_at_k"]),
            "ndcg_at_k": float(best_row["ndcg_at_k"]),
            "novelty_at_k": float(best_row["novelty_at_k"]),
            "catalog_coverage_at_k": float(best_row["catalog_coverage_at_k"]),
        },
        "metrics": hybrid_metrics.to_dict(orient="records"),
    }

    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(run_log, file, indent=2)


def main() -> None:
    """
    Run the full hybrid training and evaluation pipeline.
    """
    ensure_output_directories()
    validate_input_files()

    implicit_train = load_split(IMPLICIT_TRAIN_PATH, "implicit_train")
    implicit_valid = load_split(IMPLICIT_VALID_PATH, "implicit_valid")
    implicit_test = load_split(IMPLICIT_TEST_PATH, "implicit_test")

    required_cols = [
        "user_id",
        "recipe_id",
        "date",
        "implicit_feedback",
        "user_idx",
        "item_idx",
    ]

    validate_required_columns(implicit_train, "train", required_cols)
    validate_required_columns(implicit_valid, "valid", required_cols)
    validate_required_columns(implicit_test, "test", required_cols)

    for df in [implicit_train, implicit_valid, implicit_test]:
        df["date"] = pd.to_datetime(df["date"])

    split_summary = build_split_summary(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
    )

    coverage_summary = pd.concat(
        [
            summarise_hybrid_coverage(implicit_train, "train"),
            summarise_hybrid_coverage(implicit_valid, "valid"),
            summarise_hybrid_coverage(implicit_test, "test"),
        ],
        ignore_index=True,
    )

    print("\nSplit summary:")
    print(split_summary)

    print("\nMapping coverage summary:")
    print(coverage_summary)

    train_hybrid = filter_evaluable_rows(implicit_train)
    valid_hybrid = filter_evaluable_rows(implicit_valid)
    test_hybrid = filter_evaluable_rows(implicit_test)

    print("\nEvaluable shapes:")
    print("train:", train_hybrid.shape)
    print("valid:", valid_hybrid.shape)
    print("test:", test_hybrid.shape)

    train_matrix = build_train_matrix(train_hybrid)
    matrix_summary = build_matrix_summary(train_matrix, train_hybrid)

    print("\nTrain matrix summary:")
    print(matrix_summary)

    user_interaction_counts = np.asarray(train_matrix.getnnz(axis=1)).ravel()
    item_interaction_counts = np.asarray(train_matrix.getnnz(axis=0)).ravel()

    plot_accessible_histogram(
        values=user_interaction_counts,
        xlabel="Interactions per user",
        ylabel="Number of users",
        title="Training user interaction count distribution",
        output_path_png=USER_ACTIVITY_FIGURE_PNG,
        output_path_svg=USER_ACTIVITY_FIGURE_SVG,
    )

    plot_accessible_histogram(
        values=item_interaction_counts,
        xlabel="Interactions per item",
        ylabel="Number of items",
        title="Training item interaction count distribution",
        output_path_png=ITEM_ACTIVITY_FIGURE_PNG,
        output_path_svg=ITEM_ACTIVITY_FIGURE_SVG,
    )

    svd_model = TruncatedSVD(
        n_components=N_COMPONENTS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
    )

    user_factors = svd_model.fit_transform(train_matrix)
    item_factors = svd_model.components_.T

    component_summary = pd.DataFrame(
        {
            "component": np.arange(1, len(svd_model.explained_variance_ratio_) + 1),
            "explained_variance_ratio": svd_model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                svd_model.explained_variance_ratio_
            ),
        }
    )

    plot_accessible_explained_variance_curve(
        component_summary=component_summary,
        output_path_png=EXPLAINED_VARIANCE_FIGURE_PNG,
        output_path_svg=EXPLAINED_VARIANCE_FIGURE_SVG,
    )

    seen_items_by_user = build_seen_items_by_user(train_hybrid)
    valid_truth = build_truth_lookup(valid_hybrid)
    test_truth = build_truth_lookup(test_hybrid)

    item_popularity = build_item_popularity_table(train_hybrid)

    popularity_lookup = item_popularity.set_index("item_idx")[
        "train_interaction_count"
    ].to_dict()
    popularity_score_lookup = item_popularity.set_index("item_idx")[
        "popularity_score"
    ].to_dict()

    recipe_lookup = (
        train_hybrid[["item_idx", "recipe_id"]]
        .drop_duplicates()
        .set_index("item_idx")["recipe_id"]
        .to_dict()
    )
    user_lookup = train_hybrid[["user_idx", "user_id"]].drop_duplicates()

    ranked_popular_items = item_popularity["item_idx"].astype(int).tolist()

    item_popularity_lookup = item_popularity.set_index("item_idx")[
        ["item_popularity_rank", "train_interaction_count", "popularity_score"]
    ].to_dict("index")

    train_event_count = int(len(train_hybrid))
    n_items = train_matrix.shape[1]

    alpha_tuning_rows = (
        pd.concat(
            [
                evaluate_hybrid_on_validation(
                    alpha=alpha,
                    valid_truth=valid_truth,
                    top_k_values=TOP_K_VALUES,
                    train_event_count=train_event_count,
                    popularity_lookup=popularity_lookup,
                    user_factors=user_factors,
                    item_factors=item_factors,
                    seen_items_by_user=seen_items_by_user,
                    popularity_score_lookup=popularity_score_lookup,
                    n_items=n_items,
                )
                for alpha in ALPHA_CANDIDATES
            ],
            ignore_index=True,
        ),
    )
    alpha_tuning_rows = alpha_tuning_rows[0]

    alpha_tuning_summary = (
        alpha_tuning_rows.groupby(["alpha", "k"], as_index=False)
        .agg(
            users_evaluated=("user_idx", "nunique"),
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            hit_rate_at_k=("hit_rate_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            novelty_at_k=("novelty_at_k", "mean"),
            catalog_coverage_at_k=("catalog_coverage_at_k", "max"),
        )
        .sort_values(["alpha", "k"])
        .reset_index(drop=True)
    )

    alpha_selection = (
        alpha_tuning_summary[alpha_tuning_summary["k"] == 10]
        .sort_values(
            ["ndcg_at_k", "hit_rate_at_k", "precision_at_k"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_alpha = float(alpha_selection.loc[0, "alpha"])

    plot_accessible_alpha_tuning_curve(
        alpha_tuning_summary=alpha_tuning_summary,
        output_path_png=ALPHA_TUNING_FIGURE_PNG,
        output_path_svg=ALPHA_TUNING_FIGURE_SVG,
    )

    valid_metrics, valid_recommendations, valid_user_metrics = evaluate_top_n_hybrid(
        truth_lookup=valid_truth,
        split_name="valid",
        top_k_values=TOP_K_VALUES,
        alpha=best_alpha,
        export_top_n=EXPORT_TOP_N,
        train_event_count=train_event_count,
        popularity_lookup=popularity_lookup,
        user_factors=user_factors,
        item_factors=item_factors,
        seen_items_by_user=seen_items_by_user,
        popularity_score_lookup=popularity_score_lookup,
        n_items=n_items,
        recipe_lookup=recipe_lookup,
        item_popularity_lookup=item_popularity_lookup,
    )

    test_metrics, test_recommendations, test_user_metrics = evaluate_top_n_hybrid(
        truth_lookup=test_truth,
        split_name="test",
        top_k_values=TOP_K_VALUES,
        alpha=best_alpha,
        export_top_n=EXPORT_TOP_N,
        train_event_count=train_event_count,
        popularity_lookup=popularity_lookup,
        user_factors=user_factors,
        item_factors=item_factors,
        seen_items_by_user=seen_items_by_user,
        popularity_score_lookup=popularity_score_lookup,
        n_items=n_items,
        recipe_lookup=recipe_lookup,
        item_popularity_lookup=item_popularity_lookup,
    )

    hybrid_metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)
    hybrid_user_metrics = pd.concat(
        [valid_user_metrics, test_user_metrics], ignore_index=True
    )

    valid_recommendations = valid_recommendations.merge(
        user_lookup, on="user_idx", how="left"
    )
    test_recommendations = test_recommendations.merge(
        user_lookup, on="user_idx", how="left"
    )
    hybrid_user_metrics = hybrid_user_metrics.merge(
        user_lookup, on="user_idx", how="left"
    )

    routing_summary = pd.concat(
        [
            summarise_user_routing(implicit_valid, "valid"),
            summarise_user_routing(implicit_test, "test"),
        ],
        ignore_index=True,
    )

    valid_popularity_summary = build_recommendation_popularity_summary(
        valid_recommendations,
        "valid",
    )
    test_popularity_summary = build_recommendation_popularity_summary(
        test_recommendations,
        "test",
    )

    valid_concentration_curve = build_recommendation_concentration_curve(
        valid_popularity_summary
    )
    test_concentration_curve = build_recommendation_concentration_curve(
        test_popularity_summary
    )

    dashboard_metrics = build_dashboard_metrics_table(hybrid_metrics)
    academic_metrics = build_academic_metrics_table(hybrid_metrics)
    academic_component_summary = build_academic_component_table(component_summary)
    academic_routing_summary = build_academic_routing_table(routing_summary)
    academic_alpha_tuning = build_academic_alpha_tuning_table(alpha_tuning_summary)
    dashboard_summary = build_dashboard_summary_table(hybrid_metrics)

    plot_accessible_metric_lines(
        metrics_df=hybrid_metrics,
        metric_col="precision_at_k",
        ylabel="Precision@K",
        title="Hybrid precision across K",
        output_path_png=PRECISION_FIGURE_PNG,
        output_path_svg=PRECISION_FIGURE_SVG,
    )

    plot_accessible_metric_lines(
        metrics_df=hybrid_metrics,
        metric_col="recall_at_k",
        ylabel="Recall@K",
        title="Hybrid recall across K",
        output_path_png=RECALL_FIGURE_PNG,
        output_path_svg=RECALL_FIGURE_SVG,
    )

    plot_accessible_metric_lines(
        metrics_df=hybrid_metrics,
        metric_col="ndcg_at_k",
        ylabel="nDCG@K",
        title="Hybrid nDCG across K",
        output_path_png=NDCG_FIGURE_PNG,
        output_path_svg=NDCG_FIGURE_SVG,
    )

    plot_accessible_concentration_curve(
        curve_df=valid_concentration_curve,
        title="Validation recommendation concentration",
        output_path_png=VALID_CONCENTRATION_FIGURE_PNG,
        output_path_svg=VALID_CONCENTRATION_FIGURE_SVG,
    )

    plot_accessible_concentration_curve(
        curve_df=test_concentration_curve,
        title="Test recommendation concentration",
        output_path_png=TEST_CONCENTRATION_FIGURE_PNG,
        output_path_svg=TEST_CONCENTRATION_FIGURE_SVG,
    )

    split_summary.to_csv(SPLIT_SUMMARY_OUTPUT_PATH, index=False)
    coverage_summary.to_csv(MAPPING_COVERAGE_OUTPUT_PATH, index=False)
    matrix_summary.to_csv(MATRIX_SUMMARY_OUTPUT_PATH, index=False)
    component_summary.to_csv(COMPONENT_SUMMARY_OUTPUT_PATH, index=False)
    item_popularity.to_csv(ITEM_POPULARITY_OUTPUT_PATH, index=False)
    alpha_tuning_summary.to_csv(ALPHA_TUNING_OUTPUT_PATH, index=False)
    academic_alpha_tuning.to_csv(ALPHA_TUNING_ACADEMIC_OUTPUT_PATH, index=False)

    hybrid_metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
    hybrid_user_metrics.to_csv(USER_METRICS_OUTPUT_PATH, index=False)
    routing_summary.to_csv(ROUTING_SUMMARY_OUTPUT_PATH, index=False)

    valid_recommendations.to_csv(VALID_RECOMMENDATIONS_OUTPUT_PATH, index=False)
    test_recommendations.to_csv(TEST_RECOMMENDATIONS_OUTPUT_PATH, index=False)
    valid_popularity_summary.to_csv(
        VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH, index=False
    )
    test_popularity_summary.to_csv(
        TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH, index=False
    )

    valid_concentration_curve.to_csv(VALID_CONCENTRATION_CURVE_OUTPUT_PATH, index=False)
    test_concentration_curve.to_csv(TEST_CONCENTRATION_CURVE_OUTPUT_PATH, index=False)

    dashboard_metrics.to_csv(DASHBOARD_METRICS_OUTPUT_PATH, index=False)
    academic_metrics.to_csv(ACADEMIC_METRICS_OUTPUT_PATH, index=False)
    academic_component_summary.to_csv(ACADEMIC_COMPONENT_OUTPUT_PATH, index=False)
    academic_routing_summary.to_csv(ACADEMIC_ROUTING_OUTPUT_PATH, index=False)
    dashboard_summary.to_csv(DASHBOARD_SUMMARY_OUTPUT_PATH, index=False)

    save_model_artifact(
        svd_model=svd_model,
        user_factors=user_factors,
        item_factors=item_factors,
        recipe_lookup=recipe_lookup,
        seen_items_by_user=seen_items_by_user,
        ranked_popular_items=ranked_popular_items,
        popularity_score_lookup=popularity_score_lookup,
        item_popularity=item_popularity,
        best_alpha=best_alpha,
    )

    save_run_log(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
        train_hybrid=train_hybrid,
        valid_hybrid=valid_hybrid,
        test_hybrid=test_hybrid,
        train_matrix=train_matrix,
        coverage_summary=coverage_summary,
        routing_summary=routing_summary,
        component_summary=component_summary,
        alpha_tuning_summary=alpha_tuning_summary,
        hybrid_metrics=hybrid_metrics,
        best_alpha=best_alpha,
    )

    print("\nBest alpha:", best_alpha)
    print("\nHybrid metrics summary:")
    print(hybrid_metrics)

    print("\nSaved outputs:")
    print("-", SPLIT_SUMMARY_OUTPUT_PATH)
    print("-", MAPPING_COVERAGE_OUTPUT_PATH)
    print("-", MATRIX_SUMMARY_OUTPUT_PATH)
    print("-", COMPONENT_SUMMARY_OUTPUT_PATH)
    print("-", ITEM_POPULARITY_OUTPUT_PATH)
    print("-", ALPHA_TUNING_OUTPUT_PATH)
    print("-", ALPHA_TUNING_ACADEMIC_OUTPUT_PATH)
    print("-", METRICS_OUTPUT_PATH)
    print("-", USER_METRICS_OUTPUT_PATH)
    print("-", ROUTING_SUMMARY_OUTPUT_PATH)
    print("-", VALID_RECOMMENDATIONS_OUTPUT_PATH)
    print("-", TEST_RECOMMENDATIONS_OUTPUT_PATH)
    print("-", VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH)
    print("-", TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH)
    print("-", VALID_CONCENTRATION_CURVE_OUTPUT_PATH)
    print("-", TEST_CONCENTRATION_CURVE_OUTPUT_PATH)
    print("-", DASHBOARD_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_COMPONENT_OUTPUT_PATH)
    print("-", ACADEMIC_ROUTING_OUTPUT_PATH)
    print("-", DASHBOARD_SUMMARY_OUTPUT_PATH)
    print("-", EXPLAINED_VARIANCE_FIGURE_PNG)
    print("-", EXPLAINED_VARIANCE_FIGURE_SVG)
    print("-", ALPHA_TUNING_FIGURE_PNG)
    print("-", ALPHA_TUNING_FIGURE_SVG)
    print("-", PRECISION_FIGURE_PNG)
    print("-", PRECISION_FIGURE_SVG)
    print("-", RECALL_FIGURE_PNG)
    print("-", RECALL_FIGURE_SVG)
    print("-", NDCG_FIGURE_PNG)
    print("-", NDCG_FIGURE_SVG)
    print("-", VALID_CONCENTRATION_FIGURE_PNG)
    print("-", VALID_CONCENTRATION_FIGURE_SVG)
    print("-", TEST_CONCENTRATION_FIGURE_PNG)
    print("-", TEST_CONCENTRATION_FIGURE_SVG)
    print("-", USER_ACTIVITY_FIGURE_PNG)
    print("-", USER_ACTIVITY_FIGURE_SVG)
    print("-", ITEM_ACTIVITY_FIGURE_PNG)
    print("-", ITEM_ACTIVITY_FIGURE_SVG)
    print("-", MODEL_OUTPUT_PATH)
    print("-", MODEL_METADATA_OUTPUT_PATH)
    print("-", LOG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
