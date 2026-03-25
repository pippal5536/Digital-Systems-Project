"""
src/models/train_svd.py

Purpose:
Train and evaluate a truncated-SVD matrix factorisation recommender on the
chronological implicit interaction splits.

This module fits the latent-factor model on the implicit training split only,
generates Top-N recommendations for validation and test users, excludes items
already seen in training, evaluates ranking quality, and saves presentation-
ready tables, accessible figures, dashboard artifacts, and structured logs.

Responsibilities:
- load chronological implicit train, validation, and test splits
- validate required columns and quantify user/item index mapping coverage
- build a sparse user-item interaction matrix from training data only
- fit a TruncatedSVD latent-factor model on the training matrix only
- generate top-N recommendations for holdout users
- evaluate ranking metrics such as Precision@K, Recall@K, Hit Rate@K, and nDCG@K
- analyse novelty, catalogue coverage, and recommendation concentration
- save dashboard-friendly and academic-report-friendly tables
- save accessible figures designed for readability and long-term reuse
- save a reusable model artifact for dashboard inference
- save a compact JSON run log for pipeline monitoring and later comparison

Design notes:
- only the training split is used to fit latent factors
- validation and test rows without train-known user/item mappings are excluded
  from scoring and counted explicitly
- recommendations exclude items already observed in the training split
- holdout relevance is treated as binary user-item interaction presence
- outputs follow the existing project naming pattern with 09_* filenames
- model artifacts are saved to output/saved_models for dashboard loading
- figures are styled for readability, accessibility, and long-term reuse
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

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

MODEL_NAME = "svd"
MODEL_LABEL = "Truncated SVD"

TOP_K_VALUES = [5, 10, 20]
EXPORT_TOP_N = 10

N_COMPONENTS = 64
N_ITER = 10
RANDOM_STATE = 42

IMPLICIT_TRAIN_PATH = SPLITS_DIR / "implicit_train.parquet"
IMPLICIT_VALID_PATH = SPLITS_DIR / "implicit_valid.parquet"
IMPLICIT_TEST_PATH = SPLITS_DIR / "implicit_test.parquet"

TABLES_SUBDIR = TABLES_DIR / MODEL_NAME
FIGURES_SUBDIR = FIGURES_DIR / MODEL_NAME
LOGS_SUBDIR = LOGS_DIR / MODEL_NAME

MAPPING_COVERAGE_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_mapping_coverage_summary.csv"
SPLIT_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_split_summary.csv"
MATRIX_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_matrix_summary.csv"
COMPONENT_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_component_summary.csv"
ITEM_POPULARITY_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_item_popularity_table.csv"

VALID_RECS_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_valid_recommendations_long.csv"
TEST_RECS_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_test_recommendations_long.csv"
VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH = (
    TABLES_SUBDIR / "09_svd_valid_recommendation_popularity.csv"
)
TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH = (
    TABLES_SUBDIR / "09_svd_test_recommendation_popularity.csv"
)

METRICS_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_metrics.csv"
USER_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_user_metrics.csv"

DASHBOARD_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_metrics_dashboard.csv"
ACADEMIC_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_metrics_academic.csv"
ACADEMIC_COMPONENT_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_component_summary_academic.csv"
DASHBOARD_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "09_svd_dashboard_summary.csv"

VALID_CONCENTRATION_CURVE_OUTPUT_PATH = (
    TABLES_SUBDIR / "09_svd_valid_recommendation_concentration_curve.csv"
)
TEST_CONCENTRATION_CURVE_OUTPUT_PATH = (
    TABLES_SUBDIR / "09_svd_test_recommendation_concentration_curve.csv"
)

EXPLAINED_VARIANCE_FIGURE_PNG = (
    FIGURES_SUBDIR / "09_svd_cumulative_explained_variance.png"
)
EXPLAINED_VARIANCE_FIGURE_SVG = (
    FIGURES_SUBDIR / "09_svd_cumulative_explained_variance.svg"
)

PRECISION_FIGURE_PNG = FIGURES_SUBDIR / "09_svd_precision_by_k.png"
PRECISION_FIGURE_SVG = FIGURES_SUBDIR / "09_svd_precision_by_k.svg"

RECALL_FIGURE_PNG = FIGURES_SUBDIR / "09_svd_recall_by_k.png"
RECALL_FIGURE_SVG = FIGURES_SUBDIR / "09_svd_recall_by_k.svg"

NDCG_FIGURE_PNG = FIGURES_SUBDIR / "09_svd_ndcg_by_k.png"
NDCG_FIGURE_SVG = FIGURES_SUBDIR / "09_svd_ndcg_by_k.svg"

VALID_CONCENTRATION_FIGURE_PNG = (
    FIGURES_SUBDIR / "09_svd_valid_recommendation_concentration_curve.png"
)
VALID_CONCENTRATION_FIGURE_SVG = (
    FIGURES_SUBDIR / "09_svd_valid_recommendation_concentration_curve.svg"
)

TEST_CONCENTRATION_FIGURE_PNG = (
    FIGURES_SUBDIR / "09_svd_test_recommendation_concentration_curve.png"
)
TEST_CONCENTRATION_FIGURE_SVG = (
    FIGURES_SUBDIR / "09_svd_test_recommendation_concentration_curve.svg"
)

USER_ACTIVITY_FIGURE_PNG = (
    FIGURES_SUBDIR / "09_svd_train_user_interaction_distribution.png"
)
USER_ACTIVITY_FIGURE_SVG = (
    FIGURES_SUBDIR / "09_svd_train_user_interaction_distribution.svg"
)

ITEM_ACTIVITY_FIGURE_PNG = (
    FIGURES_SUBDIR / "09_svd_train_item_interaction_distribution.png"
)
ITEM_ACTIVITY_FIGURE_SVG = (
    FIGURES_SUBDIR / "09_svd_train_item_interaction_distribution.svg"
)

LOG_OUTPUT_PATH = LOGS_SUBDIR / "09_svd_run_log.json"

MODEL_DIR = PROJECT_ROOT / "outputs" / "saved_models"
MODEL_OUTPUT_PATH = MODEL_DIR / "09_svd_model.joblib"
MODEL_METADATA_OUTPUT_PATH = MODEL_DIR / "09_svd_model_metadata.json"


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

    Raises:
        FileNotFoundError:
            Raised when one or more required files are missing.
    """
    required_paths = [
        IMPLICIT_TRAIN_PATH,
        IMPLICIT_VALID_PATH,
        IMPLICIT_TEST_PATH,
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Missing required SVD input files:\n" + "\n".join(missing_paths)
        )


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """
    Load one parquet split file.

    Args:
        path:
            Parquet file path.
        split_name:
            Human-readable split label.

    Returns:
        pd.DataFrame:
            Loaded split dataframe.
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

    Args:
        df:
            Split dataframe.
        split_name:
            Split label.
        required_cols:
            Required column names.

    Raises:
        ValueError:
            Raised if one or more columns are missing.
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


def summarise_mapping_coverage(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Summarise user/item index availability for one split.
    """
    rows = int(len(df))
    rows_with_both = int((df["user_idx"].notna() & df["item_idx"].notna()).sum())

    return pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": rows,
                "missing_user_idx": int(df["user_idx"].isna().sum()),
                "missing_item_idx": int(df["item_idx"].isna().sum()),
                "rows_with_both_indices": rows_with_both,
                "rows_evaluable_pct": (
                    round((rows_with_both / rows) * 100, 2) if rows else 0.0
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
    train_svd: pd.DataFrame,
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
                "train_rows": int(len(train_svd)),
                "unique_user_ids": int(train_svd["user_id"].nunique()),
                "unique_recipe_ids": int(train_svd["recipe_id"].nunique()),
            }
        ]
    )


def build_item_popularity_table(train_svd: pd.DataFrame) -> pd.DataFrame:
    """
    Build an item popularity table from training interactions only.
    """
    item_popularity = (
        train_svd.groupby(["item_idx", "recipe_id"], as_index=False)
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
    item_popularity["popularity_score"] = (
        item_popularity["train_interaction_count"]
        / item_popularity["train_interaction_count"].max()
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


def build_seen_items_by_user(train_svd: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a mapping of user_idx to seen item_idx values from training.
    """
    return (
        train_svd.groupby("user_idx")["item_idx"]
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
    Compute Precision@K.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute Recall@K.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute Hit Rate@K.
    """
    return float(any(item in relevant for item in recommended[:k]))


def dcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute DCG@K for binary relevance.
    """
    dcg_value = 0.0

    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg_value += 1.0 / math.log2(rank + 1)

    return dcg_value


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute nDCG@K for binary relevance.
    """
    if not relevant:
        return 0.0

    ideal_hit_count = min(len(relevant), k)
    if ideal_hit_count == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hit_count + 1))

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(recommended, relevant, k) / ideal_dcg


def build_item_self_information(
    item_popularity: pd.DataFrame,
) -> dict[int, float]:
    """
    Build item self-information values for novelty analysis.
    """
    item_probability = (
        item_popularity.set_index("item_idx")["train_interaction_count"]
        / item_popularity["train_interaction_count"].sum()
    )

    return {
        int(item_idx): float(-math.log2(probability))
        for item_idx, probability in item_probability.items()
        if probability > 0
    }


def novelty_at_k(
    recommended: list[int],
    item_self_information: dict[int, float],
    k: int,
) -> float:
    """
    Compute novelty at K using mean self-information.
    """
    recommended_k = recommended[:k]

    if not recommended_k:
        return 0.0

    return float(
        np.mean([item_self_information.get(item, 0.0) for item in recommended_k])
    )


def score_user_all_items(
    user_idx: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> np.ndarray:
    """
    Score all items for one user using latent factors.
    """
    return user_factors[user_idx] @ item_factors.T


def recommend_top_n(
    user_idx: int,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items_by_user: dict[int, set[int]],
    top_n: int,
) -> tuple[list[int], list[float]]:
    """
    Generate Top-N item recommendations for one user and return both
    item indices and their corresponding SVD scores.
    """
    scores = score_user_all_items(user_idx, user_factors, item_factors).copy()

    seen_items = seen_items_by_user.get(user_idx, set())
    if seen_items:
        scores[list(seen_items)] = -np.inf

    finite_count = int(np.isfinite(scores).sum())
    if finite_count == 0:
        return [], []

    top_n = min(int(top_n), finite_count)

    top_items = np.argpartition(scores, -top_n)[-top_n:]
    top_items = top_items[np.argsort(scores[top_items])[::-1]]

    top_scores = scores[top_items]

    return top_items.tolist(), top_scores.astype(float).tolist()


def evaluate_holdout_split(
    split_name: str,
    truth_lookup: dict[int, set[int]],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_items_by_user: dict[int, set[int]],
    item_self_information: dict[int, float],
    recipe_lookup: dict[int, int],
    item_popularity_lookup: dict[int, dict[str, float | int]],
    catalog_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the SVD recommender on one holdout split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            User-level table and split-level metric summary table.
    """
    user_rows: list[dict[str, object]] = []
    recommended_items_by_k = {k: [] for k in TOP_K_VALUES}

    max_required_recs = max(max(TOP_K_VALUES), EXPORT_TOP_N)

    for user_idx, relevant_items in truth_lookup.items():
        recommended_items, recommended_scores = recommend_top_n(
            user_idx=user_idx,
            user_factors=user_factors,
            item_factors=item_factors,
            seen_items_by_user=seen_items_by_user,
            top_n=max_required_recs,
        )

        for k in TOP_K_VALUES:
            recommended_items_by_k[k].extend(recommended_items[:k])
            
        for recommendation_rank, (item_idx, score) in enumerate(
            zip(recommended_items[:EXPORT_TOP_N], recommended_scores[:EXPORT_TOP_N]),
            start=1,
        ):
            popularity_row = item_popularity_lookup.get(item_idx, {})

            user_rows.append(
                {
                    "split": split_name,
                    "user_idx": int(user_idx),
                    "holdout_item_count": int(len(relevant_items)),
                    "recommendation_rank": int(recommendation_rank),
                    "item_idx": int(item_idx),
                    "recipe_id": int(recipe_lookup.get(item_idx, -1)),
                    "score": float(score),
                    "global_item_popularity_rank": popularity_row.get(
                        "item_popularity_rank"
                    ),
                    "train_interaction_count": popularity_row.get(
                        "train_interaction_count"
                    ),
                    "popularity_score": popularity_row.get("popularity_score"),
                    "precision_at_5": precision_at_k(
                        recommended_items, relevant_items, 5
                    ),
                    "recall_at_5": recall_at_k(recommended_items, relevant_items, 5),
                    "hit_rate_at_5": hit_rate_at_k(
                        recommended_items, relevant_items, 5
                    ),
                    "ndcg_at_5": ndcg_at_k(recommended_items, relevant_items, 5),
                    "precision_at_10": precision_at_k(
                        recommended_items, relevant_items, 10
                    ),
                    "recall_at_10": recall_at_k(recommended_items, relevant_items, 10),
                    "hit_rate_at_10": hit_rate_at_k(
                        recommended_items, relevant_items, 10
                    ),
                    "ndcg_at_10": ndcg_at_k(recommended_items, relevant_items, 10),
                    "precision_at_20": precision_at_k(
                        recommended_items, relevant_items, 20
                    ),
                    "recall_at_20": recall_at_k(recommended_items, relevant_items, 20),
                    "hit_rate_at_20": hit_rate_at_k(
                        recommended_items, relevant_items, 20
                    ),
                    "ndcg_at_20": ndcg_at_k(recommended_items, relevant_items, 20),
                    "novelty_at_5": novelty_at_k(
                        recommended_items, item_self_information, 5
                    ),
                    "novelty_at_10": novelty_at_k(
                        recommended_items, item_self_information, 10
                    ),
                    "novelty_at_20": novelty_at_k(
                        recommended_items, item_self_information, 20
                    ),
                }
            )

    user_recommendations_df = pd.DataFrame(user_rows)

    per_user_summary = user_recommendations_df.groupby(
        ["split", "user_idx"], as_index=False
    ).agg(
        holdout_item_count=("holdout_item_count", "first"),
        precision_at_5=("precision_at_5", "first"),
        recall_at_5=("recall_at_5", "first"),
        hit_rate_at_5=("hit_rate_at_5", "first"),
        ndcg_at_5=("ndcg_at_5", "first"),
        novelty_at_5=("novelty_at_5", "first"),
        precision_at_10=("precision_at_10", "first"),
        recall_at_10=("recall_at_10", "first"),
        hit_rate_at_10=("hit_rate_at_10", "first"),
        ndcg_at_10=("ndcg_at_10", "first"),
        novelty_at_10=("novelty_at_10", "first"),
        precision_at_20=("precision_at_20", "first"),
        recall_at_20=("recall_at_20", "first"),
        hit_rate_at_20=("hit_rate_at_20", "first"),
        ndcg_at_20=("ndcg_at_20", "first"),
        novelty_at_20=("novelty_at_20", "first"),
    )

    metric_rows: list[dict[str, object]] = []

    for k in TOP_K_VALUES:
        unique_recommended_count = len(set(recommended_items_by_k[k]))

        metric_rows.append(
            {
                "split": split_name,
                "k": k,
                "users_evaluated": int(per_user_summary["user_idx"].nunique()),
                "precision_at_k": float(per_user_summary[f"precision_at_{k}"].mean()),
                "recall_at_k": float(per_user_summary[f"recall_at_{k}"].mean()),
                "hit_rate_at_k": float(per_user_summary[f"hit_rate_at_{k}"].mean()),
                "ndcg_at_k": float(per_user_summary[f"ndcg_at_{k}"].mean()),
                "novelty_at_k": float(per_user_summary[f"novelty_at_{k}"].mean()),
                "catalog_coverage_at_k": (
                    float(unique_recommended_count / catalog_size)
                    if catalog_size > 0
                    else 0.0
                ),
                "recommendation_count": int(len(recommended_items_by_k[k])),
            }
        )

    split_metrics_df = pd.DataFrame(metric_rows)

    return user_recommendations_df, split_metrics_df


def build_recommendation_popularity_summary(
    recommendations_long: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Summarise how often each item is recommended.
    """
    if recommendations_long.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "item_idx",
                "recipe_id",
                "times_recommended",
                "train_interaction_count",
                "global_item_popularity_rank",
                "popularity_score",
            ]
        )

    summary = (
        recommendations_long.groupby(
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
        .rename(columns={"size": "times_recommended"})
        .sort_values(["times_recommended", "item_idx"], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary["split"] = split_name
    summary = summary[
        [
            "split",
            "item_idx",
            "recipe_id",
            "times_recommended",
            "train_interaction_count",
            "global_item_popularity_rank",
            "popularity_score",
        ]
    ].copy()

    return summary


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

    academic_df["Users Evaluated"] = academic_df["users_evaluated"].astype(int)
    academic_df["Precision@K"] = academic_df["precision_at_k"].round(4)
    academic_df["Recall@K"] = academic_df["recall_at_k"].round(4)
    academic_df["Hit Rate@K"] = academic_df["hit_rate_at_k"].round(4)
    academic_df["nDCG@K"] = academic_df["ndcg_at_k"].round(4)
    academic_df["Novelty@K"] = academic_df["novelty_at_k"].round(4)
    academic_df["Catalogue Coverage@K"] = academic_df["catalog_coverage_at_k"].round(4)

    academic_df = academic_df[
        [
            "Model",
            "Split",
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

    return academic_df


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

    return summary_df[
        [
            "model",
            "label",
            "split",
            "k",
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

    Design notes:
    - larger figure size
    - visible black bar edges
    - readable axis labels
    - grid lines for interpretation
    - PNG and SVG export for long-term reuse
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
    plt.title("SVD cumulative explained variance", fontsize=13)
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
    item_popularity: pd.DataFrame,
) -> None:
    """
    Save the SVD artifact for dashboard reuse.
    """
    model_artifact = {
        "model_name": MODEL_NAME,
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_components": N_COMPONENTS,
        "n_iter": N_ITER,
        "random_state": RANDOM_STATE,
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "svd_model": svd_model,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "recipe_lookup": recipe_lookup,
        "seen_items_by_user": seen_items_by_user,
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
            "item_popularity_table",
        ],
    }

    with open(MODEL_METADATA_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_run_log(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
    train_svd: pd.DataFrame,
    valid_svd: pd.DataFrame,
    test_svd: pd.DataFrame,
    train_matrix: sparse.csr_matrix,
    coverage_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    svd_metrics: pd.DataFrame,
) -> None:
    """
    Save a structured JSON log for monitoring and dashboard use.
    """
    best_row = svd_metrics.sort_values(
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
            "train_rows_evaluable": int(len(train_svd)),
            "valid_rows_evaluable": int(len(valid_svd)),
            "test_rows_evaluable": int(len(test_svd)),
            "train_users_raw": int(implicit_train["user_id"].nunique()),
            "valid_users_raw": int(implicit_valid["user_id"].nunique()),
            "test_users_raw": int(implicit_test["user_id"].nunique()),
            "train_recipes_raw": int(implicit_train["recipe_id"].nunique()),
            "valid_recipes_raw": int(implicit_valid["recipe_id"].nunique()),
            "test_recipes_raw": int(implicit_test["recipe_id"].nunique()),
        },
        "mapping_coverage": coverage_summary.to_dict(orient="records"),
        "matrix_summary": {
            "shape": [int(n_users), int(n_items)],
            "nnz": int(train_matrix.nnz),
            "density": float(density),
        },
        "config": {
            "n_components": N_COMPONENTS,
            "n_iter": N_ITER,
            "random_state": RANDOM_STATE,
            "top_k_values": TOP_K_VALUES,
            "export_top_n": EXPORT_TOP_N,
        },
        "explained_variance": {
            "explained_variance_ratio_sum": float(
                component_summary["explained_variance_ratio"].sum()
            )
        },
        "artifacts": {
            "mapping_coverage_csv": str(MAPPING_COVERAGE_OUTPUT_PATH),
            "split_summary_csv": str(SPLIT_SUMMARY_OUTPUT_PATH),
            "matrix_summary_csv": str(MATRIX_SUMMARY_OUTPUT_PATH),
            "component_summary_csv": str(COMPONENT_SUMMARY_OUTPUT_PATH),
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
            "precision_at_k": float(best_row["precision_at_k"]),
            "recall_at_k": float(best_row["recall_at_k"]),
            "hit_rate_at_k": float(best_row["hit_rate_at_k"]),
            "ndcg_at_k": float(best_row["ndcg_at_k"]),
            "novelty_at_k": float(best_row["novelty_at_k"]),
            "catalog_coverage_at_k": float(best_row["catalog_coverage_at_k"]),
        },
        "metrics": svd_metrics.to_dict(orient="records"),
    }

    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(run_log, file, indent=2)


# Main pipeline


def main() -> None:
    """
    Run the full SVD training and evaluation pipeline.
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

    validate_required_columns(implicit_train, "implicit_train", required_cols)
    validate_required_columns(implicit_valid, "implicit_valid", required_cols)
    validate_required_columns(implicit_test, "implicit_test", required_cols)

    for df in [implicit_train, implicit_valid, implicit_test]:
        df["date"] = pd.to_datetime(df["date"])

    split_summary = build_split_summary(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
    )

    coverage_summary = pd.concat(
        [
            summarise_mapping_coverage(implicit_train, "train"),
            summarise_mapping_coverage(implicit_valid, "valid"),
            summarise_mapping_coverage(implicit_test, "test"),
        ],
        ignore_index=True,
    )

    print("\nSplit summary:")
    print(split_summary)

    print("\nMapping coverage summary:")
    print(coverage_summary)

    train_svd = filter_evaluable_rows(implicit_train)
    valid_svd = filter_evaluable_rows(implicit_valid)
    test_svd = filter_evaluable_rows(implicit_test)

    print("\nEvaluable shapes:")
    print("train:", train_svd.shape)
    print("valid:", valid_svd.shape)
    print("test:", test_svd.shape)

    train_matrix = build_train_matrix(train_svd)
    matrix_summary = build_matrix_summary(train_matrix, train_svd)

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

    item_popularity = build_item_popularity_table(train_svd)
    seen_items_by_user = build_seen_items_by_user(train_svd)
    valid_truth = build_truth_lookup(valid_svd)
    test_truth = build_truth_lookup(test_svd)

    recipe_lookup = (
        train_svd[["item_idx", "recipe_id"]]
        .drop_duplicates()
        .set_index("item_idx")["recipe_id"]
        .to_dict()
    )
    user_lookup = train_svd[["user_idx", "user_id"]].drop_duplicates()

    item_popularity_lookup = item_popularity.set_index("item_idx")[
        ["item_popularity_rank", "train_interaction_count", "popularity_score"]
    ].to_dict("index")

    item_self_information = build_item_self_information(item_popularity)
    catalog_size = int(item_popularity["item_idx"].nunique())

    valid_recommendations_long, valid_metrics = evaluate_holdout_split(
        split_name="valid",
        truth_lookup=valid_truth,
        user_factors=user_factors,
        item_factors=item_factors,
        seen_items_by_user=seen_items_by_user,
        item_self_information=item_self_information,
        recipe_lookup=recipe_lookup,
        item_popularity_lookup=item_popularity_lookup,
        catalog_size=catalog_size,
    )

    test_recommendations_long, test_metrics = evaluate_holdout_split(
        split_name="test",
        truth_lookup=test_truth,
        user_factors=user_factors,
        item_factors=item_factors,
        seen_items_by_user=seen_items_by_user,
        item_self_information=item_self_information,
        recipe_lookup=recipe_lookup,
        item_popularity_lookup=item_popularity_lookup,
        catalog_size=catalog_size,
    )

    svd_metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)

    valid_user_metrics = valid_recommendations_long.groupby(
        ["split", "user_idx"], as_index=False
    ).agg(
        holdout_item_count=("holdout_item_count", "first"),
        precision_at_5=("precision_at_5", "first"),
        recall_at_5=("recall_at_5", "first"),
        hit_rate_at_5=("hit_rate_at_5", "first"),
        ndcg_at_5=("ndcg_at_5", "first"),
        novelty_at_5=("novelty_at_5", "first"),
        precision_at_10=("precision_at_10", "first"),
        recall_at_10=("recall_at_10", "first"),
        hit_rate_at_10=("hit_rate_at_10", "first"),
        ndcg_at_10=("ndcg_at_10", "first"),
        novelty_at_10=("novelty_at_10", "first"),
        precision_at_20=("precision_at_20", "first"),
        recall_at_20=("recall_at_20", "first"),
        hit_rate_at_20=("hit_rate_at_20", "first"),
        ndcg_at_20=("ndcg_at_20", "first"),
        novelty_at_20=("novelty_at_20", "first"),
    )

    test_user_metrics = test_recommendations_long.groupby(
        ["split", "user_idx"], as_index=False
    ).agg(
        holdout_item_count=("holdout_item_count", "first"),
        precision_at_5=("precision_at_5", "first"),
        recall_at_5=("recall_at_5", "first"),
        hit_rate_at_5=("hit_rate_at_5", "first"),
        ndcg_at_5=("ndcg_at_5", "first"),
        novelty_at_5=("novelty_at_5", "first"),
        precision_at_10=("precision_at_10", "first"),
        recall_at_10=("recall_at_10", "first"),
        hit_rate_at_10=("hit_rate_at_10", "first"),
        ndcg_at_10=("ndcg_at_10", "first"),
        novelty_at_10=("novelty_at_10", "first"),
        precision_at_20=("precision_at_20", "first"),
        recall_at_20=("recall_at_20", "first"),
        hit_rate_at_20=("hit_rate_at_20", "first"),
        ndcg_at_20=("ndcg_at_20", "first"),
        novelty_at_20=("novelty_at_20", "first"),
    )

    svd_user_metrics = pd.concat(
        [valid_user_metrics, test_user_metrics], ignore_index=True
    )

    valid_recommendations_long = valid_recommendations_long.merge(
        user_lookup, on="user_idx", how="left"
    )
    test_recommendations_long = test_recommendations_long.merge(
        user_lookup, on="user_idx", how="left"
    )
    svd_user_metrics = svd_user_metrics.merge(user_lookup, on="user_idx", how="left")

    valid_recommendation_popularity = build_recommendation_popularity_summary(
        recommendations_long=valid_recommendations_long,
        split_name="valid",
    )
    test_recommendation_popularity = build_recommendation_popularity_summary(
        recommendations_long=test_recommendations_long,
        split_name="test",
    )

    valid_concentration_curve = build_recommendation_concentration_curve(
        valid_recommendation_popularity
    )
    test_concentration_curve = build_recommendation_concentration_curve(
        test_recommendation_popularity
    )

    dashboard_metrics = build_dashboard_metrics_table(svd_metrics)
    academic_metrics = build_academic_metrics_table(svd_metrics)
    academic_component_summary = build_academic_component_table(component_summary)
    dashboard_summary = build_dashboard_summary_table(svd_metrics)

    plot_accessible_metric_lines(
        metrics_df=svd_metrics,
        metric_col="precision_at_k",
        ylabel="Precision@K",
        title="SVD precision across K",
        output_path_png=PRECISION_FIGURE_PNG,
        output_path_svg=PRECISION_FIGURE_SVG,
    )

    plot_accessible_metric_lines(
        metrics_df=svd_metrics,
        metric_col="recall_at_k",
        ylabel="Recall@K",
        title="SVD recall across K",
        output_path_png=RECALL_FIGURE_PNG,
        output_path_svg=RECALL_FIGURE_SVG,
    )

    plot_accessible_metric_lines(
        metrics_df=svd_metrics,
        metric_col="ndcg_at_k",
        ylabel="nDCG@K",
        title="SVD nDCG across K",
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

    valid_recommendations_long.to_csv(VALID_RECS_OUTPUT_PATH, index=False)
    test_recommendations_long.to_csv(TEST_RECS_OUTPUT_PATH, index=False)
    valid_recommendation_popularity.to_csv(
        VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH, index=False
    )
    test_recommendation_popularity.to_csv(
        TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH, index=False
    )

    valid_concentration_curve.to_csv(VALID_CONCENTRATION_CURVE_OUTPUT_PATH, index=False)
    test_concentration_curve.to_csv(TEST_CONCENTRATION_CURVE_OUTPUT_PATH, index=False)

    svd_metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
    svd_user_metrics.to_csv(USER_METRICS_OUTPUT_PATH, index=False)

    dashboard_metrics.to_csv(DASHBOARD_METRICS_OUTPUT_PATH, index=False)
    academic_metrics.to_csv(ACADEMIC_METRICS_OUTPUT_PATH, index=False)
    academic_component_summary.to_csv(ACADEMIC_COMPONENT_OUTPUT_PATH, index=False)
    dashboard_summary.to_csv(DASHBOARD_SUMMARY_OUTPUT_PATH, index=False)

    save_model_artifact(
        svd_model=svd_model,
        user_factors=user_factors,
        item_factors=item_factors,
        recipe_lookup=recipe_lookup,
        seen_items_by_user=seen_items_by_user,
        item_popularity=item_popularity,
    )

    save_run_log(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
        train_svd=train_svd,
        valid_svd=valid_svd,
        test_svd=test_svd,
        train_matrix=train_matrix,
        coverage_summary=coverage_summary,
        component_summary=component_summary,
        svd_metrics=svd_metrics,
    )

    print("\nSVD metrics summary:")
    print(svd_metrics)

    print("\nSaved outputs:")
    print("-", SPLIT_SUMMARY_OUTPUT_PATH)
    print("-", MAPPING_COVERAGE_OUTPUT_PATH)
    print("-", MATRIX_SUMMARY_OUTPUT_PATH)
    print("-", COMPONENT_SUMMARY_OUTPUT_PATH)
    print("-", ITEM_POPULARITY_OUTPUT_PATH)
    print("-", VALID_RECS_OUTPUT_PATH)
    print("-", TEST_RECS_OUTPUT_PATH)
    print("-", VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH)
    print("-", TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH)
    print("-", VALID_CONCENTRATION_CURVE_OUTPUT_PATH)
    print("-", TEST_CONCENTRATION_CURVE_OUTPUT_PATH)
    print("-", METRICS_OUTPUT_PATH)
    print("-", USER_METRICS_OUTPUT_PATH)
    print("-", DASHBOARD_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_COMPONENT_OUTPUT_PATH)
    print("-", DASHBOARD_SUMMARY_OUTPUT_PATH)
    print("-", EXPLAINED_VARIANCE_FIGURE_PNG)
    print("-", EXPLAINED_VARIANCE_FIGURE_SVG)
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
