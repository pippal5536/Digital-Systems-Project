"""
src/models/train_cf.py

Purpose:
Build and evaluate the collaborative filtering recommender using sparse
implicit interactions and item-item cosine similarity.

This module implements an item-item neighbourhood recommender over the
chronological implicit interaction splits. It uses train-fitted mapped
indices from the split phase, builds a sparse user-item matrix from the
training data, computes sparse cosine-based item neighbours, generates
top-N recommendations for each user, evaluates the model on validation
and test splits, and saves presentation-ready tables, accessible figures,
dashboard artifacts, and structured logs.

Responsibilities:
- load implicit train, validation, and test split files
- validate required columns and mapped-index availability
- keep only CF-evaluable rows with mapped train-fitted indices
- build a sparse user-item interaction matrix from training data
- compute item-item cosine neighbours using sparse linear algebra
- generate top-N recommendations for each user
- evaluate ranking quality on validation and test splits
- measure coverage and neighbourhood sparsity
- save dashboard-friendly and academic-report-friendly outputs
- save a reusable CF model artifact for dashboard inference

Design notes:
- the implicit interaction split is used for Top-N recommendation
- only train-fitted mapped users and items are used for CF scoring
- rows with missing mapped indices are excluded from CF evaluation
- cosine similarity is computed between sparse item vectors
- only top neighbours per item are retained for better scaling
- items already seen in training are excluded from recommendation lists
- outputs follow the existing project naming pattern with 08_* filenames
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

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


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

TOP_K_VALUES = [5, 10, 20]
EXPORT_TOP_N = 10

ITEM_NEIGHBOUR_K = 100
RECOMMENDATION_CANDIDATE_MULTIPLIER = 5

IMPLICIT_TRAIN_PATH = SPLITS_DIR / "implicit_train.parquet"
IMPLICIT_VALID_PATH = SPLITS_DIR / "implicit_valid.parquet"
IMPLICIT_TEST_PATH = SPLITS_DIR / "implicit_test.parquet"

CF_SPLIT_SUMMARY_OUTPUT_PATH = TABLES_DIR / "08_cf_split_summary.csv"
CF_MAPPING_SUMMARY_OUTPUT_PATH = TABLES_DIR / "08_cf_mapping_summary.csv"
VALID_RECS_WIDE_OUTPUT_PATH = TABLES_DIR / "08_cf_valid_recommendations_wide.csv"
TEST_RECS_WIDE_OUTPUT_PATH = TABLES_DIR / "08_cf_test_recommendations_wide.csv"
VALID_RECS_LONG_OUTPUT_PATH = TABLES_DIR / "08_cf_valid_recommendations_long.csv"
TEST_RECS_LONG_OUTPUT_PATH = TABLES_DIR / "08_cf_test_recommendations_long.csv"
CF_METRICS_OUTPUT_PATH = TABLES_DIR / "08_cf_metrics.csv"
CF_NEIGHBOUR_SUMMARY_OUTPUT_PATH = TABLES_DIR / "08_cf_neighbour_summary.csv"

DASHBOARD_METRICS_OUTPUT_PATH = TABLES_DIR / "08_cf_metrics_dashboard.csv"
ACADEMIC_METRICS_OUTPUT_PATH = TABLES_DIR / "08_cf_metrics_academic.csv"
ACADEMIC_NEIGHBOUR_OUTPUT_PATH = TABLES_DIR / "08_cf_neighbour_summary_academic.csv"
DASHBOARD_SUMMARY_OUTPUT_PATH = TABLES_DIR / "08_cf_dashboard_summary.csv"

CF_FIGURE_OUTPUT_PATH = FIGURES_DIR / "08_cf_neighbour_distribution.png"
CF_FIGURE_OUTPUT_SVG_PATH = FIGURES_DIR / "08_cf_neighbour_distribution.svg"

CF_LOG_OUTPUT_PATH = LOGS_DIR / "08_cf_log.json"

MODEL_DIR = PROJECT_ROOT / "outputs" / "saved_models"

MODEL_OUTPUT_PATH = MODEL_DIR / "08_cf_model.joblib"
MODEL_METADATA_OUTPUT_PATH = MODEL_DIR / "08_cf_model_metadata.json"


# Helper functions


def ensure_model_directory() -> None:
    """
    Ensure the model output directory exists.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_files() -> None:
    """
    Validate that all required split files exist.

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
            "Missing required collaborative filtering input files:\n"
            + "\n".join(missing_paths)
        )


def load_implicit_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load implicit chronological train, validation, and test splits.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Train, validation, and test dataframes.
    """
    implicit_train = pd.read_parquet(IMPLICIT_TRAIN_PATH).copy()
    implicit_valid = pd.read_parquet(IMPLICIT_VALID_PATH).copy()
    implicit_test = pd.read_parquet(IMPLICIT_TEST_PATH).copy()

    return implicit_train, implicit_valid, implicit_test


def validate_required_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Validate that a split dataframe contains all required columns.

    Args:
        df:
            Split dataframe.
        dataset_name:
            Human-readable dataset name.

    Raises:
        ValueError:
            Raised when required columns are missing.
    """
    required_columns = [
        "user_id",
        "recipe_id",
        "date",
        "implicit_feedback",
        "user_idx",
        "item_idx",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def build_mapping_summary(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a summary of missing mapped indices across the three splits.
    """
    mapping_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": int(len(implicit_train)),
                "rows_missing_user_idx": int(implicit_train["user_idx"].isna().sum()),
                "rows_missing_item_idx": int(implicit_train["item_idx"].isna().sum()),
                "rows_missing_either_idx": int(
                    (
                        implicit_train["user_idx"].isna()
                        | implicit_train["item_idx"].isna()
                    ).sum()
                ),
            },
            {
                "split": "valid",
                "rows": int(len(implicit_valid)),
                "rows_missing_user_idx": int(implicit_valid["user_idx"].isna().sum()),
                "rows_missing_item_idx": int(implicit_valid["item_idx"].isna().sum()),
                "rows_missing_either_idx": int(
                    (
                        implicit_valid["user_idx"].isna()
                        | implicit_valid["item_idx"].isna()
                    ).sum()
                ),
            },
            {
                "split": "test",
                "rows": int(len(implicit_test)),
                "rows_missing_user_idx": int(implicit_test["user_idx"].isna().sum()),
                "rows_missing_item_idx": int(implicit_test["item_idx"].isna().sum()),
                "rows_missing_either_idx": int(
                    (
                        implicit_test["user_idx"].isna()
                        | implicit_test["item_idx"].isna()
                    ).sum()
                ),
            },
        ]
    )

    mapping_summary["rows_with_complete_indices"] = (
        mapping_summary["rows"] - mapping_summary["rows_missing_either_idx"]
    )
    mapping_summary["complete_index_rate"] = (
        mapping_summary["rows_with_complete_indices"] / mapping_summary["rows"]
    ).round(4)

    return mapping_summary


def keep_cf_evaluable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that have train-fitted user and item indices.
    """
    df_cf = df.dropna(subset=["user_idx", "item_idx"]).copy()
    df_cf["user_idx"] = df_cf["user_idx"].astype(int)
    df_cf["item_idx"] = df_cf["item_idx"].astype(int)

    return df_cf


def build_cf_split_summary(
    train_cf: pd.DataFrame,
    valid_cf: pd.DataFrame,
    test_cf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary table for the CF-evaluable splits.
    """
    summary_df = pd.DataFrame(
        [
            {
                "split": "train_cf",
                "rows": int(len(train_cf)),
                "users": int(train_cf["user_id"].nunique()),
                "recipes": int(train_cf["recipe_id"].nunique()),
                "min_date": train_cf["date"].min(),
                "max_date": train_cf["date"].max(),
            },
            {
                "split": "valid_cf",
                "rows": int(len(valid_cf)),
                "users": int(valid_cf["user_id"].nunique()),
                "recipes": int(valid_cf["recipe_id"].nunique()),
                "min_date": valid_cf["date"].min(),
                "max_date": valid_cf["date"].max(),
            },
            {
                "split": "test_cf",
                "rows": int(len(test_cf)),
                "users": int(test_cf["user_id"].nunique()),
                "recipes": int(test_cf["recipe_id"].nunique()),
                "min_date": test_cf["date"].min(),
                "max_date": test_cf["date"].max(),
            },
        ]
    )

    return summary_df


def build_train_matrix(train_cf: pd.DataFrame) -> sparse.csr_matrix:
    """
    Build the sparse user-item interaction matrix from training data.
    """
    n_users = int(train_cf["user_idx"].max()) + 1
    n_items = int(train_cf["item_idx"].max()) + 1

    train_matrix = sparse.csr_matrix(
        (
            train_cf["implicit_feedback"].astype(np.float32).to_numpy(),
            (train_cf["user_idx"].to_numpy(), train_cf["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    return train_matrix


def build_user_seen_history(train_cf: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a mapping of users to recipes already seen in training.
    """
    user_seen_history = (
        train_cf.groupby("user_id")["recipe_id"]
        .agg(lambda values: set(values.tolist()))
        .to_dict()
    )

    return user_seen_history


def build_user_train_item_indices(train_cf: pd.DataFrame) -> dict[int, list[int]]:
    """
    Build a mapping of users to training item indices.
    """
    user_train_items_idx = train_cf.groupby("user_id")["item_idx"].agg(list).to_dict()

    return user_train_items_idx


def build_holdout_truth(holdout_df: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a binary relevance mapping for a holdout split.
    """
    holdout_truth = (
        holdout_df.groupby("user_id")["recipe_id"]
        .agg(lambda values: set(values.tolist()))
        .to_dict()
    )

    return holdout_truth


def build_normalised_item_user_matrix(
    train_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """
    Build the normalised item-user matrix for cosine similarity.
    """
    item_user_matrix = train_matrix.T.tocsr()

    item_norms = np.sqrt(item_user_matrix.multiply(item_user_matrix).sum(axis=1)).A1
    item_norms[item_norms == 0] = 1.0

    inv_item_norms = 1.0 / item_norms
    normalised_item_user = sparse.diags(inv_item_norms).dot(item_user_matrix).tocsr()

    return normalised_item_user


def top_k_item_neighbours(
    normalised_item_user_matrix: sparse.csr_matrix,
    top_k: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build top-K cosine neighbours for each item.
    """
    neighbour_indices: list[np.ndarray] = []
    neighbour_scores: list[np.ndarray] = []

    n_items = normalised_item_user_matrix.shape[0]

    for item_idx in range(n_items):
        item_vector = normalised_item_user_matrix.getrow(item_idx)
        similarities = (item_vector @ normalised_item_user_matrix.T).tocsr()

        row_indices = similarities.indices
        row_scores = similarities.data.copy()

        if len(row_indices) == 0:
            neighbour_indices.append(np.array([], dtype=np.int32))
            neighbour_scores.append(np.array([], dtype=np.float32))
            continue

        mask_not_self = row_indices != item_idx
        row_indices = row_indices[mask_not_self]
        row_scores = row_scores[mask_not_self]

        positive_mask = row_scores > 0
        row_indices = row_indices[positive_mask]
        row_scores = row_scores[positive_mask]

        if len(row_scores) > top_k:
            top_positions = np.argpartition(-row_scores, top_k - 1)[:top_k]
            row_indices = row_indices[top_positions]
            row_scores = row_scores[top_positions]

        sort_order = np.argsort(-row_scores)
        row_indices = row_indices[sort_order].astype(np.int32)
        row_scores = row_scores[sort_order].astype(np.float32)

        neighbour_indices.append(row_indices)
        neighbour_scores.append(row_scores)

        if item_idx % 20000 == 0:
            print(f"Processed item {item_idx:,} / {n_items:,}")

    return neighbour_indices, neighbour_scores


def build_item_index_lookups(
    train_cf: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build item-index lookup dictionaries.
    """
    item_idx_to_recipe_id = (
        train_cf[["item_idx", "recipe_id"]]
        .drop_duplicates()
        .set_index("item_idx")["recipe_id"]
        .to_dict()
    )
    recipe_id_to_item_idx = {
        recipe_id: item_idx for item_idx, recipe_id in item_idx_to_recipe_id.items()
    }

    return item_idx_to_recipe_id, recipe_id_to_item_idx


def recommend_item_cf(
    user_id: int,
    user_train_items_idx: dict[int, list[int]],
    user_seen_history: dict[int, set[int]],
    item_neighbour_indices: list[np.ndarray],
    item_neighbour_scores: list[np.ndarray],
    item_idx_to_recipe_id: dict[int, int],
    top_n: int,
) -> list[tuple[int, float]]:
    """
    Recommend top-N items for one user using item-item neighbourhood scores.
    """
    seen_item_indices = user_train_items_idx.get(user_id, [])
    seen_recipe_ids = user_seen_history.get(user_id, set())

    if len(seen_item_indices) == 0:
        return []

    score_dict: dict[int, float] = {}

    for item_idx in seen_item_indices:
        neighbours = item_neighbour_indices[item_idx]
        scores = item_neighbour_scores[item_idx]

        for neighbour_idx, sim_score in zip(neighbours, scores):
            recipe_id = item_idx_to_recipe_id.get(int(neighbour_idx))
            if recipe_id is None:
                continue
            if recipe_id in seen_recipe_ids:
                continue

            score_dict[recipe_id] = score_dict.get(recipe_id, 0.0) + float(sim_score)

    if not score_dict:
        return []

    ranked = sorted(score_dict.items(), key=lambda pair: pair[1], reverse=True)

    return ranked[:top_n]


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute Precision@K for one user.
    """
    if k <= 0:
        return 0.0

    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute Recall@K for one user.
    """
    if not relevant:
        return 0.0

    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """
    Compute Hit Rate@K for one user.
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


def build_item_self_information(train_cf: pd.DataFrame) -> dict[int, float]:
    """
    Build item self-information values for novelty analysis.
    """
    train_item_popularity = train_cf.groupby("recipe_id", as_index=False).agg(
        interaction_count=("implicit_feedback", "sum")
    )

    item_probability = (
        train_item_popularity.set_index("recipe_id")["interaction_count"]
        / train_item_popularity["interaction_count"].sum()
    )

    item_self_information = {
        int(recipe_id): float(-math.log2(probability))
        for recipe_id, probability in item_probability.items()
        if probability > 0
    }

    return item_self_information


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


def evaluate_holdout_split(
    split_name: str,
    holdout_truth: dict[int, set[int]],
    user_train_items_idx: dict[int, list[int]],
    user_seen_history: dict[int, set[int]],
    item_neighbour_indices: list[np.ndarray],
    item_neighbour_scores: list[np.ndarray],
    item_idx_to_recipe_id: dict[int, int],
    item_self_information: dict[int, float],
    catalog_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the collaborative filtering model on one holdout split.
    """
    user_rows: list[dict[str, object]] = []
    recommended_items_by_k = {k: [] for k in TOP_K_VALUES}

    max_required_recs = (
        max(max(TOP_K_VALUES), EXPORT_TOP_N) * RECOMMENDATION_CANDIDATE_MULTIPLIER
    )

    for user_id, relevant_items in holdout_truth.items():
        ranked_pairs = recommend_item_cf(
            user_id=user_id,
            user_train_items_idx=user_train_items_idx,
            user_seen_history=user_seen_history,
            item_neighbour_indices=item_neighbour_indices,
            item_neighbour_scores=item_neighbour_scores,
            item_idx_to_recipe_id=item_idx_to_recipe_id,
            top_n=max_required_recs,
        )
        eval_recipe_ids = [recipe_id for recipe_id, _ in ranked_pairs]
        eval_scores = [float(score) for _, score in ranked_pairs]

        export_pairs = ranked_pairs[:EXPORT_TOP_N]
        export_recipe_ids = [int(recipe_id) for recipe_id, _ in export_pairs]
        export_scores = [float(score) for _, score in export_pairs]

        if len(eval_recipe_ids) == 0:
            continue

        row: dict[str, object] = {
            "split": split_name,
            "user_id": int(user_id),
            "holdout_item_count": int(len(relevant_items)),
            "recommended_recipe_ids": export_recipe_ids,
            "recommended_scores": export_scores,
        }

        for k in TOP_K_VALUES:
            row[f"precision_at_{k}"] = precision_at_k(
                eval_recipe_ids, relevant_items, k
            )
            row[f"recall_at_{k}"] = recall_at_k(eval_recipe_ids, relevant_items, k)
            row[f"hit_rate_at_{k}"] = hit_rate_at_k(eval_recipe_ids, relevant_items, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(eval_recipe_ids, relevant_items, k)
            row[f"novelty_at_{k}"] = novelty_at_k(
                eval_recipe_ids,
                item_self_information,
                k,
            )

            recommended_items_by_k[k].extend(eval_recipe_ids[:k])

        user_rows.append(row)

    user_metrics_df = pd.DataFrame(user_rows)

    metric_rows: list[dict[str, object]] = []

    for k in TOP_K_VALUES:
        unique_recommended_count = len(set(recommended_items_by_k[k]))

        metric_rows.append(
            {
                "split": split_name,
                "k": k,
                "users_evaluated": int(user_metrics_df["user_id"].nunique()),
                "precision_at_k": float(user_metrics_df[f"precision_at_{k}"].mean()),
                "recall_at_k": float(user_metrics_df[f"recall_at_{k}"].mean()),
                "hit_rate_at_k": float(user_metrics_df[f"hit_rate_at_{k}"].mean()),
                "ndcg_at_k": float(user_metrics_df[f"ndcg_at_{k}"].mean()),
                "novelty_at_k": float(user_metrics_df[f"novelty_at_{k}"].mean()),
                "catalog_coverage_at_k": float(unique_recommended_count / catalog_size),
                "recommendation_count": int(len(recommended_items_by_k[k])),
            }
        )

    split_metrics_df = pd.DataFrame(metric_rows)

    return user_metrics_df, split_metrics_df


def build_train_item_rank(train_cf: pd.DataFrame) -> pd.DataFrame:
    """
    Build a train popularity ranking table for recommendation exports.
    """
    train_item_rank = (
        train_cf.groupby("recipe_id", as_index=False)
        .agg(train_interaction_count=("implicit_feedback", "sum"))
        .sort_values("train_interaction_count", ascending=False)
        .reset_index(drop=True)
    )

    train_item_rank["train_popularity_rank"] = np.arange(1, len(train_item_rank) + 1)

    return train_item_rank


def expand_recommendations_long(
    recommendations_wide: pd.DataFrame,
    train_item_rank: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand wide recommendation exports into long format.
    """
    rank_lookup = train_item_rank.set_index("recipe_id").to_dict("index")

    long_rows: list[dict[str, object]] = []

    for row in recommendations_wide.itertuples(index=False):
        recipe_ids = list(row.recommended_recipe_ids)
        scores = (
            list(row.recommended_scores)
            if hasattr(row, "recommended_scores")
            else [np.nan] * len(recipe_ids)
        )

        for recommendation_rank, (recipe_id, score) in enumerate(
            zip(recipe_ids, scores),
            start=1,
        ):
            rank_metadata = rank_lookup.get(recipe_id, {})

            long_rows.append(
                {
                    "split": row.split,
                    "user_id": int(row.user_id),
                    "holdout_item_count": int(row.holdout_item_count),
                    "recommendation_rank": int(recommendation_rank),
                    "recipe_id": int(recipe_id),
                    "score": float(score) if pd.notna(score) else np.nan,
                    "train_interaction_count": rank_metadata.get(
                        "train_interaction_count"
                    ),
                    "train_popularity_rank": rank_metadata.get("train_popularity_rank"),
                }
            )

    recommendations_long = pd.DataFrame(long_rows)
    return recommendations_long


def build_neighbour_summary(
    item_neighbour_indices: list[np.ndarray],
) -> pd.DataFrame:
    """
    Build a summary table of neighbourhood sizes.
    """
    neighbour_summary = pd.DataFrame(
        {
            "item_idx": np.arange(len(item_neighbour_indices)),
            "neighbour_count": [
                len(neighbours) for neighbours in item_neighbour_indices
            ],
        }
    )
    neighbour_summary["has_neighbours"] = (
        neighbour_summary["neighbour_count"] > 0
    ).astype(int)

    return neighbour_summary


def build_dashboard_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dashboard-friendly model comparison table.
    """
    dashboard_df = metrics_df.copy()

    dashboard_df["model"] = "Collaborative Filtering"
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

    academic_df["Model"] = "Item-Item Collaborative Filtering"
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


def build_academic_neighbour_table(
    neighbour_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a report-ready neighbourhood summary table.
    """
    academic_neighbour = pd.DataFrame(
        [
            {
                "Neighbourhood Measure": "Items with neighbours",
                "Value": int(neighbour_summary["has_neighbours"].sum()),
            },
            {
                "Neighbourhood Measure": "Items without neighbours",
                "Value": int((neighbour_summary["has_neighbours"] == 0).sum()),
            },
            {
                "Neighbourhood Measure": "Mean neighbour count",
                "Value": round(float(neighbour_summary["neighbour_count"].mean()), 2),
            },
            {
                "Neighbourhood Measure": "Median neighbour count",
                "Value": round(float(neighbour_summary["neighbour_count"].median()), 2),
            },
            {
                "Neighbourhood Measure": "Maximum neighbour count",
                "Value": int(neighbour_summary["neighbour_count"].max()),
            },
        ]
    )

    return academic_neighbour


def build_dashboard_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a very compact dashboard summary table.
    """
    summary_df = metrics_df.copy()
    summary_df["model"] = "Collaborative Filtering"
    summary_df["label"] = (
        summary_df["split"].str.upper() + "_AT_" + summary_df["k"].astype(str)
    )

    summary_df["precision"] = summary_df["precision_at_k"].round(4)
    summary_df["recall"] = summary_df["recall_at_k"].round(4)
    summary_df["hit_rate"] = summary_df["hit_rate_at_k"].round(4)
    summary_df["ndcg"] = summary_df["ndcg_at_k"].round(4)
    summary_df["novelty"] = summary_df["novelty_at_k"].round(4)
    summary_df["coverage"] = summary_df["catalog_coverage_at_k"].round(4)

    summary_df = summary_df[
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

    return summary_df


def plot_neighbour_distribution(
    neighbour_summary: pd.DataFrame,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible neighbourhood size distribution figure.

    The figure is designed for academic reporting and dashboard reuse:
    - high contrast
    - readable labels
    - grid for easier visual reading
    - PNG and SVG export
    """
    plt.figure(figsize=(9.5, 6.5))

    plt.hist(
        neighbour_summary["neighbour_count"],
        bins=30,
        edgecolor="black",
        linewidth=0.8,
    )

    plt.xlabel("Neighbour count per item", fontsize=12)
    plt.ylabel("Number of items", fontsize=12)
    plt.title("Item neighbourhood size distribution", fontsize=13)
    plt.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.8)
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def save_model_artifact(
    train_matrix: sparse.csr_matrix,
    item_neighbour_indices: list[np.ndarray],
    item_neighbour_scores: list[np.ndarray],
    item_idx_to_recipe_id: dict[int, int],
    user_seen_history: dict[int, set[int]],
    user_train_items_idx: dict[int, list[int]],
    train_item_rank: pd.DataFrame,
) -> None:
    """
    Save the CF model artifact for dashboard reuse.
    """
    model_artifact = {
        "model_name": "item_item_cf_sparse_cosine",
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "item_neighbour_k": ITEM_NEIGHBOUR_K,
        "recommendation_candidate_multiplier": RECOMMENDATION_CANDIDATE_MULTIPLIER,
        "train_matrix_shape": tuple(train_matrix.shape),
        "item_neighbour_indices": item_neighbour_indices,
        "item_neighbour_scores": item_neighbour_scores,
        "item_idx_to_recipe_id": item_idx_to_recipe_id,
        "user_seen_history": user_seen_history,
        "user_train_items_idx": user_train_items_idx,
        "train_item_rank": train_item_rank,
    }

    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)

    metadata = {
        "model_name": "item_item_cf_sparse_cosine",
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_output_path": str(MODEL_OUTPUT_PATH),
        "train_matrix_shape": [int(train_matrix.shape[0]), int(train_matrix.shape[1])],
        "item_neighbour_k": ITEM_NEIGHBOUR_K,
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "fields_available": [
            "item_neighbour_indices",
            "item_neighbour_scores",
            "item_idx_to_recipe_id",
            "user_seen_history",
            "user_train_items_idx",
            "train_item_rank",
        ],
    }

    with open(MODEL_METADATA_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_run_log(
    train_cf: pd.DataFrame,
    valid_cf: pd.DataFrame,
    test_cf: pd.DataFrame,
    train_matrix: sparse.csr_matrix,
    cf_metrics: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    neighbour_summary: pd.DataFrame,
) -> None:
    """
    Save a structured JSON log for the collaborative filtering phase.
    """
    best_row = cf_metrics.sort_values(
        by=["split", "ndcg_at_k", "recall_at_k"],
        ascending=[True, False, False],
    ).iloc[0]

    run_log = {
        "model": "item_item_cf_sparse_cosine",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "train_rows_cf_evaluable": int(len(train_cf)),
            "valid_rows_cf_evaluable": int(len(valid_cf)),
            "test_rows_cf_evaluable": int(len(test_cf)),
            "train_users": int(train_cf["user_id"].nunique()),
            "valid_users": int(valid_cf["user_id"].nunique()),
            "test_users": int(test_cf["user_id"].nunique()),
            "train_items": int(train_cf["recipe_id"].nunique()),
            "valid_items": int(valid_cf["recipe_id"].nunique()),
            "test_items": int(test_cf["recipe_id"].nunique()),
        },
        "matrix_summary": {
            "matrix_shape": [int(train_matrix.shape[0]), int(train_matrix.shape[1])],
            "matrix_nnz": int(train_matrix.nnz),
            "matrix_density": float(
                train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])
            ),
        },
        "config": {
            "item_neighbour_k": ITEM_NEIGHBOUR_K,
            "top_k_values": TOP_K_VALUES,
            "export_top_n": EXPORT_TOP_N,
            "recommendation_candidate_multiplier": RECOMMENDATION_CANDIDATE_MULTIPLIER,
        },
        "artifacts": {
            "metrics_csv": str(CF_METRICS_OUTPUT_PATH),
            "dashboard_metrics_csv": str(DASHBOARD_METRICS_OUTPUT_PATH),
            "academic_metrics_csv": str(ACADEMIC_METRICS_OUTPUT_PATH),
            "academic_neighbour_csv": str(ACADEMIC_NEIGHBOUR_OUTPUT_PATH),
            "dashboard_summary_csv": str(DASHBOARD_SUMMARY_OUTPUT_PATH),
            "figure_png": str(CF_FIGURE_OUTPUT_PATH),
            "figure_svg": str(CF_FIGURE_OUTPUT_SVG_PATH),
            "model_joblib": str(MODEL_OUTPUT_PATH),
            "model_metadata_json": str(MODEL_METADATA_OUTPUT_PATH),
        },
        "mapping_summary": mapping_summary.to_dict(orient="records"),
        "neighbour_summary": {
            "items_with_neighbours": int(neighbour_summary["has_neighbours"].sum()),
            "items_without_neighbours": int(
                (neighbour_summary["has_neighbours"] == 0).sum()
            ),
            "mean_neighbour_count": float(neighbour_summary["neighbour_count"].mean()),
            "median_neighbour_count": float(
                neighbour_summary["neighbour_count"].median()
            ),
            "max_neighbour_count": int(neighbour_summary["neighbour_count"].max()),
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
        "metrics": cf_metrics.to_dict(orient="records"),
    }

    with open(CF_LOG_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(run_log, file, indent=2)


# Main pipeline


def main() -> None:
    """
    Run the full collaborative filtering pipeline.
    """
    ensure_directories()
    ensure_model_directory()
    validate_input_files()

    implicit_train, implicit_valid, implicit_test = load_implicit_splits()

    validate_required_columns(implicit_train, "implicit_train")
    validate_required_columns(implicit_valid, "implicit_valid")
    validate_required_columns(implicit_test, "implicit_test")

    for df in [implicit_train, implicit_valid, implicit_test]:
        df["date"] = pd.to_datetime(df["date"])

    mapping_summary = build_mapping_summary(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
    )

    train_cf = keep_cf_evaluable_rows(implicit_train)
    valid_cf = keep_cf_evaluable_rows(implicit_valid)
    test_cf = keep_cf_evaluable_rows(implicit_test)

    cf_split_summary = build_cf_split_summary(
        train_cf=train_cf,
        valid_cf=valid_cf,
        test_cf=test_cf,
    )

    print("CF mapping summary:")
    print(mapping_summary)

    print("\nCF split summary:")
    print(cf_split_summary)

    train_matrix = build_train_matrix(train_cf)

    print("\nTrain matrix shape:", train_matrix.shape)
    print("Train matrix nnz:", train_matrix.nnz)
    print(
        "Train matrix density:",
        train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]),
    )

    user_seen_history = build_user_seen_history(train_cf)
    user_train_items_idx = build_user_train_item_indices(train_cf)
    valid_truth = build_holdout_truth(valid_cf)
    test_truth = build_holdout_truth(test_cf)

    normalised_item_user = build_normalised_item_user_matrix(train_matrix)

    print("\nNormalised item-user matrix shape:", normalised_item_user.shape)

    item_neighbour_indices, item_neighbour_scores = top_k_item_neighbours(
        normalised_item_user_matrix=normalised_item_user,
        top_k=ITEM_NEIGHBOUR_K,
    )

    item_idx_to_recipe_id, _ = build_item_index_lookups(train_cf)
    item_self_information = build_item_self_information(train_cf)
    catalog_size = int(train_cf["recipe_id"].nunique())

    valid_user_metrics, valid_metrics = evaluate_holdout_split(
        split_name="valid",
        holdout_truth=valid_truth,
        user_train_items_idx=user_train_items_idx,
        user_seen_history=user_seen_history,
        item_neighbour_indices=item_neighbour_indices,
        item_neighbour_scores=item_neighbour_scores,
        item_idx_to_recipe_id=item_idx_to_recipe_id,
        item_self_information=item_self_information,
        catalog_size=catalog_size,
    )

    test_user_metrics, test_metrics = evaluate_holdout_split(
        split_name="test",
        holdout_truth=test_truth,
        user_train_items_idx=user_train_items_idx,
        user_seen_history=user_seen_history,
        item_neighbour_indices=item_neighbour_indices,
        item_neighbour_scores=item_neighbour_scores,
        item_idx_to_recipe_id=item_idx_to_recipe_id,
        item_self_information=item_self_information,
        catalog_size=catalog_size,
    )

    cf_metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)

    valid_recommendations_wide = valid_user_metrics[
        [
            "split",
            "user_id",
            "holdout_item_count",
            "recommended_recipe_ids",
            "recommended_scores",
        ]
    ].copy()

    test_recommendations_wide = test_user_metrics[
        [
            "split",
            "user_id",
            "holdout_item_count",
            "recommended_recipe_ids",
            "recommended_scores",
        ]
    ].copy()

    train_item_rank = build_train_item_rank(train_cf)

    valid_recommendations_long = expand_recommendations_long(
        recommendations_wide=valid_recommendations_wide,
        train_item_rank=train_item_rank,
    )
    test_recommendations_long = expand_recommendations_long(
        recommendations_wide=test_recommendations_wide,
        train_item_rank=train_item_rank,
    )

    neighbour_summary = build_neighbour_summary(item_neighbour_indices)

    dashboard_metrics = build_dashboard_metrics_table(cf_metrics)
    academic_metrics = build_academic_metrics_table(cf_metrics)
    academic_neighbour = build_academic_neighbour_table(neighbour_summary)
    dashboard_summary = build_dashboard_summary_table(cf_metrics)

    mapping_summary.to_csv(CF_MAPPING_SUMMARY_OUTPUT_PATH, index=False)
    cf_split_summary.to_csv(CF_SPLIT_SUMMARY_OUTPUT_PATH, index=False)
    valid_recommendations_wide.to_csv(VALID_RECS_WIDE_OUTPUT_PATH, index=False)
    test_recommendations_wide.to_csv(TEST_RECS_WIDE_OUTPUT_PATH, index=False)
    valid_recommendations_long.to_csv(VALID_RECS_LONG_OUTPUT_PATH, index=False)
    test_recommendations_long.to_csv(TEST_RECS_LONG_OUTPUT_PATH, index=False)
    cf_metrics.to_csv(CF_METRICS_OUTPUT_PATH, index=False)
    neighbour_summary.to_csv(CF_NEIGHBOUR_SUMMARY_OUTPUT_PATH, index=False)

    dashboard_metrics.to_csv(DASHBOARD_METRICS_OUTPUT_PATH, index=False)
    academic_metrics.to_csv(ACADEMIC_METRICS_OUTPUT_PATH, index=False)
    academic_neighbour.to_csv(ACADEMIC_NEIGHBOUR_OUTPUT_PATH, index=False)
    dashboard_summary.to_csv(DASHBOARD_SUMMARY_OUTPUT_PATH, index=False)

    plot_neighbour_distribution(
        neighbour_summary=neighbour_summary,
        output_path_png=CF_FIGURE_OUTPUT_PATH,
        output_path_svg=CF_FIGURE_OUTPUT_SVG_PATH,
    )

    save_model_artifact(
        train_matrix=train_matrix,
        item_neighbour_indices=item_neighbour_indices,
        item_neighbour_scores=item_neighbour_scores,
        item_idx_to_recipe_id=item_idx_to_recipe_id,
        user_seen_history=user_seen_history,
        user_train_items_idx=user_train_items_idx,
        train_item_rank=train_item_rank,
    )

    save_run_log(
        train_cf=train_cf,
        valid_cf=valid_cf,
        test_cf=test_cf,
        train_matrix=train_matrix,
        cf_metrics=cf_metrics,
        mapping_summary=mapping_summary,
        neighbour_summary=neighbour_summary,
    )

    print("\nNeighbour summary:")
    print(neighbour_summary["neighbour_count"].describe())

    print("\nCollaborative filtering metrics:")
    print(cf_metrics)

    print("\nSaved outputs:")
    print("-", CF_SPLIT_SUMMARY_OUTPUT_PATH)
    print("-", CF_MAPPING_SUMMARY_OUTPUT_PATH)
    print("-", VALID_RECS_WIDE_OUTPUT_PATH)
    print("-", TEST_RECS_WIDE_OUTPUT_PATH)
    print("-", VALID_RECS_LONG_OUTPUT_PATH)
    print("-", TEST_RECS_LONG_OUTPUT_PATH)
    print("-", CF_METRICS_OUTPUT_PATH)
    print("-", CF_NEIGHBOUR_SUMMARY_OUTPUT_PATH)
    print("-", DASHBOARD_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_NEIGHBOUR_OUTPUT_PATH)
    print("-", DASHBOARD_SUMMARY_OUTPUT_PATH)
    print("-", CF_FIGURE_OUTPUT_PATH)
    print("-", CF_FIGURE_OUTPUT_SVG_PATH)
    print("-", MODEL_OUTPUT_PATH)
    print("-", MODEL_METADATA_OUTPUT_PATH)
    print("-", CF_LOG_OUTPUT_PATH)


if __name__ == "__main__":
    main()
