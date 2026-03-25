"""
src/models/train_popularity.py

Purpose:
Build and evaluate the popularity baseline recommender using the
chronological implicit interaction splits.

This module implements the simplest recommendation benchmark in the
project. It learns global item popularity from the training split only,
recommends the most popular unseen recipes to each user, evaluates the
results on validation and test splits, and saves presentation-ready
tables, accessible figures, dashboard artifacts, and structured logs.

Responsibilities:
- load implicit train, validation, and test split files
- validate required columns and split integrity
- compute global recipe popularity from training data only
- build user seen-item history from training interactions
- generate top-N unseen recommendations per user
- evaluate ranking quality on validation and test splits
- measure popularity concentration and catalogue coverage
- save outputs for later comparison, reporting, and dashboard use
- export dashboard-friendly and academic-report-friendly tables
- save a reusable popularity model artifact for dashboard inference

Design notes:
- the implicit interaction split is used for Top-N recommendation
- all popularity statistics are fitted from the training split only
- items already seen in training are excluded from each user's list
- holdout relevance is treated as binary user-item interaction presence
- ranking evaluation is used instead of rating prediction metrics
- outputs follow the existing project naming pattern with 07_* filenames
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

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


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

TOP_K_VALUES = [5, 10, 20]
EXPORT_TOP_N = 10

IMPLICIT_TRAIN_PATH = SPLITS_DIR / "implicit_train.parquet"
IMPLICIT_VALID_PATH = SPLITS_DIR / "implicit_valid.parquet"
IMPLICIT_TEST_PATH = SPLITS_DIR / "implicit_test.parquet"

SPLIT_SUMMARY_OUTPUT_PATH = TABLES_DIR / "07_popularity_split_summary.csv"
ITEM_POPULARITY_OUTPUT_PATH = TABLES_DIR / "07_item_popularity_table.csv"
DISTRIBUTION_OUTPUT_PATH = TABLES_DIR / "07_popularity_distribution_summary.csv"
VALID_RECS_WIDE_OUTPUT_PATH = TABLES_DIR / "07_popularity_valid_recommendations_wide.csv"
TEST_RECS_WIDE_OUTPUT_PATH = TABLES_DIR / "07_popularity_test_recommendations_wide.csv"
VALID_RECS_LONG_OUTPUT_PATH = TABLES_DIR / "07_popularity_valid_recommendations_long.csv"
TEST_RECS_LONG_OUTPUT_PATH = TABLES_DIR / "07_popularity_test_recommendations_long.csv"
METRICS_OUTPUT_PATH = TABLES_DIR / "07_popularity_metrics.csv"

DASHBOARD_METRICS_OUTPUT_PATH = TABLES_DIR / "07_popularity_metrics_dashboard.csv"
ACADEMIC_METRICS_OUTPUT_PATH = TABLES_DIR / "07_popularity_metrics_academic.csv"
ACADEMIC_DISTRIBUTION_OUTPUT_PATH = TABLES_DIR / "07_popularity_distribution_academic.csv"
DASHBOARD_SUMMARY_OUTPUT_PATH = TABLES_DIR / "07_popularity_dashboard_summary.csv"

FIGURE_OUTPUT_PATH = FIGURES_DIR / "07_popularity_concentration_curve.png"
FIGURE_OUTPUT_SVG_PATH = FIGURES_DIR / "07_popularity_concentration_curve.svg"

LOG_OUTPUT_PATH = LOGS_DIR / "07_popularity_log.json"

MODEL_DIR = PROJECT_ROOT / "outputs" / "saved_models"

MODEL_OUTPUT_PATH = MODEL_DIR / "07_popularity_model.joblib"
MODEL_METADATA_OUTPUT_PATH = MODEL_DIR / "07_popularity_model_metadata.json"


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
            "Missing required popularity input files:\n" + "\n".join(missing_paths)
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
    required_columns = ["user_id", "recipe_id", "date", "implicit_feedback"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def build_split_summary(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary table for the three implicit splits.

    Args:
        implicit_train:
            Training split.
        implicit_valid:
            Validation split.
        implicit_test:
            Test split.

    Returns:
        pd.DataFrame:
            Summary table with row counts, user counts, item counts,
            and date ranges.
    """
    summary_df = pd.DataFrame(
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

    return summary_df


def build_item_popularity_table(implicit_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build the global item popularity table from training interactions only.

    Args:
        implicit_train:
            Training split dataframe.

    Returns:
        pd.DataFrame:
            Ranked item popularity table.
    """
    aggregation_map: dict[str, tuple[str, str]] = {
        "interaction_count": ("implicit_feedback", "sum"),
        "user_count": ("user_id", "nunique"),
        "first_seen_date": ("date", "min"),
        "last_seen_date": ("date", "max"),
    }

    if "rating" in implicit_train.columns:
        aggregation_map["mean_rating"] = ("rating", "mean")

    item_popularity = (
        implicit_train.groupby("recipe_id", as_index=False)
        .agg(**aggregation_map)
        .sort_values(
            by=[col for col in ["interaction_count", "user_count", "mean_rating", "recipe_id"] if col in aggregation_map or col == "recipe_id"],
            ascending=[False, False, False, True][:len([col for col in ["interaction_count", "user_count", "mean_rating", "recipe_id"] if col in aggregation_map or col == "recipe_id"])],
        )
        .reset_index(drop=True)
    )

    if "mean_rating" not in item_popularity.columns:
        item_popularity["mean_rating"] = np.nan

    item_popularity["popularity_rank"] = np.arange(1, len(item_popularity) + 1)
    item_popularity["popularity_score"] = (
        item_popularity["interaction_count"] / item_popularity["interaction_count"].max()
    )

    total_interactions = item_popularity["interaction_count"].sum()
    item_popularity["interaction_share"] = (
        item_popularity["interaction_count"] / total_interactions
    )
    item_popularity["cumulative_interaction_share"] = (
        item_popularity["interaction_share"].cumsum()
    )
    item_popularity["cumulative_item_share"] = (
        np.arange(1, len(item_popularity) + 1) / len(item_popularity)
    )

    return item_popularity


def build_user_seen_history(implicit_train: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a mapping of users to recipes already seen in training.

    Args:
        implicit_train:
            Training split dataframe.

    Returns:
        dict[int, set[int]]:
            Mapping of user_id to a set of training recipe IDs.
    """
    user_seen_history = (
        implicit_train.groupby("user_id")["recipe_id"]
        .agg(lambda values: set(values.tolist()))
        .to_dict()
    )

    return user_seen_history


def build_holdout_truth(holdout_df: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a binary relevance mapping for a holdout split.

    Duplicate user-item rows within the holdout split are collapsed to
    a unique set for Top-N ranking evaluation.

    Args:
        holdout_df:
            Validation or test dataframe.

    Returns:
        dict[int, set[int]]:
            Mapping of user_id to relevant holdout recipe IDs.
    """
    holdout_truth = (
        holdout_df.groupby("user_id")["recipe_id"]
        .agg(lambda values: set(values.tolist()))
        .to_dict()
    )

    return holdout_truth


def get_top_n_popular_unseen(
    user_id: int,
    ranked_recipe_ids: np.ndarray,
    user_seen_history: dict[int, set[int]],
    top_n: int,
) -> list[int]:
    """
    Return the top-N globally popular unseen items for one user.

    Args:
        user_id:
            Target user ID.
        ranked_recipe_ids:
            Globally ranked recipe IDs from the popularity table.
        user_seen_history:
            Mapping of user_id to recipes already seen in training.
        top_n:
            Number of recommendations to return.

    Returns:
        list[int]:
            Top-N unseen recipe IDs for the user.
    """
    seen_items = user_seen_history.get(user_id, set())
    recommendations: list[int] = []

    for recipe_id in ranked_recipe_ids:
        if recipe_id not in seen_items:
            recommendations.append(int(recipe_id))

        if len(recommendations) == top_n:
            break

    return recommendations


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

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hit_count + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(recommended, relevant, k) / ideal_dcg


def build_item_self_information(
    item_popularity: pd.DataFrame,
) -> dict[int, float]:
    """
    Build item self-information values for novelty analysis.

    Args:
        item_popularity:
            Ranked item popularity table.

    Returns:
        dict[int, float]:
            Mapping of recipe_id to self-information value.
    """
    item_probability = (
        item_popularity.set_index("recipe_id")["interaction_count"]
        / item_popularity["interaction_count"].sum()
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
    ranked_recipe_ids: np.ndarray,
    user_seen_history: dict[int, set[int]],
    item_self_information: dict[int, float],
    catalog_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the popularity baseline on one holdout split.

    Args:
        split_name:
            Validation or test split label.
        holdout_truth:
            Mapping of user_id to relevant holdout recipe IDs.
        ranked_recipe_ids:
            Globally ranked recipe IDs.
        user_seen_history:
            Mapping of user_id to training-seen recipe IDs.
        item_self_information:
            Mapping of recipe_id to novelty values.
        catalog_size:
            Number of ranked recipes in the train popularity table.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            User-level evaluation table and split-level metric summary table.
    """
    user_rows: list[dict[str, object]] = []
    recommended_items_by_k = {k: [] for k in TOP_K_VALUES}

    max_required_recs = max(max(TOP_K_VALUES), EXPORT_TOP_N)

    for user_id, relevant_items in holdout_truth.items():
        recommendations = get_top_n_popular_unseen(
            user_id=user_id,
            ranked_recipe_ids=ranked_recipe_ids,
            user_seen_history=user_seen_history,
            top_n=max_required_recs,
        )

        row: dict[str, object] = {
            "split": split_name,
            "user_id": int(user_id),
            "holdout_item_count": int(len(relevant_items)),
            "recommended_recipe_ids": recommendations[:EXPORT_TOP_N],
        }

        for k in TOP_K_VALUES:
            row[f"precision_at_{k}"] = precision_at_k(recommendations, relevant_items, k)
            row[f"recall_at_{k}"] = recall_at_k(recommendations, relevant_items, k)
            row[f"hit_rate_at_{k}"] = hit_rate_at_k(recommendations, relevant_items, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(recommendations, relevant_items, k)
            row[f"novelty_at_{k}"] = novelty_at_k(
                recommendations,
                item_self_information,
                k,
            )

            recommended_items_by_k[k].extend(recommendations[:k])

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


def build_popularity_distribution_summary(
    item_popularity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a summary table describing popularity concentration.
    """
    checkpoints = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    distribution_rows: list[dict[str, float | int]] = []

    for cutoff in checkpoints:
        top_item_count = max(1, int(math.ceil(len(item_popularity) * cutoff)))
        subset = item_popularity.iloc[:top_item_count]

        distribution_rows.append(
            {
                "top_item_share": cutoff,
                "top_item_count": int(top_item_count),
                "interaction_share_captured": float(subset["interaction_share"].sum()),
            }
        )

    distribution_summary = pd.DataFrame(distribution_rows)

    return distribution_summary


def build_popularity_curve(item_popularity: pd.DataFrame) -> pd.DataFrame:
    """
    Build the cumulative popularity curve table used for plotting.
    """
    return item_popularity[
        [
            "recipe_id",
            "interaction_count",
            "interaction_share",
            "cumulative_interaction_share",
            "cumulative_item_share",
        ]
    ].copy()


def expand_recommendations_long(
    recommendations_wide: pd.DataFrame,
    item_popularity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand wide recommendation exports into long format.
    """
    rank_lookup = item_popularity.set_index("recipe_id")[
        ["popularity_rank", "interaction_count", "popularity_score"]
    ].to_dict("index")

    long_rows: list[dict[str, object]] = []

    for row in recommendations_wide.itertuples(index=False):
        for recommendation_rank, recipe_id in enumerate(
            row.recommended_recipe_ids,
            start=1,
        ):
            popularity_metadata = rank_lookup.get(recipe_id, {})

            long_rows.append(
                {
                    "split": row.split,
                    "user_id": int(row.user_id),
                    "holdout_item_count": int(row.holdout_item_count),
                    "recommendation_rank": recommendation_rank,
                    "recipe_id": int(recipe_id),
                    "global_popularity_rank": popularity_metadata.get("popularity_rank"),
                    "train_interaction_count": popularity_metadata.get("interaction_count"),
                    "popularity_score": popularity_metadata.get("popularity_score"),
                }
            )

    recommendations_long = pd.DataFrame(long_rows)

    return recommendations_long


def build_dashboard_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dashboard-friendly model comparison table.
    """
    dashboard_df = metrics_df.copy()

    dashboard_df["model"] = "Popularity"
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

    academic_df["Model"] = "Popularity Baseline"
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


def build_academic_distribution_table(
    distribution_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a report-ready popularity concentration summary table.
    """
    academic_distribution = distribution_df.copy()

    academic_distribution["Top Item Share (%)"] = (
        academic_distribution["top_item_share"] * 100
    ).round(0).astype(int)

    academic_distribution["Top Item Count"] = (
        academic_distribution["top_item_count"].astype(int)
    )

    academic_distribution["Interaction Share Captured (%)"] = (
        academic_distribution["interaction_share_captured"] * 100
    ).round(2)

    academic_distribution = academic_distribution[
        [
            "Top Item Share (%)",
            "Top Item Count",
            "Interaction Share Captured (%)",
        ]
    ].copy()

    return academic_distribution


def build_dashboard_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a very compact dashboard summary table.
    """
    summary_df = metrics_df.copy()
    summary_df["model"] = "Popularity"
    summary_df["label"] = summary_df["split"].str.upper() + "_AT_" + summary_df["k"].astype(str)

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


def plot_popularity_concentration_curve(
    popularity_curve: pd.DataFrame,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible cumulative popularity concentration figure.

    The figure is designed for academic reporting and long-term reuse:
    - high contrast
    - readable font sizes
    - marker-assisted line for colour-blind accessibility
    - diagonal equality reference line
    - exported as both PNG and SVG
    """
    x = popularity_curve["cumulative_item_share"]
    y = popularity_curve["cumulative_interaction_share"]

    plt.figure(figsize=(9.5, 6.5))

    plt.plot(
        x,
        y,
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        markevery=max(1, len(popularity_curve) // 20),
        label="Observed popularity concentration",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Equality reference",
    )

    plt.xlabel("Cumulative share of recipes", fontsize=12)
    plt.ylabel("Cumulative share of interactions", fontsize=12)
    plt.title("Popularity concentration of training interactions", fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def save_model_artifact(
    item_popularity: pd.DataFrame,
    user_seen_history: dict[int, set[int]],
) -> None:
    """
    Save the popularity baseline artifact for dashboard reuse.

    The saved object contains the ranked popularity table, the ranked
    recipe ID list, user seen-history, and configuration metadata.
    """
    ranked_recipe_ids = item_popularity["recipe_id"].tolist()

    model_artifact = {
        "model_name": "popularity",
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "ranked_recipe_ids": ranked_recipe_ids,
        "item_popularity_table": item_popularity,
        "user_seen_history": user_seen_history,
    }

    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)

    metadata = {
        "model_name": "popularity",
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_output_path": str(MODEL_OUTPUT_PATH),
        "ranked_recipe_count": int(len(ranked_recipe_ids)),
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "fields_available": [
            "ranked_recipe_ids",
            "item_popularity_table",
            "user_seen_history",
        ],
    }

    with open(MODEL_METADATA_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_run_log(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
    item_popularity: pd.DataFrame,
    popularity_metrics: pd.DataFrame,
) -> None:
    """
    Save a structured JSON log for pipeline monitoring and dashboard use.
    """
    best_row = popularity_metrics.sort_values(
        by=["split", "ndcg_at_k", "recall_at_k"],
        ascending=[True, False, False],
    ).iloc[0]

    run_log = {
        "model": "popularity",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "train_rows": int(len(implicit_train)),
            "valid_rows": int(len(implicit_valid)),
            "test_rows": int(len(implicit_test)),
            "train_users": int(implicit_train["user_id"].nunique()),
            "valid_users": int(implicit_valid["user_id"].nunique()),
            "test_users": int(implicit_test["user_id"].nunique()),
            "train_recipes": int(implicit_train["recipe_id"].nunique()),
            "valid_recipes": int(implicit_valid["recipe_id"].nunique()),
            "test_recipes": int(implicit_test["recipe_id"].nunique()),
        },
        "config": {
            "top_k_values": TOP_K_VALUES,
            "export_top_n": EXPORT_TOP_N,
        },
        "artifacts": {
            "metrics_csv": str(METRICS_OUTPUT_PATH),
            "dashboard_metrics_csv": str(DASHBOARD_METRICS_OUTPUT_PATH),
            "academic_metrics_csv": str(ACADEMIC_METRICS_OUTPUT_PATH),
            "academic_distribution_csv": str(ACADEMIC_DISTRIBUTION_OUTPUT_PATH),
            "dashboard_summary_csv": str(DASHBOARD_SUMMARY_OUTPUT_PATH),
            "figure_png": str(FIGURE_OUTPUT_PATH),
            "figure_svg": str(FIGURE_OUTPUT_SVG_PATH),
            "model_joblib": str(MODEL_OUTPUT_PATH),
            "model_metadata_json": str(MODEL_METADATA_OUTPUT_PATH),
        },
        "most_popular_recipe": {
            "recipe_id": int(item_popularity.iloc[0]["recipe_id"]),
            "interaction_count": int(item_popularity.iloc[0]["interaction_count"]),
            "popularity_rank": int(item_popularity.iloc[0]["popularity_rank"]),
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
        "metrics": popularity_metrics.to_dict(orient="records"),
    }

    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(run_log, file, indent=2)


# Main pipeline

def main() -> None:
    """
    Run the full popularity baseline pipeline.
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

    split_summary = build_split_summary(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
    )

    print("Implicit split summary:")
    print(split_summary)

    item_popularity = build_item_popularity_table(implicit_train)
    user_seen_history = build_user_seen_history(implicit_train)

    ranked_recipe_ids = item_popularity["recipe_id"].to_numpy()
    item_self_information = build_item_self_information(item_popularity)
    catalog_size = int(item_popularity["recipe_id"].nunique())

    valid_truth = build_holdout_truth(implicit_valid)
    test_truth = build_holdout_truth(implicit_test)

    valid_user_metrics, valid_metrics = evaluate_holdout_split(
        split_name="valid",
        holdout_truth=valid_truth,
        ranked_recipe_ids=ranked_recipe_ids,
        user_seen_history=user_seen_history,
        item_self_information=item_self_information,
        catalog_size=catalog_size,
    )

    test_user_metrics, test_metrics = evaluate_holdout_split(
        split_name="test",
        holdout_truth=test_truth,
        ranked_recipe_ids=ranked_recipe_ids,
        user_seen_history=user_seen_history,
        item_self_information=item_self_information,
        catalog_size=catalog_size,
    )

    popularity_metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)

    valid_recommendations_wide = valid_user_metrics[
        ["split", "user_id", "holdout_item_count", "recommended_recipe_ids"]
    ].copy()

    test_recommendations_wide = test_user_metrics[
        ["split", "user_id", "holdout_item_count", "recommended_recipe_ids"]
    ].copy()

    valid_recommendations_long = expand_recommendations_long(
        recommendations_wide=valid_recommendations_wide,
        item_popularity=item_popularity,
    )
    test_recommendations_long = expand_recommendations_long(
        recommendations_wide=test_recommendations_wide,
        item_popularity=item_popularity,
    )

    popularity_distribution_summary = build_popularity_distribution_summary(
        item_popularity=item_popularity
    )
    popularity_curve = build_popularity_curve(item_popularity=item_popularity)

    dashboard_metrics = build_dashboard_metrics_table(popularity_metrics)
    academic_metrics = build_academic_metrics_table(popularity_metrics)
    academic_distribution = build_academic_distribution_table(
        popularity_distribution_summary
    )
    dashboard_summary = build_dashboard_summary_table(popularity_metrics)

    split_summary.to_csv(SPLIT_SUMMARY_OUTPUT_PATH, index=False)
    item_popularity.to_csv(ITEM_POPULARITY_OUTPUT_PATH, index=False)
    popularity_distribution_summary.to_csv(DISTRIBUTION_OUTPUT_PATH, index=False)
    valid_recommendations_wide.to_csv(VALID_RECS_WIDE_OUTPUT_PATH, index=False)
    test_recommendations_wide.to_csv(TEST_RECS_WIDE_OUTPUT_PATH, index=False)
    valid_recommendations_long.to_csv(VALID_RECS_LONG_OUTPUT_PATH, index=False)
    test_recommendations_long.to_csv(TEST_RECS_LONG_OUTPUT_PATH, index=False)
    popularity_metrics.to_csv(METRICS_OUTPUT_PATH, index=False)

    dashboard_metrics.to_csv(DASHBOARD_METRICS_OUTPUT_PATH, index=False)
    academic_metrics.to_csv(ACADEMIC_METRICS_OUTPUT_PATH, index=False)
    academic_distribution.to_csv(ACADEMIC_DISTRIBUTION_OUTPUT_PATH, index=False)
    dashboard_summary.to_csv(DASHBOARD_SUMMARY_OUTPUT_PATH, index=False)

    plot_popularity_concentration_curve(
        popularity_curve=popularity_curve,
        output_path_png=FIGURE_OUTPUT_PATH,
        output_path_svg=FIGURE_OUTPUT_SVG_PATH,
    )

    save_model_artifact(
        item_popularity=item_popularity,
        user_seen_history=user_seen_history,
    )

    save_run_log(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
        item_popularity=item_popularity,
        popularity_metrics=popularity_metrics,
    )

    print("\nTop 10 globally popular recipes:")
    print(item_popularity.head(10))

    print("\nPopularity evaluation metrics:")
    print(popularity_metrics)

    print("\nSaved outputs:")
    print("-", SPLIT_SUMMARY_OUTPUT_PATH)
    print("-", ITEM_POPULARITY_OUTPUT_PATH)
    print("-", DISTRIBUTION_OUTPUT_PATH)
    print("-", VALID_RECS_WIDE_OUTPUT_PATH)
    print("-", TEST_RECS_WIDE_OUTPUT_PATH)
    print("-", VALID_RECS_LONG_OUTPUT_PATH)
    print("-", TEST_RECS_LONG_OUTPUT_PATH)
    print("-", METRICS_OUTPUT_PATH)
    print("-", DASHBOARD_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_METRICS_OUTPUT_PATH)
    print("-", ACADEMIC_DISTRIBUTION_OUTPUT_PATH)
    print("-", DASHBOARD_SUMMARY_OUTPUT_PATH)
    print("-", FIGURE_OUTPUT_PATH)
    print("-", FIGURE_OUTPUT_SVG_PATH)
    print("-", MODEL_OUTPUT_PATH)
    print("-", MODEL_METADATA_OUTPUT_PATH)
    print("-", LOG_OUTPUT_PATH)


if __name__ == "__main__":
    main()