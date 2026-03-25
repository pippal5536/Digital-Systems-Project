"""
src/models/train_bpr.py

Purpose:
Train and evaluate a custom Bayesian Personalized Ranking (BPR)
recommender on chronological implicit interaction splits.

This module implements a pure NumPy/SciPy BPR model trained on the train
split only. It generates Top-N unseen recommendations for mapped
validation and test users, evaluates ranking quality, saves accessible
figures, dashboard-ready tables, academic-report-ready tables,
structured logs, and a reusable model artifact for dashboard inference.

Responsibilities:
- load implicit train, validation, and test split files
- validate required columns and quantify train-fitted mapping coverage
- build train-only user positive-item histories
- fit a custom BPR model using pairwise SGD updates
- generate Top-N unseen recommendations for mapped validation and test users
- exclude items already seen in training
- evaluate ranking quality on validation and test
- analyse novelty, catalogue coverage, and recommendation concentration
- save dashboard-friendly and academic-report-friendly tables
- save accessible figures designed for readability and long-term reuse
- save a reusable model artifact for dashboard inference
- save a compact JSON run log for monitoring and later comparison

Design notes:
- only the train split is used for BPR fitting
- validation and test evaluation require train-known user and item mappings
- already-seen training items are excluded at recommendation time
- holdout relevance is treated as binary user-item interaction presence
- outputs follow the existing project naming pattern with 11_* filenames
- model artifacts are saved to output/saved_models for dashboard loading
- figures are styled for readability, accessibility, and long-term reuse
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

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

MODEL_NAME = "bpr"
MODEL_LABEL = "Bayesian Personalized Ranking"

TOP_K_VALUES = [5, 10, 20]
EXPORT_TOP_N = 10

FACTORS = 64
LEARNING_RATE = 0.05
REGULARIZATION = 0.002
ITERATIONS = 30
TRAINING_SAMPLES_PER_EPOCH = 200_000
RANDOM_STATE = 42
INIT_SCALE = 0.01
EVALUATION_BATCH_USERS = 250

IMPLICIT_TRAIN_PATH = SPLITS_DIR / "implicit_train.parquet"
IMPLICIT_VALID_PATH = SPLITS_DIR / "implicit_valid.parquet"
IMPLICIT_TEST_PATH = SPLITS_DIR / "implicit_test.parquet"

TABLES_SUBDIR = TABLES_DIR / MODEL_NAME
FIGURES_SUBDIR = FIGURES_DIR / MODEL_NAME
LOGS_SUBDIR = LOGS_DIR / MODEL_NAME

SPLIT_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_split_summary.csv"
MAPPING_COVERAGE_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_mapping_coverage_summary.csv"
MATRIX_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_matrix_summary.csv"
ITEM_POPULARITY_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_item_popularity_table.csv"

VALID_RECS_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_valid_recommendations_long.csv"
TEST_RECS_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_test_recommendations_long.csv"
VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_valid_recommendation_popularity.csv"
TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_test_recommendation_popularity.csv"

VALID_CONCENTRATION_CURVE_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_valid_recommendation_concentration_curve.csv"
TEST_CONCENTRATION_CURVE_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_test_recommendation_concentration_curve.csv"

METRICS_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_metrics.csv"
USER_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_user_metrics.csv"

DASHBOARD_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_metrics_dashboard.csv"
ACADEMIC_METRICS_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_metrics_academic.csv"
ACADEMIC_COVERAGE_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_mapping_coverage_academic.csv"
DASHBOARD_SUMMARY_OUTPUT_PATH = TABLES_SUBDIR / "11_bpr_dashboard_summary.csv"

PRECISION_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_precision_by_k.png"
PRECISION_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_precision_by_k.svg"

RECALL_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_recall_by_k.png"
RECALL_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_recall_by_k.svg"

NDCG_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_ndcg_by_k.png"
NDCG_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_ndcg_by_k.svg"

VALID_CONCENTRATION_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_valid_recommendation_concentration_curve.png"
VALID_CONCENTRATION_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_valid_recommendation_concentration_curve.svg"

TEST_CONCENTRATION_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_test_recommendation_concentration_curve.png"
TEST_CONCENTRATION_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_test_recommendation_concentration_curve.svg"

USER_ACTIVITY_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_train_user_interaction_distribution.png"
USER_ACTIVITY_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_train_user_interaction_distribution.svg"

ITEM_ACTIVITY_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_train_item_interaction_distribution.png"
ITEM_ACTIVITY_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_train_item_interaction_distribution.svg"

TRAINING_LOSS_FIGURE_PNG = FIGURES_SUBDIR / "11_bpr_training_loss_proxy.png"
TRAINING_LOSS_FIGURE_SVG = FIGURES_SUBDIR / "11_bpr_training_loss_proxy.svg"

LOG_OUTPUT_PATH = LOGS_SUBDIR / "11_bpr_run_log.json"

MODEL_DIR = PROJECT_ROOT / "outputs" / "saved_models"
MODEL_OUTPUT_PATH = MODEL_DIR / "11_bpr_model.joblib"
MODEL_METADATA_OUTPUT_PATH = MODEL_DIR / "11_bpr_model_metadata.json"


# Custom BPR model

@dataclass
class BPRTrainingHistory:
    """
    Compact training history for later plotting and logging.
    """

    epoch: int
    mean_logistic_loss: float
    sampled_triplets: int


class CustomBPR:
    """
    Lightweight custom Bayesian Personalized Ranking model.

    This implementation trains latent user and item factors using
    pairwise SGD on sampled (user, positive item, negative item)
    triplets from the train split.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int = FACTORS,
        learning_rate: float = LEARNING_RATE,
        regularization: float = REGULARIZATION,
        iterations: int = ITERATIONS,
        samples_per_epoch: int = TRAINING_SAMPLES_PER_EPOCH,
        init_scale: float = INIT_SCALE,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.samples_per_epoch = int(samples_per_epoch)
        self.init_scale = float(init_scale)
        self.random_state = int(random_state)

        self.rng = np.random.default_rng(self.random_state)

        self.user_factors = self.rng.normal(
            loc=0.0,
            scale=self.init_scale,
            size=(self.n_users, self.factors),
        ).astype(np.float32)

        self.item_factors = self.rng.normal(
            loc=0.0,
            scale=self.init_scale,
            size=(self.n_items, self.factors),
        ).astype(np.float32)

        self.training_history: list[BPRTrainingHistory] = []

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Numerically stable sigmoid.
        """
        x = float(np.clip(x, -35.0, 35.0))
        return 1.0 / (1.0 + math.exp(-x))

    def _sample_negative_item(
        self,
        user_idx: int,
        user_positive_items: dict[int, np.ndarray],
        user_positive_sets: dict[int, set[int]],
    ) -> int:
        """
        Sample a negative item not observed in the user's train history.
        """
        positive_set = user_positive_sets[user_idx]

        while True:
            neg_item = int(self.rng.integers(0, self.n_items))
            if neg_item not in positive_set:
                return neg_item

    def fit(
        self,
        user_positive_items: dict[int, np.ndarray],
        user_positive_sets: dict[int, set[int]],
        train_nnz: int,
    ) -> None:
        """
        Fit the custom BPR model with sampled triplets.
        """
        users = np.array(sorted(user_positive_items.keys()), dtype=np.int32)

        if len(users) == 0:
            raise ValueError("Cannot train BPR: no training users were provided.")

        base_samples = max(10_000, int(train_nnz))
        samples_per_epoch = min(self.samples_per_epoch, max(base_samples, 50_000))

        for epoch in range(1, self.iterations + 1):
            cumulative_loss = 0.0

            for _ in range(samples_per_epoch):
                user_idx = int(users[self.rng.integers(0, len(users))])
                pos_items = user_positive_items[user_idx]
                pos_item = int(pos_items[self.rng.integers(0, len(pos_items))])
                neg_item = self._sample_negative_item(
                    user_idx=user_idx,
                    user_positive_items=user_positive_items,
                    user_positive_sets=user_positive_sets,
                )

                user_vec = self.user_factors[user_idx].copy()
                pos_vec = self.item_factors[pos_item].copy()
                neg_vec = self.item_factors[neg_item].copy()

                x_uij = float(np.dot(user_vec, pos_vec - neg_vec))
                sigmoid_neg_x = self._sigmoid(-x_uij)

                self.user_factors[user_idx] += self.learning_rate * (
                    sigmoid_neg_x * (pos_vec - neg_vec) - self.regularization * user_vec
                )

                self.item_factors[pos_item] += self.learning_rate * (
                    sigmoid_neg_x * user_vec - self.regularization * pos_vec
                )

                self.item_factors[neg_item] += self.learning_rate * (
                    -sigmoid_neg_x * user_vec - self.regularization * neg_vec
                )

                cumulative_loss += math.log1p(math.exp(-x_uij))

            mean_loss = cumulative_loss / samples_per_epoch

            self.training_history.append(
                BPRTrainingHistory(
                    epoch=epoch,
                    mean_logistic_loss=float(mean_loss),
                    sampled_triplets=int(samples_per_epoch),
                )
            )

            print(
                f"Epoch {epoch:>2}/{self.iterations} | "
                f"mean logistic loss: {mean_loss:.6f} | "
                f"sampled triplets: {samples_per_epoch:,}"
            )

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """
        Compute scores for all items for one user.
        """
        return self.item_factors @ self.user_factors[user_idx]

    def recommend(
        self,
        user_idx: int,
        seen_items: set[int],
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recommend top-N unseen items for one user.
        """
        scores = self.score_all_items(user_idx).astype(np.float32)

        if seen_items:
            seen_indices = np.fromiter(seen_items, dtype=np.int32)
            scores[seen_indices] = -np.inf

        n = min(int(n), self.n_items - len(seen_items))
        if n <= 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        candidate_idx = np.argpartition(-scores, kth=n - 1)[:n]
        ranked_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

        return ranked_idx.astype(np.int32), scores[ranked_idx].astype(np.float32)


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
    Validate that all required split files exist.
    """
    required_paths = [
        IMPLICIT_TRAIN_PATH,
        IMPLICIT_VALID_PATH,
        IMPLICIT_TEST_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Missing required BPR input files:\n" + "\n".join(missing_paths)
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


def summarise_mapping_coverage(
    df: pd.DataFrame,
    split_name: str,
    known_users: set[int],
    known_items: set[int],
) -> pd.DataFrame:
    """
    Summarise mapping coverage for one split against train-known indices.
    """
    rows = int(len(df))
    user_known_mask = df["user_idx"].isin(known_users)
    item_known_mask = df["item_idx"].isin(known_items)
    both_known_mask = user_known_mask & item_known_mask

    return pd.DataFrame(
        [
            {
                "split": split_name,
                "rows": rows,
                "known_user_rows": int(user_known_mask.sum()),
                "known_item_rows": int(item_known_mask.sum()),
                "rows_with_both_indices": int(both_known_mask.sum()),
                "known_user_pct": round(float(user_known_mask.mean() * 100), 2) if rows else 0.0,
                "known_item_pct": round(float(item_known_mask.mean() * 100), 2) if rows else 0.0,
                "rows_evaluable_pct": round(float(both_known_mask.mean() * 100), 2) if rows else 0.0,
            }
        ]
    )


def filter_evaluable_rows(
    df: pd.DataFrame,
    known_users: set[int] | None = None,
    known_items: set[int] | None = None,
) -> pd.DataFrame:
    """
    Keep only rows with valid integer user and item indices.
    """
    work = df.copy()

    work["user_idx"] = pd.to_numeric(work["user_idx"], errors="coerce")
    work["item_idx"] = pd.to_numeric(work["item_idx"], errors="coerce")

    work = work.dropna(subset=["user_idx", "item_idx"]).copy()
    work["user_idx"] = work["user_idx"].astype(int)
    work["item_idx"] = work["item_idx"].astype(int)

    if known_users is not None:
        work = work[work["user_idx"].isin(known_users)].copy()

    if known_items is not None:
        work = work[work["item_idx"].isin(known_items)].copy()

    return work


def build_train_matrix(train_df: pd.DataFrame) -> sparse.csr_matrix:
    """
    Build a sparse user-item interaction matrix from training rows only.
    """
    grouped = (
        train_df.groupby(["user_idx", "item_idx"], as_index=False)
        .size()
        .rename(columns={"size": "interaction_count"})
    )

    n_users = int(train_df["user_idx"].max()) + 1
    n_items = int(train_df["item_idx"].max()) + 1

    data = np.ones(len(grouped), dtype=np.float32)
    rows = grouped["user_idx"].to_numpy(dtype=np.int32)
    cols = grouped["item_idx"].to_numpy(dtype=np.int32)

    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )


def build_matrix_summary(
    train_matrix: sparse.csr_matrix,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact matrix summary table.
    """
    n_users, n_items = train_matrix.shape
    density = train_matrix.nnz / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0

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
    item_popularity["popularity_score"] = (
        item_popularity["train_interaction_count"]
        / item_popularity["train_interaction_count"].max()
    )

    total_interactions = item_popularity["train_interaction_count"].sum()
    item_popularity["interaction_share"] = (
        item_popularity["train_interaction_count"] / total_interactions
    )
    item_popularity["cumulative_interaction_share"] = (
        item_popularity["interaction_share"].cumsum()
    )
    item_popularity["cumulative_item_share"] = (
        np.arange(1, len(item_popularity) + 1) / len(item_popularity)
    )

    return item_popularity


def build_ground_truth(holdout_df: pd.DataFrame) -> dict[int, set[int]]:
    """
    Build a binary relevance mapping for a holdout split.
    """
    return (
        holdout_df.groupby("user_idx")["item_idx"]
        .agg(lambda values: set(map(int, values.tolist())))
        .to_dict()
    )


def build_user_positive_histories(
    train_df: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], dict[int, set[int]]]:
    """
    Build train-only user positive-item histories for BPR training.
    """
    grouped = (
        train_df.groupby("user_idx")["item_idx"]
        .agg(lambda values: sorted(set(map(int, values.tolist()))))
        .to_dict()
    )

    user_positive_items = {
        int(user_idx): np.array(item_list, dtype=np.int32)
        for user_idx, item_list in grouped.items()
    }

    user_positive_sets = {
        int(user_idx): set(item_list.tolist())
        for user_idx, item_list in user_positive_items.items()
    }

    return user_positive_items, user_positive_sets


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

    return float(np.mean([item_self_information.get(item, 0.0) for item in recommended_k]))


def recommend_for_users(
    model: CustomBPR,
    users: list[int],
    user_positive_sets: dict[int, set[int]],
    top_n: int,
) -> pd.DataFrame:
    """
    Generate recommendations for a list of mapped users.
    """
    rows: list[dict[str, float | int]] = []

    for batch_start in range(0, len(users), EVALUATION_BATCH_USERS):
        batch_users = users[batch_start : batch_start + EVALUATION_BATCH_USERS]
        print(
            f"Generating recommendations for users "
            f"{batch_start + 1:,}-{batch_start + len(batch_users):,} "
            f"of {len(users):,}"
        )

        for user_idx in batch_users:
            item_ids, scores = model.recommend(
                user_idx=user_idx,
                seen_items=user_positive_sets.get(user_idx, set()),
                n=top_n,
            )

            for rank, (item_idx, score) in enumerate(zip(item_ids, scores), start=1):
                rows.append(
                    {
                        "user_idx": int(user_idx),
                        "item_idx": int(item_idx),
                        "recommendation_rank": int(rank),
                        "score": float(score),
                    }
                )

    return pd.DataFrame(rows)


def evaluate_holdout_split(
    split_name: str,
    recommendations_long: pd.DataFrame,
    truth_lookup: dict[int, set[int]],
    item_self_information: dict[int, float],
    recipe_lookup: dict[int, int],
    item_popularity_lookup: dict[int, dict[str, float | int]],
    catalog_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the BPR recommender on one holdout split.
    """
    user_rows: list[dict[str, object]] = []
    recommended_items_by_k = {k: [] for k in TOP_K_VALUES}

    grouped_recs = (
        recommendations_long.sort_values(["user_idx", "recommendation_rank"])
        .groupby("user_idx")
    )

    for user_idx, user_recs in grouped_recs:
        relevant_items = truth_lookup.get(int(user_idx), set())
        recommended_items = user_recs["item_idx"].astype(int).tolist()

        row: dict[str, object] = {
            "split": split_name,
            "user_idx": int(user_idx),
            "holdout_item_count": int(len(relevant_items)),
        }

        for k in TOP_K_VALUES:
            row[f"precision_at_{k}"] = precision_at_k(recommended_items, relevant_items, k)
            row[f"recall_at_{k}"] = recall_at_k(recommended_items, relevant_items, k)
            row[f"hit_rate_at_{k}"] = hit_rate_at_k(recommended_items, relevant_items, k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(recommended_items, relevant_items, k)
            row[f"novelty_at_{k}"] = novelty_at_k(
                recommended_items,
                item_self_information,
                k,
            )

            recommended_items_by_k[k].extend(recommended_items[:k])

        user_rows.append(row)

    user_metrics_df = pd.DataFrame(user_rows)

    recommendation_export = recommendations_long.copy()
    recommendation_export["split"] = split_name
    recommendation_export["holdout_item_count"] = recommendation_export["user_idx"].map(
        {user_idx: len(items) for user_idx, items in truth_lookup.items()}
    )
    recommendation_export["recipe_id"] = recommendation_export["item_idx"].map(recipe_lookup)
    recommendation_export["global_item_popularity_rank"] = recommendation_export["item_idx"].map(
        lambda item_idx: item_popularity_lookup.get(item_idx, {}).get("item_popularity_rank")
    )
    recommendation_export["train_interaction_count"] = recommendation_export["item_idx"].map(
        lambda item_idx: item_popularity_lookup.get(item_idx, {}).get("train_interaction_count")
    )
    recommendation_export["popularity_score"] = recommendation_export["item_idx"].map(
        lambda item_idx: item_popularity_lookup.get(item_idx, {}).get("popularity_score")
    )

    metric_rows: list[dict[str, object]] = []

    for k in TOP_K_VALUES:
        unique_recommended_count = len(set(recommended_items_by_k[k]))

        metric_rows.append(
            {
                "split": split_name,
                "k": k,
                "users_evaluated": int(user_metrics_df["user_idx"].nunique()) if not user_metrics_df.empty else 0,
                "precision_at_k": float(user_metrics_df[f"precision_at_{k}"].mean()) if not user_metrics_df.empty else 0.0,
                "recall_at_k": float(user_metrics_df[f"recall_at_{k}"].mean()) if not user_metrics_df.empty else 0.0,
                "hit_rate_at_k": float(user_metrics_df[f"hit_rate_at_{k}"].mean()) if not user_metrics_df.empty else 0.0,
                "ndcg_at_k": float(user_metrics_df[f"ndcg_at_{k}"].mean()) if not user_metrics_df.empty else 0.0,
                "novelty_at_k": float(user_metrics_df[f"novelty_at_{k}"].mean()) if not user_metrics_df.empty else 0.0,
                "catalog_coverage_at_k": float(unique_recommended_count / catalog_size) if catalog_size > 0 else 0.0,
                "recommendation_count": int(len(recommended_items_by_k[k])),
            }
        )

    split_metrics_df = pd.DataFrame(metric_rows)

    return recommendation_export, split_metrics_df, user_metrics_df


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
    return summary[
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

    curve_df["recommendation_share"] = curve_df["times_recommended"] / total_recommendations
    curve_df["cumulative_recommendation_share"] = curve_df["recommendation_share"].cumsum()
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

    return academic_df[
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


def build_academic_coverage_table(coverage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an academic-report-friendly mapping coverage table.
    """
    academic_df = coverage_df.copy()
    academic_df["Split"] = academic_df["split"].str.capitalize()
    academic_df["Rows"] = academic_df["rows"].astype(int)
    academic_df["Known User Rows"] = academic_df["known_user_rows"].astype(int)
    academic_df["Known Item Rows"] = academic_df["known_item_rows"].astype(int)
    academic_df["Rows With Both Indices"] = academic_df["rows_with_both_indices"].astype(int)
    academic_df["Known User (%)"] = academic_df["known_user_pct"].round(2)
    academic_df["Known Item (%)"] = academic_df["known_item_pct"].round(2)
    academic_df["Evaluable Rows (%)"] = academic_df["rows_evaluable_pct"].round(2)

    return academic_df[
        [
            "Split",
            "Rows",
            "Known User Rows",
            "Known Item Rows",
            "Rows With Both Indices",
            "Known User (%)",
            "Known Item (%)",
            "Evaluable Rows (%)",
        ]
    ].copy()


def build_dashboard_summary_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact dashboard summary table.
    """
    summary_df = metrics_df.copy()
    summary_df["model"] = MODEL_NAME
    summary_df["label"] = summary_df["split"].str.upper() + "_AT_" + summary_df["k"].astype(str)

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
    log_x: bool = False,
) -> None:
    """
    Save an accessible histogram figure.
    """
    positive_values = values[values > 0] if log_x else values

    plt.figure(figsize=(9.5, 6.5))
    plt.hist(positive_values, bins=50, edgecolor="black", linewidth=0.6)
    if log_x:
        plt.xscale("log")

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


def plot_training_history(
    training_history_df: pd.DataFrame,
    output_path_png: Path,
    output_path_svg: Path,
) -> None:
    """
    Save an accessible training loss proxy figure across epochs.
    """
    if training_history_df.empty:
        return

    plt.figure(figsize=(9.5, 6.5))
    plt.plot(
        training_history_df["epoch"],
        training_history_df["mean_logistic_loss"],
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        linestyle="-",
        label="Mean sampled logistic loss",
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Mean sampled logistic loss", fontsize=12)
    plt.title("BPR training loss proxy across epochs", fontsize=13)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_svg, bbox_inches="tight")
    plt.close()


def save_model_artifact(
    model: CustomBPR,
    train_matrix: sparse.csr_matrix,
    user_positive_sets: dict[int, set[int]],
    recipe_lookup: dict[int, int],
    item_popularity: pd.DataFrame,
    training_history_df: pd.DataFrame,
) -> None:
    """
    Save the BPR artifact for dashboard reuse.
    """
    model_artifact = {
        "model_name": MODEL_NAME,
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "factors": FACTORS,
        "learning_rate": LEARNING_RATE,
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "samples_per_epoch": TRAINING_SAMPLES_PER_EPOCH,
        "random_state": RANDOM_STATE,
        "user_factors": model.user_factors,
        "item_factors": model.item_factors,
        "train_user_item_matrix": train_matrix,
        "user_seen_history": user_positive_sets,
        "recipe_lookup": recipe_lookup,
        "item_popularity_table": item_popularity,
        "training_history": training_history_df,
    }

    joblib.dump(model_artifact, MODEL_OUTPUT_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_output_path": str(MODEL_OUTPUT_PATH),
        "top_k_values": TOP_K_VALUES,
        "export_top_n": EXPORT_TOP_N,
        "factors": FACTORS,
        "learning_rate": LEARNING_RATE,
        "regularization": REGULARIZATION,
        "iterations": ITERATIONS,
        "samples_per_epoch": TRAINING_SAMPLES_PER_EPOCH,
        "random_state": RANDOM_STATE,
        "train_matrix_shape": [int(train_matrix.shape[0]), int(train_matrix.shape[1])],
        "fields_available": [
            "user_factors",
            "item_factors",
            "train_user_item_matrix",
            "user_seen_history",
            "recipe_lookup",
            "item_popularity_table",
            "training_history",
        ],
    }

    with open(MODEL_METADATA_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_run_log(
    implicit_train: pd.DataFrame,
    implicit_valid: pd.DataFrame,
    implicit_test: pd.DataFrame,
    train_bpr: pd.DataFrame,
    valid_bpr: pd.DataFrame,
    test_bpr: pd.DataFrame,
    train_matrix: sparse.csr_matrix,
    coverage_summary: pd.DataFrame,
    bpr_metrics: pd.DataFrame,
    training_history_df: pd.DataFrame,
) -> None:
    """
    Save a structured JSON log for monitoring and dashboard use.
    """
    best_row = bpr_metrics.sort_values(
        by=["split", "ndcg_at_k", "recall_at_k"],
        ascending=[True, False, False],
    ).iloc[0]

    n_users, n_items = train_matrix.shape
    density = train_matrix.nnz / (n_users * n_items) if n_users > 0 and n_items > 0 else 0.0

    run_log = {
        "model": MODEL_NAME,
        "model_label": MODEL_LABEL,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": {
            "train_rows_raw": int(len(implicit_train)),
            "valid_rows_raw": int(len(implicit_valid)),
            "test_rows_raw": int(len(implicit_test)),
            "train_rows_evaluable": int(len(train_bpr)),
            "valid_rows_evaluable": int(len(valid_bpr)),
            "test_rows_evaluable": int(len(test_bpr)),
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
            "top_k_values": TOP_K_VALUES,
            "export_top_n": EXPORT_TOP_N,
            "factors": FACTORS,
            "learning_rate": LEARNING_RATE,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "samples_per_epoch": TRAINING_SAMPLES_PER_EPOCH,
            "random_state": RANDOM_STATE,
        },
        "training_history": training_history_df.to_dict(orient="records"),
        "artifacts": {
            "mapping_coverage_csv": str(MAPPING_COVERAGE_OUTPUT_PATH),
            "split_summary_csv": str(SPLIT_SUMMARY_OUTPUT_PATH),
            "matrix_summary_csv": str(MATRIX_SUMMARY_OUTPUT_PATH),
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
        "metrics": bpr_metrics.to_dict(orient="records"),
    }

    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(run_log, file, indent=2)


# Main pipeline

def main() -> None:
    """
    Run the full BPR training and evaluation pipeline.
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

    train_bpr = filter_evaluable_rows(implicit_train)
    known_users = set(train_bpr["user_idx"].unique())
    known_items = set(train_bpr["item_idx"].unique())

    coverage_summary = pd.concat(
        [
            summarise_mapping_coverage(implicit_train, "train", known_users, known_items),
            summarise_mapping_coverage(implicit_valid, "valid", known_users, known_items),
            summarise_mapping_coverage(implicit_test, "test", known_users, known_items),
        ],
        ignore_index=True,
    )

    valid_bpr = filter_evaluable_rows(implicit_valid, known_users, known_items)
    test_bpr = filter_evaluable_rows(implicit_test, known_users, known_items)

    split_summary = build_split_summary(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
    )

    print("\nSplit summary:")
    print(split_summary)

    print("\nMapping coverage summary:")
    print(coverage_summary)

    print("\nEvaluable shapes:")
    print("train:", train_bpr.shape)
    print("valid:", valid_bpr.shape)
    print("test:", test_bpr.shape)

    train_matrix = build_train_matrix(train_bpr)
    matrix_summary = build_matrix_summary(train_matrix, train_bpr)

    print("\nTrain matrix summary:")
    print(matrix_summary)

    user_positive_items, user_positive_sets = build_user_positive_histories(train_bpr)

    user_interaction_counts = np.asarray(train_matrix.getnnz(axis=1)).ravel()
    item_interaction_counts = np.asarray(train_matrix.getnnz(axis=0)).ravel()

    plot_accessible_histogram(
        values=user_interaction_counts,
        xlabel="Interactions per user (log scale)",
        ylabel="Number of users",
        title="Training user interaction count distribution",
        output_path_png=USER_ACTIVITY_FIGURE_PNG,
        output_path_svg=USER_ACTIVITY_FIGURE_SVG,
        log_x=True,
    )

    plot_accessible_histogram(
        values=item_interaction_counts,
        xlabel="Interactions per item (log scale)",
        ylabel="Number of items",
        title="Training item interaction count distribution",
        output_path_png=ITEM_ACTIVITY_FIGURE_PNG,
        output_path_svg=ITEM_ACTIVITY_FIGURE_SVG,
        log_x=True,
    )

    print("\nTraining custom BPR model...")
    model = CustomBPR(
        n_users=train_matrix.shape[0],
        n_items=train_matrix.shape[1],
        factors=FACTORS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        samples_per_epoch=TRAINING_SAMPLES_PER_EPOCH,
        random_state=RANDOM_STATE,
    )
    model.fit(
        user_positive_items=user_positive_items,
        user_positive_sets=user_positive_sets,
        train_nnz=int(train_matrix.nnz),
    )
    print("Custom BPR training complete.")

    training_history_df = pd.DataFrame(
        [
            {
                "epoch": history.epoch,
                "mean_logistic_loss": history.mean_logistic_loss,
                "sampled_triplets": history.sampled_triplets,
            }
            for history in model.training_history
        ]
    )

    plot_training_history(
        training_history_df=training_history_df,
        output_path_png=TRAINING_LOSS_FIGURE_PNG,
        output_path_svg=TRAINING_LOSS_FIGURE_SVG,
    )

    valid_truth = build_ground_truth(valid_bpr)
    test_truth = build_ground_truth(test_bpr)

    valid_users = sorted(valid_truth.keys())
    test_users = sorted(test_truth.keys())

    print("\nEvaluation user counts:")
    print(f"valid users: {len(valid_users):,}")
    print(f"test users : {len(test_users):,}")

    valid_recommendations_raw = recommend_for_users(
        model=model,
        users=valid_users,
        user_positive_sets=user_positive_sets,
        top_n=EXPORT_TOP_N,
    )

    test_recommendations_raw = recommend_for_users(
        model=model,
        users=test_users,
        user_positive_sets=user_positive_sets,
        top_n=EXPORT_TOP_N,
    )

    item_popularity = build_item_popularity_table(train_bpr)
    item_self_information = build_item_self_information(item_popularity)

    recipe_lookup = (
        train_bpr[["item_idx", "recipe_id"]]
        .drop_duplicates()
        .set_index("item_idx")["recipe_id"]
        .to_dict()
    )
    user_lookup = train_bpr[["user_idx", "user_id"]].drop_duplicates()

    item_popularity_lookup = item_popularity.set_index("item_idx")[
        ["item_popularity_rank", "train_interaction_count", "popularity_score"]
    ].to_dict("index")

    catalog_size = int(item_popularity["item_idx"].nunique())

    valid_recommendations_long, valid_metrics, valid_user_metrics = evaluate_holdout_split(
        split_name="valid",
        recommendations_long=valid_recommendations_raw,
        truth_lookup=valid_truth,
        item_self_information=item_self_information,
        recipe_lookup=recipe_lookup,
        item_popularity_lookup=item_popularity_lookup,
        catalog_size=catalog_size,
    )

    test_recommendations_long, test_metrics, test_user_metrics = evaluate_holdout_split(
        split_name="test",
        recommendations_long=test_recommendations_raw,
        truth_lookup=test_truth,
        item_self_information=item_self_information,
        recipe_lookup=recipe_lookup,
        item_popularity_lookup=item_popularity_lookup,
        catalog_size=catalog_size,
    )

    bpr_metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)
    bpr_user_metrics = pd.concat([valid_user_metrics, test_user_metrics], ignore_index=True)

    valid_recommendations_long = valid_recommendations_long.merge(user_lookup, on="user_idx", how="left")
    test_recommendations_long = test_recommendations_long.merge(user_lookup, on="user_idx", how="left")
    bpr_user_metrics = bpr_user_metrics.merge(user_lookup, on="user_idx", how="left")

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

    dashboard_metrics = build_dashboard_metrics_table(bpr_metrics)
    academic_metrics = build_academic_metrics_table(bpr_metrics)
    academic_coverage = build_academic_coverage_table(coverage_summary)
    dashboard_summary = build_dashboard_summary_table(bpr_metrics)

    plot_accessible_metric_lines(
        metrics_df=bpr_metrics,
        metric_col="precision_at_k",
        ylabel="Precision@K",
        title="BPR precision across K",
        output_path_png=PRECISION_FIGURE_PNG,
        output_path_svg=PRECISION_FIGURE_SVG,
    )

    plot_accessible_metric_lines(
        metrics_df=bpr_metrics,
        metric_col="recall_at_k",
        ylabel="Recall@K",
        title="BPR recall across K",
        output_path_png=RECALL_FIGURE_PNG,
        output_path_svg=RECALL_FIGURE_SVG,
    )

    plot_accessible_metric_lines(
        metrics_df=bpr_metrics,
        metric_col="ndcg_at_k",
        ylabel="nDCG@K",
        title="BPR nDCG across K",
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
    item_popularity.to_csv(ITEM_POPULARITY_OUTPUT_PATH, index=False)

    valid_recommendations_long.to_csv(VALID_RECS_OUTPUT_PATH, index=False)
    test_recommendations_long.to_csv(TEST_RECS_OUTPUT_PATH, index=False)
    valid_recommendation_popularity.to_csv(VALID_RECOMMENDATION_POPULARITY_OUTPUT_PATH, index=False)
    test_recommendation_popularity.to_csv(TEST_RECOMMENDATION_POPULARITY_OUTPUT_PATH, index=False)

    valid_concentration_curve.to_csv(VALID_CONCENTRATION_CURVE_OUTPUT_PATH, index=False)
    test_concentration_curve.to_csv(TEST_CONCENTRATION_CURVE_OUTPUT_PATH, index=False)

    bpr_metrics.to_csv(METRICS_OUTPUT_PATH, index=False)
    bpr_user_metrics.to_csv(USER_METRICS_OUTPUT_PATH, index=False)

    dashboard_metrics.to_csv(DASHBOARD_METRICS_OUTPUT_PATH, index=False)
    academic_metrics.to_csv(ACADEMIC_METRICS_OUTPUT_PATH, index=False)
    academic_coverage.to_csv(ACADEMIC_COVERAGE_OUTPUT_PATH, index=False)
    dashboard_summary.to_csv(DASHBOARD_SUMMARY_OUTPUT_PATH, index=False)

    save_model_artifact(
        model=model,
        train_matrix=train_matrix,
        user_positive_sets=user_positive_sets,
        recipe_lookup=recipe_lookup,
        item_popularity=item_popularity,
        training_history_df=training_history_df,
    )

    save_run_log(
        implicit_train=implicit_train,
        implicit_valid=implicit_valid,
        implicit_test=implicit_test,
        train_bpr=train_bpr,
        valid_bpr=valid_bpr,
        test_bpr=test_bpr,
        train_matrix=train_matrix,
        coverage_summary=coverage_summary,
        bpr_metrics=bpr_metrics,
        training_history_df=training_history_df,
    )

    print("\nBPR metrics summary:")
    print(bpr_metrics)

    print("\nSaved outputs:")
    print("-", SPLIT_SUMMARY_OUTPUT_PATH)
    print("-", MAPPING_COVERAGE_OUTPUT_PATH)
    print("-", MATRIX_SUMMARY_OUTPUT_PATH)
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
    print("-", ACADEMIC_COVERAGE_OUTPUT_PATH)
    print("-", DASHBOARD_SUMMARY_OUTPUT_PATH)
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
    print("-", TRAINING_LOSS_FIGURE_PNG)
    print("-", TRAINING_LOSS_FIGURE_SVG)
    print("-", MODEL_OUTPUT_PATH)
    print("-", MODEL_METADATA_OUTPUT_PATH)
    print("-", LOG_OUTPUT_PATH)


if __name__ == "__main__":
    main()