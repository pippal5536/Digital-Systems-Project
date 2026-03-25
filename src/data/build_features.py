"""
src/data/build_features.py

Purpose:
Build Phase 6 feature tables from the chronological split datasets and save
outputs in forms suitable for pipeline logging, dashboard use, and academic
reporting.

This module creates reusable user-level and item-level features from the
training split, alongside summary outputs that support exploratory analysis,
feature inspection, and optional downstream modelling extensions.

Responsibilities:
- load chronological split datasets
- build user aggregate features from training history
- build item aggregate features from training history
- preserve recipe-side metadata as static item features
- build holdout feature tables using train-derived aggregates
- quantify missing feature coverage in validation and test splits
- save feature tables, raw summary tables, dashboard tables, report tables,
  diagnostic logs, and accessible figures

Design notes:
- this module is primarily a feature engineering and diagnostic step
- outputs are reusable for analysis, dashboards, and optional modelling
- chronological boundaries must be respected to avoid leakage
- training aggregates are fitted only on the training split
- pair-history and reorder-target experiments are not part of the core output
- figures use a colour-blind-safe palette
- figures are also designed to remain interpretable in grayscale
- both PNG and SVG outputs are saved for longevity and reuse
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import (
    FIGURES_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    ensure_directories,
)


PRIMARY_BLUE = "#0072B2"
PRIMARY_ORANGE = "#E69F00"
PRIMARY_TEAL = "#009E73"
PRIMARY_PURPLE = "#CC79A7"
NEUTRAL_GREY = "#666666"
LIGHT_GREY = "#D9D9D9"

FIG_DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOT_SIZE = 10


@dataclass(frozen=True)
class FeatureBuildOutputs:
    """
    Output file paths for Phase 6 feature engineering artefacts.
    """

    user_features_train: str = "features_user_train.parquet"
    item_features_train: str = "features_item_train.parquet"
    valid_features: str = "features_valid.parquet"
    test_features: str = "features_test.parquet"

    user_feature_summary: str = "06_user_feature_summary.csv"
    item_feature_summary: str = "06_item_feature_summary.csv"
    holdout_missing_summary: str = "06_holdout_missing_summary.csv"
    feature_null_summary: str = "06_feature_null_summary.csv"
    feature_dataset_summary: str = "06_feature_dataset_summary.csv"

    dashboard_feature_dataset_summary: str = "06_dashboard_feature_dataset_summary.csv"
    dashboard_holdout_missing_summary: str = "06_dashboard_holdout_missing_summary.csv"
    dashboard_user_feature_summary: str = "06_dashboard_user_feature_summary.csv"
    dashboard_item_feature_summary: str = "06_dashboard_item_feature_summary.csv"

    report_feature_dataset_summary: str = "06_report_feature_dataset_summary.csv"
    report_holdout_missing_summary: str = "06_report_holdout_missing_summary.csv"
    report_user_feature_summary: str = "06_report_user_feature_summary.csv"
    report_item_feature_summary: str = "06_report_item_feature_summary.csv"

    phase_log_json: str = "06_feature_build_log.json"
    phase_log_md: str = "06_feature_build_summary.md"


OUTPUTS = FeatureBuildOutputs()


def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    return f"{int(value):,}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage values for display tables."""
    return f"{value:.{decimals}f}%"


def _load_parquet(path: pd.io.common.FilePath) -> pd.DataFrame:
    """
    Load a parquet file and return a dataframe.
    """
    return pd.read_parquet(path)


def _ensure_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Return a copy of a dataframe with one column converted to datetime.
    """
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def _sort_for_time_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a sorted copy of a split dataframe.
    """
    sort_columns = [col for col in ["user_id", "recipe_id", "date"] if col in df.columns]
    out = df.copy()
    out = out.sort_values(sort_columns).reset_index(drop=True)
    return out


def _check_required_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    """
    Validate that a dataframe contains required columns.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"{df_name} is missing required columns: {missing_columns}")


def save_figure(fig: plt.Figure, stem: str) -> None:
    """
    Save a figure in both PNG and SVG formats.
    """
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def apply_axis_style(ax: plt.Axes) -> None:
    """
    Apply a clean, readable chart style suitable for dashboards and reports.
    """
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(ax: plt.Axes, decimals: int = 0, suffix: str = "") -> None:
    """
    Add direct labels above bars to reduce dependence on colour and legends.
    """
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            if decimals > 0:
                label = f"{height:,.{decimals}f}{suffix}"
            else:
                label = f"{int(height):,}{suffix}"

            ax.annotate(
                label,
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=ANNOT_SIZE,
                xytext=(0, 3),
                textcoords="offset points",
            )


def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build user aggregate features from training interactions only.
    """
    user_features = (
        train_df.groupby("user_id", dropna=False)
        .agg(
            user_total_interactions=("recipe_id", "size"),
            user_distinct_recipes=("recipe_id", "nunique"),
            user_mean_rating=("explicit_rating", "mean"),
            user_rating_count=("explicit_rating", lambda s: s.notna().sum()),
            user_review_rate=("review_exists", "mean"),
            user_unrated_count=("is_unrated_observation", "sum"),
            user_first_date=("date", "min"),
            user_last_date=("date", "max"),
        )
        .reset_index()
    )

    user_features["user_active_days"] = (
        user_features["user_last_date"] - user_features["user_first_date"]
    ).dt.days + 1

    user_features["user_interactions_per_day"] = (
        user_features["user_total_interactions"]
        / user_features["user_active_days"].replace(0, np.nan)
    )

    return user_features


def build_item_behaviour_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build item behavioural features from training interactions only.
    """
    item_features = (
        train_df.groupby("recipe_id", dropna=False)
        .agg(
            item_total_interactions=("user_id", "size"),
            item_distinct_users=("user_id", "nunique"),
            item_mean_rating=("explicit_rating", "mean"),
            item_rating_count=("explicit_rating", lambda s: s.notna().sum()),
            item_review_rate=("review_exists", "mean"),
            item_unrated_count=("is_unrated_observation", "sum"),
            item_first_date=("date", "min"),
            item_last_date=("date", "max"),
        )
        .reset_index()
    )

    item_features["item_active_days"] = (
        item_features["item_last_date"] - item_features["item_first_date"]
    ).dt.days + 1

    item_features["item_interactions_per_day"] = (
        item_features["item_total_interactions"]
        / item_features["item_active_days"].replace(0, np.nan)
    )

    item_features["item_popularity_rank"] = (
        item_features["item_total_interactions"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    return item_features


def build_recipe_static_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract recipe-side static metadata already present in the joined split.
    """
    desired_static_columns = [
        "recipe_id",
        "minutes",
        "n_steps",
        "n_ingredients",
        "name_length",
        "description_length",
        "tag_count",
        "step_count_from_list",
        "ingredient_count_from_list",
        "nutrition_vector_length",
        "nutrition_sum",
        "nutrition_mean",
        "has_description",
        "calorie_level",
        "has_pp_features",
    ]

    available_static_columns = [
        column for column in desired_static_columns if column in train_df.columns
    ]

    recipe_static_features = (
        train_df[available_static_columns]
        .drop_duplicates(subset=["recipe_id"])
        .copy()
    )

    return recipe_static_features


def build_item_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final item feature table by combining behavioural and static features.
    """
    item_behaviour_features = build_item_behaviour_features(train_df)
    recipe_static_features = build_recipe_static_features(train_df)

    item_features = item_behaviour_features.merge(
        recipe_static_features,
        on="recipe_id",
        how="left",
        validate="one_to_one",
    )

    return item_features


def build_holdout_features(
    holdout_df: pd.DataFrame,
    user_features_train: pd.DataFrame,
    item_features_train: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Build leakage-safe holdout feature tables using train-derived aggregates only.
    """
    item_behaviour_columns = [
        "recipe_id",
        "item_total_interactions",
        "item_distinct_users",
        "item_mean_rating",
        "item_rating_count",
        "item_review_rate",
        "item_unrated_count",
        "item_first_date",
        "item_last_date",
        "item_active_days",
        "item_interactions_per_day",
        "item_popularity_rank",
    ]

    holdout_features = holdout_df.copy()

    holdout_features = holdout_features.merge(
        user_features_train,
        on="user_id",
        how="left",
        validate="many_to_one",
    )

    holdout_features = holdout_features.merge(
        item_features_train[item_behaviour_columns],
        on="recipe_id",
        how="left",
        validate="many_to_one",
    )

    holdout_features["split_name"] = split_name

    return holdout_features


def build_holdout_missing_summary(
    valid_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary of feature coverage in holdout splits.
    """
    summary = pd.DataFrame(
        {
            "split": ["valid", "test"],
            "rows": [len(valid_features), len(test_features)],
            "missing_user_features": [
                valid_features["user_total_interactions"].isna().sum(),
                test_features["user_total_interactions"].isna().sum(),
            ],
            "missing_item_features": [
                valid_features["item_total_interactions"].isna().sum(),
                test_features["item_total_interactions"].isna().sum(),
            ],
        }
    )
    summary["missing_user_features_pct"] = (
        summary["missing_user_features"] / summary["rows"].replace(0, np.nan) * 100
    ).round(4)
    summary["missing_item_features_pct"] = (
        summary["missing_item_features"] / summary["rows"].replace(0, np.nan) * 100
    ).round(4)
    return summary


def build_feature_null_summary(
    user_features_train: pd.DataFrame,
    item_features_train: pd.DataFrame,
    valid_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a column-level null summary for the main Phase 6 outputs.
    """
    summary_frames: list[tuple[str, pd.DataFrame]] = [
        ("user_train", user_features_train),
        ("item_train", item_features_train),
        ("valid", valid_features),
        ("test", test_features),
    ]

    all_columns: list[str] = []
    for _, frame in summary_frames:
        all_columns.extend(frame.columns.tolist())

    all_columns = sorted(set(all_columns))
    summary = pd.DataFrame({"column": all_columns})

    for frame_name, frame in summary_frames:
        summary[f"{frame_name}_nulls"] = summary["column"].map(frame.isna().sum()).fillna(0).astype(int)

    summary["total_nulls"] = summary[
        ["user_train_nulls", "item_train_nulls", "valid_nulls", "test_nulls"]
    ].sum(axis=1)

    summary = summary.sort_values(
        by=["total_nulls", "valid_nulls", "test_nulls"],
        ascending=False,
    ).reset_index(drop=True)

    return summary


def build_user_feature_summary(user_features_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary of user feature coverage and averages.
    """
    return pd.DataFrame(
        {
            "n_users_with_features": [user_features_train["user_id"].nunique()],
            "mean_user_total_interactions": [user_features_train["user_total_interactions"].mean()],
            "mean_user_distinct_recipes": [user_features_train["user_distinct_recipes"].mean()],
            "mean_user_mean_rating": [user_features_train["user_mean_rating"].mean()],
            "mean_user_review_rate": [user_features_train["user_review_rate"].mean()],
            "mean_user_interactions_per_day": [user_features_train["user_interactions_per_day"].mean()],
        }
    )


def build_item_feature_summary(item_features_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary of item feature coverage and averages.
    """
    return pd.DataFrame(
        {
            "n_items_with_features": [item_features_train["recipe_id"].nunique()],
            "mean_item_total_interactions": [item_features_train["item_total_interactions"].mean()],
            "mean_item_distinct_users": [item_features_train["item_distinct_users"].mean()],
            "mean_item_mean_rating": [item_features_train["item_mean_rating"].mean()],
            "mean_item_review_rate": [item_features_train["item_review_rate"].mean()],
            "mean_item_interactions_per_day": [item_features_train["item_interactions_per_day"].mean()],
        }
    )


def build_feature_dataset_summary(
    user_features_train: pd.DataFrame,
    item_features_train: pd.DataFrame,
    valid_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a summary table for the main Phase 6 datasets.
    """
    return pd.DataFrame(
        {
            "dataset": [
                "user_features_train",
                "item_features_train",
                "features_valid",
                "features_test",
            ],
            "rows": [
                len(user_features_train),
                len(item_features_train),
                len(valid_features),
                len(test_features),
            ],
            "columns": [
                user_features_train.shape[1],
                item_features_train.shape[1],
                valid_features.shape[1],
                test_features.shape[1],
            ],
        }
    )


def build_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build compact, human-readable tables for dashboard usage.
    """
    dataset_summary = outputs["feature_dataset_summary"].copy()
    dataset_summary["rows"] = dataset_summary["rows"].map(format_int)
    dataset_summary["columns"] = dataset_summary["columns"].map(format_int)
    dataset_summary = dataset_summary.rename(
        columns={
            "dataset": "Dataset",
            "rows": "Rows",
            "columns": "Columns",
        }
    )

    holdout_missing = outputs["holdout_missing_summary"].copy()
    for col in ["rows", "missing_user_features", "missing_item_features"]:
        holdout_missing[col] = holdout_missing[col].map(format_int)
    holdout_missing["missing_user_features_pct"] = holdout_missing["missing_user_features_pct"].map(
        lambda x: format_pct(x, 2)
    )
    holdout_missing["missing_item_features_pct"] = holdout_missing["missing_item_features_pct"].map(
        lambda x: format_pct(x, 2)
    )
    holdout_missing = holdout_missing.rename(
        columns={
            "split": "Split",
            "rows": "Rows",
            "missing_user_features": "Missing user features",
            "missing_item_features": "Missing item features",
            "missing_user_features_pct": "Missing user features (%)",
            "missing_item_features_pct": "Missing item features (%)",
        }
    )

    user_summary = outputs["user_feature_summary"].copy().T.reset_index()
    user_summary.columns = ["Metric", "Value"]
    user_summary["Metric"] = user_summary["Metric"].replace(
        {
            "n_users_with_features": "Users with features",
            "mean_user_total_interactions": "Mean user interactions",
            "mean_user_distinct_recipes": "Mean user distinct recipes",
            "mean_user_mean_rating": "Mean user rating",
            "mean_user_review_rate": "Mean user review rate",
            "mean_user_interactions_per_day": "Mean user interactions/day",
        }
    )
    user_summary["Value"] = user_summary.apply(
        lambda row: format_pct(row["Value"] * 100, 2)
        if "rate" in row["Metric"].lower()
        else (f"{row['Value']:.2f}" if isinstance(row["Value"], (float, np.floating)) else str(row["Value"])),
        axis=1,
    )

    item_summary = outputs["item_feature_summary"].copy().T.reset_index()
    item_summary.columns = ["Metric", "Value"]
    item_summary["Metric"] = item_summary["Metric"].replace(
        {
            "n_items_with_features": "Items with features",
            "mean_item_total_interactions": "Mean item interactions",
            "mean_item_distinct_users": "Mean item distinct users",
            "mean_item_mean_rating": "Mean item rating",
            "mean_item_review_rate": "Mean item review rate",
            "mean_item_interactions_per_day": "Mean item interactions/day",
        }
    )
    item_summary["Value"] = item_summary.apply(
        lambda row: format_pct(row["Value"] * 100, 2)
        if "rate" in row["Metric"].lower()
        else (f"{row['Value']:.2f}" if isinstance(row["Value"], (float, np.floating)) else str(row["Value"])),
        axis=1,
    )

    return {
        "dashboard_feature_dataset_summary": dataset_summary,
        "dashboard_holdout_missing_summary": holdout_missing,
        "dashboard_user_feature_summary": user_summary,
        "dashboard_item_feature_summary": item_summary,
    }


def build_report_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build academic-report-friendly tables with clean labels and consistent rounding.
    """
    report_dataset_summary = outputs["feature_dataset_summary"].copy()
    report_dataset_summary["dataset"] = report_dataset_summary["dataset"].replace(
        {
            "user_features_train": "User feature training table",
            "item_features_train": "Item feature training table",
            "features_valid": "Validation feature table",
            "features_test": "Test feature table",
        }
    )
    report_dataset_summary["rows"] = report_dataset_summary["rows"].map(format_int)
    report_dataset_summary["columns"] = report_dataset_summary["columns"].map(format_int)
    report_dataset_summary = report_dataset_summary.rename(
        columns={
            "dataset": "Dataset",
            "rows": "Rows",
            "columns": "Columns",
        }
    )

    report_holdout_missing = outputs["holdout_missing_summary"].copy()
    report_holdout_missing["split"] = report_holdout_missing["split"].str.title()
    for col in ["rows", "missing_user_features", "missing_item_features"]:
        report_holdout_missing[col] = report_holdout_missing[col].map(format_int)
    report_holdout_missing["missing_user_features_pct"] = report_holdout_missing[
        "missing_user_features_pct"
    ].map(lambda x: f"{x:.2f}")
    report_holdout_missing["missing_item_features_pct"] = report_holdout_missing[
        "missing_item_features_pct"
    ].map(lambda x: f"{x:.2f}")
    report_holdout_missing = report_holdout_missing.rename(
        columns={
            "split": "Split",
            "rows": "Rows",
            "missing_user_features": "Rows missing user-derived features",
            "missing_item_features": "Rows missing item-derived features",
            "missing_user_features_pct": "Rows missing user-derived features (%)",
            "missing_item_features_pct": "Rows missing item-derived features (%)",
        }
    )

    report_user_summary = outputs["user_feature_summary"].copy().T.reset_index()
    report_user_summary.columns = ["Metric", "Value"]
    report_user_summary["Metric"] = report_user_summary["Metric"].replace(
        {
            "n_users_with_features": "Users with training-derived features",
            "mean_user_total_interactions": "Mean user total interactions",
            "mean_user_distinct_recipes": "Mean user distinct recipes",
            "mean_user_mean_rating": "Mean user mean rating",
            "mean_user_review_rate": "Mean user review rate",
            "mean_user_interactions_per_day": "Mean user interactions per day",
        }
    )
    report_user_summary["Value"] = report_user_summary.apply(
        lambda row: f"{row['Value'] * 100:.2f}"
        if "review rate" in row["Metric"].lower()
        else (format_int(row["Value"]) if "Users with" in row["Metric"] else f"{row['Value']:.2f}"),
        axis=1,
    )

    report_item_summary = outputs["item_feature_summary"].copy().T.reset_index()
    report_item_summary.columns = ["Metric", "Value"]
    report_item_summary["Metric"] = report_item_summary["Metric"].replace(
        {
            "n_items_with_features": "Items with training-derived features",
            "mean_item_total_interactions": "Mean item total interactions",
            "mean_item_distinct_users": "Mean item distinct users",
            "mean_item_mean_rating": "Mean item mean rating",
            "mean_item_review_rate": "Mean item review rate",
            "mean_item_interactions_per_day": "Mean item interactions per day",
        }
    )
    report_item_summary["Value"] = report_item_summary.apply(
        lambda row: f"{row['Value'] * 100:.2f}"
        if "review rate" in row["Metric"].lower()
        else (format_int(row["Value"]) if "Items with" in row["Metric"] else f"{row['Value']:.2f}"),
        axis=1,
    )

    return {
        "report_feature_dataset_summary": report_dataset_summary,
        "report_holdout_missing_summary": report_holdout_missing,
        "report_user_feature_summary": report_user_summary,
        "report_item_feature_summary": report_item_summary,
    }


def plot_feature_dataset_sizes(feature_dataset_summary: pd.DataFrame) -> None:
    """
    Plot row counts for the main feature datasets.
    """
    working = feature_dataset_summary.copy()
    colors = [PRIMARY_BLUE, PRIMARY_ORANGE, PRIMARY_TEAL, PRIMARY_PURPLE]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        working["dataset"],
        working["rows"],
        color=colors[: len(working)],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Rows in Phase 6 Feature Datasets", fontsize=TITLE_SIZE)
    ax.set_xlabel("Dataset", fontsize=LABEL_SIZE)
    ax.set_ylabel("Row count", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", rotation=20)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "06_feature_dataset_rows")


def plot_holdout_missingness(holdout_missing_summary: pd.DataFrame) -> None:
    """
    Plot missing user/item feature rates in holdout splits.
    """
    working = holdout_missing_summary.copy()
    x = np.arange(len(working))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(
        x - width / 2,
        working["missing_user_features_pct"],
        width=width,
        label="Missing user features",
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        working["missing_item_features_pct"],
        width=width,
        label="Missing item features",
        color=PRIMARY_ORANGE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(working["split"].str.title())
    ax.set_title("Holdout Feature Missingness Rates", fontsize=TITLE_SIZE)
    ax.set_xlabel("Split", fontsize=LABEL_SIZE)
    ax.set_ylabel("Rows missing features (%)", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, "06_holdout_feature_missingness")


def plot_top_null_columns(feature_null_summary: pd.DataFrame, top_n: int = 12) -> None:
    """
    Plot the columns with the highest total null counts across outputs.
    """
    working = feature_null_summary.copy()
    working = working[working["total_nulls"] > 0].head(top_n).copy()

    if working.empty:
        return

    working = working.sort_values("total_nulls", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(
        working["column"],
        working["total_nulls"],
        color=PRIMARY_PURPLE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Top Null-Count Feature Columns", fontsize=TITLE_SIZE)
    ax.set_xlabel("Total null count across outputs", fontsize=LABEL_SIZE)
    ax.set_ylabel("Column", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, "06_top_feature_null_columns")


def plot_feature_coverage_counts(
    user_features_train: pd.DataFrame,
    item_features_train: pd.DataFrame,
) -> None:
    """
    Plot user-feature and item-feature coverage counts.
    """
    coverage_df = pd.DataFrame(
        {
            "category": ["Users with features", "Items with features"],
            "count": [
                user_features_train["user_id"].nunique(),
                item_features_train["recipe_id"].nunique(),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        coverage_df["category"],
        coverage_df["count"],
        color=[PRIMARY_TEAL, PRIMARY_ORANGE],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Training Feature Coverage Counts", fontsize=TITLE_SIZE)
    ax.set_xlabel("Category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "06_training_feature_coverage")


def save_feature_figures(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Generate accessible Phase 6 figures.
    """
    plot_feature_dataset_sizes(outputs["feature_dataset_summary"])
    plot_holdout_missingness(outputs["holdout_missing_summary"])
    plot_top_null_columns(outputs["feature_null_summary"])
    plot_feature_coverage_counts(
        outputs["user_features_train"],
        outputs["item_features_train"],
    )


def save_raw_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save raw traceability tables and parquet outputs.
    """
    outputs["user_features_train"].to_parquet(PROCESSED_DIR / OUTPUTS.user_features_train, index=False)
    outputs["item_features_train"].to_parquet(PROCESSED_DIR / OUTPUTS.item_features_train, index=False)
    outputs["valid_features"].to_parquet(PROCESSED_DIR / OUTPUTS.valid_features, index=False)
    outputs["test_features"].to_parquet(PROCESSED_DIR / OUTPUTS.test_features, index=False)

    outputs["user_feature_summary"].to_csv(TABLES_DIR / OUTPUTS.user_feature_summary, index=False)
    outputs["item_feature_summary"].to_csv(TABLES_DIR / OUTPUTS.item_feature_summary, index=False)
    outputs["holdout_missing_summary"].to_csv(TABLES_DIR / OUTPUTS.holdout_missing_summary, index=False)
    outputs["feature_null_summary"].to_csv(TABLES_DIR / OUTPUTS.feature_null_summary, index=False)
    outputs["feature_dataset_summary"].to_csv(TABLES_DIR / OUTPUTS.feature_dataset_summary, index=False)


def save_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save dashboard-ready tables.
    """
    dashboard_tables = build_dashboard_tables(outputs)
    dashboard_tables["dashboard_feature_dataset_summary"].to_csv(
        TABLES_DIR / OUTPUTS.dashboard_feature_dataset_summary,
        index=False,
    )
    dashboard_tables["dashboard_holdout_missing_summary"].to_csv(
        TABLES_DIR / OUTPUTS.dashboard_holdout_missing_summary,
        index=False,
    )
    dashboard_tables["dashboard_user_feature_summary"].to_csv(
        TABLES_DIR / OUTPUTS.dashboard_user_feature_summary,
        index=False,
    )
    dashboard_tables["dashboard_item_feature_summary"].to_csv(
        TABLES_DIR / OUTPUTS.dashboard_item_feature_summary,
        index=False,
    )


def save_report_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save academic-report-ready tables.
    """
    report_tables = build_report_tables(outputs)
    report_tables["report_feature_dataset_summary"].to_csv(
        TABLES_DIR / OUTPUTS.report_feature_dataset_summary,
        index=False,
    )
    report_tables["report_holdout_missing_summary"].to_csv(
        TABLES_DIR / OUTPUTS.report_holdout_missing_summary,
        index=False,
    )
    report_tables["report_user_feature_summary"].to_csv(
        TABLES_DIR / OUTPUTS.report_user_feature_summary,
        index=False,
    )
    report_tables["report_item_feature_summary"].to_csv(
        TABLES_DIR / OUTPUTS.report_item_feature_summary,
        index=False,
    )


def save_logs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save both machine-readable and human-readable Phase 6 logs.
    """
    holdout_map = outputs["holdout_missing_summary"].set_index("split").to_dict(orient="index")

    phase_log = {
        "phase": "06_build_features",
        "input_shapes": {
            "joined_train_rows": int(len(outputs["joined_train"])),
            "joined_valid_rows": int(len(outputs["joined_valid"])),
            "joined_test_rows": int(len(outputs["joined_test"])),
        },
        "output_shapes": {
            "user_feature_rows": int(len(outputs["user_features_train"])),
            "item_feature_rows": int(len(outputs["item_features_train"])),
            "valid_feature_rows": int(len(outputs["valid_features"])),
            "test_feature_rows": int(len(outputs["test_features"])),
        },
        "date_ranges": {
            "train_date_min": str(outputs["joined_train"]["date"].min()),
            "train_date_max": str(outputs["joined_train"]["date"].max()),
            "valid_date_min": str(outputs["joined_valid"]["date"].min()),
            "valid_date_max": str(outputs["joined_valid"]["date"].max()),
            "test_date_min": str(outputs["joined_test"]["date"].min()),
            "test_date_max": str(outputs["joined_test"]["date"].max()),
        },
        "holdout_missing_summary": outputs["holdout_missing_summary"].to_dict(orient="records"),
        "notes": [
            "Phase 6 is retained as a feature engineering and diagnostic step.",
            "Training aggregates were fitted only on the training split.",
            "Outputs are reusable for analysis, dashboards, and optional downstream extensions.",
            "Random Forest reorder modelling is not part of the core outputs of this simplified script.",
        ],
    }

    with open(LOGS_DIR / OUTPUTS.phase_log_json, "w", encoding="utf-8") as file:
        json.dump(phase_log, file, indent=2)

    markdown_lines = [
        "# Phase 6 Feature Build Summary",
        "",
        "## Dataset outputs",
        f"- User feature training rows: {format_int(len(outputs['user_features_train']))}",
        f"- Item feature training rows: {format_int(len(outputs['item_features_train']))}",
        f"- Validation feature rows: {format_int(len(outputs['valid_features']))}",
        f"- Test feature rows: {format_int(len(outputs['test_features']))}",
        "",
        "## Holdout coverage",
        f"- Validation rows missing user-derived features: {format_int(holdout_map['valid']['missing_user_features'])} ({format_pct(holdout_map['valid']['missing_user_features_pct'], 2)})",
        f"- Validation rows missing item-derived features: {format_int(holdout_map['valid']['missing_item_features'])} ({format_pct(holdout_map['valid']['missing_item_features_pct'], 2)})",
        f"- Test rows missing user-derived features: {format_int(holdout_map['test']['missing_user_features'])} ({format_pct(holdout_map['test']['missing_user_features_pct'], 2)})",
        f"- Test rows missing item-derived features: {format_int(holdout_map['test']['missing_item_features'])} ({format_pct(holdout_map['test']['missing_item_features_pct'], 2)})",
        "",
        "## Saved artefacts",
        f"- Processed directory: `{PROCESSED_DIR}`",
        f"- Tables directory: `{TABLES_DIR}`",
        f"- Figures directory: `{FIGURES_DIR}`",
        f"- Logs directory: `{LOGS_DIR}`",
    ]

    with open(LOGS_DIR / OUTPUTS.phase_log_md, "w", encoding="utf-8") as file:
        file.write("\n".join(markdown_lines))


def save_all_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save all Phase 6 artefacts.
    """
    save_raw_tables(outputs)
    save_dashboard_tables(outputs)
    save_report_tables(outputs)
    save_logs(outputs)
    save_feature_figures(outputs)


def print_phase_summary(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Print a concise console summary for quick verification.
    """
    print("=" * 80)
    print(" PHASE 6 FEATURE ENGINEERING")
    print("=" * 80)

    print("\nFeature dataset summary:")
    print(outputs["feature_dataset_summary"].to_string(index=False))

    print("\nHoldout missing summary:")
    print(outputs["holdout_missing_summary"].to_string(index=False))

    print("\nUser feature summary:")
    print(outputs["user_feature_summary"].to_string(index=False))

    print("\nItem feature summary:")
    print(outputs["item_feature_summary"].to_string(index=False))

    print(f"\nSaved parquet outputs to: {PROCESSED_DIR}")
    print(f"Saved tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved logs to: {LOGS_DIR}")


def main() -> None:
    """
    Run the full simplified Phase 6 feature engineering pipeline.
    """
    ensure_directories()

    print("Loading chronological split datasets...")
    joined_train = _load_parquet(SPLITS_DIR / "joined_train.parquet")
    joined_valid = _load_parquet(SPLITS_DIR / "joined_valid.parquet")
    joined_test = _load_parquet(SPLITS_DIR / "joined_test.parquet")

    required_columns = [
        "user_id",
        "recipe_id",
        "date",
        "review_exists",
        "is_unrated_observation",
        "explicit_rating",
    ]

    _check_required_columns(joined_train, required_columns, "joined_train")
    _check_required_columns(joined_valid, required_columns, "joined_valid")
    _check_required_columns(joined_test, required_columns, "joined_test")

    joined_train = _ensure_datetime_column(joined_train, "date")
    joined_valid = _ensure_datetime_column(joined_valid, "date")
    joined_test = _ensure_datetime_column(joined_test, "date")

    joined_train = _sort_for_time_consistency(joined_train)
    joined_valid = _sort_for_time_consistency(joined_valid)
    joined_test = _sort_for_time_consistency(joined_test)

    print("Building user features from training history...")
    user_features_train = build_user_features(joined_train)

    print("Building item features from training history...")
    item_features_train = build_item_features(joined_train)

    print("Building leakage-safe holdout feature tables...")
    valid_features = build_holdout_features(
        holdout_df=joined_valid,
        user_features_train=user_features_train,
        item_features_train=item_features_train,
        split_name="valid",
    )
    test_features = build_holdout_features(
        holdout_df=joined_test,
        user_features_train=user_features_train,
        item_features_train=item_features_train,
        split_name="test",
    )

    print("Building summary tables...")
    user_feature_summary = build_user_feature_summary(user_features_train)
    item_feature_summary = build_item_feature_summary(item_features_train)
    holdout_missing_summary = build_holdout_missing_summary(valid_features, test_features)
    feature_null_summary = build_feature_null_summary(
        user_features_train=user_features_train,
        item_features_train=item_features_train,
        valid_features=valid_features,
        test_features=test_features,
    )
    feature_dataset_summary = build_feature_dataset_summary(
        user_features_train=user_features_train,
        item_features_train=item_features_train,
        valid_features=valid_features,
        test_features=test_features,
    )

    outputs = {
        "joined_train": joined_train,
        "joined_valid": joined_valid,
        "joined_test": joined_test,
        "user_features_train": user_features_train,
        "item_features_train": item_features_train,
        "valid_features": valid_features,
        "test_features": test_features,
        "user_feature_summary": user_feature_summary,
        "item_feature_summary": item_feature_summary,
        "holdout_missing_summary": holdout_missing_summary,
        "feature_null_summary": feature_null_summary,
        "feature_dataset_summary": feature_dataset_summary,
    }

    print("Saving outputs...")
    save_all_outputs(outputs)

    print("\nPhase 6 feature engineering complete.")
    print_phase_summary(outputs)


if __name__ == "__main__":
    main()