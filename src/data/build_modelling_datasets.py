"""
src/data/build_modelling_datasets.py

Purpose:
Build the main modelling-ready datasets required for later recommendation
and prediction stages, then save outputs in forms suitable for pipeline
logging, dashboard use, and academic reporting.

This module produces:
- explicit modelling dataset
- implicit modelling dataset
- joined interaction-recipe modelling dataset
- raw traceability tables
- dashboard-ready summary tables
- academic-report-ready tables
- machine-readable JSON log
- human-readable markdown log
- accessible static figures

Responsibilities:
- load processed interaction and recipe datasets
- validate expected columns
- build explicit and implicit modelling datasets
- build the joined interaction-recipe table
- measure join coverage and PP-feature availability
- save modelling outputs and summary tables
- generate accessible figures for reporting and dashboarding

Design notes:
- rating = 0 is treated as an observed but unrated interaction
- figures use a colour-blind-safe palette
- figures are also designed to remain interpretable in grayscale
- both PNG and SVG outputs are saved for longevity and reuse
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Any

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.paths import (  # noqa: E402
    FIGURES_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
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


def _require_columns(df: pd.DataFrame, required_columns: Iterable[str], df_name: str) -> None:
    """
    Validate that a dataframe contains all required columns.

    Args:
        df:
            DataFrame to validate.
        required_columns:
            Column names expected to be present.
        df_name:
            Human-readable dataframe name used in error messages.

    Raises:
        ValueError:
            Raised when one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _save_json_report(report: dict[str, Any], output_path: Path) -> None:
    """
    Save a JSON report to disk.

    Args:
        report:
            Serializable dictionary to save.
        output_path:
            Destination file path.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    return f"{int(value):,}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage values for display tables."""
    return f"{value:.{decimals}f}%"


def save_figure(fig: plt.Figure, stem: str) -> None:
    """
    Save a figure in both PNG and SVG formats.

    Args:
        fig:
            Matplotlib figure.
        stem:
            File stem without extension.
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


def build_outputs() -> dict[str, pd.DataFrame]:
    """
    Build all modelling datasets and summary outputs.

    Returns:
        Dictionary containing datasets and summary tables.
    """
    interactions_clean = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")
    interactions_explicit = pd.read_parquet(PROCESSED_DIR / "interactions_explicit.parquet")
    interactions_implicit = pd.read_parquet(PROCESSED_DIR / "interactions_implicit.parquet")
    recipes_joined = pd.read_parquet(PROCESSED_DIR / "recipes_joined.parquet")

    clean_required_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "review",
        "review_exists",
        "is_unrated_observation",
        "explicit_rating",
        "implicit_feedback",
    ]
    explicit_required_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "review",
        "review_exists",
        "is_unrated_observation",
        "explicit_rating",
        "implicit_feedback",
    ]
    implicit_required_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "review",
        "review_exists",
        "is_unrated_observation",
        "implicit_feedback",
    ]
    recipes_required_columns = [
        "recipe_id",
        "name",
        "minutes",
        "submitted",
        "n_steps",
        "n_ingredients",
        "has_pp_features",
    ]

    _require_columns(interactions_clean, clean_required_columns, "interactions_clean")
    _require_columns(interactions_explicit, explicit_required_columns, "interactions_explicit")
    _require_columns(interactions_implicit, implicit_required_columns, "interactions_implicit")
    _require_columns(recipes_joined, recipes_required_columns, "recipes_joined")

    explicit_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "explicit_rating",
        "review_exists",
        "implicit_feedback",
    ]
    explicit_model_df = interactions_explicit[explicit_columns].copy()
    explicit_model_df["user_id"] = explicit_model_df["user_id"].astype("int64")
    explicit_model_df["recipe_id"] = explicit_model_df["recipe_id"].astype("int64")
    explicit_model_df["explicit_rating"] = explicit_model_df["explicit_rating"].astype("int64")
    explicit_model_df["implicit_feedback"] = explicit_model_df["implicit_feedback"].astype("int8")
    explicit_model_df["review_exists"] = explicit_model_df["review_exists"].astype("int8")

    implicit_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "review_exists",
        "is_unrated_observation",
        "implicit_feedback",
    ]
    implicit_model_df = interactions_implicit[implicit_columns].copy()
    implicit_model_df["user_id"] = implicit_model_df["user_id"].astype("int64")
    implicit_model_df["recipe_id"] = implicit_model_df["recipe_id"].astype("int64")
    implicit_model_df["rating"] = implicit_model_df["rating"].astype("int64")
    implicit_model_df["review_exists"] = implicit_model_df["review_exists"].astype("int8")
    implicit_model_df["is_unrated_observation"] = implicit_model_df["is_unrated_observation"].astype("int8")
    implicit_model_df["implicit_feedback"] = implicit_model_df["implicit_feedback"].astype("int8")

    interaction_recipe_joined = interactions_clean.merge(
        recipes_joined,
        on="recipe_id",
        how="left",
        validate="many_to_one",
    )

    missing_recipe_rows = int(interaction_recipe_joined["name"].isna().sum())
    missing_recipe_pct = round(missing_recipe_rows / len(interaction_recipe_joined) * 100, 4)

    join_summary = pd.DataFrame(
        [
            {
                "interaction_rows": int(len(interaction_recipe_joined)),
                "unique_interaction_users": int(interaction_recipe_joined["user_id"].nunique()),
                "unique_interaction_recipes": int(interaction_recipe_joined["recipe_id"].nunique()),
                "missing_recipe_rows": missing_recipe_rows,
                "missing_recipe_rows_pct": missing_recipe_pct,
            }
        ]
    )

    pp_feature_summary = pd.DataFrame(
        [
            {
                "rows_with_pp_features": int(interaction_recipe_joined["has_pp_features"].fillna(0).sum()),
                "rows_without_pp_features": int(
                    (interaction_recipe_joined["has_pp_features"].fillna(0) == 0).sum()
                ),
                "rows_with_pp_features_pct": round(
                    interaction_recipe_joined["has_pp_features"].fillna(0).mean() * 100,
                    2,
                ),
            }
        ]
    )

    joined_model_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "explicit_rating",
        "review_exists",
        "is_unrated_observation",
        "implicit_feedback",
        "name",
        "minutes",
        "submitted",
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
        "num_name_tokens",
        "num_steps_tokens",
        "num_ingredient_ids",
        "num_techniques",
        "has_pp_features",
    ]
    joined_model_columns = [col for col in joined_model_columns if col in interaction_recipe_joined.columns]
    joined_model_df = interaction_recipe_joined[joined_model_columns].copy()

    modelling_summary = pd.DataFrame(
        [
            {
                "dataset": "explicit_model_df",
                "rows": int(len(explicit_model_df)),
                "users": int(explicit_model_df["user_id"].nunique()),
                "recipes": int(explicit_model_df["recipe_id"].nunique()),
                "min_date": explicit_model_df["date"].min(),
                "max_date": explicit_model_df["date"].max(),
            },
            {
                "dataset": "implicit_model_df",
                "rows": int(len(implicit_model_df)),
                "users": int(implicit_model_df["user_id"].nunique()),
                "recipes": int(implicit_model_df["recipe_id"].nunique()),
                "min_date": implicit_model_df["date"].min(),
                "max_date": implicit_model_df["date"].max(),
            },
            {
                "dataset": "joined_model_df",
                "rows": int(len(joined_model_df)),
                "users": int(joined_model_df["user_id"].nunique()),
                "recipes": int(joined_model_df["recipe_id"].nunique()),
                "min_date": joined_model_df["date"].min(),
                "max_date": joined_model_df["date"].max(),
            },
        ]
    )

    explicit_rating_dist = (
        explicit_model_df["explicit_rating"]
        .value_counts()
        .sort_index()
        .rename_axis("explicit_rating")
        .reset_index(name="count")
    )
    explicit_rating_dist["percentage"] = (
        explicit_rating_dist["count"] / len(explicit_model_df) * 100
    ).round(4)

    implicit_flag_dist = (
        implicit_model_df["is_unrated_observation"]
        .value_counts()
        .sort_index()
        .rename_axis("is_unrated_observation")
        .reset_index(name="count")
    )
    implicit_flag_dist["percentage"] = (
        implicit_flag_dist["count"] / len(implicit_model_df) * 100
    ).round(4)

    pp_coverage_dist = pd.DataFrame(
        [
            {
                "category": "With PP features",
                "count": int(interaction_recipe_joined["has_pp_features"].fillna(0).sum()),
            },
            {
                "category": "Without PP features",
                "count": int((interaction_recipe_joined["has_pp_features"].fillna(0) == 0).sum()),
            },
        ]
    )
    pp_coverage_dist["percentage"] = (pp_coverage_dist["count"] / pp_coverage_dist["count"].sum() * 100).round(4)

    return {
        "interactions_clean": interactions_clean,
        "interactions_explicit": interactions_explicit,
        "interactions_implicit": interactions_implicit,
        "recipes_joined": recipes_joined,
        "explicit_model_df": explicit_model_df,
        "implicit_model_df": implicit_model_df,
        "joined_model_df": joined_model_df,
        "modelling_summary": modelling_summary,
        "join_summary": join_summary,
        "pp_feature_summary": pp_feature_summary,
        "explicit_rating_distribution": explicit_rating_dist,
        "implicit_unrated_distribution": implicit_flag_dist,
        "pp_feature_distribution": pp_coverage_dist,
    }


def build_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build compact, human-readable tables for dashboard usage.
    """
    modelling_summary = outputs["modelling_summary"].copy()
    dashboard_modelling_summary = modelling_summary.rename(
        columns={
            "dataset": "Dataset",
            "rows": "Rows",
            "users": "Users",
            "recipes": "Recipes",
            "min_date": "Minimum date",
            "max_date": "Maximum date",
        }
    )
    for col in ["Rows", "Users", "Recipes"]:
        dashboard_modelling_summary[col] = dashboard_modelling_summary[col].map(format_int)

    dashboard_join_summary = outputs["join_summary"].copy().rename(
        columns={
            "interaction_rows": "Interaction rows",
            "unique_interaction_users": "Unique users",
            "unique_interaction_recipes": "Unique recipes",
            "missing_recipe_rows": "Missing recipe rows",
            "missing_recipe_rows_pct": "Missing recipe rows (%)",
        }
    )
    for col in ["Interaction rows", "Unique users", "Unique recipes", "Missing recipe rows"]:
        dashboard_join_summary[col] = dashboard_join_summary[col].map(format_int)
    dashboard_join_summary["Missing recipe rows (%)"] = dashboard_join_summary["Missing recipe rows (%)"].map(
        lambda x: format_pct(x, 2)
    )

    dashboard_pp_feature_summary = outputs["pp_feature_summary"].copy().rename(
        columns={
            "rows_with_pp_features": "Rows with PP features",
            "rows_without_pp_features": "Rows without PP features",
            "rows_with_pp_features_pct": "Rows with PP features (%)",
        }
    )
    for col in ["Rows with PP features", "Rows without PP features"]:
        dashboard_pp_feature_summary[col] = dashboard_pp_feature_summary[col].map(format_int)
    dashboard_pp_feature_summary["Rows with PP features (%)"] = dashboard_pp_feature_summary[
        "Rows with PP features (%)"
    ].map(lambda x: format_pct(x, 2))

    dashboard_explicit_rating = outputs["explicit_rating_distribution"].copy()
    dashboard_explicit_rating["count"] = dashboard_explicit_rating["count"].map(format_int)
    dashboard_explicit_rating["percentage"] = dashboard_explicit_rating["percentage"].map(
        lambda x: format_pct(x, 2)
    )
    dashboard_explicit_rating = dashboard_explicit_rating.rename(
        columns={
            "explicit_rating": "Explicit rating",
            "count": "Count",
            "percentage": "Percentage",
        }
    )

    dashboard_implicit_unrated = outputs["implicit_unrated_distribution"].copy()
    dashboard_implicit_unrated["is_unrated_observation"] = dashboard_implicit_unrated[
        "is_unrated_observation"
    ].map({0: "Rated interaction", 1: "Unrated observation"})
    dashboard_implicit_unrated["count"] = dashboard_implicit_unrated["count"].map(format_int)
    dashboard_implicit_unrated["percentage"] = dashboard_implicit_unrated["percentage"].map(
        lambda x: format_pct(x, 2)
    )
    dashboard_implicit_unrated = dashboard_implicit_unrated.rename(
        columns={
            "is_unrated_observation": "Category",
            "count": "Count",
            "percentage": "Percentage",
        }
    )

    return {
        "dashboard_modelling_summary": dashboard_modelling_summary,
        "dashboard_join_summary": dashboard_join_summary,
        "dashboard_pp_feature_summary": dashboard_pp_feature_summary,
        "dashboard_explicit_rating_distribution": dashboard_explicit_rating,
        "dashboard_implicit_unrated_distribution": dashboard_implicit_unrated,
    }


def build_report_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build academic-report-friendly tables with clean labels and consistent rounding.
    """
    report_modelling_summary = outputs["modelling_summary"].copy()
    report_modelling_summary["dataset"] = report_modelling_summary["dataset"].replace(
        {
            "explicit_model_df": "Explicit modelling dataset",
            "implicit_model_df": "Implicit modelling dataset",
            "joined_model_df": "Joined interaction-recipe dataset",
        }
    )
    for col in ["rows", "users", "recipes"]:
        report_modelling_summary[col] = report_modelling_summary[col].map(format_int)
    report_modelling_summary = report_modelling_summary.rename(
        columns={
            "dataset": "Dataset",
            "rows": "Rows",
            "users": "Users",
            "recipes": "Recipes",
            "min_date": "Minimum date",
            "max_date": "Maximum date",
        }
    )

    report_join_summary = outputs["join_summary"].copy()
    report_join_summary["interaction_rows"] = report_join_summary["interaction_rows"].map(format_int)
    report_join_summary["unique_interaction_users"] = report_join_summary["unique_interaction_users"].map(format_int)
    report_join_summary["unique_interaction_recipes"] = report_join_summary["unique_interaction_recipes"].map(format_int)
    report_join_summary["missing_recipe_rows"] = report_join_summary["missing_recipe_rows"].map(format_int)
    report_join_summary["missing_recipe_rows_pct"] = report_join_summary["missing_recipe_rows_pct"].map(
        lambda x: format_pct(x, 2)
    )
    report_join_summary = report_join_summary.rename(
        columns={
            "interaction_rows": "Interaction rows",
            "unique_interaction_users": "Unique users",
            "unique_interaction_recipes": "Unique recipes",
            "missing_recipe_rows": "Missing recipe-side matches",
            "missing_recipe_rows_pct": "Missing recipe-side matches (%)",
        }
    )

    report_pp_feature_summary = outputs["pp_feature_summary"].copy()
    report_pp_feature_summary["rows_with_pp_features"] = report_pp_feature_summary["rows_with_pp_features"].map(format_int)
    report_pp_feature_summary["rows_without_pp_features"] = report_pp_feature_summary["rows_without_pp_features"].map(format_int)
    report_pp_feature_summary["rows_with_pp_features_pct"] = report_pp_feature_summary["rows_with_pp_features_pct"].map(
        lambda x: format_pct(x, 2)
    )
    report_pp_feature_summary = report_pp_feature_summary.rename(
        columns={
            "rows_with_pp_features": "Rows with PP features",
            "rows_without_pp_features": "Rows without PP features",
            "rows_with_pp_features_pct": "Rows with PP features (%)",
        }
    )

    report_explicit_rating = outputs["explicit_rating_distribution"].copy()
    report_explicit_rating["count"] = report_explicit_rating["count"].map(format_int)
    report_explicit_rating["percentage"] = report_explicit_rating["percentage"].map(lambda x: f"{x:.2f}")
    report_explicit_rating = report_explicit_rating.rename(
        columns={
            "explicit_rating": "Explicit rating",
            "count": "Count",
            "percentage": "Percentage (%)",
        }
    )

    return {
        "report_modelling_summary": report_modelling_summary,
        "report_join_summary": report_join_summary,
        "report_pp_feature_summary": report_pp_feature_summary,
        "report_explicit_rating_distribution": report_explicit_rating,
    }


def plot_modelling_dataset_sizes(modelling_summary: pd.DataFrame) -> None:
    """
    Plot row counts for each modelling dataset.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(
        modelling_summary["dataset"],
        modelling_summary["rows"],
        color=[PRIMARY_BLUE, PRIMARY_ORANGE, PRIMARY_TEAL],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Rows in Modelling Datasets", fontsize=TITLE_SIZE)
    ax.set_xlabel("Dataset", fontsize=LABEL_SIZE)
    ax.set_ylabel("Row count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "04_modelling_dataset_rows")


def plot_dataset_entity_counts(modelling_summary: pd.DataFrame) -> None:
    """
    Plot user and recipe counts for each modelling dataset.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    working = modelling_summary.copy()
    x = range(len(working))
    width = 0.38

    ax.bar(
        [i - width / 2 for i in x],
        working["users"],
        width=width,
        label="Users",
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x],
        working["recipes"],
        width=width,
        label="Recipes",
        color=PRIMARY_ORANGE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(working["dataset"])
    ax.set_title("Users and Recipes Across Modelling Datasets", fontsize=TITLE_SIZE)
    ax.set_xlabel("Dataset", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, "04_modelling_dataset_entities")


def plot_explicit_rating_distribution(explicit_rating_distribution: pd.DataFrame) -> None:
    """
    Plot explicit rating distribution as an accessible bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        explicit_rating_distribution["explicit_rating"].astype(str),
        explicit_rating_distribution["count"],
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Explicit Rating Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel("Explicit rating", fontsize=LABEL_SIZE)
    ax.set_ylabel("Interaction count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "04_explicit_rating_distribution")


def plot_implicit_unrated_distribution(implicit_unrated_distribution: pd.DataFrame) -> None:
    """
    Plot rated versus unrated observations in the implicit dataset.
    """
    working = implicit_unrated_distribution.copy()
    working["label"] = working["is_unrated_observation"].map(
        {0: "Rated interaction", 1: "Unrated observation"}
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        working["label"],
        working["count"],
        color=[PRIMARY_TEAL, PRIMARY_PURPLE],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Implicit Dataset Observation Types", fontsize=TITLE_SIZE)
    ax.set_xlabel("Category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Row count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "04_implicit_observation_types")


def plot_pp_feature_coverage(pp_feature_distribution: pd.DataFrame) -> None:
    """
    Plot PP-feature coverage after joining interactions with recipes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        pp_feature_distribution["category"],
        pp_feature_distribution["percentage"],
        color=[PRIMARY_BLUE, LIGHT_GREY],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("PP Feature Coverage After Join", fontsize=TITLE_SIZE)
    ax.set_xlabel("Category", fontsize=LABEL_SIZE)
    ax.set_ylabel("Coverage (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 105)
    apply_axis_style(ax)
    add_bar_labels(ax, decimals=2, suffix="%")
    fig.tight_layout()
    save_figure(fig, "04_pp_feature_coverage")


def save_figures(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Generate accessible figures and save them to the figures directory.
    """
    plot_modelling_dataset_sizes(outputs["modelling_summary"])
    plot_dataset_entity_counts(outputs["modelling_summary"])
    plot_explicit_rating_distribution(outputs["explicit_rating_distribution"])
    plot_implicit_unrated_distribution(outputs["implicit_unrated_distribution"])
    plot_pp_feature_coverage(outputs["pp_feature_distribution"])


def save_raw_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save raw traceability tables and modelling datasets.
    """
    outputs["explicit_model_df"].to_parquet(PROCESSED_DIR / "model_explicit.parquet", index=False)
    outputs["implicit_model_df"].to_parquet(PROCESSED_DIR / "model_implicit.parquet", index=False)
    outputs["joined_model_df"].to_parquet(PROCESSED_DIR / "model_interaction_recipe_joined.parquet", index=False)

    outputs["modelling_summary"].to_csv(TABLES_DIR / "04_modelling_dataset_summary.csv", index=False)
    outputs["join_summary"].to_csv(TABLES_DIR / "04_join_coverage_summary.csv", index=False)
    outputs["pp_feature_summary"].to_csv(TABLES_DIR / "04_pp_feature_availability_summary.csv", index=False)
    outputs["explicit_rating_distribution"].to_csv(TABLES_DIR / "04_explicit_rating_distribution.csv", index=False)
    outputs["implicit_unrated_distribution"].to_csv(TABLES_DIR / "04_implicit_unrated_distribution.csv", index=False)
    outputs["pp_feature_distribution"].to_csv(TABLES_DIR / "04_pp_feature_distribution.csv", index=False)


def save_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save compact dashboard-ready tables.
    """
    dashboard_tables = build_dashboard_tables(outputs)
    for name, df in dashboard_tables.items():
        df.to_csv(TABLES_DIR / f"04_{name}.csv", index=False)


def save_report_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save cleaned academic-report-ready tables.
    """
    report_tables = build_report_tables(outputs)
    for name, df in report_tables.items():
        df.to_csv(TABLES_DIR / f"04_{name}.csv", index=False)


def save_logs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save both machine-readable and human-readable logs.
    """
    report = {
        "inputs": {
            "interactions_clean_shape": list(outputs["interactions_clean"].shape),
            "interactions_explicit_shape": list(outputs["interactions_explicit"].shape),
            "interactions_implicit_shape": list(outputs["interactions_implicit"].shape),
            "recipes_joined_shape": list(outputs["recipes_joined"].shape),
        },
        "outputs": {
            "explicit_model_shape": list(outputs["explicit_model_df"].shape),
            "implicit_model_shape": list(outputs["implicit_model_df"].shape),
            "joined_model_shape": list(outputs["joined_model_df"].shape),
        },
        "modelling_summary": outputs["modelling_summary"].to_dict(orient="records"),
        "join_summary": outputs["join_summary"].to_dict(orient="records"),
        "pp_feature_summary": outputs["pp_feature_summary"].to_dict(orient="records"),
        "explicit_rating_distribution": outputs["explicit_rating_distribution"].to_dict(orient="records"),
        "implicit_unrated_distribution": outputs["implicit_unrated_distribution"].to_dict(orient="records"),
    }
    _save_json_report(report, LOGS_DIR / "04_build_modelling_datasets_report.json")

    summary_map = outputs["modelling_summary"].set_index("dataset").to_dict(orient="index")
    join_row = outputs["join_summary"].iloc[0].to_dict()
    pp_row = outputs["pp_feature_summary"].iloc[0].to_dict()

    markdown_lines = [
        "# Build Modelling Datasets Summary",
        "",
        "## Modelling dataset outputs",
        f"- Explicit modelling dataset rows: {format_int(summary_map['explicit_model_df']['rows'])}",
        f"- Implicit modelling dataset rows: {format_int(summary_map['implicit_model_df']['rows'])}",
        f"- Joined interaction-recipe dataset rows: {format_int(summary_map['joined_model_df']['rows'])}",
        "",
        "## Join coverage",
        f"- Interaction rows after join: {format_int(join_row['interaction_rows'])}",
        f"- Missing recipe-side matches: {format_int(join_row['missing_recipe_rows'])}",
        f"- Missing recipe-side matches (%): {format_pct(join_row['missing_recipe_rows_pct'], 2)}",
        "",
        "## PP feature coverage",
        f"- Rows with PP features: {format_int(pp_row['rows_with_pp_features'])}",
        f"- Rows without PP features: {format_int(pp_row['rows_without_pp_features'])}",
        f"- Rows with PP features (%): {format_pct(pp_row['rows_with_pp_features_pct'], 2)}",
        "",
        "## Saved artefacts",
        f"- Tables directory: `{TABLES_DIR}`",
        f"- Figures directory: `{FIGURES_DIR}`",
        f"- Logs directory: `{LOGS_DIR}`",
    ]

    with open(LOGS_DIR / "04_build_modelling_datasets_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def save_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save all artefacts for this stage.
    """
    ensure_directories()
    save_raw_tables(outputs)
    save_dashboard_tables(outputs)
    save_report_tables(outputs)
    save_logs(outputs)
    save_figures(outputs)


def print_summary(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Print a concise console summary for quick verification.
    """
    print("=" * 80)
    print(" BUILD MODELLING DATASETS")
    print("=" * 80)

    print("\nModelling summary:")
    print(outputs["modelling_summary"].to_string(index=False))

    print("\nJoin summary:")
    print(outputs["join_summary"].to_string(index=False))

    print("\nPP feature summary:")
    print(outputs["pp_feature_summary"].to_string(index=False))

    print("\nExplicit rating distribution:")
    print(outputs["explicit_rating_distribution"].to_string(index=False))

    print("\nImplicit unrated distribution:")
    print(outputs["implicit_unrated_distribution"].to_string(index=False))

    print(f"\nSaved modelling datasets to: {PROCESSED_DIR}")
    print(f"Saved tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved logs to: {LOGS_DIR}")


def main() -> None:
    """
    Execute the modelling dataset build pipeline.
    """
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 200)
    pd.set_option("display.width", 200)

    outputs = build_outputs()
    save_outputs(outputs)
    print_summary(outputs)


if __name__ == "__main__":
    main()