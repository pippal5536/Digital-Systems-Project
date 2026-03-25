"""
src/data/dataset_audit.py

Purpose:
Run a reusable dataset audit step for the three raw project datasets and save
outputs in forms suitable for pipeline logging, dashboard use, and academic
reporting.

This module inspects:
- dataset shapes
- column summaries
- null values
- duplicate patterns
- rating distribution
- date coverage
- join coverage
- item popularity
- user activity

Output groups:
- raw audit tables for traceability
- dashboard-ready summary tables
- academic-report-ready tables
- machine-readable JSON log
- human-readable markdown log
- accessible static figures

Design notes:
- the script does not modify the raw datasets
- figures use a colour-blind-safe palette
- figures are also designed to remain interpretable in grayscale
- both PNG and SVG outputs are saved for longevity and reuse
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import (
    RAW_INTERACTIONS_PATH,
    RAW_RECIPES_PATH,
    PP_RECIPES_PATH,
    TABLES_DIR,
    FIGURES_DIR,
    LOGS_DIR,
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


def ensure_output_dirs() -> None:
    """Ensure all audit output directories exist."""
    for folder in [TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw datasets.

    Returns:
        Tuple containing:
            - interactions
            - raw_recipes
            - pp_recipes
    """
    interactions = pd.read_csv(RAW_INTERACTIONS_PATH)
    raw_recipes = pd.read_csv(RAW_RECIPES_PATH)
    pp_recipes = pd.read_csv(PP_RECIPES_PATH)
    return interactions, raw_recipes, pp_recipes


def column_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Build a column-level audit summary for a dataset.

    Args:
        df: Input dataframe.
        dataset_name: Dataset name for traceability.

    Returns:
        DataFrame containing dtype, null counts, unique counts, and sample values.
    """
    rows: list[dict[str, Any]] = []

    for col in df.columns:
        non_null = int(df[col].notna().sum())
        null_count = int(df[col].isna().sum())
        unique_count = int(df[col].nunique(dropna=True))

        non_null_values = df[col].dropna()
        sample_value = None if non_null_values.empty else str(non_null_values.iloc[0])[:150]

        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": non_null,
                "null_count": null_count,
                "null_percentage": round((null_count / len(df)) * 100, 4),
                "n_unique": unique_count,
                "sample_value": sample_value,
            }
        )

    return pd.DataFrame(rows)


def dataset_shape_summary(
    interactions: pd.DataFrame,
    raw_recipes: pd.DataFrame,
    pp_recipes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return row and column counts for each dataset.

    Returns:
        DataFrame containing dataset, rows, and columns.
    """
    return pd.DataFrame(
        [
            {
                "dataset": "interactions",
                "rows": interactions.shape[0],
                "columns": interactions.shape[1],
            },
            {
                "dataset": "raw_recipes",
                "rows": raw_recipes.shape[0],
                "columns": raw_recipes.shape[1],
            },
            {
                "dataset": "pp_recipes",
                "rows": pp_recipes.shape[0],
                "columns": pp_recipes.shape[1],
            },
        ]
    )


def null_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute null counts and percentages for each column.

    Args:
        df: Input dataframe.

    Returns:
        Null summary sorted by missingness.
    """
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "null_count": df.isna().sum().values,
            "null_percentage": (df.isna().mean().values * 100).round(4),
        }
    )

    return (
        summary.sort_values(by=["null_count", "null_percentage"], ascending=False)
        .reset_index(drop=True)
    )


def duplicate_summary(
    interactions: pd.DataFrame,
    raw_recipes: pd.DataFrame,
    pp_recipes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce duplicate diagnostics across the three datasets.

    Returns:
        DataFrame containing duplicate metrics.
    """
    metrics: dict[str, int] = {
        "interactions_full_row_duplicates": int(interactions.duplicated().sum()),
        "interactions_user_recipe_duplicates": int(
            interactions.duplicated(subset=["user_id", "recipe_id"]).sum()
        ),
        "interactions_user_recipe_date_rating_duplicates": int(
            interactions.duplicated(subset=["user_id", "recipe_id", "date", "rating"]).sum()
        ),
        "raw_recipes_full_row_duplicates": int(raw_recipes.duplicated().sum()),
        "pp_recipes_full_row_duplicates": int(pp_recipes.duplicated().sum()),
    }

    if "id" in raw_recipes.columns:
        metrics["raw_recipes_id_duplicates"] = int(raw_recipes.duplicated(subset=["id"]).sum())

    if "id" in pp_recipes.columns:
        metrics["pp_recipes_id_duplicates"] = int(pp_recipes.duplicated(subset=["id"]).sum())

    if "name" in raw_recipes.columns:
        metrics["raw_recipes_name_duplicates"] = int(raw_recipes.duplicated(subset=["name"]).sum())

    return pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])


def rating_distribution(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rating frequency and percentage.

    Args:
        interactions: Interaction dataframe.

    Returns:
        Rating distribution table.
    """
    dist = (
        interactions["rating"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="count")
    )
    dist["percentage"] = (dist["count"] / len(interactions) * 100).round(4)
    return dist


def date_summary(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse dates and summarise temporal coverage.

    Dates are parsed with dayfirst=True because project planning examples use
    day-first formatting.

    Args:
        interactions: Interaction dataframe.

    Returns:
        Tuple of:
            - summary dataframe
            - yearly interaction counts dataframe
    """
    dated = interactions.copy()
    dated["date_parsed"] = pd.to_datetime(
        dated["date"],
        errors="coerce",
        dayfirst=True,
    )

    summary = {
        "total_rows": int(len(dated)),
        "parsed_non_null": int(dated["date_parsed"].notna().sum()),
        "parse_failures": int(dated["date_parsed"].isna().sum()),
        "min_date": None if dated["date_parsed"].dropna().empty else str(dated["date_parsed"].min().date()),
        "max_date": None if dated["date_parsed"].dropna().empty else str(dated["date_parsed"].max().date()),
    }

    summary_df = pd.DataFrame([{"metric": k, "value": v} for k, v in summary.items()])

    year_counts = (
        dated["date_parsed"]
        .dropna()
        .dt.year
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(name="interaction_count")
    )

    return summary_df, year_counts


def join_coverage(
    interactions: pd.DataFrame,
    raw_recipes: pd.DataFrame,
    pp_recipes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure recipe ID join coverage between interactions and recipe metadata.

    Returns:
        DataFrame containing coverage for raw_recipes and pp_recipes.
    """
    interaction_recipe_ids = set(interactions["recipe_id"].dropna().unique())
    raw_recipe_ids = set(raw_recipes["id"].dropna().unique()) if "id" in raw_recipes.columns else set()
    pp_recipe_ids = set(pp_recipes["id"].dropna().unique()) if "id" in pp_recipes.columns else set()

    matched_raw = interaction_recipe_ids & raw_recipe_ids
    matched_pp = interaction_recipe_ids & pp_recipe_ids

    results = [
        {
            "comparison": "interactions vs raw_recipes",
            "unique_recipe_ids_in_interactions": len(interaction_recipe_ids),
            "unique_recipe_ids_in_recipe_file": len(raw_recipe_ids),
            "matched_ids": len(matched_raw),
            "missing_ids": len(interaction_recipe_ids - raw_recipe_ids),
            "coverage_pct": round((len(matched_raw) / len(interaction_recipe_ids)) * 100, 4)
            if interaction_recipe_ids
            else 0.0,
        },
        {
            "comparison": "interactions vs pp_recipes",
            "unique_recipe_ids_in_interactions": len(interaction_recipe_ids),
            "unique_recipe_ids_in_recipe_file": len(pp_recipe_ids),
            "matched_ids": len(matched_pp),
            "missing_ids": len(interaction_recipe_ids - pp_recipe_ids),
            "coverage_pct": round((len(matched_pp) / len(interaction_recipe_ids)) * 100, 4)
            if interaction_recipe_ids
            else 0.0,
        },
    ]

    return pd.DataFrame(results)


def dataset_statistics(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute extra dataset statistics useful for reporting and dashboarding.

    Returns:
        Tuple of:
            - dataset stats
            - item popularity table
            - user activity table
    """
    n_users = interactions["user_id"].nunique()
    n_items = interactions["recipe_id"].nunique()
    n_interactions = len(interactions)

    sparsity = 1 - (n_interactions / (n_users * n_items))

    dataset_stats = pd.DataFrame(
        [
            {"metric": "n_users", "value": n_users},
            {"metric": "n_items", "value": n_items},
            {"metric": "n_interactions", "value": n_interactions},
            {"metric": "sparsity", "value": sparsity},
        ]
    )

    item_popularity = (
        interactions["recipe_id"]
        .value_counts()
        .rename_axis("recipe_id")
        .reset_index(name="interaction_count")
    )

    user_activity = (
        interactions["user_id"]
        .value_counts()
        .rename_axis("user_id")
        .reset_index(name="interaction_count")
    )

    return dataset_stats, item_popularity, user_activity


def build_audit_outputs() -> dict[str, pd.DataFrame]:
    """
    Run the full dataset audit and return all outputs as dataframes.

    Returns:
        Dictionary of audit artefacts.
    """
    interactions, raw_recipes, pp_recipes = load_datasets()

    interactions_columns = column_summary(interactions, "interactions")
    raw_recipes_columns = column_summary(raw_recipes, "raw_recipes")
    pp_recipes_columns = column_summary(pp_recipes, "pp_recipes")

    shapes = dataset_shape_summary(interactions, raw_recipes, pp_recipes)

    interactions_nulls = null_summary(interactions)
    raw_recipes_nulls = null_summary(raw_recipes)
    pp_recipes_nulls = null_summary(pp_recipes)

    duplicates = duplicate_summary(interactions, raw_recipes, pp_recipes)
    ratings = rating_distribution(interactions)
    dates, year_counts = date_summary(interactions)
    coverage = join_coverage(interactions, raw_recipes, pp_recipes)
    dataset_stats, item_popularity, user_activity = dataset_statistics(interactions)

    return {
        "interactions_columns": interactions_columns,
        "raw_recipes_columns": raw_recipes_columns,
        "pp_recipes_columns": pp_recipes_columns,
        "dataset_shapes": shapes,
        "interactions_nulls": interactions_nulls,
        "raw_recipes_nulls": raw_recipes_nulls,
        "pp_recipes_nulls": pp_recipes_nulls,
        "duplicates": duplicates,
        "rating_distribution": ratings,
        "date_summary": dates,
        "year_counts": year_counts,
        "join_coverage": coverage,
        "dataset_stats": dataset_stats,
        "item_popularity": item_popularity,
        "user_activity": user_activity,
    }




def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    return f"{int(value):,}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage values for display tables."""
    return f"{value:.{decimals}f}%"


def build_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build compact, human-readable tables for dashboard usage.

    Returns:
        Dictionary of dashboard-ready tables.
    """
    dataset_stats = outputs["dataset_stats"].copy()
    stats_map = dict(zip(dataset_stats["metric"], dataset_stats["value"]))

    dashboard_dataset_summary = pd.DataFrame(
        [
            {"Metric": "Users", "Value": format_int(stats_map["n_users"])},
            {"Metric": "Items", "Value": format_int(stats_map["n_items"])},
            {"Metric": "Interactions", "Value": format_int(stats_map["n_interactions"])},
            {"Metric": "Sparsity", "Value": format_pct(stats_map["sparsity"] * 100, 2)},
        ]
    )

    dashboard_rating_distribution = outputs["rating_distribution"].copy()
    dashboard_rating_distribution["count"] = dashboard_rating_distribution["count"].map(format_int)
    dashboard_rating_distribution["percentage"] = dashboard_rating_distribution["percentage"].map(
        lambda x: format_pct(x, 2)
    )
    dashboard_rating_distribution = dashboard_rating_distribution.rename(
        columns={
            "rating": "Rating",
            "count": "Count",
            "percentage": "Percentage",
        }
    )

    dashboard_join_coverage = outputs["join_coverage"].copy()
    for col in [
        "unique_recipe_ids_in_interactions",
        "unique_recipe_ids_in_recipe_file",
        "matched_ids",
        "missing_ids",
    ]:
        dashboard_join_coverage[col] = dashboard_join_coverage[col].map(format_int)

    dashboard_join_coverage["coverage_pct"] = dashboard_join_coverage["coverage_pct"].map(
        lambda x: format_pct(x, 2)
    )
    dashboard_join_coverage = dashboard_join_coverage.rename(
        columns={
            "comparison": "Comparison",
            "unique_recipe_ids_in_interactions": "Recipe IDs in interactions",
            "unique_recipe_ids_in_recipe_file": "Recipe IDs in recipe file",
            "matched_ids": "Matched IDs",
            "missing_ids": "Missing IDs",
            "coverage_pct": "Coverage",
        }
    )

    dashboard_date_summary = outputs["date_summary"].copy().rename(
        columns={"metric": "Metric", "value": "Value"}
    )

    return {
        "dashboard_dataset_summary": dashboard_dataset_summary,
        "dashboard_rating_distribution": dashboard_rating_distribution,
        "dashboard_join_coverage": dashboard_join_coverage,
        "dashboard_date_summary": dashboard_date_summary,
    }


def build_report_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build academic-report-friendly tables with clean labels and consistent rounding.

    Returns:
        Dictionary of report-ready tables.
    """
    dataset_stats = outputs["dataset_stats"].copy()
    dataset_stats["metric"] = dataset_stats["metric"].replace(
        {
            "n_users": "Number of users",
            "n_items": "Number of items",
            "n_interactions": "Number of interactions",
            "sparsity": "Matrix sparsity",
        }
    )

    dataset_stats["value"] = dataset_stats.apply(
        lambda row: format_pct(row["value"] * 100, 2)
        if row["metric"] == "Matrix sparsity"
        else format_int(row["value"]),
        axis=1,
    )
    report_dataset_characteristics = dataset_stats.rename(
        columns={"metric": "Metric", "value": "Value"}
    )

    report_duplicates = outputs["duplicates"].copy()
    report_duplicates["metric"] = report_duplicates["metric"].str.replace("_", " ").str.title()
    report_duplicates["value"] = report_duplicates["value"].map(format_int)
    report_duplicates = report_duplicates.rename(columns={"metric": "Metric", "value": "Value"})

    report_join_coverage = outputs["join_coverage"].copy()
    report_join_coverage["coverage_pct"] = report_join_coverage["coverage_pct"].map(
        lambda x: format_pct(x, 2)
    )
    for col in [
        "unique_recipe_ids_in_interactions",
        "unique_recipe_ids_in_recipe_file",
        "matched_ids",
        "missing_ids",
    ]:
        report_join_coverage[col] = report_join_coverage[col].map(format_int)

    report_join_coverage = report_join_coverage.rename(
        columns={
            "comparison": "Comparison",
            "unique_recipe_ids_in_interactions": "Unique recipe IDs in interactions",
            "unique_recipe_ids_in_recipe_file": "Unique recipe IDs in recipe file",
            "matched_ids": "Matched IDs",
            "missing_ids": "Missing IDs",
            "coverage_pct": "Coverage (%)",
        }
    )

    report_rating_distribution = outputs["rating_distribution"].copy()
    report_rating_distribution["count"] = report_rating_distribution["count"].map(format_int)
    report_rating_distribution["percentage"] = report_rating_distribution["percentage"].map(
        lambda x: f"{x:.2f}"
    )
    report_rating_distribution = report_rating_distribution.rename(
        columns={
            "rating": "Rating",
            "count": "Count",
            "percentage": "Percentage (%)",
        }
    )

    report_date_summary = outputs["date_summary"].copy().rename(
        columns={"metric": "Metric", "value": "Value"}
    )

    return {
        "report_dataset_characteristics": report_dataset_characteristics,
        "report_duplicates": report_duplicates,
        "report_join_coverage": report_join_coverage,
        "report_rating_distribution": report_rating_distribution,
        "report_date_summary": report_date_summary,
    }




def save_figure(fig: plt.Figure, stem: str) -> None:
    """
    Save a figure in both PNG and SVG formats.

    Args:
        fig: Matplotlib figure.
        stem: File stem without extension.
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


def add_bar_labels(ax: plt.Axes, decimals: int = 0) -> None:
    """
    Add direct labels above bars to reduce dependence on colour and legends.
    """
    for patch in ax.patches:
        height = patch.get_height()
        if pd.notna(height):
            label = f"{height:,.{decimals}f}" if decimals > 0 else f"{int(height):,}"
            ax.annotate(
                label,
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=ANNOT_SIZE,
                xytext=(0, 3),
                textcoords="offset points",
            )


def plot_rating_distribution(rating_df: pd.DataFrame) -> None:
    """
    Plot rating distribution as an accessible bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        rating_df["rating"].astype(str),
        rating_df["count"],
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Rating Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel("Rating", fontsize=LABEL_SIZE)
    ax.set_ylabel("Interaction count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "01_rating_distribution")


def plot_interactions_by_year(year_counts: pd.DataFrame) -> None:
    """
    Plot yearly interaction counts as a line chart with markers.
    """
    if year_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        year_counts["year"],
        year_counts["interaction_count"],
        marker="o",
        linewidth=2.0,
        color=PRIMARY_TEAL,
    )
    ax.set_title("Interactions by Year", fontsize=TITLE_SIZE)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Interaction count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)

    for _, row in year_counts.iterrows():
        ax.annotate(
            f"{int(row['interaction_count']):,}",
            (row["year"], row["interaction_count"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=ANNOT_SIZE,
        )

    fig.tight_layout()
    save_figure(fig, "01_interactions_by_year")


def plot_join_coverage(coverage_df: pd.DataFrame) -> None:
    """
    Plot join coverage percentages as a compact comparison bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        coverage_df["comparison"],
        coverage_df["coverage_pct"],
        color=[PRIMARY_BLUE, PRIMARY_ORANGE],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Recipe ID Join Coverage", fontsize=TITLE_SIZE)
    ax.set_xlabel("Comparison", fontsize=LABEL_SIZE)
    ax.set_ylabel("Coverage (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 105)
    apply_axis_style(ax)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.2f}%",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE,
            xytext=(0, 3),
            textcoords="offset points",
        )

    fig.tight_layout()
    save_figure(fig, "01_join_coverage")


def plot_missingness_overview(null_df: pd.DataFrame, dataset_name: str, top_n: int = 10) -> None:
    """
    Plot top missing columns for one dataset as a horizontal bar chart.

    Args:
        null_df: Null summary dataframe.
        dataset_name: Dataset name for the figure title and file stem.
        top_n: Number of columns to plot.
    """
    working = null_df[null_df["null_count"] > 0].head(top_n).copy()

    if working.empty:
        return

    working = working.sort_values("null_percentage", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        working["column"],
        working["null_percentage"],
        color=PRIMARY_PURPLE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title(f"Top Missing Columns: {dataset_name}", fontsize=TITLE_SIZE)
    ax.set_xlabel("Missing values (%)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Column", fontsize=LABEL_SIZE)
    apply_axis_style(ax)

    for patch in ax.patches:
        width = patch.get_width()
        ax.annotate(
            f"{width:.2f}%",
            (width, patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            fontsize=ANNOT_SIZE,
            xytext=(4, 0),
            textcoords="offset points",
        )

    fig.tight_layout()
    save_figure(fig, f"01_missingness_overview_{dataset_name}")


def plot_long_tail_distribution(
    df: pd.DataFrame,
    id_col: str,
    count_col: str,
    title: str,
    xlabel: str,
    stem: str,
) -> None:
    """
    Plot long-tail popularity or activity as a rank-frequency line chart.

    This is better for recommendation datasets than a plain histogram because it
    makes head-versus-tail concentration clearer and remains interpretable in
    grayscale.

    Args:
        df: Input frequency dataframe.
        id_col: Identifier column name.
        count_col: Count column name.
        title: Figure title.
        xlabel: X-axis label.
        stem: Output file stem.
    """
    if df.empty:
        return

    ranked = df.copy().sort_values(count_col, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        ranked["rank"],
        ranked[count_col],
        linewidth=1.8,
        color=PRIMARY_BLUE,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Interaction count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, stem)


def save_audit_figures(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Generate accessible audit figures and save them to the figures directory.
    """
    plot_rating_distribution(outputs["rating_distribution"])
    plot_interactions_by_year(outputs["year_counts"])
    plot_join_coverage(outputs["join_coverage"])

    plot_missingness_overview(outputs["interactions_nulls"], "interactions")
    plot_missingness_overview(outputs["raw_recipes_nulls"], "raw_recipes")
    plot_missingness_overview(outputs["pp_recipes_nulls"], "pp_recipes")

    plot_long_tail_distribution(
        outputs["item_popularity"],
        id_col="recipe_id",
        count_col="interaction_count",
        title="Item Popularity Long-Tail Distribution",
        xlabel="Item popularity rank",
        stem="01_item_popularity_long_tail",
    )

    plot_long_tail_distribution(
        outputs["user_activity"],
        id_col="user_id",
        count_col="interaction_count",
        title="User Activity Long-Tail Distribution",
        xlabel="User activity rank",
        stem="01_user_activity_long_tail",
    )



def save_raw_audit_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save raw audit dataframes to CSV for traceability and debugging.
    """
    for name, df in outputs.items():
        df.to_csv(TABLES_DIR / f"01_{name}.csv", index=False)


def save_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save compact dashboard-ready tables.
    """
    dashboard_tables = build_dashboard_tables(outputs)
    for name, df in dashboard_tables.items():
        df.to_csv(TABLES_DIR / f"01_{name}.csv", index=False)


def save_report_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save cleaned academic-report-ready tables.
    """
    report_tables = build_report_tables(outputs)
    for name, df in report_tables.items():
        df.to_csv(TABLES_DIR / f"01_{name}.csv", index=False)


def save_logs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save both machine-readable and human-readable audit logs.
    """
    json_summary = {
        name: df.to_dict(orient="records")
        for name, df in outputs.items()
        if name not in {"item_popularity", "user_activity"}
    }

    with open(LOGS_DIR / "01_dataset_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4, ensure_ascii=False)

    stats_map = dict(zip(outputs["dataset_stats"]["metric"], outputs["dataset_stats"]["value"]))
    date_map = dict(zip(outputs["date_summary"]["metric"], outputs["date_summary"]["value"]))

    markdown_lines = [
        "# Dataset Audit Summary",
        "",
        "## Dataset characteristics",
        f"- Users: {format_int(stats_map['n_users'])}",
        f"- Items: {format_int(stats_map['n_items'])}",
        f"- Interactions: {format_int(stats_map['n_interactions'])}",
        f"- Sparsity: {format_pct(stats_map['sparsity'] * 100, 2)}",
        "",
        "## Date coverage",
        f"- Parsed rows: {format_int(date_map['parsed_non_null'])}",
        f"- Parse failures: {format_int(date_map['parse_failures'])}",
        f"- Minimum date: {date_map['min_date']}",
        f"- Maximum date: {date_map['max_date']}",
        "",
        "## Saved artefacts",
        f"- Tables directory: `{TABLES_DIR}`",
        f"- Figures directory: `{FIGURES_DIR}`",
        f"- Logs directory: `{LOGS_DIR}`",
    ]

    with open(LOGS_DIR / "01_dataset_audit_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def save_audit_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save all audit artefacts.

    This includes:
    - raw audit tables
    - dashboard-ready tables
    - report-ready tables
    - logs
    - accessible figures
    """
    ensure_output_dirs()
    save_raw_audit_tables(outputs)
    save_dashboard_tables(outputs)
    save_report_tables(outputs)
    save_logs(outputs)
    save_audit_figures(outputs)


def print_audit_summary(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Print a concise console summary for quick verification.
    """
    print("=" * 80)
    print(" DATASET AUDIT")
    print("=" * 80)

    print("\nDataset shapes:")
    print(outputs["dataset_shapes"].to_string(index=False))

    print("\nDuplicate summary:")
    print(outputs["duplicates"].to_string(index=False))

    print("\nDate summary:")
    print(outputs["date_summary"].to_string(index=False))

    print("\nJoin coverage:")
    print(outputs["join_coverage"].to_string(index=False))

    print("\nRating distribution:")
    print(outputs["rating_distribution"].to_string(index=False))

    print(f"\nSaved audit tables to: {TABLES_DIR}")
    print(f"Saved audit figures to: {FIGURES_DIR}")
    print(f"Saved audit logs to: {LOGS_DIR}")


def main() -> None:
    """
    Execute the dataset audit pipeline.
    """
    outputs = build_audit_outputs()
    save_audit_outputs(outputs)
    print_audit_summary(outputs)


if __name__ == "__main__":
    main()