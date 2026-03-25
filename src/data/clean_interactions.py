"""
src/data/clean_interactions.py

Purpose:
Clean the raw interaction dataset and produce modelling-ready interaction
outputs for downstream recommendation experiments.

This module formalises the interaction-cleaning logic previously explored
in the notebook "02 Interaction Cleaning".

Responsibilities:
- load the raw interaction dataset
- retain the required interaction columns
- standardise essential numeric fields
- parse dates for chronological use
- remove exact duplicate rows if present
- preserve review text without modelling it
- treat rating = 0 as an observed but unrated interaction
- create full cleaned, explicit, implicit, and optional filtered datasets
- save raw, dashboard-ready, and report-ready tables
- save machine-readable and human-readable logs
- generate accessible static figures for longevity and reuse

Design notes:
- raw interaction history is preserved unless a rule explicitly removes rows
- rating = 0 is retained as an observed interaction for implicit feedback
- explicit-rating modelling excludes rating = 0
- low-signal filtering is evaluated as an optional modelling variant rather
  than enforced as the main cleaning rule
- figures use a colour-blind-safe palette and remain interpretable in grayscale
- both PNG and SVG outputs are saved for dashboard and academic reuse
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import (
    FIGURES_DIR,
    LOGS_DIR,
    PROCESSED_DIR,
    RAW_INTERACTIONS_PATH,
    TABLES_DIR,
    ensure_directories,
)


REQUIRED_COLUMNS = ["user_id", "recipe_id", "date", "rating", "review"]
DEFAULT_FILTER_THRESHOLDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (3, 3),
    (5, 5),
)
OPTIONAL_FILTER_THRESHOLD = (2, 2)

PRIMARY_BLUE = "#0072B2"
PRIMARY_ORANGE = "#E69F00"
PRIMARY_TEAL = "#009E73"
PRIMARY_PURPLE = "#CC79A7"
NEUTRAL_GREY = "#666666"

FIG_DPI = 300
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOT_SIZE = 10


@dataclass(frozen=True)
class CleaningOutputs:
    """
    Container for cleaned interaction datasets and summary tables.
    """

    df_clean: pd.DataFrame
    df_explicit_full: pd.DataFrame
    df_implicit_full: pd.DataFrame
    df_filtered_optional: pd.DataFrame
    rating_distribution: pd.DataFrame
    filtering_results: pd.DataFrame
    cleaning_summary: pd.DataFrame
    yearly_interactions: pd.DataFrame
    user_activity: pd.DataFrame
    item_popularity: pd.DataFrame
    review_summary: pd.DataFrame


def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    return f"{int(value):,}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage values for display tables."""
    return f"{value:.{decimals}f}%"


def load_raw_interactions(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw interaction dataset.

    Args:
        path:
            Optional override path. If omitted, the default raw interactions
            path from src.paths is used.

    Returns:
        pd.DataFrame:
            Raw interaction dataframe.
    """
    input_path = path or RAW_INTERACTIONS_PATH
    return pd.read_csv(input_path)


def select_required_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only the columns required for interaction cleaning and modelling.

    Args:
        df_raw:
            Raw interaction dataframe.

    Returns:
        pd.DataFrame:
            Working dataframe with required columns only.

    Raises:
        KeyError:
            Raised if any required columns are missing.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
    if missing_columns:
        raise KeyError(f"Missing required interaction columns: {missing_columns}")

    return df_raw[REQUIRED_COLUMNS].copy()


def standardise_essential_numeric_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise core numeric interaction fields and remove invalid rows.

    Essential fields:
    - user_id
    - recipe_id
    - rating

    Args:
        df:
            Working interaction dataframe.

    Returns:
        pd.DataFrame:
            Dataframe with numeric core fields standardised and invalid rows
            removed.
    """
    cleaned = df.copy()

    for column in ["user_id", "recipe_id", "rating"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["user_id", "recipe_id", "rating"]).copy()

    cleaned["user_id"] = cleaned["user_id"].astype("int64")
    cleaned["recipe_id"] = cleaned["recipe_id"].astype("int64")
    cleaned["rating"] = cleaned["rating"].astype("int64")

    return cleaned


def parse_interaction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse interaction dates into datetime format and remove invalid rows.

    Args:
        df:
            Interaction dataframe with standardised core fields.

    Returns:
        pd.DataFrame:
            Dataframe with parsed date column.
    """
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"]).copy()
    return cleaned


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact full-row duplicate interaction records.

    Repeated interaction history across distinct timestamps is preserved if
    present because only exact duplicate rows are removed.

    Args:
        df:
            Interaction dataframe.

    Returns:
        pd.DataFrame:
            Deduplicated interaction dataframe.
    """
    return df.drop_duplicates().copy()


def preserve_review_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve review text as auxiliary information without modelling it
    directly in this phase.

    Processing steps:
    - convert review to pandas string dtype
    - strip surrounding whitespace
    - convert empty strings to missing values
    - create review_exists flag

    Args:
        df:
            Interaction dataframe.

    Returns:
        pd.DataFrame:
            Dataframe with cleaned review text and review_exists flag.
    """
    cleaned = df.copy()

    cleaned["review"] = cleaned["review"].astype("string")
    cleaned["review"] = cleaned["review"].str.strip()
    cleaned["review"] = cleaned["review"].replace("", pd.NA)
    cleaned["review_exists"] = cleaned["review"].notna().astype("int8")

    return cleaned


def add_rating_representation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns used to separate explicit and implicit interaction views.

    Rules:
    - rating = 0 is treated as an observed but unrated interaction
    - explicit_rating includes only values 1 to 5
    - implicit_feedback is set to 1 for all observed interactions

    Args:
        df:
            Interaction dataframe.

    Returns:
        pd.DataFrame:
            Dataframe with modelling representation columns added.
    """
    enriched = df.copy()

    enriched["is_unrated_observation"] = (enriched["rating"] == 0).astype("int8")
    enriched["explicit_rating"] = enriched["rating"].where(
        enriched["rating"].between(1, 5),
        pd.NA,
    )
    enriched["implicit_feedback"] = 1

    return enriched


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full cleaned interaction dataset.

    Sorting is applied to produce stable chronological ordering.

    Args:
        df:
            Interaction dataframe with modelling representation columns.

    Returns:
        pd.DataFrame:
            Sorted cleaned interaction dataframe.
    """
    return df.sort_values(["date", "user_id", "recipe_id"]).reset_index(drop=True).copy()


def build_explicit_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build the explicit-rating interaction dataset.

    Explicit ratings include only rows where explicit_rating is present.

    Args:
        df_clean:
            Full cleaned interaction dataframe.

    Returns:
        pd.DataFrame:
            Explicit-only interaction dataframe.
    """
    df_explicit = df_clean[df_clean["explicit_rating"].notna()].copy()
    df_explicit["explicit_rating"] = df_explicit["explicit_rating"].astype("int64")
    return df_explicit


def build_implicit_dataset(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build the implicit-feedback interaction dataset.

    Args:
        df_clean:
            Full cleaned interaction dataframe.

    Returns:
        pd.DataFrame:
            Implicit interaction dataframe with relevant columns retained.
    """
    implicit_columns = [
        "user_id",
        "recipe_id",
        "date",
        "rating",
        "review",
        "review_exists",
        "is_unrated_observation",
        "implicit_feedback",
    ]
    return df_clean[implicit_columns].copy()


def compute_rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the rating distribution table.

    Args:
        df:
            Interaction dataframe.

    Returns:
        pd.DataFrame:
            Rating frequency table with percentages.
    """
    rating_distribution = (
        df["rating"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="count")
    )
    rating_distribution["pct"] = (rating_distribution["count"] / len(df) * 100).round(2)
    return rating_distribution


def compute_yearly_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly interaction counts.

    Args:
        df:
            Cleaned interaction dataframe.

    Returns:
        pd.DataFrame:
            Yearly interaction counts.
    """
    year_counts = (
        df["date"]
        .dt.year
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(name="interaction_count")
    )
    return year_counts


def compute_user_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user activity counts.

    Args:
        df:
            Cleaned interaction dataframe.

    Returns:
        pd.DataFrame:
            User activity frequency table.
    """
    return (
        df["user_id"]
        .value_counts()
        .rename_axis("user_id")
        .reset_index(name="interaction_count")
    )


def compute_item_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute item popularity counts.

    Args:
        df:
            Cleaned interaction dataframe.

    Returns:
        pd.DataFrame:
            Item popularity frequency table.
    """
    return (
        df["recipe_id"]
        .value_counts()
        .rename_axis("recipe_id")
        .reset_index(name="interaction_count")
    )


def compute_review_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute review-text coverage summary.

    Args:
        df:
            Cleaned interaction dataframe.

    Returns:
        pd.DataFrame:
            Review presence summary.
    """
    total_rows = len(df)
    review_rows = int(df["review_exists"].sum())
    no_review_rows = int(total_rows - review_rows)

    return pd.DataFrame(
        [
            {"metric": "rows_with_review_text", "value": review_rows},
            {"metric": "rows_without_review_text", "value": no_review_rows},
            {
                "metric": "review_text_coverage_pct",
                "value": round((review_rows / total_rows) * 100, 2) if total_rows else 0.0,
            },
        ]
    )


def iterative_filter(
    data: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    """
    Iteratively filter the dataset until all remaining users and items meet
    the specified minimum interaction thresholds.

    Args:
        data:
            Interaction dataframe to filter.
        min_user_interactions:
            Minimum number of interactions required per user.
        min_item_interactions:
            Minimum number of interactions required per item.

    Returns:
        pd.DataFrame:
            Filtered dataframe.
    """
    filtered = data.copy()
    previous_shape: tuple[int, int] | None = None

    while previous_shape != filtered.shape:
        previous_shape = filtered.shape

        valid_users = (
            filtered["user_id"]
            .value_counts()
            .loc[lambda series: series >= min_user_interactions]
            .index
        )
        filtered = filtered[filtered["user_id"].isin(valid_users)]

        valid_items = (
            filtered["recipe_id"]
            .value_counts()
            .loc[lambda series: series >= min_item_interactions]
            .index
        )
        filtered = filtered[filtered["recipe_id"].isin(valid_items)]

    return filtered.copy()


def evaluate_filtering_thresholds(
    df_clean: pd.DataFrame,
    threshold_grid: Iterable[tuple[int, int]] = DEFAULT_FILTER_THRESHOLDS,
) -> pd.DataFrame:
    """
    Evaluate multiple user/item minimum interaction thresholds.

    Args:
        df_clean:
            Full cleaned interaction dataframe.
        threshold_grid:
            Iterable of (min_user_interactions, min_item_interactions) pairs.

    Returns:
        pd.DataFrame:
            Summary table describing retention under each threshold pair.
    """
    results: list[dict[str, int | float]] = []

    for min_user_interactions, min_item_interactions in threshold_grid:
        filtered = iterative_filter(
            df_clean,
            min_user_interactions=min_user_interactions,
            min_item_interactions=min_item_interactions,
        )
        results.append(
            {
                "min_user_interactions": min_user_interactions,
                "min_item_interactions": min_item_interactions,
                "rows_remaining": int(len(filtered)),
                "rows_retained_pct": round(len(filtered) / len(df_clean) * 100, 2),
                "users_remaining": int(filtered["user_id"].nunique()),
                "items_remaining": int(filtered["recipe_id"].nunique()),
            }
        )

    return pd.DataFrame(results)


def build_cleaning_summary(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_explicit_full: pd.DataFrame,
    df_implicit_full: pd.DataFrame,
    df_filtered_optional: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact interaction-cleaning summary table.

    Args:
        df_raw:
            Raw interaction dataframe.
        df_clean:
            Full cleaned interaction dataframe.
        df_explicit_full:
            Explicit-only interaction dataframe.
        df_implicit_full:
            Implicit interaction dataframe.
        df_filtered_optional:
            Optional filtered interaction dataframe.

    Returns:
        pd.DataFrame:
            One-row summary table.
    """
    return pd.DataFrame(
        [
            {
                "raw_rows": int(len(df_raw)),
                "rows_after_core_cleaning": int(len(df_clean)),
                "explicit_rows_full": int(len(df_explicit_full)),
                "implicit_rows_full": int(len(df_implicit_full)),
                "rating_zero_count": int(df_clean["is_unrated_observation"].sum()),
                "rows_with_review_text": int(df_clean["review_exists"].sum()),
                "min_date": str(df_clean["date"].min()),
                "max_date": str(df_clean["date"].max()),
                "optional_filtered_rows_2_2": int(len(df_filtered_optional)),
                "optional_filtered_retention_pct_2_2": round(
                    len(df_filtered_optional) / len(df_clean) * 100,
                    2,
                ),
            }
        ]
    )


def build_dashboard_tables(outputs: CleaningOutputs) -> dict[str, pd.DataFrame]:
    """
    Build compact, human-readable tables for dashboard usage.

    Args:
        outputs:
            Cleaning outputs produced by clean_interactions().

    Returns:
        dict[str, pd.DataFrame]:
            Dashboard-ready tables.
    """
    summary = outputs.cleaning_summary.iloc[0]

    dashboard_cleaning_summary = pd.DataFrame(
        [
            {"Metric": "Raw rows", "Value": format_int(summary["raw_rows"])},
            {
                "Metric": "Rows after core cleaning",
                "Value": format_int(summary["rows_after_core_cleaning"]),
            },
            {"Metric": "Explicit rows", "Value": format_int(summary["explicit_rows_full"])},
            {"Metric": "Implicit rows", "Value": format_int(summary["implicit_rows_full"])},
            {"Metric": "Rating = 0 rows", "Value": format_int(summary["rating_zero_count"])},
            {
                "Metric": "Rows with review text",
                "Value": format_int(summary["rows_with_review_text"]),
            },
            {
                "Metric": "Optional filtered retention (2,2)",
                "Value": format_pct(summary["optional_filtered_retention_pct_2_2"], 2),
            },
        ]
    )

    dashboard_rating_distribution = outputs.rating_distribution.copy()
    dashboard_rating_distribution["count"] = dashboard_rating_distribution["count"].map(format_int)
    dashboard_rating_distribution["pct"] = dashboard_rating_distribution["pct"].map(
        lambda x: format_pct(x, 2)
    )
    dashboard_rating_distribution = dashboard_rating_distribution.rename(
        columns={"rating": "Rating", "count": "Count", "pct": "Percentage"}
    )

    dashboard_filtering_results = outputs.filtering_results.copy()
    for col in ["rows_remaining", "users_remaining", "items_remaining"]:
        dashboard_filtering_results[col] = dashboard_filtering_results[col].map(format_int)
    dashboard_filtering_results["rows_retained_pct"] = dashboard_filtering_results[
        "rows_retained_pct"
    ].map(lambda x: format_pct(x, 2))
    dashboard_filtering_results = dashboard_filtering_results.rename(
        columns={
            "min_user_interactions": "Min user interactions",
            "min_item_interactions": "Min item interactions",
            "rows_remaining": "Rows remaining",
            "rows_retained_pct": "Rows retained",
            "users_remaining": "Users remaining",
            "items_remaining": "Items remaining",
        }
    )

    dashboard_review_summary = outputs.review_summary.copy()
    dashboard_review_summary["value"] = dashboard_review_summary.apply(
        lambda row: format_pct(row["value"], 2)
        if row["metric"] == "review_text_coverage_pct"
        else format_int(row["value"]),
        axis=1,
    )
    dashboard_review_summary["metric"] = dashboard_review_summary["metric"].replace(
        {
            "rows_with_review_text": "Rows with review text",
            "rows_without_review_text": "Rows without review text",
            "review_text_coverage_pct": "Review text coverage",
        }
    )
    dashboard_review_summary = dashboard_review_summary.rename(
        columns={"metric": "Metric", "value": "Value"}
    )

    return {
        "dashboard_interaction_cleaning_summary": dashboard_cleaning_summary,
        "dashboard_rating_distribution": dashboard_rating_distribution,
        "dashboard_filtering_results": dashboard_filtering_results,
        "dashboard_review_summary": dashboard_review_summary,
    }


def build_report_tables(outputs: CleaningOutputs) -> dict[str, pd.DataFrame]:
    """
    Build academic-report-friendly tables with clean labels and consistent rounding.

    Args:
        outputs:
            Cleaning outputs produced by clean_interactions().

    Returns:
        dict[str, pd.DataFrame]:
            Report-ready tables.
    """
    summary = outputs.cleaning_summary.iloc[0]

    report_cleaning_summary = pd.DataFrame(
        [
            {"Metric": "Raw interaction rows", "Value": format_int(summary["raw_rows"])},
            {
                "Metric": "Rows after core cleaning",
                "Value": format_int(summary["rows_after_core_cleaning"]),
            },
            {"Metric": "Explicit interaction rows", "Value": format_int(summary["explicit_rows_full"])},
            {"Metric": "Implicit interaction rows", "Value": format_int(summary["implicit_rows_full"])},
            {"Metric": "Observed but unrated rows (rating = 0)", "Value": format_int(summary["rating_zero_count"])},
            {"Metric": "Rows containing review text", "Value": format_int(summary["rows_with_review_text"])},
            {"Metric": "Minimum interaction date", "Value": str(summary["min_date"])},
            {"Metric": "Maximum interaction date", "Value": str(summary["max_date"])},
            {
                "Metric": "Optional filtering retention at threshold (2,2)",
                "Value": format_pct(summary["optional_filtered_retention_pct_2_2"], 2),
            },
        ]
    )

    report_rating_distribution = outputs.rating_distribution.copy()
    report_rating_distribution["count"] = report_rating_distribution["count"].map(format_int)
    report_rating_distribution["pct"] = report_rating_distribution["pct"].map(lambda x: f"{x:.2f}")
    report_rating_distribution = report_rating_distribution.rename(
        columns={"rating": "Rating", "count": "Count", "pct": "Percentage (%)"}
    )

    report_filtering_results = outputs.filtering_results.copy()
    for col in ["rows_remaining", "users_remaining", "items_remaining"]:
        report_filtering_results[col] = report_filtering_results[col].map(format_int)
    report_filtering_results["rows_retained_pct"] = report_filtering_results[
        "rows_retained_pct"
    ].map(lambda x: f"{x:.2f}")
    report_filtering_results = report_filtering_results.rename(
        columns={
            "min_user_interactions": "Minimum user interactions",
            "min_item_interactions": "Minimum item interactions",
            "rows_remaining": "Rows remaining",
            "rows_retained_pct": "Rows retained (%)",
            "users_remaining": "Users remaining",
            "items_remaining": "Items remaining",
        }
    )

    report_review_summary = outputs.review_summary.copy()
    report_review_summary["value"] = report_review_summary.apply(
        lambda row: f"{row['value']:.2f}"
        if row["metric"] == "review_text_coverage_pct"
        else format_int(row["value"]),
        axis=1,
    )
    report_review_summary["metric"] = report_review_summary["metric"].replace(
        {
            "rows_with_review_text": "Rows with review text",
            "rows_without_review_text": "Rows without review text",
            "review_text_coverage_pct": "Review text coverage (%)",
        }
    )
    report_review_summary = report_review_summary.rename(
        columns={"metric": "Metric", "value": "Value"}
    )

    return {
        "report_interaction_cleaning_summary": report_cleaning_summary,
        "report_rating_distribution": report_rating_distribution,
        "report_filtering_results": report_filtering_results,
        "report_review_summary": report_review_summary,
    }


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
    Apply a clean chart style suitable for dashboards and reports.
    """
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(ax: plt.Axes, decimals: int = 0, suffix: str = "") -> None:
    """
    Add direct labels above bars to reduce dependence on colour and legends.

    Args:
        ax:
            Matplotlib axis.
        decimals:
            Decimal places for label formatting.
        suffix:
            Optional suffix such as '%'.
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
    ax.set_title("Interaction Rating Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel("Rating", fontsize=LABEL_SIZE)
    ax.set_ylabel("Interaction count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "02_rating_distribution")


def plot_filtering_retention(filtering_df: pd.DataFrame) -> None:
    """
    Plot retention percentage across filtering thresholds.
    """
    working = filtering_df.copy()
    working["threshold_label"] = working.apply(
        lambda row: f"({int(row['min_user_interactions'])},{int(row['min_item_interactions'])})",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        working["threshold_label"],
        working["rows_retained_pct"],
        color=PRIMARY_ORANGE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Filtering Threshold Retention", fontsize=TITLE_SIZE)
    ax.set_xlabel("Minimum interaction thresholds (user, item)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Rows retained (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 105)
    apply_axis_style(ax)
    add_bar_labels(ax, decimals=2, suffix="%")
    fig.tight_layout()
    save_figure(fig, "02_filtering_threshold_retention")


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
    ax.set_title("Cleaned Interactions by Year", fontsize=TITLE_SIZE)
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
    save_figure(fig, "02_interactions_by_year")


def plot_long_tail_distribution(
    df: pd.DataFrame,
    count_col: str,
    title: str,
    xlabel: str,
    stem: str,
) -> None:
    """
    Plot long-tail activity or popularity as a rank-frequency line chart.

    Args:
        df:
            Input frequency dataframe.
        count_col:
            Count column name.
        title:
            Figure title.
        xlabel:
            X-axis label.
        stem:
            Output file stem.
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


def plot_review_coverage(review_summary: pd.DataFrame) -> None:
    """
    Plot review-text coverage as a compact bar chart.
    """
    working = review_summary[review_summary["metric"] != "review_text_coverage_pct"].copy()
    working["label"] = working["metric"].replace(
        {
            "rows_with_review_text": "With review text",
            "rows_without_review_text": "Without review text",
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(
        working["label"],
        working["value"],
        color=[PRIMARY_BLUE, PRIMARY_PURPLE],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Review Text Coverage", fontsize=TITLE_SIZE)
    ax.set_xlabel("Review text presence", fontsize=LABEL_SIZE)
    ax.set_ylabel("Row count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "02_review_text_coverage")


def save_figures(outputs: CleaningOutputs) -> None:
    """
    Generate accessible figures and save them to the figures directory.
    """
    plot_rating_distribution(outputs.rating_distribution)
    plot_filtering_retention(outputs.filtering_results)
    plot_interactions_by_year(outputs.yearly_interactions)
    plot_long_tail_distribution(
        outputs.user_activity,
        count_col="interaction_count",
        title="User Activity Long-Tail Distribution",
        xlabel="User activity rank",
        stem="02_user_activity_long_tail",
    )
    plot_long_tail_distribution(
        outputs.item_popularity,
        count_col="interaction_count",
        title="Item Popularity Long-Tail Distribution",
        xlabel="Item popularity rank",
        stem="02_item_popularity_long_tail",
    )
    plot_review_coverage(outputs.review_summary)


def clean_interactions() -> CleaningOutputs:
    """
    Run the full interaction-cleaning pipeline in memory.

    Returns:
        CleaningOutputs:
            Container holding cleaned datasets and summary tables.
    """
    df_raw = load_raw_interactions()
    df = select_required_columns(df_raw)
    df = standardise_essential_numeric_fields(df)
    df = parse_interaction_dates(df)
    df = remove_exact_duplicates(df)
    df = preserve_review_text(df)
    df = add_rating_representation_columns(df)

    df_clean = build_clean_dataset(df)
    df_explicit_full = build_explicit_dataset(df_clean)
    df_implicit_full = build_implicit_dataset(df_clean)

    rating_distribution = compute_rating_distribution(df_clean)
    filtering_results = evaluate_filtering_thresholds(df_clean)

    optional_min_user, optional_min_item = OPTIONAL_FILTER_THRESHOLD
    df_filtered_optional = iterative_filter(
        df_clean,
        min_user_interactions=optional_min_user,
        min_item_interactions=optional_min_item,
    )

    cleaning_summary = build_cleaning_summary(
        df_raw=df_raw,
        df_clean=df_clean,
        df_explicit_full=df_explicit_full,
        df_implicit_full=df_implicit_full,
        df_filtered_optional=df_filtered_optional,
    )

    yearly_interactions = compute_yearly_interactions(df_clean)
    user_activity = compute_user_activity(df_clean)
    item_popularity = compute_item_popularity(df_clean)
    review_summary = compute_review_summary(df_clean)

    return CleaningOutputs(
        df_clean=df_clean,
        df_explicit_full=df_explicit_full,
        df_implicit_full=df_implicit_full,
        df_filtered_optional=df_filtered_optional,
        rating_distribution=rating_distribution,
        filtering_results=filtering_results,
        cleaning_summary=cleaning_summary,
        yearly_interactions=yearly_interactions,
        user_activity=user_activity,
        item_popularity=item_popularity,
        review_summary=review_summary,
    )


def save_raw_tables(outputs: CleaningOutputs) -> None:
    """
    Save raw summary tables for traceability.
    """
    outputs.rating_distribution.to_csv(
        TABLES_DIR / "02_rating_distribution.csv",
        index=False,
    )
    outputs.filtering_results.to_csv(
        TABLES_DIR / "02_filtering_threshold_comparison.csv",
        index=False,
    )
    outputs.cleaning_summary.to_csv(
        TABLES_DIR / "02_interaction_cleaning_summary.csv",
        index=False,
    )
    outputs.yearly_interactions.to_csv(
        TABLES_DIR / "02_yearly_interactions.csv",
        index=False,
    )
    outputs.user_activity.to_csv(
        TABLES_DIR / "02_user_activity.csv",
        index=False,
    )
    outputs.item_popularity.to_csv(
        TABLES_DIR / "02_item_popularity.csv",
        index=False,
    )
    outputs.review_summary.to_csv(
        TABLES_DIR / "02_review_summary.csv",
        index=False,
    )


def save_dashboard_tables(outputs: CleaningOutputs) -> None:
    """
    Save compact dashboard-ready tables.
    """
    dashboard_tables = build_dashboard_tables(outputs)
    for name, df in dashboard_tables.items():
        df.to_csv(TABLES_DIR / f"02_{name}.csv", index=False)


def save_report_tables(outputs: CleaningOutputs) -> None:
    """
    Save academic-report-ready tables.
    """
    report_tables = build_report_tables(outputs)
    for name, df in report_tables.items():
        df.to_csv(TABLES_DIR / f"02_{name}.csv", index=False)


def save_logs(outputs: CleaningOutputs) -> None:
    """
    Save both machine-readable and human-readable logs.
    """
    json_summary = {
        "cleaning_summary": outputs.cleaning_summary.to_dict(orient="records"),
        "rating_distribution": outputs.rating_distribution.to_dict(orient="records"),
        "filtering_results": outputs.filtering_results.to_dict(orient="records"),
        "yearly_interactions": outputs.yearly_interactions.to_dict(orient="records"),
        "review_summary": outputs.review_summary.to_dict(orient="records"),
    }

    with open(LOGS_DIR / "02_interaction_cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4, ensure_ascii=False)

    summary = outputs.cleaning_summary.iloc[0]
    review_map = dict(zip(outputs.review_summary["metric"], outputs.review_summary["value"]))

    markdown_lines = [
        "# Interaction Cleaning Summary",
        "",
        "## Core cleaning outputs",
        f"- Raw rows: {format_int(summary['raw_rows'])}",
        f"- Rows after core cleaning: {format_int(summary['rows_after_core_cleaning'])}",
        f"- Explicit rows: {format_int(summary['explicit_rows_full'])}",
        f"- Implicit rows: {format_int(summary['implicit_rows_full'])}",
        f"- Rating = 0 rows retained as observed interactions: {format_int(summary['rating_zero_count'])}",
        "",
        "## Review text coverage",
        f"- Rows with review text: {format_int(review_map['rows_with_review_text'])}",
        f"- Rows without review text: {format_int(review_map['rows_without_review_text'])}",
        f"- Review text coverage: {format_pct(review_map['review_text_coverage_pct'], 2)}",
        "",
        "## Date coverage",
        f"- Minimum interaction date: {summary['min_date']}",
        f"- Maximum interaction date: {summary['max_date']}",
        "",
        "## Optional filtering",
        f"- Rows retained at threshold (2,2): {format_int(summary['optional_filtered_rows_2_2'])}",
        f"- Retention at threshold (2,2): {format_pct(summary['optional_filtered_retention_pct_2_2'], 2)}",
        "",
        "## Saved artefacts",
        f"- Tables directory: `{TABLES_DIR}`",
        f"- Figures directory: `{FIGURES_DIR}`",
        f"- Logs directory: `{LOGS_DIR}`",
        f"- Processed parquet directory: `{PROCESSED_DIR}`",
    ]

    with open(LOGS_DIR / "02_interaction_cleaning_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def save_parquet_outputs(outputs: CleaningOutputs) -> None:
    """
    Save cleaned interaction datasets to parquet files.

    Args:
        outputs:
            Cleaning outputs produced by clean_interactions().
    """
    outputs.df_clean.to_parquet(
        PROCESSED_DIR / "interactions_clean.parquet",
        index=False,
    )
    outputs.df_explicit_full.to_parquet(
        PROCESSED_DIR / "interactions_explicit.parquet",
        index=False,
    )
    outputs.df_implicit_full.to_parquet(
        PROCESSED_DIR / "interactions_implicit.parquet",
        index=False,
    )
    outputs.df_filtered_optional.to_parquet(
        PROCESSED_DIR / "interactions_filtered_optional_2_2.parquet",
        index=False,
    )


def save_outputs(outputs: CleaningOutputs) -> None:
    """
    Save all cleaned outputs, summary tables, logs, and figures.

    Args:
        outputs:
            Cleaning outputs produced by clean_interactions().
    """
    save_raw_tables(outputs)
    save_dashboard_tables(outputs)
    save_report_tables(outputs)
    save_logs(outputs)
    save_figures(outputs)
    save_parquet_outputs(outputs)


def print_summary(outputs: CleaningOutputs) -> None:
    """
    Print a concise console summary for quick verification.
    """
    print("=" * 80)
    print(" INTERACTION CLEANING")
    print("=" * 80)

    print("\nCleaning summary:")
    print(outputs.cleaning_summary.to_string(index=False))

    print("\nRating distribution:")
    print(outputs.rating_distribution.to_string(index=False))

    print("\nFiltering threshold comparison:")
    print(outputs.filtering_results.to_string(index=False))

    print(f"\nSaved tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved logs to: {LOGS_DIR}")
    print(f"Saved parquet files to: {PROCESSED_DIR}")


def main() -> None:
    """
    Execute the interaction-cleaning pipeline and save outputs.
    """
    ensure_directories()

    outputs = clean_interactions()
    save_outputs(outputs)
    print_summary(outputs)


if __name__ == "__main__":
    main()