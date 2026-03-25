"""
src/data/make_chronological_splits.py

Purpose:
Create chronological train, validation, and test splits for the modelling
datasets produced in the earlier preprocessing stages, then save outputs in
forms suitable for pipeline logging, dashboard use, and academic reporting.

This module prepares time-aware split datasets for recommendation experiments
so that later interactions are never used to train models evaluated on earlier
ones. It also fits user and item ID mappings using train data only, then applies
those mappings to validation and test to avoid leakage.

Output groups:
- raw split summary tables for traceability
- dashboard-ready summary tables
- academic-report-ready tables
- machine-readable JSON log
- human-readable markdown log
- accessible static figures

Design notes:
- one shared temporal split policy is applied across explicit, implicit, and
  joined modelling datasets
- splitting is per-user chronological rather than random so each user's
  earlier interactions are used to predict that same user's later behaviour
- train, validation, and test proportions default to 70 / 15 / 15
- ID mappings are learned from train only
- unseen validation/test users or recipes are retained and reported rather
  than silently removed
- figures use a colour-blind-safe palette
- figures are also designed to remain interpretable in grayscale
- both PNG and SVG outputs are saved for longevity and reuse
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import (
    FIGURES_DIR,
    LOGS_DIR,
    MAPPINGS_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    ensure_directories,
)


TRAIN_FRAC = 0.70
VALID_FRAC = 0.15
TEST_FRAC = 0.15

SPLIT_POLICY_NAME = "per_user_chronological"

EXPLICIT_INPUT_NAME = "model_explicit.parquet"
IMPLICIT_INPUT_NAME = "model_implicit.parquet"
JOINED_INPUT_NAME = "model_interaction_recipe_joined.parquet"

EXPLICIT_REQUIRED_COLS = ["user_id", "recipe_id", "date", "explicit_rating"]
IMPLICIT_REQUIRED_COLS = ["user_id", "recipe_id", "date", "implicit_feedback"]
JOINED_REQUIRED_COLS = ["user_id", "recipe_id", "date"]


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
class SplitResult:
    """
    Container for train, validation, and test split dataframes.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    return f"{int(value):,}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage values for display tables."""
    return f"{value:.{decimals}f}%"


def validate_split_fractions(train_frac: float, valid_frac: float, test_frac: float) -> None:
    """
    Validate that split fractions sum to 1.
    """
    total = round(train_frac + valid_frac + test_frac, 8)
    if total != 1.0:
        raise ValueError(
            f"Split fractions must sum to 1.0, but received {train_frac}, {valid_frac}, {test_frac}."
        )


def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load a parquet dataset from the processed directory.
    """
    path = PROCESSED_DIR / file_name
    return pd.read_parquet(path)


def ensure_required_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """
    Ensure that a dataframe contains all required columns.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def ensure_datetime_column(df: pd.DataFrame, column_name: str = "date") -> pd.DataFrame:
    """
    Ensure that the specified column is stored as datetime.
    """
    if column_name not in df.columns:
        raise ValueError(f"Required datetime column '{column_name}' is missing.")

    out = df.copy()
    out[column_name] = pd.to_datetime(out[column_name], errors="coerce")

    if out[column_name].isna().all():
        raise ValueError(f"Column '{column_name}' could not be converted to datetime.")

    return out


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a dataframe chronologically with stable tie-breakers.
    """
    sort_columns = [col for col in ["user_id", "date", "recipe_id"] if col in df.columns]
    return df.sort_values(by=sort_columns).reset_index(drop=True)


def _allocate_per_user_counts(
    n_rows: int,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
) -> tuple[int, int, int]:
    """
    Allocate per-user train, validation, and test row counts.

    The allocation keeps each user's history ordered in time, guarantees at
    least one training interaction when data exists, and creates later-user
    holdout rows only when enough interactions are available.
    """
    validate_split_fractions(train_frac, valid_frac, test_frac)

    if n_rows <= 0:
        return 0, 0, 0
    if n_rows == 1:
        return 1, 0, 0
    if n_rows == 2:
        return 1, 1, 0

    train_count = int(np.floor(n_rows * train_frac))
    valid_count = int(np.floor(n_rows * valid_frac))
    test_count = n_rows - train_count - valid_count

    train_count = max(train_count, 1)
    valid_count = max(valid_count, 1)
    test_count = max(test_count, 1)

    while train_count + valid_count + test_count > n_rows:
        if train_count > valid_count and train_count > 1:
            train_count -= 1
        elif test_count > 1:
            test_count -= 1
        elif valid_count > 1:
            valid_count -= 1
        else:
            train_count -= 1

    while train_count + valid_count + test_count < n_rows:
        train_count += 1

    return train_count, valid_count, test_count


def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
) -> SplitResult:
    """
    Split a chronologically sorted dataframe into train, validation, and test.

    The split is performed within each user so that every retained holdout row
    occurs later than that same user's training history.
    """
    validate_split_fractions(train_frac, valid_frac, test_frac)

    if df.empty:
        empty = df.copy()
        return SplitResult(train=empty, valid=empty, test=empty)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    grouped = df.groupby("user_id", sort=False, dropna=False)

    for _, user_df in grouped:
        ordered_user_df = user_df.sort_values(by=["date", "recipe_id"]).reset_index(drop=True)
        train_count, valid_count, test_count = _allocate_per_user_counts(
            n_rows=len(ordered_user_df),
            train_frac=train_frac,
            valid_frac=valid_frac,
            test_frac=test_frac,
        )

        train_end = train_count
        valid_end = train_count + valid_count

        if train_count > 0:
            train_parts.append(ordered_user_df.iloc[:train_end].copy())
        if valid_count > 0:
            valid_parts.append(ordered_user_df.iloc[train_end:valid_end].copy())
        if test_count > 0:
            test_parts.append(ordered_user_df.iloc[valid_end:valid_end + test_count].copy())

    empty = df.iloc[0:0].copy()
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty.copy()
    valid_df = pd.concat(valid_parts, ignore_index=True) if valid_parts else empty.copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else empty.copy()

    train_df = sort_chronologically(train_df)
    valid_df = sort_chronologically(valid_df)
    test_df = sort_chronologically(test_df)

    return SplitResult(train=train_df, valid=valid_df, test=test_df)


def build_id_maps_from_train(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build train-only user and recipe mapping tables.
    """
    user_map = (
        pd.DataFrame({"user_id": sorted(train_df["user_id"].dropna().unique())})
        .reset_index()
        .rename(columns={"index": "user_idx"})
    )
    user_map["user_idx"] = user_map["user_idx"].astype("int64")
    user_map["user_id"] = user_map["user_id"].astype("int64")

    recipe_map = (
        pd.DataFrame({"recipe_id": sorted(train_df["recipe_id"].dropna().unique())})
        .reset_index()
        .rename(columns={"index": "item_idx"})
    )
    recipe_map["item_idx"] = recipe_map["item_idx"].astype("int64")
    recipe_map["recipe_id"] = recipe_map["recipe_id"].astype("int64")

    return user_map, recipe_map


def apply_id_maps(
    df: pd.DataFrame,
    user_map: pd.DataFrame,
    recipe_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply user and recipe mapping tables to a split dataframe.
    """
    out = df.merge(user_map, on="user_id", how="left")
    out = out.merge(recipe_map, on="recipe_id", how="left")
    return out


def create_split_summary(df: pd.DataFrame, split_name: str, dataset_name: str) -> dict[str, object]:
    """
    Create a compact summary for a split dataframe.
    """
    return {
        "dataset": dataset_name,
        "split": split_name,
        "split_policy": SPLIT_POLICY_NAME,
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()) if "user_id" in df.columns else 0,
        "recipes": int(df["recipe_id"].nunique()) if "recipe_id" in df.columns else 0,
        "min_date": str(df["date"].min()) if len(df) > 0 else None,
        "max_date": str(df["date"].max()) if len(df) > 0 else None,
    }


def create_unseen_mapping_summary(df: pd.DataFrame, split_name: str, dataset_name: str) -> dict[str, object]:
    """
    Measure missing mapped IDs in a validation or test split.
    """
    missing_either = df["user_idx"].isna() | df["item_idx"].isna()

    return {
        "dataset": dataset_name,
        "split": split_name,
        "rows": int(len(df)),
        "rows_missing_user_idx": int(df["user_idx"].isna().sum()),
        "rows_missing_item_idx": int(df["item_idx"].isna().sum()),
        "rows_missing_either_idx": int(missing_either.sum()),
        "pct_missing_either_idx": round(float(missing_either.mean() * 100), 4) if len(df) > 0 else 0.0,
    }


def create_user_history_summary(
    split_result: SplitResult,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Summarise how many users contribute to each per-user temporal pattern.
    """
    user_frames: list[pd.DataFrame] = []
    for split_name, df in [("train", split_result.train), ("valid", split_result.valid), ("test", split_result.test)]:
        if df.empty:
            continue
        user_frames.append(
            df.groupby("user_id", as_index=False)
            .size()
            .rename(columns={"size": "rows"})
            .assign(split=split_name)
        )

    if not user_frames:
        return pd.DataFrame(columns=["dataset", "user_history_pattern", "users", "share_of_users_pct"])

    combined = pd.concat(user_frames, ignore_index=True)
    presence = (
        combined.assign(has_split=1)
        .pivot_table(index="user_id", columns="split", values="has_split", aggfunc="max", fill_value=0)
        .reset_index()
    )

    for col in ["train", "valid", "test"]:
        if col not in presence.columns:
            presence[col] = 0

    def _label(row: pd.Series) -> str:
        flags = [name for name in ["train", "valid", "test"] if int(row.get(name, 0)) == 1]
        return "_".join(flags) if flags else "none"

    presence["user_history_pattern"] = presence.apply(_label, axis=1)
    summary = (
        presence.groupby("user_history_pattern", as_index=False)
        .size()
        .rename(columns={"size": "users"})
        .sort_values("users", ascending=False)
        .reset_index(drop=True)
    )

    total_users = int(summary["users"].sum())
    summary["share_of_users_pct"] = (
        summary["users"] / total_users * 100 if total_users > 0 else 0.0
    ).round(4)
    summary.insert(0, "dataset", dataset_name)
    return summary


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save a dataframe to parquet.
    """
    df.to_parquet(path, index=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save a dataframe to CSV.
    """
    df.to_csv(path, index=False)


def prepare_dataset_for_splitting(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Validate and prepare a dataset for chronological splitting.
    """
    ensure_required_columns(df, required_columns, dataset_name)
    df = ensure_datetime_column(df, column_name="date")
    df = sort_chronologically(df)
    return df


def build_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build compact, human-readable tables for dashboard usage.
    """
    dashboard_split_summary = outputs["all_split_summary"].copy()
    for col in ["rows", "users", "recipes"]:
        dashboard_split_summary[col] = dashboard_split_summary[col].map(format_int)

    dashboard_split_summary = dashboard_split_summary.rename(
        columns={
            "dataset": "Dataset",
            "split": "Split",
            "split_policy": "Split policy",
            "rows": "Rows",
            "users": "Users",
            "recipes": "Recipes",
            "min_date": "Minimum date",
            "max_date": "Maximum date",
        }
    )

    dashboard_unseen_summary = outputs["all_unseen_summary"].copy()
    for col in [
        "rows",
        "rows_missing_user_idx",
        "rows_missing_item_idx",
        "rows_missing_either_idx",
    ]:
        dashboard_unseen_summary[col] = dashboard_unseen_summary[col].map(format_int)

    dashboard_unseen_summary["pct_missing_either_idx"] = dashboard_unseen_summary[
        "pct_missing_either_idx"
    ].map(lambda x: format_pct(x, 2))

    dashboard_unseen_summary = dashboard_unseen_summary.rename(
        columns={
            "dataset": "Dataset",
            "split": "Split",
            "rows": "Rows",
            "rows_missing_user_idx": "Rows missing user index",
            "rows_missing_item_idx": "Rows missing item index",
            "rows_missing_either_idx": "Rows missing either index",
            "pct_missing_either_idx": "Rows missing either index (%)",
        }
    )

    dashboard_mapping_summary = outputs["mapping_summary"].copy()
    dashboard_mapping_summary["count"] = dashboard_mapping_summary["count"].map(format_int)
    dashboard_mapping_summary = dashboard_mapping_summary.rename(
        columns={
            "dataset": "Dataset",
            "mapping_type": "Mapping type",
            "count": "Count",
        }
    )

    return {
        "dashboard_split_summary": dashboard_split_summary,
        "dashboard_unseen_summary": dashboard_unseen_summary,
        "dashboard_mapping_summary": dashboard_mapping_summary,
    }


def build_report_tables(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build academic-report-friendly tables with clean labels and consistent rounding.
    """
    report_split_summary = outputs["all_split_summary"].copy()
    report_split_summary["dataset"] = report_split_summary["dataset"].replace(
        {
            "explicit": "Explicit modelling dataset",
            "implicit": "Implicit modelling dataset",
            "joined": "Joined interaction-recipe dataset",
        }
    )
    report_split_summary["split"] = report_split_summary["split"].str.replace("_", " ").str.title()

    for col in ["rows", "users", "recipes"]:
        report_split_summary[col] = report_split_summary[col].map(format_int)

    report_split_summary = report_split_summary.rename(
        columns={
            "dataset": "Dataset",
            "split": "Split",
            "split_policy": "Split policy",
            "rows": "Rows",
            "users": "Users",
            "recipes": "Recipes",
            "min_date": "Minimum date",
            "max_date": "Maximum date",
        }
    )

    report_unseen_summary = outputs["all_unseen_summary"].copy()
    report_unseen_summary["dataset"] = report_unseen_summary["dataset"].replace(
        {
            "explicit": "Explicit modelling dataset",
            "implicit": "Implicit modelling dataset",
        }
    )
    report_unseen_summary["split"] = report_unseen_summary["split"].str.replace("_", " ").str.title()

    for col in [
        "rows",
        "rows_missing_user_idx",
        "rows_missing_item_idx",
        "rows_missing_either_idx",
    ]:
        report_unseen_summary[col] = report_unseen_summary[col].map(format_int)

    report_unseen_summary["pct_missing_either_idx"] = report_unseen_summary[
        "pct_missing_either_idx"
    ].map(lambda x: f"{x:.2f}")

    report_unseen_summary = report_unseen_summary.rename(
        columns={
            "dataset": "Dataset",
            "split": "Split",
            "rows": "Rows",
            "rows_missing_user_idx": "Rows missing user index",
            "rows_missing_item_idx": "Rows missing item index",
            "rows_missing_either_idx": "Rows missing either index",
            "pct_missing_either_idx": "Rows missing either index (%)",
        }
    )

    report_mapping_summary = outputs["mapping_summary"].copy()
    report_mapping_summary["dataset"] = report_mapping_summary["dataset"].replace(
        {
            "explicit": "Explicit modelling dataset",
            "implicit": "Implicit modelling dataset",
        }
    )
    report_mapping_summary["mapping_type"] = report_mapping_summary["mapping_type"].replace(
        {
            "user_map": "User mapping table rows",
            "recipe_map": "Recipe mapping table rows",
        }
    )
    report_mapping_summary["count"] = report_mapping_summary["count"].map(format_int)
    report_mapping_summary = report_mapping_summary.rename(
        columns={
            "dataset": "Dataset",
            "mapping_type": "Mapping type",
            "count": "Count",
        }
    )

    return {
        "report_split_summary": report_split_summary,
        "report_unseen_summary": report_unseen_summary,
        "report_mapping_summary": report_mapping_summary,
    }


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


def plot_rows_by_split(all_split_summary: pd.DataFrame) -> None:
    """
    Plot split row counts for each dataset.
    """
    working = all_split_summary.copy()

    labels = working["dataset"] + " | " + working["split"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        labels,
        working["rows"],
        color=[PRIMARY_BLUE, PRIMARY_ORANGE, PRIMARY_TEAL] * (len(working) // 3),
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Rows Across Per-User Chronological Splits", fontsize=TITLE_SIZE)
    ax.set_xlabel("Dataset and split", fontsize=LABEL_SIZE)
    ax.set_ylabel("Row count", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", rotation=35)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "05_rows_across_splits")


def plot_entity_counts_by_split(all_split_summary: pd.DataFrame) -> None:
    """
    Plot user and recipe counts across splits.
    """
    working = all_split_summary.copy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(working))
    width = 0.38

    ax.bar(
        x - width / 2,
        working["users"],
        width=width,
        label="Users",
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        working["recipes"],
        width=width,
        label="Recipes",
        color=PRIMARY_ORANGE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(working["dataset"] + " | " + working["split"], rotation=35, ha="right")
    ax.set_title("Users and Recipes Across Per-User Splits", fontsize=TITLE_SIZE)
    ax.set_xlabel("Dataset and split", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, "05_entities_across_splits")


def plot_unseen_mapping_rates(all_unseen_summary: pd.DataFrame) -> None:
    """
    Plot unseen mapping rates for validation and test splits.
    """
    if all_unseen_summary.empty:
        return

    working = all_unseen_summary.copy()
    labels = working["dataset"] + " | " + working["split"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        labels,
        working["pct_missing_either_idx"],
        color=[PRIMARY_PURPLE, PRIMARY_TEAL, PRIMARY_PURPLE, PRIMARY_TEAL],
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Rows with Unseen User or Recipe IDs", fontsize=TITLE_SIZE)
    ax.set_xlabel("Dataset and split", fontsize=LABEL_SIZE)
    ax.set_ylabel("Rows missing either index (%)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", rotation=25)
    apply_axis_style(ax)
    add_bar_labels(ax, decimals=2, suffix="%")
    fig.tight_layout()
    save_figure(fig, "05_unseen_mapping_rates")


def plot_date_coverage(all_split_summary: pd.DataFrame) -> None:
    """
    Plot split start and end dates as a compact timeline-like chart.
    """
    working = all_split_summary.copy()
    working["min_date"] = pd.to_datetime(working["min_date"], errors="coerce")
    working["max_date"] = pd.to_datetime(working["max_date"], errors="coerce")
    working = working.dropna(subset=["min_date", "max_date"])

    if working.empty:
        return

    working = working.reset_index(drop=True)
    y = np.arange(len(working))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, row in working.iterrows():
        ax.hlines(
            y=i,
            xmin=row["min_date"],
            xmax=row["max_date"],
            color=PRIMARY_BLUE if row["dataset"] != "joined" else PRIMARY_ORANGE,
            linewidth=3,
        )
        ax.plot(row["min_date"], i, marker="o", color=PRIMARY_TEAL)
        ax.plot(row["max_date"], i, marker="o", color=PRIMARY_PURPLE)

    ax.set_yticks(y)
    ax.set_yticklabels(working["dataset"] + " | " + working["split"])
    ax.set_title("Date Coverage Across Per-User Temporal Splits", fontsize=TITLE_SIZE)
    ax.set_xlabel("Date", fontsize=LABEL_SIZE)
    ax.set_ylabel("Dataset and split", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, "05_date_coverage_across_splits")


def save_split_figures(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Generate accessible split figures and save them.
    """
    plot_rows_by_split(outputs["all_split_summary"])
    plot_entity_counts_by_split(outputs["all_split_summary"])
    plot_unseen_mapping_rates(outputs["all_unseen_summary"])
    plot_date_coverage(outputs["all_split_summary"])


def save_raw_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save raw summary tables for traceability.
    """
    outputs["explicit_split_summary"].to_csv(TABLES_DIR / "05_explicit_split_summary.csv", index=False)
    outputs["implicit_split_summary"].to_csv(TABLES_DIR / "05_implicit_split_summary.csv", index=False)
    outputs["joined_split_summary"].to_csv(TABLES_DIR / "05_joined_split_summary.csv", index=False)
    outputs["explicit_unseen_summary"].to_csv(TABLES_DIR / "05_explicit_unseen_mapping_summary.csv", index=False)
    outputs["implicit_unseen_summary"].to_csv(TABLES_DIR / "05_implicit_unseen_mapping_summary.csv", index=False)
    outputs["mapping_summary"].to_csv(TABLES_DIR / "05_mapping_summary.csv", index=False)
    outputs["all_split_summary"].to_csv(TABLES_DIR / "05_all_split_summary.csv", index=False)
    outputs["all_unseen_summary"].to_csv(TABLES_DIR / "05_all_unseen_summary.csv", index=False)
    outputs["user_history_summary"].to_csv(TABLES_DIR / "05_user_history_summary.csv", index=False)


def save_dashboard_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save compact dashboard-ready tables.
    """
    dashboard_tables = build_dashboard_tables(outputs)
    for name, df in dashboard_tables.items():
        df.to_csv(TABLES_DIR / f"05_{name}.csv", index=False)


def save_report_tables(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save cleaned academic-report-ready tables.
    """
    report_tables = build_report_tables(outputs)
    for name, df in report_tables.items():
        df.to_csv(TABLES_DIR / f"05_{name}.csv", index=False)


def save_logs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save both machine-readable and human-readable logs.
    """
    json_summary = {
        "split_policy": SPLIT_POLICY_NAME,
        "split_fractions": {
            "train_fraction": TRAIN_FRAC,
            "valid_fraction": VALID_FRAC,
            "test_fraction": TEST_FRAC,
        },
        "all_split_summary": outputs["all_split_summary"].to_dict(orient="records"),
        "all_unseen_summary": outputs["all_unseen_summary"].to_dict(orient="records"),
        "mapping_summary": outputs["mapping_summary"].to_dict(orient="records"),
        "user_history_summary": outputs["user_history_summary"].to_dict(orient="records"),
        "input_shapes": outputs["input_shapes"],
    }

    with open(LOGS_DIR / "05_chronological_split_report.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4, ensure_ascii=False)

    markdown_lines = [
        "# Per-User Chronological Split Summary",
        "",
        "## Split policy",
        f"- Policy: {SPLIT_POLICY_NAME}",
        "",
        "## Split proportions",
        f"- Train: {format_pct(TRAIN_FRAC * 100, 2)}",
        f"- Validation: {format_pct(VALID_FRAC * 100, 2)}",
        f"- Test: {format_pct(TEST_FRAC * 100, 2)}",
        "",
        "## Input shapes",
        f"- Explicit modelling dataset rows: {format_int(outputs['input_shapes']['explicit_rows'])}",
        f"- Implicit modelling dataset rows: {format_int(outputs['input_shapes']['implicit_rows'])}",
        f"- Joined modelling dataset rows: {format_int(outputs['input_shapes']['joined_rows'])}",
        "",
        "## Saved artefacts",
        f"- Splits directory: `{SPLITS_DIR}`",
        f"- Mapping directory: `{MAPPINGS_DIR}`",
        f"- Tables directory: `{TABLES_DIR}`",
        f"- Figures directory: `{FIGURES_DIR}`",
        f"- Logs directory: `{LOGS_DIR}`",
    ]

    with open(LOGS_DIR / "05_chronological_split_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def save_split_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Save all split artefacts.
    """
    save_raw_tables(outputs)
    save_dashboard_tables(outputs)
    save_report_tables(outputs)
    save_logs(outputs)
    save_split_figures(outputs)


def print_split_summary(outputs: dict[str, pd.DataFrame]) -> None:
    """
    Print a concise console summary for quick verification.
    """
    print("=" * 80)
    print(" PER-USER CHRONOLOGICAL SPLITS")
    print("=" * 80)

    print("\nAll split summary:")
    print(outputs["all_split_summary"].to_string(index=False))

    print("\nUnseen mapping summary:")
    print(outputs["all_unseen_summary"].to_string(index=False))

    print("\nMapping summary:")
    print(outputs["mapping_summary"].to_string(index=False))

    print(f"\nSaved split files to: {SPLITS_DIR}")
    print(f"Saved mapping tables to: {MAPPINGS_DIR}")
    print(f"Saved summary tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved logs to: {LOGS_DIR}")


def main() -> None:
    """
    Execute the chronological split pipeline.
    """
    ensure_directories()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    validate_split_fractions(TRAIN_FRAC, VALID_FRAC, TEST_FRAC)

    explicit_df = load_dataset(EXPLICIT_INPUT_NAME)
    implicit_df = load_dataset(IMPLICIT_INPUT_NAME)
    joined_df = load_dataset(JOINED_INPUT_NAME)

    input_shapes = {
        "explicit_rows": int(len(explicit_df)),
        "implicit_rows": int(len(implicit_df)),
        "joined_rows": int(len(joined_df)),
    }

    explicit_df = prepare_dataset_for_splitting(
        explicit_df,
        dataset_name="Explicit modelling dataset",
        required_columns=EXPLICIT_REQUIRED_COLS,
    )
    implicit_df = prepare_dataset_for_splitting(
        implicit_df,
        dataset_name="Implicit modelling dataset",
        required_columns=IMPLICIT_REQUIRED_COLS,
    )
    joined_df = prepare_dataset_for_splitting(
        joined_df,
        dataset_name="Joined modelling dataset",
        required_columns=JOINED_REQUIRED_COLS,
    )

    explicit_split = chronological_split(explicit_df, TRAIN_FRAC, VALID_FRAC, TEST_FRAC)
    implicit_split = chronological_split(implicit_df, TRAIN_FRAC, VALID_FRAC, TEST_FRAC)
    joined_split = chronological_split(joined_df, TRAIN_FRAC, VALID_FRAC, TEST_FRAC)

    explicit_user_map, explicit_recipe_map = build_id_maps_from_train(explicit_split.train)
    implicit_user_map, implicit_recipe_map = build_id_maps_from_train(implicit_split.train)

    explicit_train_mapped = apply_id_maps(explicit_split.train, explicit_user_map, explicit_recipe_map)
    explicit_valid_mapped = apply_id_maps(explicit_split.valid, explicit_user_map, explicit_recipe_map)
    explicit_test_mapped = apply_id_maps(explicit_split.test, explicit_user_map, explicit_recipe_map)

    implicit_train_mapped = apply_id_maps(implicit_split.train, implicit_user_map, implicit_recipe_map)
    implicit_valid_mapped = apply_id_maps(implicit_split.valid, implicit_user_map, implicit_recipe_map)
    implicit_test_mapped = apply_id_maps(implicit_split.test, implicit_user_map, implicit_recipe_map)

    explicit_split_summary = pd.DataFrame(
        [
            create_split_summary(explicit_train_mapped, "train", "explicit"),
            create_split_summary(explicit_valid_mapped, "valid", "explicit"),
            create_split_summary(explicit_test_mapped, "test", "explicit"),
        ]
    )

    implicit_split_summary = pd.DataFrame(
        [
            create_split_summary(implicit_train_mapped, "train", "implicit"),
            create_split_summary(implicit_valid_mapped, "valid", "implicit"),
            create_split_summary(implicit_test_mapped, "test", "implicit"),
        ]
    )

    joined_split_summary = pd.DataFrame(
        [
            create_split_summary(joined_split.train, "train", "joined"),
            create_split_summary(joined_split.valid, "valid", "joined"),
            create_split_summary(joined_split.test, "test", "joined"),
        ]
    )

    explicit_unseen_summary = pd.DataFrame(
        [
            create_unseen_mapping_summary(explicit_valid_mapped, "valid", "explicit"),
            create_unseen_mapping_summary(explicit_test_mapped, "test", "explicit"),
        ]
    )

    implicit_unseen_summary = pd.DataFrame(
        [
            create_unseen_mapping_summary(implicit_valid_mapped, "valid", "implicit"),
            create_unseen_mapping_summary(implicit_test_mapped, "test", "implicit"),
        ]
    )

    mapping_summary = pd.DataFrame(
        [
            {"dataset": "explicit", "mapping_type": "user_map", "count": int(len(explicit_user_map))},
            {"dataset": "explicit", "mapping_type": "recipe_map", "count": int(len(explicit_recipe_map))},
            {"dataset": "implicit", "mapping_type": "user_map", "count": int(len(implicit_user_map))},
            {"dataset": "implicit", "mapping_type": "recipe_map", "count": int(len(implicit_recipe_map))},
        ]
    )

    all_split_summary = pd.concat(
        [explicit_split_summary, implicit_split_summary, joined_split_summary],
        ignore_index=True,
    )

    all_unseen_summary = pd.concat(
        [explicit_unseen_summary, implicit_unseen_summary],
        ignore_index=True,
    )

    user_history_summary = pd.concat(
        [
            create_user_history_summary(explicit_split, "explicit"),
            create_user_history_summary(implicit_split, "implicit"),
            create_user_history_summary(joined_split, "joined"),
        ],
        ignore_index=True,
    )

    explicit_user_map_path = MAPPINGS_DIR / "05_explicit_user_id_map.csv"
    explicit_recipe_map_path = MAPPINGS_DIR / "05_explicit_recipe_id_map.csv"
    implicit_user_map_path = MAPPINGS_DIR / "05_implicit_user_id_map.csv"
    implicit_recipe_map_path = MAPPINGS_DIR / "05_implicit_recipe_id_map.csv"

    save_csv(explicit_user_map, explicit_user_map_path)
    save_csv(explicit_recipe_map, explicit_recipe_map_path)
    save_csv(implicit_user_map, implicit_user_map_path)
    save_csv(implicit_recipe_map, implicit_recipe_map_path)

    save_dataframe(explicit_train_mapped, SPLITS_DIR / "explicit_train.parquet")
    save_dataframe(explicit_valid_mapped, SPLITS_DIR / "explicit_valid.parquet")
    save_dataframe(explicit_test_mapped, SPLITS_DIR / "explicit_test.parquet")

    save_dataframe(implicit_train_mapped, SPLITS_DIR / "implicit_train.parquet")
    save_dataframe(implicit_valid_mapped, SPLITS_DIR / "implicit_valid.parquet")
    save_dataframe(implicit_test_mapped, SPLITS_DIR / "implicit_test.parquet")

    save_dataframe(joined_split.train, SPLITS_DIR / "joined_train.parquet")
    save_dataframe(joined_split.valid, SPLITS_DIR / "joined_valid.parquet")
    save_dataframe(joined_split.test, SPLITS_DIR / "joined_test.parquet")

    outputs = {
        "explicit_split_summary": explicit_split_summary,
        "implicit_split_summary": implicit_split_summary,
        "joined_split_summary": joined_split_summary,
        "explicit_unseen_summary": explicit_unseen_summary,
        "implicit_unseen_summary": implicit_unseen_summary,
        "mapping_summary": mapping_summary,
        "all_split_summary": all_split_summary,
        "all_unseen_summary": all_unseen_summary,
        "user_history_summary": user_history_summary,
        "input_shapes": input_shapes,
    }

    save_split_outputs(outputs)
    print_split_summary(outputs)


if __name__ == "__main__":
    main()