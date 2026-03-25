"""
src/data/clean_recipes.py

Purpose:
Clean and preprocess the recipe metadata tables so that they can be used
in downstream feature engineering, modelling, and dashboard display.

This module formalises the recipe preprocessing logic previously explored
in the notebook "03 Recipe Preprocessing".

Responsibilities:
- load RAW_recipes.csv and PP_recipes.csv
- confirm recipe-table coverage assumptions
- retain the required recipe columns
- standardise recipe identifiers and essential numeric fields
- parse list-like columns safely
- clean basic text and date fields
- derive compact recipe features from readable metadata
- derive compact recipe features from tokenised metadata
- build simple tag indicators
- join RAW_recipes and PP_recipes using recipe_id
- check alignment with the cleaned interaction dataset
- save cleaned recipe outputs and summary tables
- save dashboard-ready and academic-report-ready tables
- save machine-readable and human-readable logs
- generate accessible static figures for longevity and reuse

Design notes:
- RAW_recipes.csv is treated as the base recipe metadata source because it
  preserves the full recipe catalogue, while PP_recipes.csv is used as an
  auxiliary enrichment source with incomplete coverage
- the recipe join is a left join from raw_recipes to pp_recipes on
  recipe_id so that no raw recipe rows are lost during enrichment
- figures use a colour-blind-safe palette and remain interpretable in grayscale
- both PNG and SVG outputs are saved for dashboard and academic reuse
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import (
    FIGURES_DIR,
    INTERIM_DIR,
    LOGS_DIR,
    PP_RECIPES_PATH,
    PROCESSED_DIR,
    RAW_RECIPES_PATH,
    TABLES_DIR,
    ensure_directories,
)


RAW_REQUIRED_COLUMNS = [
    "id",
    "name",
    "minutes",
    "contributor_id",
    "submitted",
    "tags",
    "nutrition",
    "n_steps",
    "steps",
    "description",
    "ingredients",
    "n_ingredients",
]

PP_REQUIRED_COLUMNS = [
    "id",
    "i",
    "name_tokens",
    "ingredient_tokens",
    "steps_tokens",
    "techniques",
    "calorie_level",
    "ingredient_ids",
]

RAW_LIST_COLUMNS = ["tags", "nutrition", "steps", "ingredients"]
PP_LIST_COLUMNS = [
    "name_tokens",
    "ingredient_tokens",
    "steps_tokens",
    "techniques",
    "ingredient_ids",
]

SELECTED_TAGS = [
    "breakfast",
    "lunch",
    "dinner",
    "dessert",
    "healthy",
    "easy",
    "vegetarian",
    "vegan",
]

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
class RecipeCleaningOutputs:
    """
    Container for cleaned recipe datasets and summary tables.
    """

    raw_recipe_clean: pd.DataFrame
    pp_recipe_clean: pd.DataFrame
    recipes_joined: pd.DataFrame
    recipe_table_comparison: pd.DataFrame
    recipe_join_coverage: pd.DataFrame
    recipes_join_summary: pd.DataFrame
    interaction_recipe_coverage: pd.DataFrame
    recipe_stats: pd.DataFrame
    compact_feature_nulls: pd.DataFrame
    tag_summary: pd.DataFrame
    calorie_level_distribution: pd.DataFrame
    recipe_year_counts: pd.DataFrame
    minutes_distribution: pd.DataFrame
    step_distribution: pd.DataFrame
    ingredient_distribution: pd.DataFrame


def format_int(value: float | int) -> str:
    """Format integer-like values with thousands separators."""
    return f"{int(value):,}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage values for display tables."""
    return f"{value:.{decimals}f}%"


def load_recipe_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw and preprocessed recipe datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Raw recipe dataframe and preprocessed recipe dataframe.
    """
    raw_recipes = pd.read_csv(RAW_RECIPES_PATH)
    pp_recipes = pd.read_csv(PP_RECIPES_PATH)
    return raw_recipes, pp_recipes


def confirm_recipe_table_assumptions(
    raw_recipes: pd.DataFrame,
    pp_recipes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconfirm recipe-table coverage and overlap assumptions.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Comparison table and coverage table.
    """
    raw_recipe_ids = set(raw_recipes["id"].dropna().unique())
    pp_recipe_ids = set(pp_recipes["id"].dropna().unique())

    recipe_table_comparison = pd.DataFrame(
        [
            {
                "table": "raw_recipes",
                "rows": int(len(raw_recipes)),
                "unique_recipe_ids": int(raw_recipes["id"].nunique()),
            },
            {
                "table": "pp_recipes",
                "rows": int(len(pp_recipes)),
                "unique_recipe_ids": int(pp_recipes["id"].nunique()),
            },
            {
                "table": "recipes_in_both_tables",
                "rows": int(len(raw_recipe_ids & pp_recipe_ids)),
                "unique_recipe_ids": int(len(raw_recipe_ids & pp_recipe_ids)),
            },
            {
                "table": "raw_only_recipe_ids",
                "rows": int(len(raw_recipe_ids - pp_recipe_ids)),
                "unique_recipe_ids": int(len(raw_recipe_ids - pp_recipe_ids)),
            },
        ]
    )

    recipe_join_coverage = pd.DataFrame(
        [
            {
                "comparison": "raw_recipes vs pp_recipes",
                "raw_recipe_ids": int(len(raw_recipe_ids)),
                "pp_recipe_ids": int(len(pp_recipe_ids)),
                "matched_ids": int(len(raw_recipe_ids & pp_recipe_ids)),
                "missing_pp_ids_from_raw": int(len(raw_recipe_ids - pp_recipe_ids)),
                "coverage_pct_of_raw": round(
                    len(raw_recipe_ids & pp_recipe_ids) / len(raw_recipe_ids) * 100,
                    4,
                ),
            }
        ]
    )

    return recipe_table_comparison, recipe_join_coverage


def select_required_recipe_columns(
    raw_recipes: pd.DataFrame,
    pp_recipes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain only the columns required for recipe preprocessing.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Working raw and preprocessed recipe dataframes.
    """
    raw_df = raw_recipes[RAW_REQUIRED_COLUMNS].copy()
    pp_df = pp_recipes[PP_REQUIRED_COLUMNS].copy()
    return raw_df, pp_df


def standardise_recipe_identifiers_and_numeric_fields(
    raw_df: pd.DataFrame,
    pp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardise recipe identifiers and key numeric fields.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Cleaned raw and preprocessed recipe dataframes.
    """
    raw_clean = raw_df.rename(columns={"id": "recipe_id"}).copy()
    pp_clean = pp_df.rename(columns={"id": "recipe_id"}).copy()

    for column in ["recipe_id", "minutes", "contributor_id", "n_steps", "n_ingredients"]:
        raw_clean[column] = pd.to_numeric(raw_clean[column], errors="coerce")

    for column in ["recipe_id", "i", "calorie_level"]:
        pp_clean[column] = pd.to_numeric(pp_clean[column], errors="coerce")

    raw_clean = raw_clean.dropna(subset=["recipe_id"]).copy()
    pp_clean = pp_clean.dropna(subset=["recipe_id"]).copy()

    raw_clean["recipe_id"] = raw_clean["recipe_id"].astype("int64")
    pp_clean["recipe_id"] = pp_clean["recipe_id"].astype("int64")

    for column in ["minutes", "contributor_id", "n_steps", "n_ingredients"]:
        raw_clean[column] = raw_clean[column].astype("Int64")

    for column in ["i", "calorie_level"]:
        pp_clean[column] = pp_clean[column].astype("Int64")

    return raw_clean, pp_clean


def parse_list_like(value):
    """
    Parse a Python-style list string into a Python object.

    Rules:
    - existing list values are returned as-is
    - missing values return pd.NA
    - malformed values return pd.NA
    """
    if pd.isna(value):
        return pd.NA

    if isinstance(value, list):
        return value

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return pd.NA
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return pd.NA

    return pd.NA


def parse_recipe_list_columns(
    raw_df: pd.DataFrame,
    pp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse list-like columns in the raw and preprocessed recipe tables.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Dataframes with parsed list-like columns.
    """
    raw_clean = raw_df.copy()
    pp_clean = pp_df.copy()

    for column in RAW_LIST_COLUMNS:
        raw_clean[column] = raw_clean[column].apply(parse_list_like)

    for column in PP_LIST_COLUMNS:
        pp_clean[column] = pp_clean[column].apply(parse_list_like)

    return raw_clean, pp_clean


def clean_raw_recipe_text_and_dates(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean recipe name, description, and submitted date fields.

    Returns:
        pd.DataFrame:
            Raw recipe dataframe with cleaned text and parsed date.
    """
    cleaned = raw_df.copy()

    cleaned["name"] = cleaned["name"].astype("string").str.strip()
    cleaned["name"] = cleaned["name"].replace("", pd.NA)

    cleaned["description"] = cleaned["description"].astype("string").str.strip()
    cleaned["description"] = cleaned["description"].replace("", pd.NA)

    cleaned["submitted"] = pd.to_datetime(cleaned["submitted"], errors="coerce")

    return cleaned


def safe_len(value):
    """
    Return the length of a list-like value, otherwise pd.NA.
    """
    if isinstance(value, list):
        return len(value)
    return pd.NA


def safe_numeric_list_stats(values):
    """
    Extract simple statistics from a numeric list.

    Returns:
        dict[str, object]:
            Dictionary containing length, sum, and mean.
    """
    if not isinstance(values, list) or len(values) == 0:
        return {
            "len": pd.NA,
            "sum": pd.NA,
            "mean": pd.NA,
        }

    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()

    if numeric_values.empty:
        return {
            "len": len(values),
            "sum": pd.NA,
            "mean": pd.NA,
        }

    return {
        "len": len(values),
        "sum": float(numeric_values.sum()),
        "mean": float(numeric_values.mean()),
    }


def derive_raw_recipe_compact_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive compact summary features from readable raw recipe metadata.

    Returns:
        pd.DataFrame:
            Raw recipe dataframe enriched with compact features.
    """
    enriched = raw_df.copy()

    enriched["name_length"] = enriched["name"].str.len().astype("Int64")
    enriched["description_length"] = enriched["description"].str.len().astype("Int64")

    enriched["tag_count"] = enriched["tags"].apply(safe_len).astype("Int64")
    enriched["step_count_from_list"] = enriched["steps"].apply(safe_len).astype("Int64")
    enriched["ingredient_count_from_list"] = enriched["ingredients"].apply(safe_len).astype("Int64")

    enriched["has_description"] = enriched["description"].notna().astype("int8")
    enriched["has_tags"] = enriched["tags"].apply(
        lambda value: isinstance(value, list) and len(value) > 0
    ).astype("int8")
    enriched["has_steps"] = enriched["steps"].apply(
        lambda value: isinstance(value, list) and len(value) > 0
    ).astype("int8")
    enriched["has_ingredients"] = enriched["ingredients"].apply(
        lambda value: isinstance(value, list) and len(value) > 0
    ).astype("int8")

    nutrition_stats = enriched["nutrition"].apply(safe_numeric_list_stats)
    enriched["nutrition_vector_length"] = nutrition_stats.apply(lambda value: value["len"]).astype("Int64")
    enriched["nutrition_sum"] = nutrition_stats.apply(lambda value: value["sum"])
    enriched["nutrition_mean"] = nutrition_stats.apply(lambda value: value["mean"])

    return enriched


def count_active_techniques(value):
    """
    Count the number of active techniques in the technique vector.
    """
    if not isinstance(value, list):
        return pd.NA

    numeric = pd.to_numeric(pd.Series(value), errors="coerce").fillna(0)
    return int((numeric > 0).sum())


def derive_pp_recipe_compact_features(pp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive compact summary features from tokenised PP recipe metadata.

    Returns:
        pd.DataFrame:
            PP recipe dataframe enriched with compact features.
    """
    enriched = pp_df.copy()

    enriched["name_token_count"] = enriched["name_tokens"].apply(safe_len).astype("Int64")
    enriched["ingredient_token_group_count"] = enriched["ingredient_tokens"].apply(safe_len).astype("Int64")
    enriched["step_token_count"] = enriched["steps_tokens"].apply(safe_len).astype("Int64")
    enriched["technique_vector_length"] = enriched["techniques"].apply(safe_len).astype("Int64")
    enriched["ingredient_id_count"] = enriched["ingredient_ids"].apply(safe_len).astype("Int64")
    enriched["technique_count_active"] = enriched["techniques"].apply(count_active_techniques).astype("Int64")
    enriched["has_pp_features"] = 1

    return enriched


def contains_tag(tag_list, target: str) -> int:
    """
    Return 1 if the target tag is present, otherwise 0.
    """
    if not isinstance(tag_list, list):
        return 0

    normalised = {str(value).strip().lower() for value in tag_list}
    return int(target.lower() in normalised)


def add_selected_tag_indicators(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple tag-based binary indicators to the raw recipe table.

    Returns:
        pd.DataFrame:
            Raw recipe dataframe with selected tag indicators added.
    """
    enriched = raw_df.copy()

    for tag in SELECTED_TAGS:
        enriched[f"tag_{tag}"] = enriched["tags"].apply(
            lambda values, target=tag: contains_tag(values, target)
        ).astype("int8")

    return enriched


def build_compact_cleaned_recipe_tables(
    raw_df: pd.DataFrame,
    pp_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain the main downstream fields in compact cleaned recipe tables.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Compact cleaned raw recipe table and compact cleaned PP recipe table.
    """
    raw_recipe_clean = raw_df[
        [
            "recipe_id",
            "name",
            "minutes",
            "contributor_id",
            "submitted",
            "description",
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
            "has_tags",
            "has_steps",
            "has_ingredients",
        ]
        + [f"tag_{tag}" for tag in SELECTED_TAGS]
    ].copy()

    pp_recipe_clean = pp_df[
        [
            "recipe_id",
            "i",
            "calorie_level",
            "name_token_count",
            "ingredient_token_group_count",
            "step_token_count",
            "technique_vector_length",
            "technique_count_active",
            "ingredient_id_count",
            "has_pp_features",
        ]
    ].copy()

    return raw_recipe_clean, pp_recipe_clean


def join_recipe_tables(
    raw_recipe_clean: pd.DataFrame,
    pp_recipe_clean: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Left join the raw and PP recipe tables on recipe_id.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            Joined recipe table and join summary table.
    """
    recipes_joined = raw_recipe_clean.merge(
        pp_recipe_clean,
        on="recipe_id",
        how="left",
        validate="one_to_one",
    )

    recipes_join_summary = pd.DataFrame(
        [
            {
                "raw_recipe_rows": int(len(raw_recipe_clean)),
                "pp_recipe_rows": int(len(pp_recipe_clean)),
                "joined_rows": int(len(recipes_joined)),
                "recipes_with_pp_features": int(recipes_joined["has_pp_features"].fillna(0).sum()),
                "recipes_without_pp_features": int(recipes_joined["has_pp_features"].isna().sum()),
                "pp_feature_coverage_pct": round(
                    recipes_joined["has_pp_features"].fillna(0).mean() * 100,
                    4,
                ),
            }
        ]
    )

    return recipes_joined, recipes_join_summary


def check_alignment_with_cleaned_interactions(
    recipes_joined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check how well the joined recipe table covers recipe IDs used in
    interactions_clean.parquet.

    Returns:
        pd.DataFrame:
            Interaction-to-recipe coverage table.
    """
    interactions_clean = pd.read_parquet(PROCESSED_DIR / "interactions_clean.parquet")

    interaction_recipe_ids = set(interactions_clean["recipe_id"].dropna().unique())
    joined_recipe_ids = set(recipes_joined["recipe_id"].dropna().unique())

    interaction_recipe_coverage = pd.DataFrame(
        [
            {
                "comparison": "interactions_clean vs recipes_joined",
                "unique_recipe_ids_in_interactions": int(len(interaction_recipe_ids)),
                "unique_recipe_ids_in_joined_recipes": int(len(joined_recipe_ids)),
                "matched_ids": int(len(interaction_recipe_ids & joined_recipe_ids)),
                "missing_ids": int(len(interaction_recipe_ids - joined_recipe_ids)),
                "coverage_pct": round(
                    len(interaction_recipe_ids & joined_recipe_ids) / len(interaction_recipe_ids) * 100,
                    4,
                ),
            }
        ]
    )

    return interaction_recipe_coverage


def build_recipe_stats(recipes_joined: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary statistics describing the cleaned recipe feature space.

    Returns:
        pd.DataFrame:
            Recipe statistics table.
    """
    return pd.DataFrame(
        [
            {"metric": "n_recipes_joined", "value": int(len(recipes_joined))},
            {"metric": "n_unique_recipe_ids", "value": int(recipes_joined["recipe_id"].nunique())},
            {
                "metric": "recipes_with_pp_features",
                "value": int(recipes_joined["has_pp_features"].fillna(0).sum()),
            },
            {
                "metric": "recipes_without_pp_features",
                "value": int(recipes_joined["has_pp_features"].isna().sum()),
            },
            {
                "metric": "median_minutes",
                "value": float(recipes_joined["minutes"].dropna().median()),
            },
            {
                "metric": "median_n_steps",
                "value": float(recipes_joined["n_steps"].dropna().median()),
            },
            {
                "metric": "median_n_ingredients",
                "value": float(recipes_joined["n_ingredients"].dropna().median()),
            },
        ]
    )


def build_compact_feature_nulls(recipes_joined: pd.DataFrame) -> pd.DataFrame:
    """
    Build null-count and null-percentage summary for joined recipe features.

    Returns:
        pd.DataFrame:
            Null summary table sorted by highest missingness.
    """
    return (
        pd.DataFrame(
            {
                "column": recipes_joined.columns,
                "null_count": recipes_joined.isna().sum().values,
                "null_percentage": (recipes_joined.isna().mean().values * 100).round(4),
            }
        )
        .sort_values(by=["null_count", "null_percentage"], ascending=False)
        .reset_index(drop=True)
    )


def build_tag_summary(recipes_joined: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary counts for selected tag indicators.

    Returns:
        pd.DataFrame:
            Tag count and percentage summary.
    """
    rows: list[dict[str, object]] = []
    total = len(recipes_joined)

    for tag in SELECTED_TAGS:
        column = f"tag_{tag}"
        count = int(recipes_joined[column].fillna(0).sum())
        rows.append(
            {
                "tag": tag,
                "recipe_count": count,
                "recipe_pct": round((count / total) * 100, 4) if total else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("recipe_count", ascending=False).reset_index(drop=True)


def build_calorie_level_distribution(pp_recipe_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build calorie level distribution from the PP recipe table.

    Returns:
        pd.DataFrame:
            Calorie level frequency table.
    """
    dist = (
        pp_recipe_clean["calorie_level"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("calorie_level")
        .reset_index(name="count")
    )
    dist["pct"] = (dist["count"] / len(pp_recipe_clean) * 100).round(4)
    return dist


def build_recipe_year_counts(raw_recipe_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build yearly counts from the submitted date column.

    Returns:
        pd.DataFrame:
            Yearly recipe submission counts.
    """
    working = raw_recipe_clean.dropna(subset=["submitted"]).copy()

    if working.empty:
        return pd.DataFrame(columns=["year", "recipe_count"])

    return (
        working["submitted"]
        .dt.year
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(name="recipe_count")
    )


def build_univariate_distribution(
    series: pd.Series,
    label: str,
) -> pd.DataFrame:
    """
    Build descriptive statistics for a numeric feature.

    Returns:
        pd.DataFrame:
            Summary statistics table.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return pd.DataFrame(
            [{"feature": label, "count": 0, "median": pd.NA, "mean": pd.NA, "p90": pd.NA}]
        )

    return pd.DataFrame(
        [
            {
                "feature": label,
                "count": int(numeric.shape[0]),
                "median": float(numeric.median()),
                "mean": float(numeric.mean()),
                "p90": float(numeric.quantile(0.90)),
            }
        ]
    )


def clean_recipes() -> RecipeCleaningOutputs:
    """
    Run the full recipe-cleaning pipeline in memory.

    Returns:
        RecipeCleaningOutputs:
            Container holding cleaned datasets and summary tables.
    """
    raw_recipes, pp_recipes = load_recipe_datasets()

    recipe_table_comparison, recipe_join_coverage = confirm_recipe_table_assumptions(
        raw_recipes=raw_recipes,
        pp_recipes=pp_recipes,
    )

    raw_df, pp_df = select_required_recipe_columns(raw_recipes, pp_recipes)
    raw_df, pp_df = standardise_recipe_identifiers_and_numeric_fields(raw_df, pp_df)
    raw_df, pp_df = parse_recipe_list_columns(raw_df, pp_df)
    raw_df = clean_raw_recipe_text_and_dates(raw_df)
    raw_df = derive_raw_recipe_compact_features(raw_df)
    pp_df = derive_pp_recipe_compact_features(pp_df)
    raw_df = add_selected_tag_indicators(raw_df)

    raw_recipe_clean, pp_recipe_clean = build_compact_cleaned_recipe_tables(raw_df, pp_df)
    recipes_joined, recipes_join_summary = join_recipe_tables(raw_recipe_clean, pp_recipe_clean)

    interaction_recipe_coverage = check_alignment_with_cleaned_interactions(recipes_joined)
    recipe_stats = build_recipe_stats(recipes_joined)
    compact_feature_nulls = build_compact_feature_nulls(recipes_joined)

    tag_summary = build_tag_summary(recipes_joined)
    calorie_level_distribution = build_calorie_level_distribution(pp_recipe_clean)
    recipe_year_counts = build_recipe_year_counts(raw_recipe_clean)
    minutes_distribution = build_univariate_distribution(recipes_joined["minutes"], "minutes")
    step_distribution = build_univariate_distribution(recipes_joined["n_steps"], "n_steps")
    ingredient_distribution = build_univariate_distribution(recipes_joined["n_ingredients"], "n_ingredients")

    return RecipeCleaningOutputs(
        raw_recipe_clean=raw_recipe_clean,
        pp_recipe_clean=pp_recipe_clean,
        recipes_joined=recipes_joined,
        recipe_table_comparison=recipe_table_comparison,
        recipe_join_coverage=recipe_join_coverage,
        recipes_join_summary=recipes_join_summary,
        interaction_recipe_coverage=interaction_recipe_coverage,
        recipe_stats=recipe_stats,
        compact_feature_nulls=compact_feature_nulls,
        tag_summary=tag_summary,
        calorie_level_distribution=calorie_level_distribution,
        recipe_year_counts=recipe_year_counts,
        minutes_distribution=minutes_distribution,
        step_distribution=step_distribution,
        ingredient_distribution=ingredient_distribution,
    )


def build_dashboard_tables(outputs: RecipeCleaningOutputs) -> dict[str, pd.DataFrame]:
    """
    Build compact, human-readable tables for dashboard usage.

    Returns:
        dict[str, pd.DataFrame]:
            Dashboard-ready tables.
    """
    stats_map = dict(zip(outputs.recipe_stats["metric"], outputs.recipe_stats["value"]))
    join_row = outputs.recipes_join_summary.iloc[0]
    coverage_row = outputs.interaction_recipe_coverage.iloc[0]

    dashboard_recipe_summary = pd.DataFrame(
        [
            {"Metric": "Recipes joined", "Value": format_int(stats_map["n_recipes_joined"])},
            {"Metric": "Unique recipe IDs", "Value": format_int(stats_map["n_unique_recipe_ids"])},
            {"Metric": "Recipes with PP features", "Value": format_int(stats_map["recipes_with_pp_features"])},
            {"Metric": "Recipes without PP features", "Value": format_int(stats_map["recipes_without_pp_features"])},
            {"Metric": "Median minutes", "Value": f"{stats_map['median_minutes']:.1f}"},
            {"Metric": "Median steps", "Value": f"{stats_map['median_n_steps']:.1f}"},
            {"Metric": "Median ingredients", "Value": f"{stats_map['median_n_ingredients']:.1f}"},
            {"Metric": "PP feature coverage", "Value": format_pct(join_row["pp_feature_coverage_pct"], 2)},
            {"Metric": "Interaction recipe coverage", "Value": format_pct(coverage_row["coverage_pct"], 2)},
        ]
    )

    dashboard_tag_summary = outputs.tag_summary.copy()
    dashboard_tag_summary["recipe_count"] = dashboard_tag_summary["recipe_count"].map(format_int)
    dashboard_tag_summary["recipe_pct"] = dashboard_tag_summary["recipe_pct"].map(lambda x: format_pct(x, 2))
    dashboard_tag_summary = dashboard_tag_summary.rename(
        columns={"tag": "Tag", "recipe_count": "Recipe count", "recipe_pct": "Recipe percentage"}
    )

    dashboard_recipe_join_coverage = outputs.recipe_join_coverage.copy()
    for col in ["raw_recipe_ids", "pp_recipe_ids", "matched_ids", "missing_pp_ids_from_raw"]:
        dashboard_recipe_join_coverage[col] = dashboard_recipe_join_coverage[col].map(format_int)
    dashboard_recipe_join_coverage["coverage_pct_of_raw"] = dashboard_recipe_join_coverage[
        "coverage_pct_of_raw"
    ].map(lambda x: format_pct(x, 2))
    dashboard_recipe_join_coverage = dashboard_recipe_join_coverage.rename(
        columns={
            "comparison": "Comparison",
            "raw_recipe_ids": "Raw recipe IDs",
            "pp_recipe_ids": "PP recipe IDs",
            "matched_ids": "Matched IDs",
            "missing_pp_ids_from_raw": "Missing PP IDs from raw",
            "coverage_pct_of_raw": "Coverage",
        }
    )

    dashboard_interaction_recipe_coverage = outputs.interaction_recipe_coverage.copy()
    for col in [
        "unique_recipe_ids_in_interactions",
        "unique_recipe_ids_in_joined_recipes",
        "matched_ids",
        "missing_ids",
    ]:
        dashboard_interaction_recipe_coverage[col] = dashboard_interaction_recipe_coverage[col].map(format_int)
    dashboard_interaction_recipe_coverage["coverage_pct"] = dashboard_interaction_recipe_coverage[
        "coverage_pct"
    ].map(lambda x: format_pct(x, 2))
    dashboard_interaction_recipe_coverage = dashboard_interaction_recipe_coverage.rename(
        columns={
            "comparison": "Comparison",
            "unique_recipe_ids_in_interactions": "Recipe IDs in interactions",
            "unique_recipe_ids_in_joined_recipes": "Recipe IDs in joined recipes",
            "matched_ids": "Matched IDs",
            "missing_ids": "Missing IDs",
            "coverage_pct": "Coverage",
        }
    )

    return {
        "dashboard_recipe_summary": dashboard_recipe_summary,
        "dashboard_tag_summary": dashboard_tag_summary,
        "dashboard_recipe_join_coverage": dashboard_recipe_join_coverage,
        "dashboard_interaction_recipe_coverage": dashboard_interaction_recipe_coverage,
    }


def build_report_tables(outputs: RecipeCleaningOutputs) -> dict[str, pd.DataFrame]:
    """
    Build academic-report-friendly tables with clean labels and consistent rounding.

    Returns:
        dict[str, pd.DataFrame]:
            Report-ready tables.
    """
    report_recipe_stats = outputs.recipe_stats.copy()
    report_recipe_stats["metric"] = report_recipe_stats["metric"].replace(
        {
            "n_recipes_joined": "Number of joined recipes",
            "n_unique_recipe_ids": "Number of unique recipe IDs",
            "recipes_with_pp_features": "Recipes with PP features",
            "recipes_without_pp_features": "Recipes without PP features",
            "median_minutes": "Median preparation time (minutes)",
            "median_n_steps": "Median number of steps",
            "median_n_ingredients": "Median number of ingredients",
        }
    )

    def _format_recipe_stat(row: pd.Series) -> str:
        if "Median" in row["metric"]:
            return f"{row['value']:.2f}"
        return format_int(row["value"])

    report_recipe_stats["value"] = report_recipe_stats.apply(_format_recipe_stat, axis=1)
    report_recipe_stats = report_recipe_stats.rename(columns={"metric": "Metric", "value": "Value"})

    report_recipe_join_coverage = outputs.recipe_join_coverage.copy()
    for col in ["raw_recipe_ids", "pp_recipe_ids", "matched_ids", "missing_pp_ids_from_raw"]:
        report_recipe_join_coverage[col] = report_recipe_join_coverage[col].map(format_int)
    report_recipe_join_coverage["coverage_pct_of_raw"] = report_recipe_join_coverage[
        "coverage_pct_of_raw"
    ].map(lambda x: f"{x:.2f}")
    report_recipe_join_coverage = report_recipe_join_coverage.rename(
        columns={
            "comparison": "Comparison",
            "raw_recipe_ids": "Raw recipe IDs",
            "pp_recipe_ids": "PP recipe IDs",
            "matched_ids": "Matched IDs",
            "missing_pp_ids_from_raw": "Missing PP IDs from raw",
            "coverage_pct_of_raw": "Coverage (%)",
        }
    )

    report_interaction_recipe_coverage = outputs.interaction_recipe_coverage.copy()
    for col in [
        "unique_recipe_ids_in_interactions",
        "unique_recipe_ids_in_joined_recipes",
        "matched_ids",
        "missing_ids",
    ]:
        report_interaction_recipe_coverage[col] = report_interaction_recipe_coverage[col].map(format_int)
    report_interaction_recipe_coverage["coverage_pct"] = report_interaction_recipe_coverage[
        "coverage_pct"
    ].map(lambda x: f"{x:.2f}")
    report_interaction_recipe_coverage = report_interaction_recipe_coverage.rename(
        columns={
            "comparison": "Comparison",
            "unique_recipe_ids_in_interactions": "Unique recipe IDs in interactions",
            "unique_recipe_ids_in_joined_recipes": "Unique recipe IDs in joined recipes",
            "matched_ids": "Matched IDs",
            "missing_ids": "Missing IDs",
            "coverage_pct": "Coverage (%)",
        }
    )

    report_tag_summary = outputs.tag_summary.copy()
    report_tag_summary["recipe_count"] = report_tag_summary["recipe_count"].map(format_int)
    report_tag_summary["recipe_pct"] = report_tag_summary["recipe_pct"].map(lambda x: f"{x:.2f}")
    report_tag_summary = report_tag_summary.rename(
        columns={
            "tag": "Tag",
            "recipe_count": "Recipe count",
            "recipe_pct": "Recipe percentage (%)",
        }
    )

    report_nulls = outputs.compact_feature_nulls.copy()
    report_nulls["null_count"] = report_nulls["null_count"].map(format_int)
    report_nulls["null_percentage"] = report_nulls["null_percentage"].map(lambda x: f"{x:.2f}")
    report_nulls = report_nulls.rename(
        columns={
            "column": "Column",
            "null_count": "Null count",
            "null_percentage": "Null percentage (%)",
        }
    )

    return {
        "report_recipe_stats": report_recipe_stats,
        "report_recipe_join_coverage": report_recipe_join_coverage,
        "report_interaction_recipe_coverage": report_interaction_recipe_coverage,
        "report_tag_summary": report_tag_summary,
        "report_recipe_feature_nulls": report_nulls,
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


def plot_recipe_join_coverage(recipe_join_coverage: pd.DataFrame) -> None:
    """
    Plot recipe table join coverage as a compact comparison bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        recipe_join_coverage["comparison"],
        recipe_join_coverage["coverage_pct_of_raw"],
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Recipe Table Join Coverage", fontsize=TITLE_SIZE)
    ax.set_xlabel("Comparison", fontsize=LABEL_SIZE)
    ax.set_ylabel("Coverage of raw recipes (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 105)
    apply_axis_style(ax)
    add_bar_labels(ax, decimals=2, suffix="%")
    fig.tight_layout()
    save_figure(fig, "03_recipe_join_coverage")


def plot_interaction_recipe_coverage(interaction_recipe_coverage: pd.DataFrame) -> None:
    """
    Plot interaction-to-recipe coverage as a compact bar chart.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        interaction_recipe_coverage["comparison"],
        interaction_recipe_coverage["coverage_pct"],
        color=PRIMARY_ORANGE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Interaction Recipe Coverage", fontsize=TITLE_SIZE)
    ax.set_xlabel("Comparison", fontsize=LABEL_SIZE)
    ax.set_ylabel("Coverage (%)", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 105)
    apply_axis_style(ax)
    add_bar_labels(ax, decimals=2, suffix="%")
    fig.tight_layout()
    save_figure(fig, "03_interaction_recipe_coverage")


def plot_tag_summary(tag_summary: pd.DataFrame) -> None:
    """
    Plot selected tag frequencies as an accessible bar chart.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        tag_summary["tag"],
        tag_summary["recipe_count"],
        color=PRIMARY_TEAL,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Selected Tag Frequency", fontsize=TITLE_SIZE)
    ax.set_xlabel("Tag", fontsize=LABEL_SIZE)
    ax.set_ylabel("Recipe count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "03_selected_tag_frequency")


def plot_calorie_level_distribution(calorie_level_distribution: pd.DataFrame) -> None:
    """
    Plot calorie level distribution from PP recipes.
    """
    working = calorie_level_distribution.dropna(subset=["calorie_level"]).copy()
    if working.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        working["calorie_level"].astype(str),
        working["count"],
        color=PRIMARY_PURPLE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title("Calorie Level Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel("Calorie level", fontsize=LABEL_SIZE)
    ax.set_ylabel("Recipe count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    add_bar_labels(ax)
    fig.tight_layout()
    save_figure(fig, "03_calorie_level_distribution")


def plot_recipe_submissions_by_year(recipe_year_counts: pd.DataFrame) -> None:
    """
    Plot yearly recipe submission counts as a line chart.
    """
    if recipe_year_counts.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        recipe_year_counts["year"],
        recipe_year_counts["recipe_count"],
        marker="o",
        linewidth=2.0,
        color=PRIMARY_TEAL,
    )
    ax.set_title("Recipe Submissions by Year", fontsize=TITLE_SIZE)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Recipe count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)

    for _, row in recipe_year_counts.iterrows():
        ax.annotate(
            f"{int(row['recipe_count']):,}",
            (row["year"], row["recipe_count"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=ANNOT_SIZE,
        )

    fig.tight_layout()
    save_figure(fig, "03_recipe_submissions_by_year")


def plot_missingness_overview(null_df: pd.DataFrame, top_n: int = 10) -> None:
    """
    Plot the highest-missing compact features as a horizontal bar chart.
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
    ax.set_title("Top Missing Recipe Features", fontsize=TITLE_SIZE)
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
    save_figure(fig, "03_recipe_feature_missingness")


def plot_numeric_distribution(series: pd.Series, title: str, xlabel: str, stem: str) -> None:
    """
    Plot a numeric feature distribution as a histogram.

    Histograms are kept single-series and high-contrast for accessibility.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        numeric,
        bins=30,
        color=PRIMARY_BLUE,
        edgecolor=NEUTRAL_GREY,
        linewidth=0.8,
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Recipe count", fontsize=LABEL_SIZE)
    apply_axis_style(ax)
    fig.tight_layout()
    save_figure(fig, stem)


def save_figures(outputs: RecipeCleaningOutputs) -> None:
    """
    Generate accessible figures and save them to the figures directory.
    """
    plot_recipe_join_coverage(outputs.recipe_join_coverage)
    plot_interaction_recipe_coverage(outputs.interaction_recipe_coverage)
    plot_tag_summary(outputs.tag_summary)
    plot_calorie_level_distribution(outputs.calorie_level_distribution)
    plot_recipe_submissions_by_year(outputs.recipe_year_counts)
    plot_missingness_overview(outputs.compact_feature_nulls)
    plot_numeric_distribution(
        outputs.recipes_joined["minutes"],
        title="Recipe Preparation Time Distribution",
        xlabel="Minutes",
        stem="03_recipe_minutes_distribution",
    )
    plot_numeric_distribution(
        outputs.recipes_joined["n_steps"],
        title="Recipe Step Count Distribution",
        xlabel="Number of steps",
        stem="03_recipe_step_distribution",
    )
    plot_numeric_distribution(
        outputs.recipes_joined["n_ingredients"],
        title="Recipe Ingredient Count Distribution",
        xlabel="Number of ingredients",
        stem="03_recipe_ingredient_distribution",
    )


def save_raw_tables(outputs: RecipeCleaningOutputs) -> None:
    """
    Save cleaned recipe datasets and raw summary tables.
    """
    outputs.raw_recipe_clean.to_parquet(
        INTERIM_DIR / "recipes_raw_clean.parquet",
        index=False,
    )
    outputs.pp_recipe_clean.to_parquet(
        INTERIM_DIR / "recipes_pp_clean.parquet",
        index=False,
    )
    outputs.recipes_joined.to_parquet(
        PROCESSED_DIR / "recipes_joined.parquet",
        index=False,
    )

    outputs.recipe_table_comparison.to_csv(
        TABLES_DIR / "03_recipe_table_comparison.csv",
        index=False,
    )
    outputs.recipe_join_coverage.to_csv(
        TABLES_DIR / "03_recipe_join_coverage.csv",
        index=False,
    )
    outputs.recipes_join_summary.to_csv(
        TABLES_DIR / "03_recipe_join_summary.csv",
        index=False,
    )
    outputs.interaction_recipe_coverage.to_csv(
        TABLES_DIR / "03_interaction_recipe_coverage.csv",
        index=False,
    )
    outputs.recipe_stats.to_csv(
        TABLES_DIR / "03_recipe_stats.csv",
        index=False,
    )
    outputs.compact_feature_nulls.to_csv(
        TABLES_DIR / "03_recipe_feature_nulls.csv",
        index=False,
    )
    outputs.tag_summary.to_csv(
        TABLES_DIR / "03_tag_summary.csv",
        index=False,
    )
    outputs.calorie_level_distribution.to_csv(
        TABLES_DIR / "03_calorie_level_distribution.csv",
        index=False,
    )
    outputs.recipe_year_counts.to_csv(
        TABLES_DIR / "03_recipe_year_counts.csv",
        index=False,
    )
    outputs.minutes_distribution.to_csv(
        TABLES_DIR / "03_minutes_distribution_summary.csv",
        index=False,
    )
    outputs.step_distribution.to_csv(
        TABLES_DIR / "03_step_distribution_summary.csv",
        index=False,
    )
    outputs.ingredient_distribution.to_csv(
        TABLES_DIR / "03_ingredient_distribution_summary.csv",
        index=False,
    )


def save_dashboard_tables(outputs: RecipeCleaningOutputs) -> None:
    """
    Save compact dashboard-ready tables.
    """
    dashboard_tables = build_dashboard_tables(outputs)
    for name, df in dashboard_tables.items():
        df.to_csv(TABLES_DIR / f"03_{name}.csv", index=False)


def save_report_tables(outputs: RecipeCleaningOutputs) -> None:
    """
    Save academic-report-ready tables.
    """
    report_tables = build_report_tables(outputs)
    for name, df in report_tables.items():
        df.to_csv(TABLES_DIR / f"03_{name}.csv", index=False)


def save_logs(outputs: RecipeCleaningOutputs) -> None:
    """
    Save both machine-readable and human-readable logs.
    """
    json_summary = {
        "recipe_table_comparison": outputs.recipe_table_comparison.to_dict(orient="records"),
        "recipe_join_coverage": outputs.recipe_join_coverage.to_dict(orient="records"),
        "recipes_join_summary": outputs.recipes_join_summary.to_dict(orient="records"),
        "interaction_recipe_coverage": outputs.interaction_recipe_coverage.to_dict(orient="records"),
        "recipe_stats": outputs.recipe_stats.to_dict(orient="records"),
        "compact_feature_nulls_top_20": outputs.compact_feature_nulls.head(20).to_dict(orient="records"),
        "tag_summary": outputs.tag_summary.to_dict(orient="records"),
        "calorie_level_distribution": outputs.calorie_level_distribution.to_dict(orient="records"),
        "recipe_year_counts": outputs.recipe_year_counts.to_dict(orient="records"),
    }

    with open(LOGS_DIR / "03_recipe_cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=4, ensure_ascii=False)

    stats_map = dict(zip(outputs.recipe_stats["metric"], outputs.recipe_stats["value"]))
    join_row = outputs.recipes_join_summary.iloc[0]
    coverage_row = outputs.interaction_recipe_coverage.iloc[0]

    markdown_lines = [
        "# Recipe Cleaning Summary",
        "",
        "## Recipe coverage",
        f"- Joined recipes: {format_int(stats_map['n_recipes_joined'])}",
        f"- Unique recipe IDs: {format_int(stats_map['n_unique_recipe_ids'])}",
        f"- Recipes with PP features: {format_int(stats_map['recipes_with_pp_features'])}",
        f"- Recipes without PP features: {format_int(stats_map['recipes_without_pp_features'])}",
        f"- PP feature coverage: {format_pct(join_row['pp_feature_coverage_pct'], 2)}",
        f"- Interaction recipe coverage: {format_pct(coverage_row['coverage_pct'], 2)}",
        "",
        "## Recipe feature medians",
        f"- Median minutes: {stats_map['median_minutes']:.2f}",
        f"- Median steps: {stats_map['median_n_steps']:.2f}",
        f"- Median ingredients: {stats_map['median_n_ingredients']:.2f}",
        "",
        "## Saved artefacts",
        f"- Tables directory: `{TABLES_DIR}`",
        f"- Figures directory: `{FIGURES_DIR}`",
        f"- Logs directory: `{LOGS_DIR}`",
        f"- Interim parquet directory: `{INTERIM_DIR}`",
        f"- Processed parquet directory: `{PROCESSED_DIR}`",
    ]

    with open(LOGS_DIR / "03_recipe_cleaning_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))


def save_outputs(outputs: RecipeCleaningOutputs) -> None:
    """
    Save cleaned recipe datasets, tables, logs, and figures.
    """
    save_raw_tables(outputs)
    save_dashboard_tables(outputs)
    save_report_tables(outputs)
    save_logs(outputs)
    save_figures(outputs)


def print_summary(outputs: RecipeCleaningOutputs) -> None:
    """
    Print a concise console summary for quick verification.
    """
    print("=" * 80)
    print(" RECIPE CLEANING")
    print("=" * 80)

    print("\nRecipe join summary:")
    print(outputs.recipes_join_summary.to_string(index=False))

    print("\nInteraction recipe coverage:")
    print(outputs.interaction_recipe_coverage.to_string(index=False))

    print("\nRecipe stats:")
    print(outputs.recipe_stats.to_string(index=False))

    print(f"\nSaved tables to: {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved logs to: {LOGS_DIR}")
    print(f"Saved interim parquet files to: {INTERIM_DIR}")
    print(f"Saved processed parquet files to: {PROCESSED_DIR}")


def main() -> None:
    """
    Execute the recipe-cleaning pipeline and save outputs.
    """
    ensure_directories()

    outputs = clean_recipes()
    save_outputs(outputs)
    print_summary(outputs)


if __name__ == "__main__":
    main()