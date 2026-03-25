from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from utils.constants import DATASET_FIGURES, DATASET_TABLES, POPULARITY_TABLE_CANDIDATES
from utils.paths import figures_dir, mappings_dir, processed_dir, tables_dir


@st.cache_data(show_spinner=False)
def read_csv_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_parquet_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_dataset_tables() -> dict[str, pd.DataFrame]:
    base = tables_dir()
    return {key: read_csv_safe(base / rel_path) for key, rel_path in DATASET_TABLES.items()}


@st.cache_data(show_spinner=False)
def load_dataset_figures() -> list[Path]:
    base = figures_dir()
    return [base / rel_path for rel_path in DATASET_FIGURES if (base / rel_path).exists()]


@st.cache_data(show_spinner=False)
def load_recipes_metadata() -> pd.DataFrame:
    recipes = read_parquet_safe(processed_dir() / "recipes_joined.parquet")
    if recipes.empty:
        return recipes

    rename_map = {}
    for candidate in ["id", "recipe_id", "recipeid"]:
        if candidate in recipes.columns:
            rename_map[candidate] = "recipe_id"
            break

    for candidate in ["name", "recipe_name", "title"]:
        if candidate in recipes.columns:
            rename_map[candidate] = "recipe_name"
            break

    recipes = recipes.rename(columns=rename_map)
    if "recipe_id" in recipes.columns:
        recipes["recipe_id"] = pd.to_numeric(recipes["recipe_id"], errors="coerce")
        recipes = recipes.dropna(subset=["recipe_id"]).copy()
        recipes["recipe_id"] = recipes["recipe_id"].astype(int)
    return recipes


@st.cache_data(show_spinner=False)
def load_popularity_metadata() -> pd.DataFrame:
    base = tables_dir()
    popularity = pd.DataFrame()
    for relative_path in POPULARITY_TABLE_CANDIDATES:
        candidate = base / relative_path
        if candidate.exists():
            popularity = read_csv_safe(candidate)
            if not popularity.empty:
                break

    if popularity.empty:
        return popularity

    rename_map = {}
    lowered = {str(col).lower(): col for col in popularity.columns}

    if "recipe_id" not in popularity.columns:
        for candidate in ["recipe_id", "item_id", "recipe"]:
            if candidate in lowered:
                rename_map[lowered[candidate]] = "recipe_id"
                break

    if "train_interaction_count" not in popularity.columns:
        for candidate in ["train_interaction_count", "interaction_count", "count"]:
            if candidate in lowered:
                rename_map[lowered[candidate]] = "train_interaction_count"
                break

    if "popularity_rank" not in popularity.columns:
        for candidate in ["popularity_rank", "train_popularity_rank", "global_popularity_rank", "item_popularity_rank"]:
            if candidate in lowered:
                rename_map[lowered[candidate]] = "popularity_rank"
                break

    popularity = popularity.rename(columns=rename_map)
    if "recipe_id" in popularity.columns:
        popularity["recipe_id"] = pd.to_numeric(popularity["recipe_id"], errors="coerce")
        popularity = popularity.dropna(subset=["recipe_id"]).copy()
        popularity["recipe_id"] = popularity["recipe_id"].astype(int)
        popularity = popularity.drop_duplicates(subset=["recipe_id"])

    return popularity


@st.cache_data(show_spinner=False)
def load_user_id_map() -> pd.DataFrame:
    user_map = read_csv_safe(mappings_dir() / "05_implicit_user_id_map.csv")
    if user_map.empty:
        return user_map
    expected = {"user_idx", "user_id"}
    if not expected.issubset(set(user_map.columns)):
        return pd.DataFrame()
    user_map["user_idx"] = pd.to_numeric(user_map["user_idx"], errors="coerce")
    user_map["user_id"] = pd.to_numeric(user_map["user_id"], errors="coerce")
    user_map = user_map.dropna(subset=["user_idx", "user_id"]).copy()
    user_map["user_idx"] = user_map["user_idx"].astype(int)
    user_map["user_id"] = user_map["user_id"].astype(int)
    return user_map.drop_duplicates(subset=["user_idx"])


@st.cache_data(show_spinner=False)
def load_recipe_id_map() -> pd.DataFrame:
    item_map = read_csv_safe(mappings_dir() / "05_implicit_recipe_id_map.csv")
    if item_map.empty:
        return item_map
    expected = {"item_idx", "recipe_id"}
    if not expected.issubset(set(item_map.columns)):
        return pd.DataFrame()
    item_map["item_idx"] = pd.to_numeric(item_map["item_idx"], errors="coerce")
    item_map["recipe_id"] = pd.to_numeric(item_map["recipe_id"], errors="coerce")
    item_map = item_map.dropna(subset=["item_idx", "recipe_id"]).copy()
    item_map["item_idx"] = item_map["item_idx"].astype(int)
    item_map["recipe_id"] = item_map["recipe_id"].astype(int)
    return item_map.drop_duplicates(subset=["item_idx"])


@st.cache_data(show_spinner=False)
def load_home_counts() -> dict[str, str]:
    tables = load_dataset_tables()
    stats = tables.get("dataset_stats", pd.DataFrame())
    modelling = tables.get("modelling_summary", pd.DataFrame())
    recipe_summary = tables.get("recipe_summary", pd.DataFrame())

    counts = {
        "users": _extract_value(stats, ["user", "users"]),
        "recipes": _extract_value(recipe_summary, ["recipe", "recipes", "items"]),
        "interactions": _extract_value(stats, ["interaction", "interactions"]),
        "explicit_rows": _extract_value(modelling, ["explicit"]),
        "implicit_rows": _extract_value(modelling, ["implicit"]),
    }
    return counts


@st.cache_data(show_spinner=False)
def load_generic_table(file_name: str) -> pd.DataFrame:
    return read_csv_safe(tables_dir() / file_name)


@st.cache_data(show_spinner=False)
def list_existing_tables(prefix: str | None = None) -> list[Path]:
    table_root = tables_dir()
    files = sorted(table_root.rglob("*.csv"))
    if prefix:
        files = [path for path in files if path.name.startswith(prefix)]
    return files


@st.cache_data(show_spinner=False)
def list_existing_figures(prefixes: tuple[str, ...] | None = None) -> list[Path]:
    figure_root = figures_dir()
    files = sorted(figure_root.rglob("*.png"))
    if prefixes:
        files = [path for path in files if path.name.startswith(prefixes)]
    return files


def _extract_value(df: pd.DataFrame, keywords: Iterable[str]) -> str:
    if df.empty:
        return "N/A"

    lowered_keywords = [keyword.lower() for keyword in keywords]

    for _, row in df.iterrows():
        row_text = " ".join(str(value).lower() for value in row.tolist())
        if any(keyword in row_text for keyword in lowered_keywords):
            for value in row.tolist()[::-1]:
                if isinstance(value, (int, float)) and not pd.isna(value):
                    return f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.2f}"
                if isinstance(value, str) and value.strip() and value.strip().replace(",", "").replace(".", "", 1).isdigit():
                    return value

    numeric_cols = df.select_dtypes(include=["number"])
    if not numeric_cols.empty:
        value = numeric_cols.iloc[0, 0]
        return f"{int(value):,}" if float(value).is_integer() else f"{float(value):,.2f}"

    return "N/A"
