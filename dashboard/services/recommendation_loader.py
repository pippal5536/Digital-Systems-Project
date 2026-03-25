from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.data_loader import (
    load_popularity_metadata,
    load_recipe_id_map,
    load_recipes_metadata,
    load_user_id_map,
    read_csv_safe,
)
from utils.constants import MODEL_DISPLAY_NAMES, RECOMMENDATION_FILE_CANDIDATES
from utils.paths import tables_dir

USER_CANDIDATES = ["user_id", "user", "raw_user_id", "encoded_user_id", "user_idx"]
RECIPE_CANDIDATES = ["recipe_id", "item_id", "item", "raw_recipe_id", "recipe", "recipe_idx", "item_idx"]
SCORE_CANDIDATES = ["score", "prediction", "pred", "estimated_rating", "recommendation_score", "popularity_score"]
RANK_CANDIDATES = ["recommendation_rank", "rank", "position", "top_n_rank"]


@st.cache_data(show_spinner=False)
def available_models() -> list[str]:
    models: list[str] = []
    for model_key in RECOMMENDATION_FILE_CANDIDATES:
        if find_recommendation_file(model_key, "test") or find_recommendation_file(model_key, "valid"):
            models.append(model_key)
    return models


@st.cache_data(show_spinner=False)
def find_recommendation_file(model_key: str, split: str) -> Path | None:
    base = tables_dir()
    for relative_name in RECOMMENDATION_FILE_CANDIDATES.get(model_key, {}).get(split, []):
        candidate = base / relative_name
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_recommendations(model_key: str, split: str) -> pd.DataFrame:
    path = find_recommendation_file(model_key, split)
    if not path:
        return _empty_recommendation_frame()

    raw = read_csv_safe(path)
    if raw.empty:
        return _empty_recommendation_frame()

    standardized = _standardize_recommendation_frame(raw)
    if standardized.empty:
        return _empty_recommendation_frame()

    standardized["model_name"] = MODEL_DISPLAY_NAMES.get(model_key, model_key.upper())
    standardized["split"] = split

    standardized = _attach_recipe_metadata(standardized)
    standardized = _attach_popularity_metadata(standardized)

    standardized = standardized.sort_values(["user_id", "rank", "score"], ascending=[True, True, False]).reset_index(drop=True)
    return standardized


@st.cache_data(show_spinner=False)
def list_users_for_model(model_key: str, split: str) -> list[int | str]:
    df = load_recommendations(model_key, split)
    if df.empty or "user_id" not in df.columns:
        return []
    users = df["user_id"].dropna().drop_duplicates().tolist()
    try:
        return sorted(users)
    except Exception:
        return users


@st.cache_data(show_spinner=False)
def get_common_users(model_keys: tuple[str, ...], split: str) -> list[int | str]:
    if not model_keys:
        return []
    user_sets = []
    for model_key in model_keys:
        users = set(list_users_for_model(model_key, split))
        if not users:
            return []
        user_sets.append(users)
    common = set.intersection(*user_sets) if user_sets else set()
    try:
        return sorted(common)
    except Exception:
        return list(common)


@st.cache_data(show_spinner=False)
def get_top_n_recommendations(model_key: str, split: str, user_id: str, top_n: int) -> pd.DataFrame:
    df = load_recommendations(model_key, split)
    if df.empty:
        return df

    filtered = df[df["user_id"].astype(str) == str(user_id)].copy()
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values(["rank", "score"], ascending=[True, False])
    return filtered.head(top_n).reset_index(drop=True)


def _empty_recommendation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "user_id",
            "user_idx",
            "recipe_id",
            "item_idx",
            "score",
            "rank",
            "model_name",
            "split",
        ]
    )


def _standardize_recommendation_frame(df: pd.DataFrame) -> pd.DataFrame:
    lowered = {str(col).lower(): col for col in df.columns}

    user_col = _first_existing(lowered, USER_CANDIDATES)
    recipe_col = _first_existing(lowered, RECIPE_CANDIDATES)
    score_col = _first_existing(lowered, SCORE_CANDIDATES)
    rank_col = _first_existing(lowered, RANK_CANDIDATES)

    wide_recipe_columns = [col for col in df.columns if str(col).lower().startswith(("top_", "rec_", "recommendation_"))]
    if not recipe_col and wide_recipe_columns and user_col:
        melted = _melt_wide_recommendations(df, user_col, wide_recipe_columns)
        return _standardize_recommendation_frame(melted)

    standardized = pd.DataFrame()

    if user_col:
        standardized["raw_user_key"] = df[user_col]
    if recipe_col:
        standardized["raw_recipe_key"] = df[recipe_col]

    standardized["score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col else pd.NA
    standardized["rank"] = pd.to_numeric(df[rank_col], errors="coerce") if rank_col else pd.NA

    for optional_col in [
        "holdout_item_count",
        "route",
        "alpha",
        "train_interaction_count",
        "global_item_popularity_rank",
        "train_popularity_rank",
        "global_popularity_rank",
        "popularity_score",
    ]:
        if optional_col in df.columns:
            standardized[optional_col] = df[optional_col]

    if "user_id" in lowered:
        standardized["user_id"] = pd.to_numeric(df[lowered["user_id"]], errors="coerce")
    if "user_idx" in lowered:
        standardized["user_idx"] = pd.to_numeric(df[lowered["user_idx"]], errors="coerce")

    if "recipe_id" in lowered:
        standardized["recipe_id"] = pd.to_numeric(df[lowered["recipe_id"]], errors="coerce")
    if "item_idx" in lowered:
        standardized["item_idx"] = pd.to_numeric(df[lowered["item_idx"]], errors="coerce")

    if "user_id" not in standardized.columns and "raw_user_key" in standardized.columns:
        standardized["user_id"] = pd.to_numeric(standardized["raw_user_key"], errors="coerce")
    if "recipe_id" not in standardized.columns and "raw_recipe_key" in standardized.columns:
        standardized["recipe_id"] = pd.to_numeric(standardized["raw_recipe_key"], errors="coerce")

    standardized = _attach_id_mappings(standardized)

    standardized = standardized.dropna(subset=["user_id", "recipe_id"]).copy()
    standardized["user_id"] = standardized["user_id"].astype(int)
    standardized["recipe_id"] = standardized["recipe_id"].astype(int)

    if "user_idx" in standardized.columns:
        standardized["user_idx"] = pd.to_numeric(standardized["user_idx"], errors="coerce")
    if "item_idx" in standardized.columns:
        standardized["item_idx"] = pd.to_numeric(standardized["item_idx"], errors="coerce")

    standardized["score"] = pd.to_numeric(standardized["score"], errors="coerce")
    standardized["rank"] = pd.to_numeric(standardized["rank"], errors="coerce")

    # Remove exact duplicate user-recipe pairs, preferring:
    # - lower rank first if present
    # - higher score first otherwise
    standardized["_rank_missing"] = standardized["rank"].isna()
    standardized = standardized.sort_values(
        ["user_id", "recipe_id", "_rank_missing", "rank", "score"],
        ascending=[True, True, True, True, False],
        kind="stable",
    )
    standardized = standardized.drop_duplicates(subset=["user_id", "recipe_id"], keep="first")

    has_any_rank = standardized["rank"].notna().any()

    if has_any_rank:
        # Use provided rank where available; missing ranks go after ranked rows and are score-sorted
        standardized["_rank_missing"] = standardized["rank"].isna()
        standardized = standardized.sort_values(
            ["user_id", "_rank_missing", "rank", "score", "recipe_id"],
            ascending=[True, True, True, False, True],
            kind="stable",
        )
    else:
        # No usable rank provided -> derive from score descending
        standardized = standardized.sort_values(
            ["user_id", "score", "recipe_id"],
            ascending=[True, False, True],
            kind="stable",
        )

    # Rebuild rank sequentially after final per-user ordering
    standardized["rank"] = standardized.groupby("user_id").cumcount() + 1
    standardized["rank"] = standardized["rank"].astype(int)

    if standardized["score"].isna().all():
        standardized["score"] = 0.0

    keep_cols = [
        col
        for col in [
            "user_id",
            "user_idx",
            "recipe_id",
            "item_idx",
            "score",
            "rank",
            "holdout_item_count",
            "route",
            "alpha",
            "train_interaction_count",
            "global_item_popularity_rank",
            "train_popularity_rank",
            "global_popularity_rank",
            "popularity_score",
        ]
        if col in standardized.columns
    ]
    return standardized[keep_cols].copy()


def _attach_id_mappings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    user_map = load_user_id_map()
    if "user_id" in out.columns:
        out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    if "user_idx" in out.columns:
        out["user_idx"] = pd.to_numeric(out["user_idx"], errors="coerce")

    if ("user_id" not in out.columns or out["user_id"].isna().all()) and "user_idx" in out.columns and not user_map.empty:
        out = out.merge(user_map, on="user_idx", how="left")
    elif "user_idx" in out.columns and not user_map.empty and "user_id" in out.columns:
        missing_mask = out["user_id"].isna()
        if missing_mask.any():
            fill_map = user_map.set_index("user_idx")["user_id"]
            out.loc[missing_mask, "user_id"] = out.loc[missing_mask, "user_idx"].map(fill_map)

    recipe_map = load_recipe_id_map()
    if "recipe_id" in out.columns:
        out["recipe_id"] = pd.to_numeric(out["recipe_id"], errors="coerce")
    if "item_idx" in out.columns:
        out["item_idx"] = pd.to_numeric(out["item_idx"], errors="coerce")

    if ("recipe_id" not in out.columns or out["recipe_id"].isna().all()) and "item_idx" in out.columns and not recipe_map.empty:
        out = out.merge(recipe_map, on="item_idx", how="left")
    elif "item_idx" in out.columns and not recipe_map.empty and "recipe_id" in out.columns:
        missing_mask = out["recipe_id"].isna()
        if missing_mask.any():
            fill_map = recipe_map.set_index("item_idx")["recipe_id"]
            out.loc[missing_mask, "recipe_id"] = out.loc[missing_mask, "item_idx"].map(fill_map)

    return out


def _attach_recipe_metadata(df: pd.DataFrame) -> pd.DataFrame:
    recipes = load_recipes_metadata()
    if recipes.empty or "recipe_id" not in recipes.columns:
        return df
    join_cols = [col for col in ["recipe_id", "recipe_name", "minutes", "avg_rating", "n_ingredients", "calorie_level", "tag_count"] if col in recipes.columns]
    return df.merge(recipes[join_cols].drop_duplicates(subset=["recipe_id"]), on="recipe_id", how="left")


def _attach_popularity_metadata(df: pd.DataFrame) -> pd.DataFrame:
    popularity = load_popularity_metadata()
    if popularity.empty or "recipe_id" not in popularity.columns:
        return df

    join_cols = [col for col in ["recipe_id", "train_interaction_count", "popularity_rank", "popularity_score"] if col in popularity.columns]
    out = df.merge(popularity[join_cols], on="recipe_id", how="left", suffixes=("", "_joined"))

    fill_pairs = [
        ("train_interaction_count", "train_interaction_count_joined"),
        ("global_item_popularity_rank", "popularity_rank"),
        ("train_popularity_rank", "popularity_rank"),
    ]
    for target_col, source_col in fill_pairs:
        if source_col in out.columns:
            if target_col not in out.columns:
                out[target_col] = out[source_col]
            else:
                out[target_col] = out[target_col].fillna(out[source_col])

    joined_score_col = "popularity_score_joined"
    if joined_score_col in out.columns:
        if "popularity_score" not in out.columns:
            out["popularity_score"] = out[joined_score_col]
        else:
            out["popularity_score"] = out["popularity_score"].fillna(out[joined_score_col])

    drop_cols = [col for col in ["train_interaction_count_joined", "popularity_rank", "popularity_score_joined"] if col in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _melt_wide_recommendations(df: pd.DataFrame, user_col: str, recipe_cols: list[str]) -> pd.DataFrame:
    melted = df.melt(id_vars=[user_col], value_vars=recipe_cols, var_name="rank", value_name="recipe_id")
    melted["rank"] = melted["rank"].astype(str).str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    melted["score"] = pd.NA
    return melted


def _first_existing(lowered_column_map: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in lowered_column_map:
            return lowered_column_map[candidate]
    return None
