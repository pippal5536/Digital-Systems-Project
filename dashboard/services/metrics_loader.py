from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.data_loader import read_csv_safe
from utils.constants import MODEL_DISPLAY_NAMES, MODEL_FILE_PATTERNS
from utils.paths import tables_dir


@st.cache_data(show_spinner=False)
def load_model_metric_tables() -> dict[str, pd.DataFrame]:
    table_root = tables_dir()
    result: dict[str, pd.DataFrame] = {}
    for model_key, file_names in MODEL_FILE_PATTERNS.items():
        chosen = _find_first_existing(table_root, file_names)
        result[model_key] = read_csv_safe(chosen) if chosen else pd.DataFrame()
    return result


@st.cache_data(show_spinner=False)
def build_combined_metrics_long() -> pd.DataFrame:
    tables = load_model_metric_tables()
    rows: list[dict] = []

    for model_key, df in tables.items():
        if df.empty:
            continue

        model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key.upper())
        metric_columns = [col for col in df.columns if _looks_like_metric_name(str(col))]

        if {"metric", "value"}.issubset({str(c).lower() for c in df.columns}):
            rename_map = {}
            for col in df.columns:
                lower = str(col).lower()
                if lower == "metric":
                    rename_map[col] = "metric"
                elif lower == "value":
                    rename_map[col] = "value"
            normalized = df.rename(columns=rename_map)
            for _, row in normalized.iterrows():
                rows.append(
                    {
                        "model": model_name,
                        "metric": str(row.get("metric")),
                        "value": pd.to_numeric(row.get("value"), errors="coerce"),
                    }
                )
            continue

        if metric_columns:
            first_row = df.iloc[0]
            for metric_col in metric_columns:
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric_col,
                        "value": pd.to_numeric(first_row.get(metric_col), errors="coerce"),
                    }
                )
            continue

        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            first_row = df.iloc[0]
            for metric_col in numeric_cols:
                rows.append(
                    {
                        "model": model_name,
                        "metric": metric_col,
                        "value": pd.to_numeric(first_row.get(metric_col), errors="coerce"),
                    }
                )

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined
    combined = combined.dropna(subset=["value"]).sort_values(["metric", "model"]).reset_index(drop=True)
    return combined


@st.cache_data(show_spinner=False)
def build_combined_metrics_wide() -> pd.DataFrame:
    combined = build_combined_metrics_long()
    if combined.empty:
        return combined
    wide = combined.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    return wide.reset_index()


@st.cache_data(show_spinner=False)
def load_best_available_metric(metric_keywords: list[str]) -> pd.DataFrame:
    combined = build_combined_metrics_long()
    if combined.empty:
        return combined
    metric_keywords = [keyword.lower() for keyword in metric_keywords]
    filtered = combined[combined["metric"].astype(str).str.lower().apply(lambda x: any(k in x for k in metric_keywords))]
    return filtered.sort_values("value", ascending=False).reset_index(drop=True)


def _find_first_existing(base: Path, file_names: list[str]) -> Path | None:
    for file_name in file_names:
        candidate = base / file_name
        if candidate.exists():
            return candidate
    return None


def _looks_like_metric_name(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["precision", "recall", "ndcg", "map", "rmse", "coverage", "novelty", "hit"])
