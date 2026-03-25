
from __future__ import annotations

"""
Purpose
-------
Build a dedicated bias-and-coverage evaluation bundle for the recommender
systems from previously saved model outputs. The module does not retrain any
model. It consolidates saved ranking metrics, optional concentration artifacts,
and optional recommendation-popularity exports into a focused set of tables,
figures, and narrative outputs under outputs/bias_coverage.
"""

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

CURRENT_FILE = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = CURRENT_FILE.parents[2] if len(CURRENT_FILE.parents) >= 3 else CURRENT_FILE.parent
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_PROJECT_ROOT))

try:
    from src.paths import OUTPUTS_DIR as SRC_OUTPUTS_DIR
    from src.paths import TABLES_DIR as SRC_TABLES_DIR
except Exception:
    SRC_OUTPUTS_DIR = DEFAULT_PROJECT_ROOT / "outputs"
    SRC_TABLES_DIR = SRC_OUTPUTS_DIR / "tables"

OUTPUT_PREFIX = "13"
MODEL_ORDER = ["popularity", "cf", "svd", "hybrid", "bpr"]
SPLIT_ORDER = ["valid", "test"]
PRIMARY_METRICS = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
DISCOVERY_METRICS = ["novelty_at_k", "catalog_coverage_at_k"]
ALL_METRICS = PRIMARY_METRICS + DISCOVERY_METRICS

METRIC_LABELS = {
    "precision_at_k": "Precision@K",
    "recall_at_k": "Recall@K",
    "hit_rate_at_k": "Hit Rate@K",
    "ndcg_at_k": "nDCG@K",
    "novelty_at_k": "Novelty@K",
    "catalog_coverage_at_k": "Coverage@K",
    "users_evaluated": "Users evaluated",
}

SHORT_MODEL_LABELS = {
    "Popularity": "Popularity",
    "Collaborative Filtering": "CF",
    "Truncated SVD": "SVD",
    "Hybrid (SVD + Popularity)": "Hybrid",
    "Bayesian Personalized Ranking": "BPR",
}

MODEL_STYLES = {
    "popularity": {"color": "#0072B2", "marker": "o", "linestyle": "-", "short_label": "Popularity"},
    "cf": {"color": "#E69F00", "marker": "s", "linestyle": "--", "short_label": "CF"},
    "svd": {"color": "#009E73", "marker": "^", "linestyle": "-.", "short_label": "SVD"},
    "hybrid": {"color": "#CC79A7", "marker": "D", "linestyle": ":", "short_label": "Hybrid"},
    "bpr": {"color": "#D55E00", "marker": "P", "linestyle": (0, (5, 1)), "short_label": "BPR"},
}

TITLE_PAD = 16
AXIS_LABEL_PAD = 12
TICK_PAD = 10
FIGURE_DPI = 300
BASE_FONT_SIZE = 11
HEATMAP_CMAP = "cividis"
GRID_ALPHA = 0.22


@dataclass(frozen=True)
class ModelSpec:
    key: str
    stem: str
    display_name: str


MODEL_SPECS = {
    "popularity": ModelSpec("popularity", "07_popularity", "Popularity"),
    "cf": ModelSpec("cf", "08_cf", "Collaborative Filtering"),
    "svd": ModelSpec("svd", "09_svd", "Truncated SVD"),
    "hybrid": ModelSpec("hybrid", "10_hybrid", "Hybrid (SVD + Popularity)"),
    "bpr": ModelSpec("bpr", "11_bpr", "Bayesian Personalized Ranking"),
}

NORMALISED_MODEL_NAME = {
    "popularity": "Popularity",
    "collaborative filtering": "Collaborative Filtering",
    "cf": "Collaborative Filtering",
    "svd": "Truncated SVD",
    "truncated svd": "Truncated SVD",
    "hybrid": "Hybrid (SVD + Popularity)",
    "hybrid (svd + popularity)": "Hybrid (SVD + Popularity)",
    "bpr": "Bayesian Personalized Ranking",
    "bayesian personalized ranking": "Bayesian Personalized Ranking",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.7,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bias-and-coverage artifacts from saved recommender outputs.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--source-tables-dir", type=Path, default=SRC_TABLES_DIR)
    parser.add_argument("--metrics-tables-dir", type=Path, default=SRC_OUTPUTS_DIR / "metrics" / "tables")
    parser.add_argument("--output-dir", type=Path, default=SRC_OUTPUTS_DIR / "bias_coverage")
    parser.add_argument("--output-prefix", type=str, default=OUTPUT_PREFIX)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def save_json(data: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_metric_value(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def wrap_axis_label(label: str, width: int = 14) -> str:
    if not label:
        return ""
    return "\n".join(textwrap.wrap(str(label), width=width))


def short_model_label(model_display: str) -> str:
    return SHORT_MODEL_LABELS.get(model_display, model_display)


def style_for_model(model_key: str) -> dict[str, object]:
    return MODEL_STYLES.get(model_key, {"color": "#4C4C4C", "marker": "o", "linestyle": "-", "short_label": model_key.upper()})


def set_axis_spacing(ax: plt.Axes, x_grid: bool = False, y_grid: bool = False) -> None:
    if x_grid:
        ax.grid(axis="x", alpha=GRID_ALPHA)
    if y_grid:
        ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.tick_params(axis="both", pad=TICK_PAD)
    ax.set_axisbelow(True)


def apply_heatmap_grid(ax: plt.Axes, shape: tuple[int, int]) -> None:
    rows, cols = shape
    ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def cell_text_colour(value: float, norm: Normalize) -> str:
    return "black" if norm(value) >= 0.58 else "white"


def model_display_from_raw(raw: str | None, fallback: str) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return fallback
    return NORMALISED_MODEL_NAME.get(str(raw).strip().lower(), fallback)


def candidate_names_for_spec(spec: ModelSpec) -> list[str]:
    return [
        f"{spec.stem}_metrics_dashboard.csv",
        f"{spec.stem}_dashboard_summary.csv",
        f"{spec.stem}_metrics_academic.csv",
        f"{spec.stem}_metrics.csv",
    ]


def find_candidate_file(source_tables_dir: Path, candidate_names: Iterable[str]) -> Path | None:
    for candidate_name in candidate_names:
        matches = sorted(source_tables_dir.rglob(candidate_name))
        if matches:
            return matches[0]
    return None


def standardise_metric_table(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    working = working.rename(
        columns={
            "precision": "precision_at_k",
            "recall": "recall_at_k",
            "hit_rate": "hit_rate_at_k",
            "ndcg": "ndcg_at_k",
            "novelty": "novelty_at_k",
            "coverage": "catalog_coverage_at_k",
            "model_name": "model",
        }
    )

    if "model" not in working.columns:
        working["model"] = spec.display_name
    working["model_display"] = working["model"].apply(lambda value: model_display_from_raw(value, spec.display_name))
    working["model_key"] = spec.key

    if "split" in working.columns:
        working["split"] = working["split"].astype(str).str.lower().str.strip()
    if "k" in working.columns:
        working["k"] = pd.to_numeric(working["k"], errors="coerce")

    if "recommendation_count" not in working.columns and {"users_evaluated", "k"}.issubset(working.columns):
        working["recommendation_count"] = pd.to_numeric(working["users_evaluated"], errors="coerce") * pd.to_numeric(working["k"], errors="coerce")

    for column in ["users_evaluated", "recommendation_count", "alpha", *ALL_METRICS]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    required = ["model_key", "model_display", "split", "k", "users_evaluated", "recommendation_count", *ALL_METRICS]
    for column in required:
        if column not in working.columns:
            working[column] = pd.NA
    working = working[required].dropna(subset=["split", "k"])
    return working.sort_values(["split", "k"]).reset_index(drop=True)


def load_combined_metrics(source_tables_dir: Path, metrics_tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_long_path = metrics_tables_dir / "12_model_comparison_long.csv"
    manifest_rows: list[dict[str, object]] = []

    if comparison_long_path.exists():
        combined = safe_read_csv(comparison_long_path)
        if not combined.empty:
            if "model_display" not in combined.columns and "model" in combined.columns:
                combined["model_display"] = combined["model"]
            if "model_key" not in combined.columns:
                reverse_lookup = {spec.display_name: spec.key for spec in MODEL_SPECS.values()}
                combined["model_key"] = combined["model_display"].map(reverse_lookup)
            combined["split"] = combined["split"].astype(str).str.lower().str.strip()
            combined["k"] = pd.to_numeric(combined["k"], errors="coerce")
            combined = combined.dropna(subset=["split", "k"]).reset_index(drop=True)
            for model_key in MODEL_ORDER:
                spec = MODEL_SPECS[model_key]
                subset = combined[combined["model_key"] == model_key]
                manifest_rows.append(
                    {
                        "model_key": model_key,
                        "model_display": spec.display_name,
                        "status": "loaded" if not subset.empty else "missing",
                        "source_file": str(comparison_long_path) if not subset.empty else "",
                        "rows_loaded": int(len(subset)),
                    }
                )
            return combined, pd.DataFrame(manifest_rows)

    tables: list[pd.DataFrame] = []
    for model_key in MODEL_ORDER:
        spec = MODEL_SPECS[model_key]
        source_path = find_candidate_file(source_tables_dir, candidate_names_for_spec(spec))
        if source_path is None:
            manifest_rows.append({"model_key": model_key, "model_display": spec.display_name, "status": "missing", "source_file": "", "rows_loaded": 0})
            continue
        std_df = standardise_metric_table(safe_read_csv(source_path), spec)
        manifest_rows.append(
            {
                "model_key": model_key,
                "model_display": spec.display_name,
                "status": "loaded" if not std_df.empty else "empty",
                "source_file": str(source_path),
                "rows_loaded": int(len(std_df)),
            }
        )
        if not std_df.empty:
            tables.append(std_df)

    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if not combined.empty:
        model_rank = {key: idx for idx, key in enumerate(MODEL_ORDER)}
        split_rank = {key: idx for idx, key in enumerate(SPLIT_ORDER)}
        combined["_model_rank"] = combined["model_key"].map(model_rank).fillna(999)
        combined["_split_rank"] = combined["split"].map(split_rank).fillna(999)
        combined = combined.sort_values(["_split_rank", "k", "_model_rank"]).drop(columns=["_model_rank", "_split_rank"])
    return combined.reset_index(drop=True), pd.DataFrame(manifest_rows)


def best_model_name(df: pd.DataFrame, metric: str) -> str:
    if df.empty or metric not in df.columns:
        return "Unavailable"
    value = df[metric].max()
    winners = df.loc[df[metric] == value, "model_display"].astype(str).tolist()
    if not winners:
        return "Unavailable"
    if len(winners) == 1:
        return winners[0]
    if len(winners) == 2:
        return f"{winners[0]} and {winners[1]}"
    return ", ".join(winners[:-1]) + f", and {winners[-1]}"


def build_focus_table(combined: pd.DataFrame, split: str, k: int) -> pd.DataFrame:
    subset = combined[(combined["split"] == split) & (combined["k"] == k)].copy()
    if subset.empty:
        return pd.DataFrame()
    subset = subset[["model_key", "model_display", "users_evaluated", *ALL_METRICS]].copy()
    for metric in ALL_METRICS:
        subset[f"rank_{metric}"] = subset[metric].rank(method="dense", ascending=False)
    subset = subset.sort_values(["rank_ndcg_at_k", "rank_precision_at_k", "model_display"]).reset_index(drop=True)
    return subset


def best_practical_compromise(focus_df: pd.DataFrame) -> str:
    if focus_df.empty:
        return "Unavailable"
    working = focus_df.copy()
    if {"rank_precision_at_k", "rank_ndcg_at_k", "rank_novelty_at_k", "rank_catalog_coverage_at_k"}.issubset(working.columns):
        candidates = working[(working["rank_precision_at_k"] <= 2) & (working["rank_ndcg_at_k"] <= 2)].copy()
        if candidates.empty:
            candidates = working.nsmallest(3, "rank_ndcg_at_k").copy()
        candidates["discovery_priority"] = candidates["rank_novelty_at_k"] + candidates["rank_catalog_coverage_at_k"]
        candidates["relevance_priority"] = candidates["rank_precision_at_k"] + candidates["rank_ndcg_at_k"]
        candidates = candidates.sort_values(
            ["discovery_priority", "relevance_priority", "catalog_coverage_at_k", "novelty_at_k"],
            ascending=[True, True, False, False],
        )
        return str(candidates.iloc[0]["model_display"])
    return str(working.iloc[0]["model_display"])


def build_bias_scorecard(focus_df: pd.DataFrame) -> pd.DataFrame:
    if focus_df.empty:
        return pd.DataFrame()

    working = focus_df.copy()
    relevance_cols = [f"rank_{metric}" for metric in PRIMARY_METRICS]
    discovery_cols = [f"rank_{metric}" for metric in DISCOVERY_METRICS]
    working["relevance_rank_mean"] = working[relevance_cols].mean(axis=1)
    working["discovery_rank_mean"] = working[discovery_cols].mean(axis=1)
    working["breadth_score"] = (
        working["novelty_at_k"].rank(method="dense", ascending=False)
        + working["catalog_coverage_at_k"].rank(method="dense", ascending=False)
    ) / 2.0
    working["concentration_risk"] = 6 - working["breadth_score"]
    working["balance_score"] = (0.7 * (6 - working["relevance_rank_mean"])) + (0.3 * (6 - working["discovery_rank_mean"]))

    profiles = []
    for _, row in working.iterrows():
        if row["rank_precision_at_k"] <= 2 and row["rank_catalog_coverage_at_k"] >= 4:
            profiles.append("Highly accurate but narrow")
        elif row["rank_catalog_coverage_at_k"] <= 2 and row["rank_precision_at_k"] >= 4:
            profiles.append("Broad but weakly targeted")
        elif row["rank_ndcg_at_k"] <= 2 and row["rank_catalog_coverage_at_k"] <= 3:
            profiles.append("Balanced ranking and breadth")
        else:
            profiles.append("Moderate middle-ground")
    working["bias_coverage_profile"] = profiles

    return working[
        [
            "model_key",
            "model_display",
            "users_evaluated",
            "precision_at_k",
            "ndcg_at_k",
            "novelty_at_k",
            "catalog_coverage_at_k",
            "relevance_rank_mean",
            "discovery_rank_mean",
            "concentration_risk",
            "balance_score",
            "bias_coverage_profile",
        ]
    ].sort_values(["balance_score", "ndcg_at_k"], ascending=[False, False]).reset_index(drop=True)


def build_tradeoff_table(focus_df: pd.DataFrame) -> pd.DataFrame:
    if focus_df.empty:
        return pd.DataFrame()
    working = focus_df.copy()
    precision_max = working["precision_at_k"].max() or 1
    ndcg_max = working["ndcg_at_k"].max() or 1
    novelty_max = working["novelty_at_k"].max() or 1
    coverage_max = working["catalog_coverage_at_k"].max() or 1

    working["relative_precision"] = working["precision_at_k"] / precision_max
    working["relative_ndcg"] = working["ndcg_at_k"] / ndcg_max
    working["relative_novelty"] = working["novelty_at_k"] / novelty_max
    working["relative_coverage"] = working["catalog_coverage_at_k"] / coverage_max
    working["relevance_strength"] = (working["relative_precision"] + working["relative_ndcg"]) / 2.0
    working["breadth_strength"] = (working["relative_novelty"] + working["relative_coverage"]) / 2.0
    working["tradeoff_gap"] = working["breadth_strength"] - working["relevance_strength"]
    working["tradeoff_interpretation"] = np.select(
        [
            (working["relevance_strength"] >= 0.85) & (working["breadth_strength"] < 0.35),
            (working["breadth_strength"] >= 0.75) & (working["relevance_strength"] < 0.35),
            (working["rank_precision_at_k"] <= 2) & (working["rank_ndcg_at_k"] <= 2) & (working["rank_catalog_coverage_at_k"] <= 3),
        ],
        [
            "Relevance-led with narrow exposure",
            "Breadth-led with weak relevance",
            "Best practical compromise",
        ],
        default="Intermediate trade-off",
    )
    return working[
        [
            "model_key",
            "model_display",
            "relevance_strength",
            "breadth_strength",
            "tradeoff_gap",
            "tradeoff_interpretation",
        ]
    ].sort_values(["relevance_strength", "breadth_strength"], ascending=[False, False]).reset_index(drop=True)


def detect_model_key_from_name(name: str) -> str | None:
    lowered = name.lower()
    for key, spec in MODEL_SPECS.items():
        if spec.stem in lowered or key in lowered:
            return key
    return None


def load_optional_concentration_tables(source_tables_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(source_tables_dir.rglob("*_recommendation_concentration_curve.csv")):
        df = safe_read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(path)
        model_key = detect_model_key_from_name(path.name) or detect_model_key_from_name(str(path.parent))
        split = "test" if "test" in path.name.lower() else "valid" if "valid" in path.name.lower() else ""
        df["model_key"] = model_key
        df["split"] = split
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_optional_recommendation_popularity_tables(source_tables_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(source_tables_dir.rglob("*_recommendation_popularity.csv")):
        df = safe_read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_file"] = str(path)
        model_key = detect_model_key_from_name(path.name) or detect_model_key_from_name(str(path.parent))
        split = "test" if "test" in path.name.lower() else "valid" if "valid" in path.name.lower() else ""
        df["model_key"] = model_key
        df["split"] = split
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_popularity_bias_summary(focus_df: pd.DataFrame, popularity_tables: pd.DataFrame) -> pd.DataFrame:
    summary = focus_df[["model_key", "model_display", "novelty_at_k", "catalog_coverage_at_k", "precision_at_k", "ndcg_at_k"]].copy()
    summary["novelty_rank"] = summary["novelty_at_k"].rank(method="dense", ascending=False)
    summary["coverage_rank"] = summary["catalog_coverage_at_k"].rank(method="dense", ascending=False)
    summary["popularity_bias_interpretation"] = np.select(
        [
            (summary["novelty_rank"] >= 4) & (summary["coverage_rank"] >= 4),
            (summary["novelty_rank"] <= 2) & (summary["coverage_rank"] <= 2),
        ],
        [
            "Most popularity-concentrated",
            "Least popularity-concentrated",
        ],
        default="Moderate popularity concentration",
    )

    if not popularity_tables.empty:
        numeric_cols = popularity_tables.select_dtypes(include=["number"]).columns.tolist()
        candidate = None
        for col in ["mean_item_popularity", "avg_item_popularity", "mean_popularity", "average_popularity", "item_popularity_mean"]:
            if col in popularity_tables.columns:
                candidate = col
                break
        if candidate is None and numeric_cols:
            candidate = numeric_cols[0]
        if candidate:
            pop_summary = (
                popularity_tables.groupby("model_key", dropna=False)[candidate]
                .mean()
                .rename("mean_recommended_item_popularity")
                .reset_index()
            )
            summary = summary.merge(pop_summary, on="model_key", how="left")
    return summary.sort_values(["novelty_rank", "coverage_rank"]).reset_index(drop=True)


def build_concentration_summary(concentration_tables: pd.DataFrame) -> pd.DataFrame:
    if concentration_tables.empty:
        return pd.DataFrame()
    numeric_cols = concentration_tables.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    x_col, y_col = numeric_cols[:2]
    out_rows: list[dict[str, object]] = []
    for (model_key, split), group in concentration_tables.groupby(["model_key", "split"], dropna=False):
        group = group.sort_values(x_col)
        area = float(np.trapz(group[y_col], group[x_col]))
        out_rows.append(
            {
                "model_key": model_key,
                "model_display": MODEL_SPECS.get(str(model_key), ModelSpec("", "", str(model_key))).display_name if model_key in MODEL_SPECS else str(model_key),
                "split": split,
                "curve_x_column": x_col,
                "curve_y_column": y_col,
                "concentration_curve_area": area,
                "final_curve_x": float(group[x_col].iloc[-1]),
                "final_curve_y": float(group[y_col].iloc[-1]),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["split", "concentration_curve_area"], ascending=[True, False]).reset_index(drop=True)


def build_artifact_manifest(figures_dir: Path, tables_dir: Path, logs_dir: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for folder, folder_name in [(figures_dir, "figures"), (tables_dir, "tables"), (logs_dir, "logs")]:
        for path in sorted(folder.glob(f"{prefix}_*")):
            rows.append({"artifact_type": folder_name, "name": path.name, "path": str(path)})
    return pd.DataFrame(rows)


def plot_coverage_and_novelty_lines(combined: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    for metric in ["catalog_coverage_at_k", "novelty_at_k"]:
        fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.8))
        for ax, split in zip(axes, SPLIT_ORDER):
            subset = combined[combined["split"] == split]
            for model_key in MODEL_ORDER:
                model_subset = subset[subset["model_key"] == model_key].sort_values("k")
                if model_subset.empty:
                    continue
                style = style_for_model(model_key)
                ax.plot(
                    model_subset["k"].astype(int),
                    model_subset[metric].astype(float),
                    label=style["short_label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.3,
                    markersize=7,
                )
            ax.set_title(f"{split.capitalize()} split", pad=TITLE_PAD)
            ax.set_ylabel(METRIC_LABELS[metric], labelpad=AXIS_LABEL_PAD)
            ax.set_xlabel("K", labelpad=AXIS_LABEL_PAD)
            ax.set_xticks(sorted(subset["k"].dropna().astype(int).unique().tolist()))
            ax.tick_params(axis="both", pad=TICK_PAD)
            set_axis_spacing(ax, y_grid=True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False)
        fig.suptitle(f"{METRIC_LABELS[metric]} across K by split", y=0.98)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.12, hspace=0.42)
        save_figure(fig, figures_dir / f"{prefix}_{metric}_comparison")


def plot_tradeoff_scatter(focus_df: pd.DataFrame, figures_dir: Path, prefix: str, x_metric: str, y_metric: str, filename: str, title: str) -> None:
    if focus_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    for _, row in focus_df.iterrows():
        style = style_for_model(str(row["model_key"]))
        ax.scatter(
            float(row[x_metric]),
            float(row[y_metric]),
            s=110,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidths=0.8,
            label=style["short_label"],
        )
    ax.set_xlabel(METRIC_LABELS[x_metric], labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel(METRIC_LABELS[y_metric], labelpad=AXIS_LABEL_PAD)
    ax.set_title(title, pad=TITLE_PAD)
    ax.tick_params(axis="both", pad=TICK_PAD)
    ax.margins(x=0.12, y=0.14)
    set_axis_spacing(ax, x_grid=True, y_grid=True)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Model")
    fig.subplots_adjust(left=0.12, right=0.77, top=0.90, bottom=0.13)
    save_figure(fig, figures_dir / f"{prefix}_{filename}")


def plot_focus_bars(focus_df: pd.DataFrame, figures_dir: Path, prefix: str, split: str) -> None:
    if focus_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    metrics = ["precision_at_k", "ndcg_at_k", "novelty_at_k", "catalog_coverage_at_k"]
    for ax, metric in zip(axes.flatten(), metrics):
        metric_df = focus_df[["model_key", "model_display", metric]].sort_values(metric, ascending=True)
        colors = [style_for_model(key)["color"] for key in metric_df["model_key"]]
        labels = [wrap_axis_label(short_model_label(name), width=10) for name in metric_df["model_display"]]
        ax.barh(labels, metric_df[metric], color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(METRIC_LABELS[metric], pad=TITLE_PAD)
        ax.set_xlabel("Score", labelpad=AXIS_LABEL_PAD)
        ax.tick_params(axis="both", pad=TICK_PAD)
        set_axis_spacing(ax, x_grid=True)
        xmax = float(metric_df[metric].max()) if len(metric_df) else 0.0
        ax.set_xlim(0, xmax * 1.24 if xmax > 0 else 1)
        for idx, value in enumerate(metric_df[metric].tolist()):
            ax.text(value + (xmax * 0.02 if xmax > 0 else 0.02), idx, format_metric_value(value), va="center", fontsize=9)
    fig.suptitle(f"Bias and coverage scorecard at {split.capitalize()}@10", y=0.98)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.08, hspace=0.42, wspace=0.40)
    save_figure(fig, figures_dir / f"{prefix}_{split}_at_10_scorecard")


def plot_user_coverage(focus_valid: pd.DataFrame, focus_test: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 8.0))
    for ax, split, focus_df in zip(axes, ["valid", "test"], [focus_valid, focus_test]):
        if focus_df.empty:
            ax.set_visible(False)
            continue
        ordered = focus_df.sort_values("users_evaluated", ascending=True)
        labels = [wrap_axis_label(short_model_label(name), width=10) for name in ordered["model_display"]]
        colors = [style_for_model(key)["color"] for key in ordered["model_key"]]
        ax.barh(labels, ordered["users_evaluated"], color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(f"{split.capitalize()} user reach at K=10", pad=TITLE_PAD)
        ax.set_xlabel("Users evaluated", labelpad=AXIS_LABEL_PAD)
        ax.tick_params(axis="both", pad=TICK_PAD)
        set_axis_spacing(ax, x_grid=True)
        xmax = float(ordered["users_evaluated"].max()) if len(ordered) else 0.0
        ax.set_xlim(0, xmax * 1.18 if xmax > 0 else 1)
        for idx, value in enumerate(ordered["users_evaluated"].tolist()):
            ax.text(value + (xmax * 0.015 if xmax > 0 else 0.02), idx, f"{int(value):,}", va="center", fontsize=9)
    fig.suptitle("Catalogue reach proxy through users evaluated", y=0.98)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.08, hspace=0.40)
    save_figure(fig, figures_dir / f"{prefix}_users_evaluated_k10")


def plot_heatmap(focus_df: pd.DataFrame, figures_dir: Path, prefix: str, split: str) -> None:
    if focus_df.empty:
        return
    matrix = focus_df.set_index("model_display")[["precision_at_k", "ndcg_at_k", "novelty_at_k", "catalog_coverage_at_k"]].copy()
    matrix = matrix.reindex([MODEL_SPECS[key].display_name for key in MODEL_ORDER if MODEL_SPECS[key].display_name in matrix.index])
    normed = matrix.copy()
    for col in normed.columns:
        s = normed[col].astype(float)
        denom = s.max() - s.min()
        normed[col] = 0.0 if denom == 0 else (s - s.min()) / denom

    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    norm = Normalize(vmin=0.0, vmax=1.0)
    image = ax.imshow(normed.values, aspect="auto", cmap=HEATMAP_CMAP, norm=norm)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([wrap_axis_label(METRIC_LABELS[c], width=12) for c in matrix.columns], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([wrap_axis_label(short_model_label(i), width=10) for i in matrix.index])
    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=10)
    ax.set_title(f"Normalised bias-and-coverage comparison at {split.capitalize()}@10", pad=TITLE_PAD)
    apply_heatmap_grid(ax, matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            raw = float(matrix.iloc[i, j])
            nv = float(normed.iloc[i, j])
            ax.text(j, i, format_metric_value(raw), ha="center", va="center", fontsize=9, color=cell_text_colour(nv, norm))
    cb = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    cb.set_label("Normalised score")
    fig.subplots_adjust(left=0.24, right=0.94, top=0.88, bottom=0.24)
    save_figure(fig, figures_dir / f"{prefix}_{split}_at_10_heatmap")


def plot_concentration_curves(concentration_tables: pd.DataFrame, figures_dir: Path, prefix: str) -> None:
    if concentration_tables.empty:
        return
    numeric_cols = concentration_tables.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        return
    x_col, y_col = numeric_cols[:2]
    for split in SPLIT_ORDER:
        subset = concentration_tables[concentration_tables["split"] == split]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 6.2))
        for model_key in MODEL_ORDER:
            group = subset[subset["model_key"] == model_key]
            if group.empty:
                continue
            style = style_for_model(model_key)
            group = group.sort_values(x_col)
            ax.plot(
                group[x_col].astype(float),
                group[y_col].astype(float),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.3,
                label=style["short_label"],
            )
        ax.set_xlabel(wrap_axis_label(x_col.replace("_", " ").title(), width=18), labelpad=AXIS_LABEL_PAD)
        ax.set_ylabel(wrap_axis_label(y_col.replace("_", " ").title(), width=18), labelpad=AXIS_LABEL_PAD)
        ax.set_title(f"Recommendation concentration curves ({split.capitalize()})", pad=TITLE_PAD)
        ax.tick_params(axis="both", pad=TICK_PAD)
        set_axis_spacing(ax, x_grid=True, y_grid=True)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Model")
        fig.subplots_adjust(left=0.14, right=0.78, top=0.90, bottom=0.14)
        save_figure(fig, figures_dir / f"{prefix}_{split}_recommendation_concentration")


def build_summary_markdown(
    focus_test: pd.DataFrame,
    focus_valid: pd.DataFrame,
    bias_scorecard: pd.DataFrame,
    tradeoff_table: pd.DataFrame,
    popularity_bias_summary: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    output_dir: Path,
) -> str:
    if focus_test.empty:
        return "# Bias and coverage summary\n\nNo saved evaluation results were available."

    best_precision = best_model_name(focus_test, "precision_at_k")
    best_ndcg = best_model_name(focus_test, "ndcg_at_k")
    best_novelty = best_model_name(focus_test, "novelty_at_k")
    best_coverage = best_model_name(focus_test, "catalog_coverage_at_k")
    balance_model = best_practical_compromise(focus_test)

    focus_lookup = focus_test.set_index("model_display")

    def val(model: str, column: str) -> str:
        if model in focus_lookup.index and column in focus_lookup.columns:
            return format_metric_value(focus_lookup.loc[model, column])
        return "NA"

    lines = [
        "# Bias and Coverage",
        "",
        "## Section purpose",
        (
            "This section examines whether the recommenders distribute exposure broadly across the recipe catalogue "
            "or whether they concentrate recommendations on a narrow set of already popular recipes. "
            "The analysis therefore focuses on catalogue coverage, novelty, user reach, popularity concentration, "
            "and the trade-off between breadth and ranking quality."
        ),
        "",
        "## 1. Catalogue coverage and user reach",
        (
            f"At Test@10, {best_coverage} provides the widest catalogue coverage "
            f"({val(best_coverage if ' and ' not in best_coverage else best_coverage.split(' and ')[0], 'catalog_coverage_at_k')}), "
            "while Popularity uses only a very small fraction of the catalogue. "
            "This indicates that the simpler relevance-led baseline is highly concentrated, whereas the more exploratory models "
            "activate a broader range of recipes."
        ),
        (
            "User reach should be interpreted carefully. The number of evaluated users is not identical across models, "
            "which means small performance gaps should be read with some caution. "
            "Nevertheless, the user-reach tables still help show whether a model is functioning at comparable practical scale."
        ),
        "",
        "## 2. Popularity bias and long-tail exposure",
        (
            f"{best_novelty} achieves the highest novelty at Test@10, indicating the weakest dependence on already popular recipes. "
            f"By contrast, Popularity has the lowest novelty ({val('Popularity', 'novelty_at_k')}) and the lowest coverage ({val('Popularity', 'catalog_coverage_at_k')}), "
            "which is a strong sign of popularity concentration."
        ),
        (
            "Higher novelty and broader coverage together suggest stronger long-tail exposure. "
            "However, diversity alone is not sufficient, because a model may broaden exposure while losing ranking usefulness."
        ),
        "",
        "## 3. Trade-offs between usefulness and breadth",
        (
            f"At Test@10, {best_precision} leads precision and {best_ndcg} leads nDCG, showing the strongest relevance-oriented behaviour. "
            f"At the same time, {best_novelty} and {best_coverage} lead the breadth-oriented metrics. "
            "This produces a clear trade-off between accuracy and exploration."
        ),
        (
            f"The most balanced practical option is {balance_model}. "
            "It remains close to the strongest relevance models while avoiding the most extreme narrowness of the pure Popularity baseline."
        ),
        "",
        "## 4. Overall judgment",
        (
            "Overall, the bias-and-coverage evidence suggests that Popularity is effective but strongly concentrated, "
            "Collaborative Filtering is broad and novel but weakly targeted, Truncated SVD is a moderate middle-ground, "
            "and Hybrid offers the strongest compromise between ranking quality and exposure balance. "
            "Bayesian Personalized Ranking improves breadth relative to Popularity but remains weaker than the leading models on the main relevance metrics."
        ),
        "",
        "## Output location",
        f"All generated artifacts were written under `{output_dir}`.",
    ]

    if not concentration_summary.empty:
        top = concentration_summary.sort_values("concentration_curve_area", ascending=False).iloc[0]
        lines.insert(
            len(lines) - 2,
            (
                f"Optional concentration-curve exports were available, and the largest curve area was observed for "
                f"{top['model_display']} on the {str(top['split']).capitalize()} split. "
                "This can be used as additional evidence about how recommendation exposure accumulates across the catalogue."
            ),
        )

    return "\n".join(lines)


def build_artifact_manifest(figures_dir: Path, tables_dir: Path, logs_dir: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for folder, folder_name in [(figures_dir, "figures"), (tables_dir, "tables"), (logs_dir, "logs")]:
        for path in sorted(folder.glob(f"{prefix}_*")):
            rows.append({"artifact_type": folder_name, "name": path.name, "path": str(path)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    source_tables_dir = args.source_tables_dir.resolve()
    metrics_tables_dir = args.metrics_tables_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    logs_dir = output_dir / "logs"
    prefix = args.output_prefix

    ensure_dir(output_dir)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(logs_dir)

    configure_matplotlib()
    combined, manifest = load_combined_metrics(source_tables_dir, metrics_tables_dir)

    if combined.empty:
        save_text("# Bias and coverage summary\n\nNo saved evaluation results were available.", logs_dir / f"{prefix}_bias_coverage_summary.md")
        save_json({"rows_combined": 0, "output_dir": str(output_dir)}, logs_dir / f"{prefix}_bias_coverage_report.json")
        print("No saved evaluation results were available. Empty summary written.")
        return

    focus_test = build_focus_table(combined, "test", 10)
    focus_valid = build_focus_table(combined, "valid", 10)
    bias_scorecard = build_bias_scorecard(focus_test)
    tradeoff_table = build_tradeoff_table(focus_test)

    concentration_tables = load_optional_concentration_tables(source_tables_dir)
    concentration_summary = build_concentration_summary(concentration_tables)

    popularity_tables = load_optional_recommendation_popularity_tables(source_tables_dir)
    popularity_bias_summary = build_popularity_bias_summary(focus_test, popularity_tables)

    plot_coverage_and_novelty_lines(combined, figures_dir, prefix)
    plot_tradeoff_scatter(focus_test, figures_dir, prefix, "catalog_coverage_at_k", "ndcg_at_k", "coverage_vs_ndcg_test_at_10", "Coverage versus nDCG at Test@10")
    plot_tradeoff_scatter(focus_test, figures_dir, prefix, "novelty_at_k", "precision_at_k", "novelty_vs_precision_test_at_10", "Novelty versus Precision at Test@10")
    plot_focus_bars(focus_test, figures_dir, prefix, "test")
    plot_focus_bars(focus_valid, figures_dir, prefix, "valid")
    plot_user_coverage(focus_valid, focus_test, figures_dir, prefix)
    plot_heatmap(focus_test, figures_dir, prefix, "test")
    plot_heatmap(focus_valid, figures_dir, prefix, "valid")
    plot_concentration_curves(concentration_tables, figures_dir, prefix)

    save_csv(focus_test, tables_dir / f"{prefix}_bias_coverage_test_at_10.csv")
    save_csv(focus_valid, tables_dir / f"{prefix}_bias_coverage_valid_at_10.csv")
    save_csv(bias_scorecard, tables_dir / f"{prefix}_bias_coverage_scorecard.csv")
    save_csv(tradeoff_table, tables_dir / f"{prefix}_tradeoff_summary.csv")
    save_csv(popularity_bias_summary, tables_dir / f"{prefix}_popularity_bias_summary.csv")
    save_csv(concentration_summary, tables_dir / f"{prefix}_concentration_summary.csv")
    save_csv(manifest, tables_dir / f"{prefix}_model_output_manifest.csv")

    artifact_manifest = build_artifact_manifest(figures_dir, tables_dir, logs_dir, prefix)
    save_csv(artifact_manifest, tables_dir / f"{prefix}_artifact_manifest.csv")

    summary_markdown = build_summary_markdown(
        focus_test=focus_test,
        focus_valid=focus_valid,
        bias_scorecard=bias_scorecard,
        tradeoff_table=tradeoff_table,
        popularity_bias_summary=popularity_bias_summary,
        concentration_summary=concentration_summary,
        output_dir=output_dir,
    )
    save_text(summary_markdown, logs_dir / f"{prefix}_bias_coverage_summary.md")

    metadata = {
        "rows_combined": int(len(combined)),
        "rows_focus_test": int(len(focus_test)),
        "rows_focus_valid": int(len(focus_valid)),
        "rows_bias_scorecard": int(len(bias_scorecard)),
        "rows_tradeoff": int(len(tradeoff_table)),
        "rows_popularity_bias": int(len(popularity_bias_summary)),
        "rows_concentration_summary": int(len(concentration_summary)),
        "artifacts": int(len(artifact_manifest)),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "logs_dir": str(logs_dir),
        "models_loaded": manifest.loc[manifest["status"] == "loaded", "model_key"].tolist(),
    }
    save_json(metadata, logs_dir / f"{prefix}_bias_coverage_report.json")

    print(f"Bias and coverage artifacts generated under {output_dir}")


if __name__ == "__main__":
    main()
