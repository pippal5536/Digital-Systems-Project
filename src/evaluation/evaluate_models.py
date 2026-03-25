from __future__ import annotations

"""
Purpose
-------
Build consolidated evaluation artifacts for the recommender models from
existing saved model outputs. The module does not retrain any model. It reads
previously exported tables, standardises them into a shared schema, then writes
comparison tables, figures, and narrative summaries under the metrics folder.

Responsibilities
----------------
- Discover per-model evaluation tables under the existing source tables folder.
- Normalise inconsistent model names and metric column names.
- Build combined long-form and wide-form comparison tables.
- Rank models by metric for each split and K.
- Export report-oriented and academic summary tables.
- Produce comparison figures directly from saved outputs.
- Write a markdown summary describing the strongest observed models.

Design notes
------------
- No training is performed in this module.
- Existing exported CSV outputs are treated as the source of truth.
- Input and output locations are intentionally separated.
  The script reads source tables from outputs/tables by default and writes
  consolidated evaluation artifacts to outputs/metrics.
- Model discovery uses recursive search so that root-level tables and nested
  subdirectories such as outputs/tables/svd are both supported.
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


OUTPUT_PREFIX = "12"
MODEL_ORDER = ["popularity", "cf", "svd", "hybrid", "bpr"]
SPLIT_ORDER = ["valid", "test"]
DEFAULT_K_VALUES = [5, 10, 20]
PRIMARY_METRICS = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
SECONDARY_METRICS = ["novelty_at_k", "catalog_coverage_at_k"]
ALL_METRICS = PRIMARY_METRICS + SECONDARY_METRICS
METRIC_LABELS = {
    "precision_at_k": "Precision@K",
    "recall_at_k": "Recall@K",
    "hit_rate_at_k": "Hit Rate@K",
    "ndcg_at_k": "nDCG@K",
    "novelty_at_k": "Novelty@K",
    "catalog_coverage_at_k": "Coverage@K",
}


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

FIGURE_DPI = 300
BASE_FONT_SIZE = 11
TITLE_FONT_SIZE = 14
SUBTITLE_FONT_SIZE = 12
ANNOTATION_FONT_SIZE = 9
GRID_ALPHA = 0.22
TITLE_PAD = 14
AXIS_LABEL_PAD = 10
TICK_PAD = 8
HEATMAP_CMAP = "cividis"
SCATTER_LABEL_OFFSETS = {
    "popularity": (8, 8),
    "cf": (8, -12),
    "svd": (-44, 8),
    "hybrid": (-44, -12),
    "bpr": (10, 0),
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": SUBTITLE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": TITLE_FONT_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.7,
            "xtick.major.pad": TICK_PAD,
            "ytick.major.pad": TICK_PAD,
        }
    )


def wrap_axis_label(label: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width)) if label else ""


def short_model_label(model_display: str) -> str:
    return SHORT_MODEL_LABELS.get(model_display, model_display)


def style_for_model(model_key: str) -> dict[str, object]:
    return MODEL_STYLES.get(
        model_key,
        {"color": "#4C4C4C", "marker": "o", "linestyle": "-", "short_label": model_key.upper()},
    )


def set_clean_axis_style(ax: plt.Axes, y_grid: bool = False, x_grid: bool = False) -> None:
    if y_grid:
        ax.grid(axis="y", alpha=GRID_ALPHA)
    if x_grid:
        ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.set_axisbelow(True)


def apply_heatmap_grid(ax: plt.Axes, shape: tuple[int, int]) -> None:
    rows, cols = shape
    ax.set_xticks([x - 0.5 for x in range(1, cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, rows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def cell_text_colour(value: float, norm: Normalize) -> str:
    return "black" if norm(value) >= 0.58 else "white"


def nice_metric_ticklabels(metrics: list[str]) -> list[str]:
    return [wrap_axis_label(METRIC_LABELS.get(metric, metric), width=14) for metric in metrics]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison tables and figures from saved model outputs.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--source-tables-dir",
        type=Path,
        default=SRC_TABLES_DIR,
        help="Folder containing previously exported model result CSV files.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=SRC_OUTPUTS_DIR / "metrics",
        help="Root folder where consolidated evaluation tables, figures, and logs will be written.",
    )
    parser.add_argument("--output-prefix", type=str, default=OUTPUT_PREFIX)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_candidate_file(source_tables_dir: Path, candidate_names: Iterable[str]) -> Path | None:
    for candidate_name in candidate_names:
        matches = sorted(source_tables_dir.rglob(candidate_name))
        if matches:
            return matches[0]
    return None


def candidate_names_for_spec(spec: ModelSpec) -> list[str]:
    return [
        f"{spec.stem}_dashboard_summary.csv",
        f"{spec.stem}_metrics_dashboard.csv",
        f"{spec.stem}_metrics_academic.csv",
        f"{spec.stem}_metrics.csv",
    ]


def normalise_model_display(raw_value: str | None, spec: ModelSpec) -> str:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return spec.display_name
    value = str(raw_value).strip().lower()
    return NORMALISED_MODEL_NAME.get(value, spec.display_name)


def standardise_metric_table(df: pd.DataFrame, spec: ModelSpec, source_path: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    rename_map = {
        "precision": "precision_at_k",
        "recall": "recall_at_k",
        "hit_rate": "hit_rate_at_k",
        "ndcg": "ndcg_at_k",
        "novelty": "novelty_at_k",
        "coverage": "catalog_coverage_at_k",
        "model_name": "model",
    }
    working = working.rename(columns=rename_map)

    if "split" in working.columns:
        working["split"] = working["split"].astype(str).str.lower().str.strip()

    if "k" in working.columns:
        working["k"] = pd.to_numeric(working["k"], errors="coerce").astype("Int64")

    if "model" not in working.columns:
        working["model"] = spec.display_name

    working["model_display"] = working["model"].apply(lambda value: normalise_model_display(value, spec))
    working["model_key"] = spec.key

    if "split_k" not in working.columns and {"split", "k"}.issubset(working.columns):
        working["split_k"] = working.apply(
            lambda row: f"{str(row['split']).upper()}@{int(row['k'])}" if pd.notna(row["split"]) and pd.notna(row["k"]) else pd.NA,
            axis=1,
        )

    if "label" not in working.columns and {"split", "k"}.issubset(working.columns):
        working["label"] = working.apply(
            lambda row: f"{str(row['split']).upper()}_AT_{int(row['k'])}" if pd.notna(row["split"]) and pd.notna(row["k"]) else pd.NA,
            axis=1,
        )

    if "recommendation_count" not in working.columns and {"users_evaluated", "k"}.issubset(working.columns):
        users = pd.to_numeric(working.get("users_evaluated"), errors="coerce")
        ks = pd.to_numeric(working.get("k"), errors="coerce")
        working["recommendation_count"] = (users * ks).astype("Int64")

    if "alpha" not in working.columns:
        working["alpha"] = pd.NA

    numeric_columns = ["k", "users_evaluated", "recommendation_count", "alpha", *ALL_METRICS]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    required_columns = [
        "model_key",
        "model_display",
        "model",
        "split",
        "k",
        "split_k",
        "label",
        "alpha",
        "users_evaluated",
        "recommendation_count",
        *ALL_METRICS,
    ]
    for column in required_columns:
        if column not in working.columns:
            working[column] = pd.NA

    working["source_file"] = str(source_path)
    working["source_name"] = source_path.name
    working = working[required_columns + ["source_file", "source_name"]].copy()
    working = working.dropna(subset=["split", "k"], how="any")
    working = working.sort_values(["split", "k"]).reset_index(drop=True)
    return working


def load_all_model_tables(source_tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for model_key in MODEL_ORDER:
        spec = MODEL_SPECS[model_key]
        source_path = find_candidate_file(source_tables_dir, candidate_names_for_spec(spec))
        if source_path is None:
            manifest_rows.append(
                {
                    "model_key": spec.key,
                    "model_display": spec.display_name,
                    "source_file": pd.NA,
                    "status": "missing",
                    "rows_loaded": 0,
                }
            )
            continue

        raw_df = safe_read_csv(source_path)
        std_df = standardise_metric_table(raw_df, spec, source_path)
        status = "loaded" if not std_df.empty else "empty"
        manifest_rows.append(
            {
                "model_key": spec.key,
                "model_display": spec.display_name,
                "source_file": str(source_path),
                "status": status,
                "rows_loaded": len(std_df),
            }
        )
        if not std_df.empty:
            tables.append(std_df)

    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    return combined, manifest


def sort_combined(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return combined
    model_rank = {model_key: idx for idx, model_key in enumerate(MODEL_ORDER)}
    split_rank = {split: idx for idx, split in enumerate(SPLIT_ORDER)}
    ordered = combined.copy()
    ordered["_model_rank"] = ordered["model_key"].map(model_rank).fillna(999)
    ordered["_split_rank"] = ordered["split"].map(split_rank).fillna(999)
    ordered = ordered.sort_values(["_split_rank", "k", "_model_rank"]).drop(columns=["_model_rank", "_split_rank"])
    ordered = ordered.reset_index(drop=True)
    return ordered


def build_wide_table(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    columns = [
        "model_key",
        "model_display",
        "split",
        "k",
        "users_evaluated",
        "recommendation_count",
        *ALL_METRICS,
    ]
    return combined[columns].copy()


def build_metric_ranks(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    rank_rows: list[pd.DataFrame] = []
    for metric in ALL_METRICS:
        if metric not in combined.columns:
            continue
        subset = combined[["model_key", "model_display", "split", "k", metric]].copy()
        subset = subset.dropna(subset=[metric])
        if subset.empty:
            continue
        subset["metric"] = metric
        subset["metric_value"] = subset[metric]
        subset["rank"] = subset.groupby(["split", "k"])["metric_value"].rank(method="dense", ascending=False)
        rank_rows.append(subset[["model_key", "model_display", "split", "k", "metric", "metric_value", "rank"]])

    if not rank_rows:
        return pd.DataFrame()

    ranks = pd.concat(rank_rows, ignore_index=True)
    ranks = ranks.sort_values(["split", "k", "metric", "rank", "model_display"]).reset_index(drop=True)
    return ranks


def build_best_model_table(ranks: pd.DataFrame) -> pd.DataFrame:
    if ranks.empty:
        return pd.DataFrame()
    best = ranks.loc[ranks.groupby(["split", "k", "metric"])["rank"].idxmin()].copy()
    best = best.rename(columns={"metric_value": "best_value", "model_display": "best_model"})
    best = best[["split", "k", "metric", "best_model", "best_value", "rank"]]
    best = best.sort_values(["split", "k", "metric"]).reset_index(drop=True)
    return best


def build_report_focus_table(combined: pd.DataFrame, split: str = "test", k: int = 10) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    focus = combined[(combined["split"] == split) & (combined["k"] == k)].copy()
    if focus.empty:
        return pd.DataFrame()
    focus = focus[["model_key", "model_display", "users_evaluated", *ALL_METRICS]].copy()
    for metric in ALL_METRICS:
        focus[f"rank_{metric}"] = focus[metric].rank(method="dense", ascending=False)
    focus = focus.sort_values(["rank_ndcg_at_k", "rank_precision_at_k", "model_display"]).reset_index(drop=True)
    return focus


def build_academic_summary_table(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    academic = combined[["model_key", "model_display", "split", "k", "users_evaluated", *ALL_METRICS]].copy()
    academic = academic.sort_values(["split", "k", "model_display"]).reset_index(drop=True)
    return academic


def build_figure_manifest(figures_dir: Path, output_prefix: str) -> pd.DataFrame:
    figure_rows: list[dict[str, str]] = []
    for path in sorted(figures_dir.glob(f"{output_prefix}_*.png")):
        figure_rows.append(
            {
                "figure_file": path.name,
                "figure_path": str(path),
                "svg_pair": str(path.with_suffix(".svg")) if path.with_suffix(".svg").exists() else "",
            }
        )
    return pd.DataFrame(figure_rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def save_json(data: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def format_metric_value(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def build_summary_markdown(combined: pd.DataFrame, best: pd.DataFrame, manifest: pd.DataFrame, metrics_root: Path) -> str:
    if combined.empty:
        return "# Model evaluation summary\n\nNo combined evaluation data could be built from the saved outputs."

    def best_row(split: str, k: int, metric: str) -> tuple[str, str]:
        subset = best[(best["split"] == split) & (best["k"] == k) & (best["metric"] == metric)]
        if subset.empty:
            return "Unavailable", "NA"
        row = subset.iloc[0]
        return str(row["best_model"]), format_metric_value(row["best_value"])

    ndcg_model, ndcg_value = best_row("test", 10, "ndcg_at_k")
    precision_model, precision_value = best_row("test", 10, "precision_at_k")
    recall_model, recall_value = best_row("test", 10, "recall_at_k")
    novelty_model, novelty_value = best_row("test", 10, "novelty_at_k")
    coverage_model, coverage_value = best_row("test", 10, "catalog_coverage_at_k")

    loaded_models = manifest[manifest["status"] == "loaded"]["model_display"].tolist()
    missing_models = manifest[manifest["status"] != "loaded"]["model_display"].tolist()

    lines = [
        "# Model evaluation summary",
        "",
        "## Output location",
        f"All generated evaluation artifacts were written under: `{metrics_root}`.",
        "",
        "## Data availability",
        f"Loaded models: {', '.join(loaded_models) if loaded_models else 'None'}.",
        f"Missing or empty models: {', '.join(missing_models) if missing_models else 'None'}.",
        "",
        "## Academic interpretation",
        (
            "The comparison table was generated entirely from previously exported model outputs. "
            "No recommender was retrained during this stage. The strongest report-facing cut point is Test@10, "
            "because it balances ranking difficulty with practical recommendation list length."
        ),
        (
            f"At Test@10, the highest nDCG was achieved by {ndcg_model} ({ndcg_value}), "
            f"the highest precision by {precision_model} ({precision_value}), and the highest recall by {recall_model} ({recall_value})."
        ),
        (
            f"For beyond-accuracy behaviour, the highest novelty at Test@10 was observed for {novelty_model} ({novelty_value}), "
            f"while the highest catalog coverage at Test@10 was observed for {coverage_model} ({coverage_value})."
        ),
        (
            "This pattern should be interpreted as an accuracy-versus-discovery trade-off: "
            "some models optimise ranking relevance, whereas others distribute recommendations more broadly across the catalogue."
        ),
        "",
        "## Plain-language interpretation",
        (
            "The saved outputs show which model is best at giving useful recommendations and which model is best at variety. "
            "Higher precision, recall, hit rate, and nDCG indicate more useful recommendations. "
            "Higher novelty and coverage indicate less repetitive recommendations and broader catalogue use."
        ),
    ]
    return "\n".join(lines)



def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(path_without_suffix.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)



def plot_metric_panels(combined: pd.DataFrame, figures_dir: Path, output_prefix: str) -> None:
    if combined.empty:
        return

    for metric in ALL_METRICS:
        fig, axes = plt.subplots(2, 1, figsize=(10.8, 9.4), sharex=True)
        axes = list(axes)

        for axis, split in zip(axes, SPLIT_ORDER):
            subset = combined[combined["split"] == split]
            if subset.empty:
                axis.set_visible(False)
                continue

            available_ks = sorted(subset["k"].dropna().astype(int).unique().tolist())
            for model_key in MODEL_ORDER:
                model_subset = subset[subset["model_key"] == model_key].sort_values("k")
                if model_subset.empty or metric not in model_subset.columns:
                    continue

                style = style_for_model(model_key)
                label = short_model_label(str(model_subset["model_display"].iloc[0]))
                axis.plot(
                    model_subset["k"].astype(int),
                    model_subset[metric].astype(float),
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.2,
                    markersize=7,
                )

            axis.set_title(f"{split.capitalize()} split", pad=TITLE_PAD)
            axis.set_ylabel(METRIC_LABELS[metric], labelpad=AXIS_LABEL_PAD)
            axis.tick_params(axis="both", pad=TICK_PAD)
            axis.set_xticks(available_ks)
            axis.set_xlim(min(available_ks) - 0.5, max(available_ks) + 0.5)
            set_clean_axis_style(axis, y_grid=True)

        axes[-1].set_xlabel("Top-K recommendation list length", labelpad=AXIS_LABEL_PAD)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(5, len(labels)),
                frameon=False,
                title=None,
                columnspacing=1.8,
                handlelength=2.4,
                handletextpad=0.8,
                borderaxespad=0.6,
                labelspacing=1.0,
            )

        fig.suptitle(f"{METRIC_LABELS[metric]} across K by split", y=0.985)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.14, hspace=0.28)
        save_figure(fig, figures_dir / f"{output_prefix}_{metric}_comparison")


def plot_focus_metric_bars(combined: pd.DataFrame, figures_dir: Path, output_prefix: str, split: str, k: int) -> None:
    subset = combined[(combined["split"] == split) & (combined["k"] == k)].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(15.0, 12.8))
    axes_flat = axes.flatten()

    for axis, metric in zip(axes_flat, ALL_METRICS):
        metric_df = subset[["model_key", "model_display", metric]].dropna().sort_values(metric, ascending=True)
        colors = [style_for_model(model_key)["color"] for model_key in metric_df["model_key"]]
        labels = [wrap_axis_label(short_model_label(model_display), width=10) for model_display in metric_df["model_display"]]

        axis.barh(labels, metric_df[metric], color=colors, edgecolor="black", linewidth=0.6, height=0.72)
        axis.set_title(METRIC_LABELS[metric], pad=TITLE_PAD)
        axis.set_xlabel("Score", labelpad=AXIS_LABEL_PAD)
        axis.tick_params(axis="both", pad=TICK_PAD)
        set_clean_axis_style(axis, x_grid=True)

        x_max = float(metric_df[metric].max()) if len(metric_df) else 0.0
        axis.set_xlim(0, x_max * 1.22 if x_max > 0 else 1)
        for index, value in enumerate(metric_df[metric].tolist()):
            axis.text(value + (x_max * 0.02 if x_max > 0 else 0.02), index, format_metric_value(value), va="center", fontsize=9)

    fig.suptitle(f"Model leaderboard by metric at {split.capitalize()}@{k}", y=0.98)
    fig.tight_layout(rect=[0.03, 0.03, 0.98, 0.95])
    save_figure(fig, figures_dir / f"{output_prefix}_{split}_at_{k}_metric_bars")


def plot_metric_rank_heatmap(ranks: pd.DataFrame, figures_dir: Path, output_prefix: str, split: str, k: int) -> None:
    subset = ranks[(ranks["split"] == split) & (ranks["k"] == k)].copy()
    if subset.empty:
        return

    pivot = subset.pivot(index="model_display", columns="metric", values="rank")
    ordered_columns = [metric for metric in ALL_METRICS if metric in pivot.columns]
    pivot = pivot.reindex(columns=ordered_columns)
    pivot = pivot.reindex([MODEL_SPECS[key].display_name for key in MODEL_ORDER if MODEL_SPECS[key].display_name in pivot.index])

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    max_rank = float(pivot.max().max()) if not pivot.empty else 1.0
    norm = Normalize(vmin=1, vmax=max(max_rank, 1.0))
    image = ax.imshow(pivot.values, aspect="auto", cmap=HEATMAP_CMAP, norm=norm)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(nice_metric_ticklabels(list(pivot.columns)), rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([wrap_axis_label(short_model_label(label), width=12) for label in pivot.index])
    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=10)
    ax.set_title(f"Metric rank heatmap at {split.capitalize()}@{k} – lower rank is better", pad=TITLE_PAD)
    apply_heatmap_grid(ax, pivot.shape)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=10, color=cell_text_colour(value, norm))

    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
    colorbar.set_label("Rank position")
    fig.subplots_adjust(left=0.24, right=0.94, top=0.88, bottom=0.24)
    save_figure(fig, figures_dir / f"{output_prefix}_{split}_at_{k}_rank_heatmap")


def plot_test_at_10_scatter(combined: pd.DataFrame, figures_dir: Path, output_prefix: str) -> None:
    subset = combined[(combined["split"] == "test") & (combined["k"] == 10)].copy()
    if subset.empty:
        return

    scatter_specs = [
        ("catalog_coverage_at_k", "ndcg_at_k", "Coverage versus nDCG at Test@10", "coverage_vs_ndcg_test_at_10"),
        ("novelty_at_k", "precision_at_k", "Novelty versus precision at Test@10", "novelty_vs_precision_test_at_10"),
    ]

    for x_metric, y_metric, title, filename in scatter_specs:
        fig, ax = plt.subplots(figsize=(9.8, 6.8))

        for _, row in subset.iterrows():
            style = style_for_model(str(row["model_key"]))
            x_value = float(row[x_metric])
            y_value = float(row[y_metric])
            ax.scatter(
                x_value,
                y_value,
                s=105,
                color=style["color"],
                marker=style["marker"],
                edgecolors="black",
                linewidths=0.8,
                label=short_model_label(str(row["model_display"])),
                zorder=3,
            )

        ax.set_xlabel(METRIC_LABELS[x_metric], labelpad=AXIS_LABEL_PAD)
        ax.set_ylabel(METRIC_LABELS[y_metric], labelpad=AXIS_LABEL_PAD)
        ax.set_title(title, pad=TITLE_PAD)
        ax.tick_params(axis="both", pad=TICK_PAD)
        set_clean_axis_style(ax, x_grid=True, y_grid=True)
        ax.margins(x=0.12, y=0.12)

        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(
            dedup.values(),
            dedup.keys(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title="Model",
        )

        fig.subplots_adjust(left=0.12, right=0.76, top=0.90, bottom=0.12)
        save_figure(fig, figures_dir / f"{output_prefix}_{filename}")


def plot_focus_heatmap(combined: pd.DataFrame, figures_dir: Path, output_prefix: str, split: str, k: int) -> None:
    subset = combined[(combined["split"] == split) & (combined["k"] == k)].copy()
    if subset.empty:
        return

    display_metrics = [metric for metric in ALL_METRICS if metric in subset.columns]
    matrix = subset.set_index("model_display")[display_metrics].copy()
    matrix = matrix.reindex([MODEL_SPECS[key].display_name for key in MODEL_ORDER if MODEL_SPECS[key].display_name in matrix.index])

    normed = matrix.copy()
    for column in normed.columns:
        col = normed[column].astype(float)
        denominator = col.max() - col.min()
        normed[column] = 0.0 if denominator == 0 else (col - col.min()) / denominator

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    norm = Normalize(vmin=0.0, vmax=1.0)
    image = ax.imshow(normed.values, aspect="auto", cmap=HEATMAP_CMAP, norm=norm)

    ax.set_xticks(range(len(display_metrics)))
    ax.set_xticklabels(nice_metric_ticklabels(display_metrics), rotation=32, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([wrap_axis_label(short_model_label(label), width=12) for label in matrix.index])
    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=10)
    ax.set_title(f"Normalised model comparison at {split.capitalize()}@{k}", pad=TITLE_PAD)
    apply_heatmap_grid(ax, matrix.shape)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            raw_value = float(matrix.iloc[i, j])
            norm_value = float(normed.iloc[i, j])
            ax.text(
                j,
                i,
                format_metric_value(raw_value),
                ha="center",
                va="center",
                fontsize=9,
                color=cell_text_colour(norm_value, norm),
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.02)
    colorbar.set_label("Normalised score")
    fig.subplots_adjust(left=0.24, right=0.94, top=0.88, bottom=0.24)
    save_figure(fig, figures_dir / f"{output_prefix}_{split}_at_{k}_heatmap")


def plot_users_evaluated(combined: pd.DataFrame, figures_dir: Path, output_prefix: str) -> None:
    subset = combined[combined["k"] == 10].copy()
    if subset.empty or "users_evaluated" not in subset.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.8))
    axes = list(axes)
    for axis, split in zip(axes, SPLIT_ORDER):
        split_df = subset[subset["split"] == split].copy()
        if split_df.empty:
            axis.set_visible(False)
            continue

        split_df = split_df.sort_values("users_evaluated", ascending=True)
        colors = [style_for_model(model_key)["color"] for model_key in split_df["model_key"]]
        labels = [wrap_axis_label(short_model_label(model_display), width=10) for model_display in split_df["model_display"]]

        axis.barh(labels, split_df["users_evaluated"], color=colors, edgecolor="black", linewidth=0.6, height=0.72)
        axis.set_title(f"{split.capitalize()} users evaluated at K=10", pad=TITLE_PAD)
        axis.set_xlabel("Users evaluated", labelpad=AXIS_LABEL_PAD)
        axis.tick_params(axis="both", pad=TICK_PAD)
        set_clean_axis_style(axis, x_grid=True)

        x_max = float(split_df["users_evaluated"].max()) if len(split_df) else 0.0
        axis.set_xlim(0, x_max * 1.16 if x_max > 0 else 1)
        for index, value in enumerate(split_df["users_evaluated"].tolist()):
            axis.text(value + (x_max * 0.015 if x_max > 0 else 0.02), index, f"{int(value):,}", va="center", fontsize=9)

    fig.suptitle("Users evaluated by split at K=10", y=0.985)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.08, hspace=0.38)
    save_figure(fig, figures_dir / f"{output_prefix}_users_evaluated_k10")

def plot_all_figures(combined: pd.DataFrame, ranks: pd.DataFrame, figures_dir: Path, output_prefix: str) -> None:
    if combined.empty:
        return
    configure_matplotlib()
    plot_metric_panels(combined, figures_dir, output_prefix)
    plot_focus_metric_bars(combined, figures_dir, output_prefix, split="valid", k=10)
    plot_focus_metric_bars(combined, figures_dir, output_prefix, split="test", k=10)
    plot_metric_rank_heatmap(ranks, figures_dir, output_prefix, split="valid", k=10)
    plot_metric_rank_heatmap(ranks, figures_dir, output_prefix, split="test", k=10)
    plot_focus_heatmap(combined, figures_dir, output_prefix, split="valid", k=10)
    plot_focus_heatmap(combined, figures_dir, output_prefix, split="test", k=10)
    plot_test_at_10_scatter(combined, figures_dir, output_prefix)
    plot_users_evaluated(combined, figures_dir, output_prefix)


def write_outputs(
    combined: pd.DataFrame,
    manifest: pd.DataFrame,
    wide: pd.DataFrame,
    ranks: pd.DataFrame,
    best: pd.DataFrame,
    report_test_at_10: pd.DataFrame,
    report_valid_at_10: pd.DataFrame,
    academic_summary: pd.DataFrame,
    figure_manifest: pd.DataFrame,
    tables_dir: Path,
    logs_dir: Path,
    metrics_root: Path,
    output_prefix: str,
) -> None:
    save_csv(combined, tables_dir / f"{output_prefix}_model_comparison_long.csv")
    save_csv(wide, tables_dir / f"{output_prefix}_model_comparison_wide.csv")
    save_csv(ranks, tables_dir / f"{output_prefix}_model_metric_ranks.csv")
    save_csv(best, tables_dir / f"{output_prefix}_best_model_by_metric.csv")
    save_csv(report_test_at_10, tables_dir / f"{output_prefix}_report_model_comparison_test_at_10.csv")
    save_csv(report_valid_at_10, tables_dir / f"{output_prefix}_report_model_comparison_valid_at_10.csv")
    save_csv(academic_summary, tables_dir / f"{output_prefix}_academic_model_comparison.csv")
    save_csv(manifest, tables_dir / f"{output_prefix}_model_output_manifest.csv")
    save_csv(figure_manifest, tables_dir / f"{output_prefix}_figure_manifest.csv")

    summary_markdown = build_summary_markdown(combined, best, manifest, metrics_root)
    (logs_dir / f"{output_prefix}_model_evaluation_summary.md").write_text(summary_markdown, encoding="utf-8")

    metadata = {
        "rows_combined": int(len(combined)),
        "rows_ranks": int(len(ranks)),
        "rows_best": int(len(best)),
        "rows_report_test_at_10": int(len(report_test_at_10)),
        "rows_report_valid_at_10": int(len(report_valid_at_10)),
        "rows_academic_summary": int(len(academic_summary)),
        "figures_generated": int(len(figure_manifest)),
        "metrics_root": str(metrics_root),
        "models_loaded": manifest[manifest["status"] == "loaded"]["model_key"].tolist(),
        "models_missing": manifest[manifest["status"] != "loaded"]["model_key"].tolist(),
    }
    save_json(metadata, logs_dir / f"{output_prefix}_model_evaluation_report.json")


def main() -> None:
    args = parse_args()
    source_tables_dir = args.source_tables_dir.resolve()
    metrics_root = args.metrics_dir.resolve()
    tables_dir = metrics_root / "tables"
    figures_dir = metrics_root / "figures"
    logs_dir = metrics_root / "logs"
    output_prefix = args.output_prefix

    ensure_dir(metrics_root)
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)
    ensure_dir(logs_dir)

    combined, manifest = load_all_model_tables(source_tables_dir)
    combined = sort_combined(combined)

    if combined.empty:
        write_outputs(
            combined=combined,
            manifest=manifest,
            wide=pd.DataFrame(),
            ranks=pd.DataFrame(),
            best=pd.DataFrame(),
            report_test_at_10=pd.DataFrame(),
            report_valid_at_10=pd.DataFrame(),
            academic_summary=pd.DataFrame(),
            figure_manifest=pd.DataFrame(),
            tables_dir=tables_dir,
            logs_dir=logs_dir,
            metrics_root=metrics_root,
            output_prefix=output_prefix,
        )
        print("No model evaluation tables were found. Manifest and empty summary outputs were written.")
        return

    wide = build_wide_table(combined)
    ranks = build_metric_ranks(combined)
    best = build_best_model_table(ranks)
    report_test_at_10 = build_report_focus_table(combined, split="test", k=10)
    report_valid_at_10 = build_report_focus_table(combined, split="valid", k=10)
    academic_summary = build_academic_summary_table(combined)

    plot_all_figures(combined, ranks, figures_dir, output_prefix)
    figure_manifest = build_figure_manifest(figures_dir, output_prefix)

    write_outputs(
        combined=combined,
        manifest=manifest,
        wide=wide,
        ranks=ranks,
        best=best,
        report_test_at_10=report_test_at_10,
        report_valid_at_10=report_valid_at_10,
        academic_summary=academic_summary,
        figure_manifest=figure_manifest,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
        metrics_root=metrics_root,
        output_prefix=output_prefix,
    )

    loaded = manifest[manifest["status"] == "loaded"]["model_display"].tolist()
    print(f"Model evaluation artifacts generated under {metrics_root} from saved outputs for: {', '.join(loaded)}")


if __name__ == "__main__":
    main()
