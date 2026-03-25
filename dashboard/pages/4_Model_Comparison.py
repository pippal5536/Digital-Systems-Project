from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from utils.formatters import prettify_columns  # type: ignore
except Exception:
    def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
        pretty = df.copy()
        pretty.columns = [str(col).replace("_", " ").title() for col in pretty.columns]
        return pretty


METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics"
TABLES_DIR = METRICS_ROOT / "tables"
FIGURES_DIR = METRICS_ROOT / "figures"
LOGS_DIR = METRICS_ROOT / "logs"


TABLE_FILES = {
    "academic": "12_academic_model_comparison.csv",
    "test_at_10": "12_report_model_comparison_test_at_10.csv",
    "valid_at_10": "12_report_model_comparison_valid_at_10.csv",
    "best_by_metric": "12_best_model_by_metric.csv",
    "metric_ranks": "12_model_metric_ranks.csv",
    "wide": "12_model_comparison_wide.csv",
    "long": "12_model_comparison_long.csv",
    "output_manifest": "12_model_output_manifest.csv",
    "figure_manifest": "12_figure_manifest.csv",
}

FIGURE_GROUPS = {
    "ranking_trends": [
        (
            "12_precision_at_k_comparison.png",
            "Precision by K",
            "Higher precision means a larger share of the recommended items were relevant.",
        ),
        (
            "12_recall_at_k_comparison.png",
            "Recall by K",
            "Higher recall means the system recovered more of the relevant items available in the holdout set.",
        ),
        (
            "12_hit_rate_at_k_comparison.png",
            "Hit rate by K",
            "Hit rate shows how often at least one useful recommendation appeared in the top-K list.",
        ),
        (
            "12_ndcg_at_k_comparison.png",
            "nDCG by K",
            "nDCG rewards models that place relevant items nearer the top of the recommendation list.",
        ),
    ],
    "beyond_accuracy": [
        (
            "12_novelty_at_k_comparison.png",
            "Novelty by K",
            "Higher novelty indicates that recommendations are less dominated by highly popular items.",
        ),
        (
            "12_catalog_coverage_at_k_comparison.png",
            "Catalogue coverage by K",
            "Coverage shows how much of the recipe catalogue is being used by each recommender.",
        ),
        (
            "12_novelty_vs_precision_test_at_10.png",
            "Novelty versus precision at Test@10",
            "This plot highlights the trade-off between usefulness and discovery. Models higher and further right are more novel and accurate at the same time.",
        ),
        (
            "12_coverage_vs_ndcg_test_at_10.png",
            "Coverage versus nDCG at Test@10",
            "This plot contrasts breadth of catalogue use with ranking quality.",
        ),
    ],
    "test_cut": [
        (
            "12_test_at_10_metric_bars.png",
            "Test@10 metric bars",
            "A practical snapshot of the main evaluation cut point used for the final comparison.",
        ),
        (
            "12_test_at_10_heatmap.png",
            "Test@10 standardised heatmap",
            "This heatmap makes model strengths and weaknesses easier to compare across different metric scales.",
        ),
        (
            "12_test_at_10_rank_heatmap.png",
            "Test@10 rank heatmap",
            "Rank positions simplify comparison by showing which model finished first, second, and so on for each metric.",
        ),
        (
            "12_users_evaluated_k10.png",
            "Users evaluated at K=10",
            "Differences in users evaluated should be interpreted as a fairness caveat when comparing models.",
        ),
    ],
    "validation_cut": [
        (
            "12_valid_at_10_metric_bars.png",
            "Validation@10 metric bars",
            "Validation results indicate whether the broad ordering of models remains stable before the final test split.",
        ),
        (
            "12_valid_at_10_heatmap.png",
            "Validation@10 standardised heatmap",
            "The heatmap makes the validation-stage performance pattern easier to read.",
        ),
        (
            "12_valid_at_10_rank_heatmap.png",
            "Validation@10 rank heatmap",
            "Rank positions on validation help check whether the final test comparison is consistent.",
        ),
    ],
}

METRIC_LABELS = {
    "precision_at_k": "Precision",
    "recall_at_k": "Recall",
    "hit_rate_at_k": "Hit rate",
    "ndcg_at_k": "nDCG",
    "novelty_at_k": "Novelty",
    "catalog_coverage_at_k": "Catalogue coverage",
}

RELEVANCE_METRICS = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
DISCOVERY_METRICS = ["novelty_at_k", "catalog_coverage_at_k"]


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""



def human_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)



def human_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return str(value)



def metric_title(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())



def metric_rank_col(metric: str) -> str:
    return f"rank_{metric}"



def winners(df: pd.DataFrame, metric: str) -> tuple[list[str], float | None]:
    if df.empty or metric not in df.columns:
        return [], None
    value = df[metric].max()
    rows = df[df[metric] == value]
    names = rows["model_display"].astype(str).tolist() if "model_display" in rows.columns else []
    return names, float(value)



def joined_names(names: Iterable[str]) -> str:
    names = [str(name) for name in names if str(name).strip()]
    if not names:
        return "No model"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"



def best_balance_model(df: pd.DataFrame) -> str:
    if df.empty:
        return "No model"
    working = df.copy()
    relevance_rank_cols = [metric_rank_col(metric) for metric in RELEVANCE_METRICS if metric_rank_col(metric) in working.columns]
    discovery_rank_cols = [metric_rank_col(metric) for metric in DISCOVERY_METRICS if metric_rank_col(metric) in working.columns]
    if not relevance_rank_cols:
        return str(working.iloc[0].get("model_display", "No model"))
    working["relevance_rank_sum"] = working[relevance_rank_cols].sum(axis=1)
    if discovery_rank_cols:
        working["discovery_rank_sum"] = working[discovery_rank_cols].sum(axis=1)
        working["balance_rank_sum"] = working["relevance_rank_sum"] + (0.5 * working["discovery_rank_sum"])
    else:
        working["balance_rank_sum"] = working["relevance_rank_sum"]
    working = working.sort_values(["balance_rank_sum", "relevance_rank_sum", "model_display"])
    return str(working.iloc[0].get("model_display", "No model"))



def render_figure(filename: str, title: str, caption: str) -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.markdown(f"**{title}**")
        st.image(str(path), use_container_width=True)
        st.caption(caption)
    else:
        st.info(f"Figure not found: {filename}")



def analysis_paragraphs(test_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    academic: list[str] = []
    accessible: list[str] = []
    caution: list[str] = []

    if test_df.empty:
        return academic, accessible, caution

    precision_winners, precision_value = winners(test_df, "precision_at_k")
    recall_winners, recall_value = winners(test_df, "recall_at_k")
    ndcg_winners, ndcg_value = winners(test_df, "ndcg_at_k")
    novelty_winners, novelty_value = winners(test_df, "novelty_at_k")
    coverage_winners, coverage_value = winners(test_df, "catalog_coverage_at_k")
    balance = best_balance_model(test_df)

    academic.append(
        "The saved evaluation outputs indicate that the strongest relevance-oriented models at Test@10 are Popularity and Hybrid, with Popularity leading precision and recall while Hybrid remains joint-best on nDCG. "
        f"At this cut point, {joined_names(precision_winners)} achieves the highest precision ({human_float(precision_value)}), {joined_names(recall_winners)} achieves the highest recall ({human_float(recall_value)}), and {joined_names(ndcg_winners)} achieves the highest nDCG ({human_float(ndcg_value)})."
    )
    academic.append(
        f"The beyond-accuracy figures show a clear accuracy-versus-discovery trade-off. {joined_names(novelty_winners)} produces the highest novelty ({human_float(novelty_value)}), while {joined_names(coverage_winners)} produces the widest catalogue coverage ({human_float(coverage_value)}). However, these benefits come with weaker ranking effectiveness than the best-performing relevance models."
    )
    academic.append(
        f"Taken together, the tables and figures suggest that {balance} offers the most balanced overall behaviour, because it remains near the top on the core relevance metrics while avoiding the extreme catalogue narrowness of the Popularity baseline. Truncated SVD acts as a middle-ground model, whereas Bayesian Personalized Ranking remains weaker on the primary ranking metrics in this experiment."
    )

    

    if not valid_df.empty:
        caution.append(
            "The validation figures broadly support the same ordering seen on the test split, which strengthens confidence that the final comparison is not driven by a single split anomaly."
        )
    if "users_evaluated" in test_df.columns:
        user_min = int(test_df["users_evaluated"].min())
        user_max = int(test_df["users_evaluated"].max())
        if user_min != user_max:
            caution.append(
                f"The number of users evaluated is not identical across models at Test@10 ({human_int(user_min)} to {human_int(user_max)} users). This should be acknowledged as a fairness caveat when interpreting small performance gaps."
            )
    caution.append(
        "All results shown on this page are loaded from saved output files produced earlier in the pipeline. No models are retrained here."
    )
    return academic, accessible, caution



def load_all_tables() -> dict[str, pd.DataFrame]:
    return {key: load_csv(TABLES_DIR / filename) for key, filename in TABLE_FILES.items()}



def inject_page_css() -> None:
    st.markdown(
        """
        <style>
        .mc-card {
            background: linear-gradient(180deg, rgba(17,24,39,0.96) 0%, rgba(12,18,30,0.96) 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 165px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }
        .mc-label {
            font-size: 0.84rem;
            color: #cbd5e1;
            margin-bottom: 0.45rem;
            line-height: 1.35;
        }
        .mc-model {
            font-size: 1.5rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.16;
            margin-bottom: 0.65rem;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
        }
        .mc-score {
            display: inline-block;
            padding: 0.26rem 0.68rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 700;
            color: #22c55e;
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.20);
            margin-bottom: 0.55rem;
        }
        .mc-note {
            font-size: 0.8rem;
            color: #94a3b8;
            line-height: 1.35;
        }
        .mc-kicker {
            display: inline-block;
            margin-bottom: 0.65rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #93c5fd;
            background: rgba(59,130,246,0.12);
            border: 1px solid rgba(59,130,246,0.18);
            letter-spacing: 0.01em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_summary_card(label: str, model_text: str, value_text: str, note: str, kicker: str = "Test@10 leader") -> str:
    return f"""
    <div class="mc-card">
        <div class="mc-kicker">{html.escape(kicker)}</div>
        <div class="mc-label">{html.escape(label)}</div>
        <div class="mc-model">{html.escape(model_text)}</div>
        <div class="mc-score">{html.escape(value_text)}</div>
        <div class="mc-note">{html.escape(note)}</div>
    </div>
    """



def render_summary_cards(test_df: pd.DataFrame) -> None:
    if test_df.empty:
        return

    precision_winners, value_precision = winners(test_df, "precision_at_k")
    recall_winners, value_recall = winners(test_df, "recall_at_k")
    ndcg_winners, value_ndcg = winners(test_df, "ndcg_at_k")
    novelty_winners, value_novelty = winners(test_df, "novelty_at_k")
    coverage_winners, value_coverage = winners(test_df, "catalog_coverage_at_k")
    balance = best_balance_model(test_df)

    first_row = st.columns(3)
    cards_row_1 = [
        render_summary_card(
            label="Best precision",
            model_text=joined_names(precision_winners),
            value_text=human_float(value_precision),
            note="Higher is better. More of the top recommendations were relevant.",
        ),
        render_summary_card(
            label="Best recall",
            model_text=joined_names(recall_winners),
            value_text=human_float(value_recall),
            note="Higher is better. More relevant holdout items were recovered.",
        ),
        render_summary_card(
            label="Best nDCG",
            model_text=joined_names(ndcg_winners),
            value_text=human_float(value_ndcg),
            note="Higher is better. Relevant items were ranked nearer the top.",
        ),
    ]
    for column, card_html in zip(first_row, cards_row_1):
        with column:
            st.markdown(card_html, unsafe_allow_html=True)

    second_row = st.columns(3)
    cards_row_2 = [
        render_summary_card(
            label="Best novelty",
            model_text=joined_names(novelty_winners),
            value_text=human_float(value_novelty),
            note="Higher is better. Recommendations are less dominated by very popular items.",
            kicker="Discovery leader",
        ),
        render_summary_card(
            label="Best catalogue coverage",
            model_text=joined_names(coverage_winners),
            value_text=human_float(value_coverage),
            note="Higher is better. A larger share of the recipe catalogue is used.",
            kicker="Breadth leader",
        ),
        render_summary_card(
            label="Most balanced overall",
            model_text=balance,
            value_text="Interpretive summary",
            note="Strong relevance, better ranking balance, and less extreme catalogue narrowness.",
            kicker="Recommended headline",
        ),
    ]
    for column, card_html in zip(second_row, cards_row_2):
        with column:
            st.markdown(card_html, unsafe_allow_html=True)


st.set_page_config(page_title="Model Comparison", layout="wide")
inject_page_css()

st.title("Model Comparison")
st.markdown(
    "This page compares the saved evaluation outputs across the trained recommenders using the generated `outputs/metrics` artifacts. "
    "It combines report-ready figures, academic interpretation, and the underlying comparison tables in one place."
)

if not METRICS_ROOT.exists():
    st.error(
        "The `outputs/metrics` folder was not found. Run `python src/evaluation/evaluate_models.py` first so the comparison tables and figures are generated."
    )
    st.stop()

frames = load_all_tables()
academic_df = frames["academic"]
test_df = frames["test_at_10"]
valid_df = frames["valid_at_10"]
best_df = frames["best_by_metric"]
ranks_df = frames["metric_ranks"]
wide_df = frames["wide"]
long_df = frames["long"]
output_manifest_df = frames["output_manifest"]
figure_manifest_df = frames["figure_manifest"]

loaded_models = []
if not output_manifest_df.empty and {"model_display", "status"}.issubset(output_manifest_df.columns):
    loaded_models = output_manifest_df.loc[output_manifest_df["status"].astype(str).str.lower() == "loaded", "model_display"].astype(str).tolist()

academic_paras, accessible_paras, caution_paras = analysis_paragraphs(test_df, valid_df)

render_summary_cards(test_df)

if loaded_models:
    st.caption("Saved model outputs loaded: " + ", ".join(loaded_models))

section_overview, section_figures, section_tables, section_outputs = st.tabs(
    ["Interpretation", "Figures", "Tables", "Artifacts"]
)

with section_overview:
    st.subheader("Interpretation")
    for paragraph in academic_paras:
        st.markdown(paragraph)

    

    st.subheader("Important interpretation notes")
    for paragraph in caution_paras:
        st.markdown(f"- {paragraph}")

    if not test_df.empty:
        st.subheader("Practical comparison cut point: Test@10")
        st.dataframe(prettify_columns(test_df), use_container_width=True)

    if not valid_df.empty:
        with st.expander("Validation reference table (Valid@10)"):
            st.dataframe(prettify_columns(valid_df), use_container_width=True)

  
with section_figures:
    st.subheader("Ranking performance across recommendation list length")
    for filename, title, caption in FIGURE_GROUPS["ranking_trends"]:
        render_figure(filename, title, caption)

    st.subheader("Beyond-accuracy behaviour")
    for filename, title, caption in FIGURE_GROUPS["beyond_accuracy"]:
        render_figure(filename, title, caption)

    st.subheader("Final comparison focus: Test@10")
    for filename, title, caption in FIGURE_GROUPS["test_cut"]:
        render_figure(filename, title, caption)

    st.subheader("Validation consistency check")
    for filename, title, caption in FIGURE_GROUPS["validation_cut"]:
        render_figure(filename, title, caption)

with section_tables:
    st.subheader("Report-ready tables")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Test@10 comparison**")
        if not test_df.empty:
            st.dataframe(prettify_columns(test_df), use_container_width=True)
        else:
            st.info("Test@10 report table not found.")
    with col_b:
        st.markdown("**Valid@10 comparison**")
        if not valid_df.empty:
            st.dataframe(prettify_columns(valid_df), use_container_width=True)
        else:
            st.info("Valid@10 report table not found.")

    st.subheader("Academic comparison table")
    if not academic_df.empty:
        st.dataframe(prettify_columns(academic_df), use_container_width=True)
    else:
        st.info("Academic comparison table not found.")

    st.subheader("Best model by metric and cut point")
    if not best_df.empty:
        st.dataframe(prettify_columns(best_df), use_container_width=True)
    else:
        st.info("Best-by-metric table not found.")

    with st.expander("Metric ranks for every split and K"):
        if not ranks_df.empty:
            st.dataframe(prettify_columns(ranks_df), use_container_width=True)
        else:
            st.info("Metric rank table not found.")

    with st.expander("Combined wide comparison table"):
        if not wide_df.empty:
            st.dataframe(prettify_columns(wide_df), use_container_width=True)
        else:
            st.info("Wide comparison table not found.")

    with st.expander("Combined long comparison table"):
        if not long_df.empty:
            st.dataframe(prettify_columns(long_df), use_container_width=True)
        else:
            st.info("Long comparison table not found.")

with section_outputs:
    st.subheader("Evaluation artifact manifest")
    if not output_manifest_df.empty:
        st.dataframe(prettify_columns(output_manifest_df), use_container_width=True)
    else:
        st.info("Model output manifest not found.")

    st.subheader("Generated figure manifest")
    if not figure_manifest_df.empty:
        st.dataframe(prettify_columns(figure_manifest_df), use_container_width=True)
    else:
        st.info("Figure manifest not found.")

    st.caption(
        f"Metrics root: {METRICS_ROOT} | Tables: {TABLES_DIR} | Figures: {FIGURES_DIR} | Logs: {LOGS_DIR}"
    )
