
from __future__ import annotations

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


BIAS_ROOT = PROJECT_ROOT / "outputs" / "bias_coverage"
TABLES_DIR = BIAS_ROOT / "tables"
FIGURES_DIR = BIAS_ROOT / "figures"
LOGS_DIR = BIAS_ROOT / "logs"

TABLE_FILES = {
    "test": "13_bias_coverage_test_at_10.csv",
    "valid": "13_bias_coverage_valid_at_10.csv",
    "scorecard": "13_bias_coverage_scorecard.csv",
    "tradeoff": "13_tradeoff_summary.csv",
    "popularity_bias": "13_popularity_bias_summary.csv",
    "concentration": "13_concentration_summary.csv",
    "artifact_manifest": "13_artifact_manifest.csv",
    "model_manifest": "13_model_output_manifest.csv",
}

FIGURE_GROUPS = {
    "coverage_and_reach": [
        (
            "13_catalog_coverage_at_k_comparison.png",
            "Catalogue coverage across K",
            "Catalogue coverage shows how much of the recipe catalogue each recommender activates. Higher values indicate broader item exposure.",
        ),
        (
            "13_users_evaluated_k10.png",
            "Users evaluated at K=10",
            "This figure acts as a user-reach proxy and should be read as a fairness caveat when comparing small metric differences.",
        ),
    ],
    "bias_and_breadth": [
        (
            "13_novelty_at_k_comparison.png",
            "Novelty across K",
            "Novelty captures the extent to which recommendations are less dominated by highly popular recipes.",
        ),
        (
            "13_test_recommendation_concentration.png",
            "Recommendation concentration curves (test)",
            "These curves show whether recommendation exposure becomes concentrated in a smaller set of repeatedly recommended items.",
        ),
        (
            "13_valid_recommendation_concentration.png",
            "Recommendation concentration curves (validation)",
            "Validation concentration patterns help check whether repetition behaviour is stable before the final test split.",
        ),
    ],
    "tradeoffs": [
        (
            "13_coverage_vs_ndcg_test_at_10.png",
            "Coverage versus nDCG at Test@10",
            "This figure makes the breadth-versus-ranking-quality trade-off easier to interpret.",
        ),
        (
            "13_novelty_vs_precision_test_at_10.png",
            "Novelty versus precision at Test@10",
            "This figure contrasts recommendation usefulness with discovery and long-tail exposure.",
        ),
        (
            "13_test_at_10_heatmap.png",
            "Normalised bias-and-coverage heatmap at Test@10",
            "The heatmap summarises relative strengths across relevance and breadth metrics on the final test split.",
        ),
        (
            "13_valid_at_10_heatmap.png",
            "Normalised bias-and-coverage heatmap at Valid@10",
            "The validation heatmap checks whether the same broad structure appears before the final test split.",
        ),
    ],
    "scorecards": [
        (
            "13_test_at_10_scorecard.png",
            "Bias-and-coverage scorecard at Test@10",
            "This scorecard compares the models directly on the main metrics used in this section.",
        ),
        (
            "13_valid_at_10_scorecard.png",
            "Bias-and-coverage scorecard at Valid@10",
            "This scorecard shows whether the test-stage ranking pattern is already visible on validation.",
        ),
    ],
}

MODEL_SHORT = {
    "Popularity": "Popularity",
    "Collaborative Filtering": "CF",
    "Truncated SVD": "SVD",
    "Hybrid (SVD + Popularity)": "Hybrid",
    "Bayesian Personalized Ranking": "BPR",
}


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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


def short_model(name: str) -> str:
    return MODEL_SHORT.get(str(name), str(name))


def get_row(df: pd.DataFrame, model_key: str) -> pd.Series | None:
    if df.empty or "model_key" not in df.columns:
        return None
    subset = df[df["model_key"].astype(str).str.lower() == model_key.lower()]
    return None if subset.empty else subset.iloc[0]


def find_best(df: pd.DataFrame, metric: str, ascending: bool = False) -> tuple[str, float | None]:
    if df.empty or metric not in df.columns:
        return "Unavailable", None
    working = df.dropna(subset=[metric]).copy()
    if working.empty:
        return "Unavailable", None
    idx = working[metric].idxmin() if ascending else working[metric].idxmax()
    row = working.loc[idx]
    return short_model(str(row.get("model_display", "Unavailable"))), float(row[metric])


def render_figure(filename: str, title: str, caption: str) -> None:
    path = FIGURES_DIR / filename
    if path.exists():
        st.markdown(f"**{title}**")
        st.image(str(path), use_container_width=True)
        st.caption(caption)
    else:
        st.info(f"Figure not found: {filename}")


def build_analysis(test_df: pd.DataFrame, valid_df: pd.DataFrame, scorecard_df: pd.DataFrame,
                   tradeoff_df: pd.DataFrame, bias_df: pd.DataFrame, concentration_df: pd.DataFrame) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "overview": [],
        "coverage": [],
        "bias": [],
        "tradeoff": [],
        "concentration": [],
        "conclusion": [],
        "cautions": [],
    }

    if test_df.empty:
        return sections

    pop_test = get_row(test_df, "popularity")
    hyb_test = get_row(test_df, "hybrid")
    svd_test = get_row(test_df, "svd")
    bpr_test = get_row(test_df, "bpr")
    cf_test = get_row(test_df, "cf")

    pop_valid = get_row(valid_df, "popularity")
    hyb_valid = get_row(valid_df, "hybrid")
    svd_valid = get_row(valid_df, "svd")
    cf_valid = get_row(valid_df, "cf")

    best_precision_model, best_precision = find_best(test_df, "precision_at_k")
    best_ndcg_model, best_ndcg = find_best(test_df, "ndcg_at_k")
    best_novelty_model, best_novelty = find_best(test_df, "novelty_at_k")
    best_coverage_model, best_coverage = find_best(test_df, "catalog_coverage_at_k")

    sections["overview"].append(
        "This section evaluates recommender behaviour beyond ranking accuracy by examining catalogue breadth, popularity concentration, novelty, and user reach. "
        "The purpose is not to identify a single universally superior system, but to determine whether stronger top-K relevance is achieved through narrow exposure to a small subset of recipes or through broader, less concentrated recommendation behaviour."
    )
    sections["overview"].append(
        f"At Test@10, {best_precision_model} records the highest precision ({human_float(best_precision)}), while {best_novelty_model} and {best_coverage_model} dominate the discovery-oriented metrics with the highest novelty ({human_float(best_novelty)}) and coverage ({human_float(best_coverage)}), respectively. "
        "This immediately suggests that the most accurate model is not the broadest model, and that the page must therefore be interpreted as a trade-off analysis rather than a simple leaderboard."
    )

    if cf_test is not None and pop_test is not None and svd_test is not None and hyb_test is not None:
        coverage_ratio = float(cf_test["catalog_coverage_at_k"]) / max(float(pop_test["catalog_coverage_at_k"]), 1e-9)
        sections["coverage"].append(
            f"Catalogue coverage at Test@10 is dominated by Collaborative Filtering, which reaches {human_float(cf_test['catalog_coverage_at_k'])}, compared with only {human_float(pop_test['catalog_coverage_at_k'])} for Popularity, {human_float(hyb_test['catalog_coverage_at_k'])} for Hybrid, and {human_float(svd_test['catalog_coverage_at_k'])} for Truncated SVD. "
            f"In relative terms, CF covers nearly {coverage_ratio:,.0f} times as much of the catalogue as the Popularity baseline. This indicates an extremely broad exposure pattern for CF and an extremely narrow one for Popularity."
        )
    if cf_valid is not None and pop_valid is not None:
        sections["coverage"].append(
            f"The same structure remains visible on the validation split, where CF reaches coverage of {human_float(cf_valid['catalog_coverage_at_k'])} and Popularity remains at {human_float(pop_valid['catalog_coverage_at_k'])}. "
            "This stability across validation and test suggests that the breadth ordering is not a one-off split artefact."
        )
    if pop_test is not None and cf_test is not None:
        sections["coverage"].append(
            f"User reach differs by model, but far less dramatically than item coverage. At Test@10, Popularity evaluates {human_int(pop_test['users_evaluated'])} users and CF evaluates {human_int(cf_test['users_evaluated'])}. "
            "The user-level gap is therefore modest relative to the item-exposure gap, which strengthens the interpretation that the coverage differences are mainly driven by recommendation breadth rather than by radically different evaluation populations."
        )

    if cf_test is not None and pop_test is not None and bpr_test is not None and svd_test is not None:
        sections["bias"].append(
            f"Novelty results point to a strong popularity-bias gradient across the models. CF records the highest novelty at Test@10 ({human_float(cf_test['novelty_at_k'])}), followed by BPR ({human_float(bpr_test['novelty_at_k'])}) and SVD ({human_float(svd_test['novelty_at_k'])}), while Popularity remains lowest ({human_float(pop_test['novelty_at_k'])}). "
            "In practical terms, this means that Popularity is most dominated by well-known recipes, whereas CF is the least constrained by head-item exposure."
        )
        sections["bias"].append(
            f"The contrast between Popularity and Hybrid is especially important. Hybrid raises novelty from {human_float(pop_test['novelty_at_k'])} to {human_float(hyb_test['novelty_at_k'])} and increases coverage from {human_float(pop_test['catalog_coverage_at_k'])} to {human_float(hyb_test['catalog_coverage_at_k'])}, yet still preserves high ranking quality. "
            "This indicates that modest reductions in popularity concentration are possible without fully abandoning relevance."
        )
    if not bias_df.empty and {"model_display", "popularity_bias_interpretation"}.issubset(bias_df.columns):
        bias_lines = []
        for _, row in bias_df.iterrows():
            bias_lines.append(f"{short_model(str(row['model_display']))}: {row['popularity_bias_interpretation']}")
        sections["bias"].append(
            "The table-based bias summary reinforces this reading: " + "; ".join(bias_lines) + "."
        )

    if pop_test is not None and hyb_test is not None and svd_test is not None and cf_test is not None and bpr_test is not None:
        sections["tradeoff"].append(
            f"The trade-off figures show a clear separation between relevance-led and breadth-led models. Popularity and Hybrid occupy the high-precision, high-nDCG region, with both reaching nDCG of {human_float(pop_test['ndcg_at_k']) if pop_test is not None else '—'} at Test@10, while CF occupies the high-coverage, high-novelty region but with much lower precision ({human_float(cf_test['precision_at_k'])}) and nDCG ({human_float(cf_test['ndcg_at_k'])})."
        )
        sections["tradeoff"].append(
            f"Truncated SVD sits between these extremes. Its Test@10 precision ({human_float(svd_test['precision_at_k'])}) and nDCG ({human_float(svd_test['ndcg_at_k'])}) are below Popularity and Hybrid, but its coverage ({human_float(svd_test['catalog_coverage_at_k'])}) and novelty ({human_float(svd_test['novelty_at_k'])}) are materially higher. "
            "This makes SVD a true middle-ground model rather than a weak variant of either extreme."
        )
        sections["tradeoff"].append(
            f"BPR is more ambiguous. Although it achieves the second-highest novelty at Test@10 ({human_float(bpr_test['novelty_at_k'])}), its coverage remains low ({human_float(bpr_test['catalog_coverage_at_k'])}) and its ranking quality stays clearly below the strongest models. "
            "That pattern suggests that higher novelty alone is not sufficient evidence of balanced or practically useful recommendation behaviour."
        )
    if not tradeoff_df.empty and {"model_display", "tradeoff_interpretation"}.issubset(tradeoff_df.columns):
        mapping = "; ".join(
            f"{short_model(str(row['model_display']))}: {row['tradeoff_interpretation']}"
            for _, row in tradeoff_df.iterrows()
        )
        sections["tradeoff"].append("The saved trade-off table classifies the models in the same direction: " + mapping + ".")

    if not concentration_df.empty:
        test_conc = concentration_df[concentration_df["split"].astype(str).str.lower() == "test"].copy()
        valid_conc = concentration_df[concentration_df["split"].astype(str).str.lower() == "valid"].copy()
        if not test_conc.empty:
            top_test = test_conc.sort_values("concentration_curve_area", ascending=False).iloc[0]
            low_test = test_conc.sort_values("concentration_curve_area", ascending=True).iloc[0]
            sections["concentration"].append(
                f"Among the models for which concentration-curve artefacts are available, {short_model(str(top_test['model_display']))} has the largest concentration-curve area on the test split ({human_float(top_test['concentration_curve_area'])}), while {short_model(str(low_test['model_display']))} has the smallest ({human_float(low_test['concentration_curve_area'])}). "
                "Within this subset, the evidence suggests that Hybrid repeats a smaller set of recipes more intensely than SVD, whereas BPR is the least repetitive of the three."
            )
        if not valid_conc.empty:
            sections["concentration"].append(
                "The same relative ordering is preserved on the validation split, which adds confidence that this partial concentration pattern is stable rather than accidental."
            )
        sections["concentration"].append(
            "However, the available concentration curves do not include Popularity or Collaborative Filtering in the provided outputs. Consequently, the concentration analysis should be treated as partial evidence that complements, rather than replaces, the novelty and coverage results."
        )

    if not scorecard_df.empty and {"model_display", "balance_score", "bias_coverage_profile"}.issubset(scorecard_df.columns):
        score_sorted = scorecard_df.sort_values("balance_score", ascending=False).reset_index(drop=True)
        top = score_sorted.iloc[0]
        sections["conclusion"].append(
            f"The scorecard assigns the highest composite balance score to {short_model(str(top['model_display']))} ({human_float(top['balance_score'], 3)}), but this result should be interpreted critically rather than mechanically."
        )
    if pop_test is not None and hyb_test is not None:
        sections["conclusion"].append(
            f"Although Popularity is strongest on raw relevance metrics and reaches the same Test@10 nDCG as Hybrid ({human_float(pop_test['ndcg_at_k'])}), its coverage is essentially negligible ({human_float(pop_test['catalog_coverage_at_k'])}). "
            f"Hybrid is therefore the more defensible practical compromise: it remains near-best on relevance, ties the top model on test nDCG, and offers modest but real gains in novelty ({human_float(hyb_test['novelty_at_k'])}) and coverage ({human_float(hyb_test['catalog_coverage_at_k'])}) over the Popularity baseline."
        )
    if cf_test is not None:
        sections["conclusion"].append(
            f"Collaborative Filtering is best understood as the broadest but least targeted recommender. Its extremely high novelty ({human_float(cf_test['novelty_at_k'])}) and coverage ({human_float(cf_test['catalog_coverage_at_k'])}) make it valuable as evidence that breadth is possible, but its very low precision ({human_float(cf_test['precision_at_k'])}) and nDCG ({human_float(cf_test['ndcg_at_k'])}) limit its suitability as the primary deployed model."
        )
    if svd_test is not None:
        sections["conclusion"].append(
            f"Truncated SVD provides the strongest middle-ground alternative. It does not match the top relevance of Popularity or Hybrid, but its combination of precision ({human_float(svd_test['precision_at_k'])}), nDCG ({human_float(svd_test['ndcg_at_k'])}), novelty ({human_float(svd_test['novelty_at_k'])}), and coverage ({human_float(svd_test['catalog_coverage_at_k'])}) makes it the clearest compromise between narrow accuracy and broad exploration."
        )

    sections["cautions"].append(
        "This page analyses item-exposure bias and coverage, not demographic fairness. The available dataset does not provide protected-group attributes suitable for demographic fairness auditing."
    )
    sections["cautions"].append(
        "Users evaluated are not identical across models. Small performance gaps should therefore be interpreted alongside the reach figure rather than in isolation."
    )
    sections["cautions"].append(
        "Coverage and novelty can improve simply because a model spreads recommendations widely, even when those recommendations are weakly targeted. These metrics must therefore be interpreted together with precision and nDCG."
    )
    sections["cautions"].append(
        "Concentration-curve artefacts are incomplete for the current output bundle, so the repetition analysis is informative but not fully comprehensive."
    )
    return sections


def inject_page_css() -> None:
    st.markdown(
        """
        <style>
        .bc-card {
            background: linear-gradient(180deg, rgba(17,24,39,0.98) 0%, rgba(12,18,30,0.98) 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem 1rem;
            min-height: 155px;
            margin-bottom: 0.6rem;
        }
        .bc-kicker {
            display: inline-block;
            margin-bottom: 0.55rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            color: #93c5fd;
            background: rgba(59,130,246,0.12);
            border: 1px solid rgba(59,130,246,0.18);
        }
        .bc-label {
            font-size: 0.86rem;
            color: #cbd5e1;
            margin-bottom: 0.35rem;
        }
        .bc-model {
            font-size: 1.45rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.15;
            margin-bottom: 0.55rem;
        }
        .bc-score {
            font-size: 1.0rem;
            font-weight: 700;
            color: #22c55e;
            margin-bottom: 0.4rem;
        }
        .bc-note {
            font-size: 0.82rem;
            color: #94a3b8;
            line-height: 1.36;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(label: str, model_text: str, score_text: str, note: str, kicker: str) -> str:
    return f"""
    <div class="bc-card">
        <div class="bc-kicker">{kicker}</div>
        <div class="bc-label">{label}</div>
        <div class="bc-model">{model_text}</div>
        <div class="bc-score">{score_text}</div>
        <div class="bc-note">{note}</div>
    </div>
    """


st.set_page_config(page_title="Bias and Coverage", layout="wide")
inject_page_css()

st.title("Bias and Coverage")
st.markdown(
    "This page evaluates whether the recommender systems deliver balanced exposure across the recipe catalogue or whether stronger ranking quality is achieved by repeatedly recommending a narrow set of popular items. "
    "The analysis focuses on catalogue coverage, novelty, recommendation concentration, trade-offs with relevance, and user reach."
)

if not BIAS_ROOT.exists():
    st.error(
        "The `outputs/bias_coverage` folder was not found. Run `python src/evaluation/bias_coverage.py` first so the page can load the generated tables and figures."
    )
    st.stop()

frames = {key: load_csv(TABLES_DIR / filename) for key, filename in TABLE_FILES.items()}
test_df = frames["test"]
valid_df = frames["valid"]
scorecard_df = frames["scorecard"]
tradeoff_df = frames["tradeoff"]
bias_df = frames["popularity_bias"]
concentration_df = frames["concentration"]
artifact_manifest_df = frames["artifact_manifest"]
model_manifest_df = frames["model_manifest"]

analysis = build_analysis(test_df, valid_df, scorecard_df, tradeoff_df, bias_df, concentration_df)

best_precision_model, best_precision = find_best(test_df, "precision_at_k")
best_ndcg_model, best_ndcg = find_best(test_df, "ndcg_at_k")
best_novelty_model, best_novelty = find_best(test_df, "novelty_at_k")
best_coverage_model, best_coverage = find_best(test_df, "catalog_coverage_at_k")

row1 = st.columns(4)
cards = [
    render_card(
        "Best Test@10 precision",
        best_precision_model,
        human_float(best_precision),
        "Highest share of relevant items in the top-10 list.",
        "Relevance leader",
    ),
    render_card(
        "Best Test@10 nDCG",
        best_ndcg_model,
        human_float(best_ndcg),
        "Strongest ranking quality near the top of the list.",
        "Ranking leader",
    ),
    render_card(
        "Best Test@10 novelty",
        best_novelty_model,
        human_float(best_novelty),
        "Lowest dependence on already popular recipes.",
        "Discovery leader",
    ),
    render_card(
        "Best Test@10 coverage",
        best_coverage_model,
        human_float(best_coverage),
        "Broadest use of the available recipe catalogue.",
        "Breadth leader",
    ),
]
for col, html_card in zip(row1, cards):
    with col:
        st.markdown(html_card, unsafe_allow_html=True)

section_overview, section_figures, section_tables, section_artifacts = st.tabs(
    ["Interpretation", "Figures", "Tables", "Artifacts"]
)

with section_overview:
    st.subheader("Why this section matters")
    for paragraph in analysis["overview"]:
        st.markdown(paragraph)

    st.subheader("1. Catalogue coverage and user reach")
    for paragraph in analysis["coverage"]:
        st.markdown(paragraph)

    if not test_df.empty:
        with st.expander("Key test table: bias and coverage at Test@10"):
            st.dataframe(prettify_columns(test_df), use_container_width=True)

    st.subheader("2. Popularity bias and long-tail exposure")
    for paragraph in analysis["bias"]:
        st.markdown(paragraph)

    st.subheader("3. Usefulness versus breadth trade-offs")
    for paragraph in analysis["tradeoff"]:
        st.markdown(paragraph)

    st.subheader("4. Recommendation concentration and repetition")
    for paragraph in analysis["concentration"]:
        st.markdown(paragraph)

    st.subheader("5. Overall critical judgment")
    for paragraph in analysis["conclusion"]:
        st.markdown(paragraph)

    st.subheader("Important interpretation notes")
    for paragraph in analysis["cautions"]:
        st.markdown(f"- {paragraph}")

with section_figures:
    st.subheader("Coverage and reach")
    for filename, title, caption in FIGURE_GROUPS["coverage_and_reach"]:
        render_figure(filename, title, caption)

    st.subheader("Popularity bias, novelty, and exposure concentration")
    for filename, title, caption in FIGURE_GROUPS["bias_and_breadth"]:
        render_figure(filename, title, caption)

    st.subheader("Trade-off analysis")
    for filename, title, caption in FIGURE_GROUPS["tradeoffs"]:
        render_figure(filename, title, caption)

    st.subheader("Compact scorecards")
    for filename, title, caption in FIGURE_GROUPS["scorecards"]:
        render_figure(filename, title, caption)

with section_tables:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Bias and coverage at Test@10**")
        if not test_df.empty:
            st.dataframe(prettify_columns(test_df), use_container_width=True)
        else:
            st.info("Test@10 table not found.")

        st.markdown("**Popularity bias summary**")
        if not bias_df.empty:
            st.dataframe(prettify_columns(bias_df), use_container_width=True)
        else:
            st.info("Popularity-bias summary table not found.")

        st.markdown("**Trade-off summary**")
        if not tradeoff_df.empty:
            st.dataframe(prettify_columns(tradeoff_df), use_container_width=True)
        else:
            st.info("Trade-off table not found.")

    with col_right:
        st.markdown("**Bias and coverage at Valid@10**")
        if not valid_df.empty:
            st.dataframe(prettify_columns(valid_df), use_container_width=True)
        else:
            st.info("Valid@10 table not found.")

        st.markdown("**Bias-and-coverage scorecard**")
        if not scorecard_df.empty:
            st.dataframe(prettify_columns(scorecard_df), use_container_width=True)
        else:
            st.info("Scorecard table not found.")

        st.markdown("**Concentration summary**")
        if not concentration_df.empty:
            st.dataframe(prettify_columns(concentration_df), use_container_width=True)
        else:
            st.info("Concentration summary table not found.")

with section_artifacts:
    st.subheader("Model output manifest")
    if not model_manifest_df.empty:
        st.dataframe(prettify_columns(model_manifest_df), use_container_width=True)
    else:
        st.info("Model output manifest not found.")

    st.subheader("Bias-and-coverage artifact manifest")
    if not artifact_manifest_df.empty:
        st.dataframe(prettify_columns(artifact_manifest_df), use_container_width=True)
    else:
        st.info("Artifact manifest not found.")

    st.caption(
        f"Bias-and-coverage root: {BIAS_ROOT} | Tables: {TABLES_DIR} | Figures: {FIGURES_DIR} | Logs: {LOGS_DIR}"
    )
