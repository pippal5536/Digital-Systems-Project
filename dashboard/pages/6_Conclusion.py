from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from utils.formatters import human_float, prettify_columns  # type: ignore
except Exception:
    def human_float(value: object, digits: int = 4) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "—"

    def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
        pretty = df.copy()
        pretty.columns = [str(col).replace("_", " ").title() for col in pretty.columns]
        return pretty


METRICS_ROOT = PROJECT_ROOT / "outputs" / "metrics"
METRICS_TABLES = METRICS_ROOT / "tables"

BIAS_ROOT = PROJECT_ROOT / "outputs" / "bias_coverage"
BIAS_TABLES = BIAS_ROOT / "tables"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def joined_names(names: list[str]) -> str:
    clean = [str(name) for name in names if str(name).strip()]
    if not clean:
        return "No model"
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def winners(df: pd.DataFrame, metric: str) -> tuple[list[str], float | None]:
    if df.empty or metric not in df.columns:
        return [], None
    values = pd.to_numeric(df[metric], errors="coerce")
    if values.dropna().empty:
        return [], None
    best_value = float(values.max())
    rows = df.loc[values == best_value]
    if "model_display" in rows.columns:
        names = rows["model_display"].astype(str).tolist()
    elif "model" in rows.columns:
        names = rows["model"].astype(str).tolist()
    else:
        names = []
    return names, best_value


def best_balance_model(model_df: pd.DataFrame, bias_df: pd.DataFrame) -> str:
    if model_df.empty:
        return "No model"

    working = model_df.copy()
    for metric in ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]:
        if metric in working.columns:
            working[f"rank_{metric}"] = working[metric].rank(method="dense", ascending=False)

    working["relevance_rank_sum"] = working[
        [col for col in working.columns if col.startswith("rank_")]
    ].sum(axis=1)

    if not bias_df.empty and {"model_display", "novelty_at_k", "catalog_coverage_at_k"}.issubset(bias_df.columns):
        bias_subset = bias_df[["model_display", "novelty_at_k", "catalog_coverage_at_k"]].copy()
        working = working.merge(bias_subset, on="model_display", how="left", suffixes=("", "_bias"))
        if "novelty_at_k_bias" in working.columns:
            working["novelty_at_k"] = working["novelty_at_k_bias"].combine_first(working.get("novelty_at_k"))
        if "catalog_coverage_at_k_bias" in working.columns:
            working["catalog_coverage_at_k"] = working["catalog_coverage_at_k_bias"].combine_first(
                working.get("catalog_coverage_at_k")
            )

    if "novelty_at_k" in working.columns:
        working["rank_novelty"] = working["novelty_at_k"].rank(method="dense", ascending=False)
    else:
        working["rank_novelty"] = 0

    if "catalog_coverage_at_k" in working.columns:
        working["rank_coverage"] = working["catalog_coverage_at_k"].rank(method="dense", ascending=False)
    else:
        working["rank_coverage"] = 0

    working["balance_score"] = working["relevance_rank_sum"] + 0.5 * (
        working["rank_novelty"] + working["rank_coverage"]
    )
    working = working.sort_values(["balance_score", "relevance_rank_sum", "model_display"])
    return str(working.iloc[0]["model_display"])


def card(label: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(148,163,184,0.20);
            border-radius: 16px;
            padding: 1rem 1rem 0.85rem 1rem;
            background: linear-gradient(180deg, rgba(17,24,39,0.96), rgba(12,18,30,0.96));
            min-height: 132px;
        ">
            <div style="font-size: 0.86rem; color: #cbd5e1; margin-bottom: 0.45rem;">{label}</div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #f8fafc; line-height: 1.15; margin-bottom: 0.55rem;">{value}</div>
            <div style="font-size: 0.92rem; color: #86efac;">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Conclusion", layout="wide")

st.title("Conclusion")
st.markdown(
    "This section synthesises the evidence from the ranking comparison and the bias-and-coverage analysis, "
    "then identifies the most defensible overall recommender for the project."
)

metrics_test = load_csv(METRICS_TABLES / "12_report_model_comparison_test_at_10.csv")
metrics_valid = load_csv(METRICS_TABLES / "12_report_model_comparison_valid_at_10.csv")
bias_test = load_csv(BIAS_TABLES / "13_bias_coverage_test_at_10.csv")
bias_valid = load_csv(BIAS_TABLES / "13_bias_coverage_valid_at_10.csv")
tradeoff_df = load_csv(BIAS_TABLES / "13_tradeoff_summary.csv")
scorecard_df = load_csv(BIAS_TABLES / "13_bias_coverage_scorecard.csv")

if metrics_test.empty and bias_test.empty:
    st.error(
        "The final conclusion could not be built because the saved evaluation outputs were not found. "
        "Generate the comparison and bias-and-coverage artifacts first."
    )
    st.stop()

precision_winners, precision_value = winners(metrics_test, "precision_at_k")
recall_winners, recall_value = winners(metrics_test, "recall_at_k")
ndcg_winners, ndcg_value = winners(metrics_test, "ndcg_at_k")
novelty_winners, novelty_value = winners(bias_test if not bias_test.empty else metrics_test, "novelty_at_k")
coverage_winners, coverage_value = winners(bias_test if not bias_test.empty else metrics_test, "catalog_coverage_at_k")
recommended_model = best_balance_model(metrics_test, bias_test)

top_cards = st.columns(3)
with top_cards[0]:
    card("Best precision at Test@10", joined_names(precision_winners), human_float(precision_value))
with top_cards[1]:
    card("Best recall at Test@10", joined_names(recall_winners), human_float(recall_value))
with top_cards[2]:
    card("Recommended overall model", recommended_model, "Best practical balance")

st.markdown("### Final conclusion")

st.markdown(
    f"""
The evaluation indicates that **{recommended_model}** is the strongest overall model for this project when the evidence is considered as a whole rather than through a single metric. The ranking comparison shows that **{joined_names(precision_winners)}** achieves the highest precision at Test@10 ({human_float(precision_value)}), **{joined_names(recall_winners)}** achieves the highest recall ({human_float(recall_value)}), and **{joined_names(ndcg_winners)}** achieves the highest nDCG ({human_float(ndcg_value)}). However, the bias-and-coverage analysis shows that the models with the strongest relevance do not necessarily provide the broadest or most balanced recommendation behaviour.
"""
)

st.markdown(
    f"""
A clear trade-off is present between **usefulness** and **breadth of exposure**. **{joined_names(novelty_winners)}** achieves the highest novelty at Test@10 ({human_float(novelty_value)}), while **{joined_names(coverage_winners)}** achieves the highest catalogue coverage ({human_float(coverage_value)}). This means that the most exploratory models reduce popularity concentration and reach more of the recipe catalogue, but they do so at the cost of weaker ranking quality. In practical terms, a model that exposes many more recipes is not automatically preferable if the recommendations themselves are much less relevant.
"""
)

st.markdown(
    """
From a critical perspective, the **Popularity** baseline remains an important benchmark because it performs strongly on the core relevance metrics. This shows that the dataset contains a strong head of popular items that are difficult to beat. Nevertheless, the same model is highly narrow in catalogue use and therefore not the most balanced recommendation strategy. At the other extreme, **Collaborative Filtering** provides the broadest coverage and strongest novelty, which makes it useful for demonstrating reduced popularity bias, but its ranking performance is too weak for it to be selected as the primary deployment candidate.
"""
)

st.markdown(
    """
**Truncated SVD** acts as the clearest middle-ground model, preserving more relevance than the exploratory baselines while also offering broader exposure than the most concentrated approach. **Bayesian Personalized Ranking** performs better than Collaborative Filtering on some relevance dimensions but still does not offer a convincing enough balance between ranking strength and exposure breadth to become the preferred model. The most defensible final choice is therefore the model that stays close to the top on ranking quality while moderating the narrowness of the popularity baseline.
"""
)

st.markdown(
    f"""
Taken together, the results support the selection of **{recommended_model}** as the most suitable recommender for the final system. It does not dominate every single metric, but it provides the strongest practical compromise between ranking effectiveness, exposure balance, and resistance to excessive popularity concentration. For a food recommendation setting, that is a more persuasive deployment argument than maximising one metric in isolation.
"""
)

st.markdown("### Key takeaways")

takeaways = [
    f"**Ranking quality:** {joined_names(precision_winners)} and {joined_names(ndcg_winners)} lead the most important top-K relevance metrics.",
    f"**Bias and coverage:** {joined_names(novelty_winners)} and {joined_names(coverage_winners)} provide the broadest and least popularity-dominated behaviour.",
    f"**Best compromise:** {recommended_model} offers the strongest overall balance between usefulness and breadth.",
    "**Interpretation caution:** the number of evaluated users is not perfectly identical across models, so very small performance gaps should be interpreted carefully.",
    "**Practical implication:** the best recommender for deployment is the one that remains accurate while avoiding extreme concentration on a narrow set of recipes.",
]
for item in takeaways:
    st.markdown(f"- {item}")

st.markdown("### Limitations and future work")

st.markdown(
    """
The present comparison should still be interpreted with appropriate caution. First, the evaluation is based on offline saved outputs rather than live user testing, so the results measure ranking performance and exposure behaviour rather than real-time satisfaction. Second, not every model appears to have been evaluated on exactly the same number of users, which introduces a modest fairness caveat when differences are small. Third, the available bias analysis focuses on item-exposure behaviour rather than demographic fairness, because the dataset does not provide rich protected-attribute information. Future work could therefore examine stronger per-user chronological evaluation, richer content-aware hybridisation, calibration of popularity bias, and online user studies that assess whether broader exposure also improves perceived usefulness.
"""
)

tabs = st.tabs(["Final evidence tables", "Supporting bias tables"])

with tabs[0]:
    st.subheader("Test@10 model comparison")
    if not metrics_test.empty:
        st.dataframe(prettify_columns(metrics_test), use_container_width=True)
    else:
        st.info("Test@10 comparison table not found.")

    if not metrics_valid.empty:
        with st.expander("Validation comparison (Valid@10)"):
            st.dataframe(prettify_columns(metrics_valid), use_container_width=True)

with tabs[1]:
    st.subheader("Bias and coverage evidence")
    if not bias_test.empty:
        st.dataframe(prettify_columns(bias_test), use_container_width=True)
    else:
        st.info("Bias-and-coverage Test@10 table not found.")

    if not tradeoff_df.empty:
        with st.expander("Trade-off summary"):
            st.dataframe(prettify_columns(tradeoff_df), use_container_width=True)

    if not scorecard_df.empty:
        with st.expander("Bias and coverage scorecard"):
            st.dataframe(prettify_columns(scorecard_df), use_container_width=True)

    if not bias_valid.empty:
        with st.expander("Bias-and-coverage validation table (Valid@10)"):
            st.dataframe(prettify_columns(bias_valid), use_container_width=True)
