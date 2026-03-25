from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from services.recommendation_loader import available_models, get_top_n_recommendations, list_users_for_model
from utils.constants import MODEL_DISPLAY_NAMES
from utils.formatters import prettify_columns


MODEL_EXPLANATIONS = {
    "popularity": {
        "title": "Popularity baseline",
        "summary": (
            "This model recommends recipes that received the highest interaction volume in the training data. "
            "It is a non-personalised baseline: items rise to the top because many users interacted with them before, "
            "not because the model learned a deep preference profile for the selected user."
        ),
        "learning_points": [
            "Useful as a benchmark because it is simple, transparent, and easy to interpret.",
            "Often recommends very similar items to many users.",
            "Usually strong on head items, but weak on diversity and long-tail discovery.",
        ],
    },
    "svd": {
        "title": "SVD collaborative filtering",
        "summary": (
            "This model uses matrix factorisation to learn latent patterns between users and recipes. "
            "Instead of recommending only the globally most popular items, it ranks items using learned hidden factors "
            "that capture behavioural similarity across users and items."
        ),
        "learning_points": [
            "More personalised than a pure popularity baseline.",
            "Still often leans toward popular items because collaborative data are sparse and popularity influences what the model can learn.",
            "Useful for showing how latent-factor models move beyond simple frequency counts.",
        ],
    },
    "hybrid": {
        "title": "Hybrid recommender",
        "summary": (
            "This model combines more than one signal source, typically collaborative information and recipe metadata. "
            "The goal is to improve recommendation quality by balancing behavioural patterns with content information."
        ),
        "learning_points": [
            "Can improve personalisation when interaction data alone are sparse.",
            "Can expose items that are not purely the most popular.",
            "Useful for teaching how recommender systems combine collaborative and content signals.",
        ],
    },
    "bpr": {
        "title": "BPR ranking model",
        "summary": (
            "This model is trained with pairwise ranking logic: it learns to score observed items above unobserved ones. "
            "That makes it especially suitable for implicit-feedback recommendation tasks."
        ),
        "learning_points": [
            "Optimises ranking behaviour directly rather than only reconstructing observed interactions.",
            "Often better aligned with top-N recommendation tasks.",
            "Useful for explaining why ranking-focused models are common in recommender systems.",
        ],
    },
}
def _render_wrapped_recommendation_table(recommendations: pd.DataFrame, model_key: str) -> None:
    display = _prepare_display_table(recommendations, model_key).copy()
    html = display.to_html(index=False, escape=True)

    st.markdown(
        """
        <style>
        .recommendation-table-wrapper {
            overflow-x: auto;
            width: 100%;
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-radius: 10px;
        }

        .recommendation-table-wrapper table {
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
            font-size: 0.95rem;
            background-color: transparent;
        }

        .recommendation-table-wrapper thead th {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid rgba(250, 250, 250, 0.12);
            background-color: rgba(255, 255, 255, 0.06);
            color: inherit;
            white-space: nowrap;
        }

        .recommendation-table-wrapper tbody td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(250, 250, 250, 0.08);
            vertical-align: top;
            color: inherit;
            white-space: normal !important;
            word-break: break-word;
        }

        .recommendation-table-wrapper th:last-child,
        .recommendation-table-wrapper td:last-child {
            min-width: 420px;
            max-width: 700px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="recommendation-table-wrapper">{html}</div>',
        unsafe_allow_html=True,
    )


def _model_family(model_key: str) -> str:
    key = (model_key or "").lower()
    for family in MODEL_EXPLANATIONS:
        if family in key:
            return family
    return "popularity" if "pop" in key else key



def _default_model_index(models: list[str]) -> int:
    for i, model in enumerate(models):
        if "pop" in model.lower():
            return i
    return 0



def _build_interpretation(row: pd.Series, family: str) -> str:
    parts: list[str] = []

    rank = row.get("rank", row.get("recommendation_rank"))
    popularity_rank = (
    row.get("global_item_popularity_rank")
    if pd.notna(row.get("global_item_popularity_rank"))
    else row.get("global_popularity_rank")
)
    if pd.isna(popularity_rank):
        popularity_rank = row.get("train_popularity_rank")
    train_count = row.get("train_interaction_count")
    popularity_score = row.get("popularity_score")

    if pd.notna(rank):
        parts.append(f"Recommended at rank {int(rank)}")

    if family == "popularity":
        parts.append("because it is one of the most interacted-with recipes in the training data")
    elif family == "svd":
        parts.append("because the latent-factor model estimated it as relevant for this user")
    elif family == "hybrid":
        parts.append("because the hybrid model found it relevant using behavioural and metadata signals")
    elif family == "bpr":
        parts.append("because the ranking model placed it above other candidate recipes")

    if pd.notna(popularity_rank):
        parts.append(f"global popularity rank #{int(popularity_rank)}")
    if pd.notna(train_count):
        parts.append(f"{int(train_count)} training interactions")
    if pd.notna(popularity_score):
        try:
            parts.append(f"popularity score {float(popularity_score):.3f}")
        except (TypeError, ValueError):
            pass

    return " | ".join(parts)

def _coalesce_columns(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    existing = [col for col in candidates if col in frame.columns]
    if not existing:
        return None
    if len(existing) == 1:
        return frame[existing[0]]
    return frame[existing].bfill(axis=1).iloc[:, 0]



def _prepare_display_table(recommendations: pd.DataFrame, model_key: str) -> pd.DataFrame:
    frame = recommendations.copy()
    family = _model_family(model_key)

    columns_to_build = [
        ("Recommendation Rank", ["rank", "recommendation_rank"]),
        ("Recipe Name", ["recipe_name"]),
        ("Recipe ID", ["recipe_id"]),
        ("Model Score", ["score", "prediction", "pred", "estimated_rating"]),
        ("Training Interactions", ["train_interaction_count"]),
        ("Global Popularity Rank", ["global_item_popularity_rank", "global_popularity_rank"]),
        ("Training Popularity Rank", ["train_popularity_rank"]),
        ("Popularity Score", ["popularity_score"]),
        ("Minutes", ["minutes"]),
        ("Average Rating", ["avg_rating"]),
        ("Ingredient Count", ["n_ingredients"]),
        ("Calorie Level", ["calorie_level"]),
    ]

    data: dict[str, pd.Series] = {}

    for display_name, source_candidates in columns_to_build:
        series = _coalesce_columns(frame, source_candidates)
        if series is not None:
            data[display_name] = series

    display = pd.DataFrame(data, index=frame.index)
    display["What This Means"] = frame.apply(lambda row: _build_interpretation(row, family), axis=1)

    return display



def _render_model_explanation(model_key: str) -> None:
    family = _model_family(model_key)
    meta = MODEL_EXPLANATIONS.get(family)
    if not meta:
        return

    st.info(meta["summary"])
    with st.expander("How to read this model", expanded=True):
        st.markdown("**Key ideas**")
        for point in meta["learning_points"]:
            st.markdown(f"- {point}")

        if family == "popularity":
            st.markdown(
                "**How to interpret the table:** higher-ranked items are recipes that were interacted with more often in the training data. "
                "This means the list is easy to explain, but it may look very similar across many users."
            )
        elif family == "svd":
            st.markdown(
                "**How to interpret the table:** the ranking is produced by learned latent factors rather than simple global frequency. "
                "The saved table still includes popularity context so it is possible to see whether the SVD list is relying on highly popular recipes or not."
            )
        else:
            st.markdown(
                "**How to interpret the table:** the ranked list reflects the model's estimate of relevance for the selected user. "
                "Auxiliary columns such as popularity or metadata help explain why specific items may have appeared."
            )



def _render_user_level_metrics(recommendations: pd.DataFrame) -> None:
    metric_candidates = [
        ("precision_at_10", "Precision@10"),
        ("recall_at_10", "Recall@10"),
        ("hit_rate_at_10", "Hit Rate@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("novelty_at_10", "Novelty@10"),
    ]
    metric_values = [(col, label) for col, label in metric_candidates if col in recommendations.columns]
    if not metric_values:
        return

    first_row = recommendations.iloc[0]
    st.markdown("### What the selected user's results mean")
    cols = st.columns(len(metric_values))
    for streamlit_col, (col, label) in zip(cols, metric_values):
        value = first_row[col]
        if label == "Hit Rate@10":
            display_value = "Hit" if float(value) >= 1 else "No hit"
        else:
            display_value = f"{float(value):.4f}"
        streamlit_col.metric(label, display_value)

    with st.expander("Metric explanations"):
        st.markdown(
            "- **Precision@10**: the fraction of the top 10 recommendations that were actually relevant in the holdout set.\n"
            "- **Recall@10**: the fraction of the user's relevant holdout items that were recovered inside the top 10.\n"
            "- **Hit Rate@10**: whether at least one relevant item appeared in the top 10 list.\n"
            "- **nDCG@10**: rewards not only finding relevant items, but placing them high in the ranking.\n"
            "- **Novelty@10**: indicates how uncommon the recommended items are relative to the training data. Higher values generally mean less popular recommendations."
        )


st.set_page_config(page_title="Recommendation Demo", layout="wide")

st.title("Recommendation Demo")
st.markdown(
    "Explore saved top-N recommendation outputs for existing users. The page is designed to show **what a model recommends** and also **what the recommendation list means**, so it can be used both as a demo and as a learning aid."
)

models = available_models()
if not models:
    st.error("No recommendation files were found under outputs/tables.")
    st.stop()

model_index = _default_model_index(models)
default_model = models[model_index]

control_col1, control_col2 = st.columns([1.6, 1])
with control_col1:
    selected_model = st.selectbox(
        "Recommendation model",
        models,
        index=model_index,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.upper()),
        help="The default selection is the popularity baseline because it is the clearest starting point for understanding recommender systems.",
    )
with control_col2:
    split = st.selectbox("Evaluation split", ["test", "valid"], index=0)

top_n = 10
st.caption("The saved demo outputs currently contain top 10 recommendations per user, so this page shows the top 10 list for the selected model and user.")

users = list_users_for_model(selected_model, split)
if not users:
    st.warning("No users were found for the current model and split selection.")
    st.stop()

user_id = st.selectbox(
    "User ID",
    users,
    help=f"Users available for {MODEL_DISPLAY_NAMES.get(selected_model, selected_model.upper())} on the {split} split.",
)

selected_label = MODEL_DISPLAY_NAMES.get(selected_model, selected_model.upper())
st.success(f"Showing the top {top_n} saved recommendations for user {user_id} using {selected_label} on the {split} split.")

st.markdown(f"## {selected_label}")
_render_model_explanation(selected_model)

recommendations = get_top_n_recommendations(selected_model, split, str(user_id), int(top_n))
if recommendations.empty:
    st.info("This model has no saved recommendations for the selected user under the current filters.")
    st.stop()

if "holdout_item_count" in recommendations.columns:
    holdout_count = recommendations["holdout_item_count"].iloc[0]
    st.caption(f"Saved evaluation context: this user has {int(holdout_count)} holdout item(s) for the selected split.")

_render_user_level_metrics(recommendations)

st.markdown("### Recommended recipes")
_render_wrapped_recommendation_table(recommendations, selected_model)

with st.expander("How to interpret the ranked list"):
    st.markdown(
        "A top-N recommendation list is a ranking, not just a random set of recipes. The first item is the model's strongest suggestion, "
        "followed by lower-ranked alternatives. For the popularity baseline, the ranking is driven mainly by historical interaction frequency. "
        "For collaborative or ranking models such as SVD or BPR, the ordering reflects learned patterns from user-item behaviour."
    )

with st.expander("Raw recommendation data"):
    st.dataframe(prettify_columns(recommendations), use_container_width=True, hide_index=True)
