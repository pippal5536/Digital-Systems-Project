from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from services.data_loader import load_dataset_tables, load_home_counts
from utils.formatters import prettify_columns

st.title("Dashboard for the Comparative Evaluation of Recommender Systems for Personalised Food Recommendations")
st.subheader("Project Overview")

counts = load_home_counts()
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Users", counts.get("users", "N/A"))
col2.metric("Recipes", counts.get("recipes", "N/A"))
col3.metric("Interactions", counts.get("interactions", "N/A"))
col4.metric("Explicit rows", counts.get("explicit_rows", "N/A"))
col5.metric("Implicit rows", counts.get("implicit_rows", "N/A"))

st.markdown(
    """
This dashboard is designed as a **presentation and inference layer** rather than a training pipeline.
It reads existing artefacts from `outputs/`, `data/processed/`, and `outputs/saved_models/`. 
The evaluation story is built around a **shared preprocessed foundation** and a **shared time-aware split policy**:
- the same cleaned interaction base feeds the implicit recommenders
- the same recipe metadata supports display and hybrid enrichment
- the same chronological split rule is reused across evaluation outputs
"""
)

st.markdown(
    """
### What this dashboard covers
- Dataset Overview: This page provides a structured overview of the Genius Kitchen dataset and the preprocessing pipeline used to prepare it for recommender-system experimentation. It brings together the main stages of the project, including dataset audit, interaction cleaning, recipe preprocessing, modelling dataset construction, per-user chronological splitting, and feature engineering. The aim is to show how the raw data were transformed into modelling-ready artefacts while preserving temporal realism, structural consistency, and compatibility across the different recommendation approaches evaluated in the dashboard.
- Recommendation Demo: This page allows the selection of a user in order to display the top 10 recipe recommendations produced by the different recommender systems.
- Model Comparison: The Model Comparison page shows the comparative performance of the recommender systems in a highly sparse dataset.
- Bias and Coverage: This page examines how the recommender systems distribute recommendations across the recipe catalogue, with particular attention to popularity bias and coverage.
- Conclusion: This page summarises the overall findings of the project and presents the final interpretation of the recommender-system results.
"""
)


