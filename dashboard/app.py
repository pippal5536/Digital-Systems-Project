from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Comparative Evaluation of Recommender Systems for Personalised Food Recommendations: Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    {
        "Dashboard": [
            st.Page("pages/1_Home.py", title="Home"),
            st.Page("pages/3_Recommendation_Demo.py", title="Recommendation Demo"),
            st.Page("pages/4_Model_Comparison.py", title="Model Comparison"),
            st.Page("pages/5_Bias_and_Coverage.py", title="Bias and Coverage"),
            st.Page("pages/6_Conclusion.py", title="Conclusion"),
        ]
    },
    position="sidebar",
    expanded=True,
)

pg.run()