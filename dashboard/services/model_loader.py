from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.paths import saved_models_dir


@st.cache_data(show_spinner=False)
def list_saved_model_files() -> list[Path]:
    root = saved_models_dir()
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())
