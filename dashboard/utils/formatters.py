from __future__ import annotations

import math
from typing import Any

import pandas as pd


def human_int(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return str(value)


def human_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
        if math.isnan(number):
            return "N/A"
        return f"{number:.{digits}f}"
    except Exception:
        return str(value)


def title_from_key(text: str) -> str:
    return text.replace("_", " ").strip().title()


def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    copy = df.copy()
    copy.columns = [title_from_key(str(col)) for col in copy.columns]
    return copy
