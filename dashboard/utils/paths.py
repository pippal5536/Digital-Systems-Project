from __future__ import annotations

from pathlib import Path


def dashboard_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return dashboard_dir().parent


def project_root_exists() -> Path:
    root = project_root()
    return root


def data_dir() -> Path:
    return project_root() / "data"


def outputs_dir() -> Path:
    return project_root() / "outputs"


def tables_dir() -> Path:
    return outputs_dir() / "tables"


def figures_dir() -> Path:
    return outputs_dir() / "figures"


def logs_dir() -> Path:
    return outputs_dir() / "logs"


def saved_models_dir() -> Path:
    return outputs_dir() / "saved_models"


def processed_dir() -> Path:
    return data_dir() / "processed"

def mappings_dir() -> Path:
    return data_dir() / "mappings"

