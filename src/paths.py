"""
paths.py

This module centralises all project directory and file system path definitions.

Purpose
-------
The purpose of this module is to provide a single source of truth for the
location of all major project directories, including raw data, intermediate
outputs, processed datasets, experimental splits, saved models, evaluation
outputs, dashboard files, notebooks, and tests.

Rationale
---------
In a data-driven machine learning project, file paths are used repeatedly across
multiple scripts, notebooks, and pipeline stages. Hardcoding absolute or
relative paths throughout the codebase reduces maintainability, increases the
risk of path inconsistencies, and makes the project harder to reproduce or
relocate across environments.

By defining all important paths in one module:

1. path management becomes consistent across the project;
2. changes to directory structure can be made in one place only;
3. scripts remain cleaner and more readable;
4. reproducibility and portability are improved.

Design
------
The root project directory is inferred dynamically from the current file
location using pathlib. All other directories are then defined relative to this
root. This design avoids dependence on machine-specific absolute paths.

The module also provides a helper function, `ensure_directories()`, which
creates the required directory structure if it does not already exist. This is
useful when initialising the project or running the pipeline in a new
environment.

Typical Usage
-------------
Other modules import the required path objects from this file rather than
constructing paths manually. For example, raw datasets should be loaded from
`RAW_DIR`, processed outputs should be saved to `PROCESSED_DIR`, and trained
models should be stored in `SAVED_MODELS_DIR`.

Academic Relevance
------------------
This module supports good software engineering practice through separation of
concerns, maintainability, and reproducibility. In the context of this project,
it contributes to a structured experimental workflow in which data ingestion,
preprocessing, modelling, evaluation, and dashboard outputs are organised in a
clear and systematic manner.
"""
from pathlib import Path

# project/
ROOT_DIR = Path(__file__).resolve().parents[1]

# data/
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MAPPINGS_DIR = DATA_DIR / "mappings"

# Raw data files
RAW_INTERACTIONS_PATH = RAW_DIR / "RAW_interactions.csv"
RAW_RECIPES_PATH = RAW_DIR / "RAW_recipes.csv"
PP_RECIPES_PATH = RAW_DIR / "PP_recipes.csv"
INTERACTIONS_TRAIN_PATH = RAW_DIR / "interactions_train.csv"
INTERACTIONS_VALIDATION_PATH = RAW_DIR / "interactions_validation.csv"
INTERACTIONS_TEST_PATH = RAW_DIR / "interactions_test.csv"

# notebooks/
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# src/
SRC_DIR = ROOT_DIR / "src"
SRC_DATA_DIR = SRC_DIR / "data"
SRC_FEATURES_DIR = SRC_DIR / "features"
SRC_MODELS_DIR = SRC_DIR / "models"
SRC_EVAL_DIR = SRC_DIR / "evaluation"
SRC_INFERENCE_DIR = SRC_DIR / "inference"

# dashboard/
DASHBOARD_DIR = ROOT_DIR / "dashboard"
DASHBOARD_PAGES_DIR = DASHBOARD_DIR / "pages"
DASHBOARD_COMPONENTS_DIR = DASHBOARD_DIR / "components"

# outputs/
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"
SAVED_MODELS_DIR = OUTPUTS_DIR / "saved_models"

# tests/
TESTS_DIR = ROOT_DIR / "tests"


def ensure_directories() -> None:
    dirs = [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        MAPPINGS_DIR,
        NOTEBOOKS_DIR,
        SRC_DIR,
        SRC_DATA_DIR,
        SRC_FEATURES_DIR,
        SRC_MODELS_DIR,
        SRC_EVAL_DIR,
        SRC_INFERENCE_DIR,
        DASHBOARD_DIR,
        DASHBOARD_PAGES_DIR,
        DASHBOARD_COMPONENTS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        METRICS_DIR,
        LOGS_DIR,
        SAVED_MODELS_DIR,
        TESTS_DIR
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()