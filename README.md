# Comparative Evaluation of Recommender Systems for Personalised Food Recommendations

This repository contains an end-to-end recommender-systems project built around the Genius Kitchen dataset. The project combines a reproducible data pipeline, multiple recommendation models, consolidated evaluation scripts, and a Streamlit dashboard for presenting the results.

The workflow is organised around a shared preprocessing foundation and a shared time-aware evaluation policy. Raw interaction and recipe data are cleaned, transformed into modelling-ready datasets, split chronologically on a per-user basis, used to train multiple recommenders, and then summarised through comparison and bias/coverage reporting. The dashboard acts as a presentation and inference layer on top of the saved artefacts rather than retraining models live.

## Project goals

- audit and preprocess the Genius Kitchen interaction and recipe datasets
- construct comparable modelling datasets for explicit and implicit recommendation tasks
- apply a shared per-user chronological split policy to reduce temporal leakage
- train and evaluate several recommender-system baselines and stronger ranking models
- compare relevance, novelty, and catalogue coverage in a single reporting workflow
- present the results through an academic-style Streamlit dashboard

## Implemented recommenders

The project includes the following models:

- **Popularity** baseline
- **Collaborative Filtering** (item-item cosine similarity on sparse implicit interactions)
- **Truncated SVD**
- **Hybrid (SVD + Popularity)**
- **Bayesian Personalized Ranking (BPR)**

## End-to-end workflow

The main pipeline is organised into the following stages:

1. **Dataset audit**
2. **Interaction cleaning**
3. **Recipe preprocessing**
4. **Modelling dataset construction**
5. **Per-user chronological train/validation/test splitting**
6. **Feature engineering**
7. **Popularity training**
8. **Collaborative Filtering training**
9. **SVD training**
10. **Hybrid training**
11. **BPR training**
12. **Consolidated model evaluation**
13. **Bias and coverage analysis**

## Repository structure

A recommended repository layout for this codebase is shown below:

```text
project/
├─ data/
│  ├─ raw/
│  │  ├─ RAW_interactions.csv
│  │  ├─ RAW_recipes.csv
│  │  └─ PP_recipes.csv
│  ├─ interim/
│  ├─ processed/
│  ├─ splits/
│  └─ mappings/
├─ src/
│  ├─ data/
│  │  ├─ dataset_audit.py
│  │  ├─ clean_interactions.py
│  │  ├─ clean_recipes.py
│  │  ├─ build_modelling_datasets.py
│  │  ├─ make_chronological_splits.py
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ train_popularity.py
│  │  ├─ train_cf.py
│  │  ├─ train_svd.py
│  │  ├─ train_hybrid.py
│  │  └─ train_bpr.py
│  ├─ evaluation/
│  │  ├─ evaluate_models.py
│  │  └─ bias_coverage.py
│  └─ paths.py
├─ dashboard/
│  ├─ 1_Home.py
│  ├─ pages/
│  │  ├─ 2_Dataset_Overview.py
│  │  ├─ 3_Recommendation_Demo.py
│  │  ├─ 4_Model_Comparison.py
│  │  ├─ 5_Bias_and_Coverage.py
│  │  └─ 6_Conclusion.py
│  ├─ services/
│  │  ├─ data_loader.py
│  │  ├─ metrics_loader.py
│  │  ├─ model_loader.py
│  │  └─ recommendation_loader.py
│  └─ utils/
│     ├─ constants.py
│     ├─ formatters.py
│     └─ paths.py
├─ outputs/
│  ├─ figures/
│  ├─ tables/
│  ├─ metrics/
│  ├─ bias_coverage/
│  ├─ logs/
│  └─ saved_models/
├─ run_pipeline.py
├─ requirements.txt
└─ README.md
```

## Input data

Download the **Food.com Recipes and Interactions** dataset from the Kaggle webpage as a ZIP file:

`https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions`

After downloading the ZIP file from the web, extract the materials into:

```text
project/
├─ data/
│  └─ raw/
```

The project expects the following three raw source files inside `data/raw/`:

- `RAW_interactions.csv`
- `RAW_recipes.csv`
- `PP_recipes.csv`

A correct input-data layout should look like this:

```text
project/
├─ data/
│  ├─ raw/
│  │  ├─ RAW_interactions.csv
│  │  ├─ RAW_recipes.csv
│  │  └─ PP_recipes.csv
```

These filenames should remain unchanged, as the pipeline expects them exactly as shown above.

## Environment setup

Create and activate a virtual environment, then install the project dependencies.

### Windows (PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux / Git Bash

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the full pipeline

If the repository root contains `run_pipeline.py`, run:

```bash
python run_pipeline.py
```

This generates the processed datasets, split artefacts, trained model files, evaluation tables, figures, logs, and dashboard-ready outputs.

## Run the dashboard

Run the dashboard entry script with Streamlit.

If your home page file lives at the repository root of the dashboard folder, use a command such as:

```bash
streamlit run dashboard/1_Home.py
```

If your current working layout is still flat while developing, adjust the path accordingly, for example:

```bash
streamlit run 1_Home.py
```

## Dashboard pages

The dashboard is designed as a reporting layer over saved artefacts. The main pages are:

- **Home** – project overview and headline dataset counts
- **Dataset Overview** – audit, preprocessing, modelling dataset construction, splitting, and feature engineering outputs
- **Recommendation Demo** – user-level top-10 recommendations from the saved recommender outputs
- **Model Comparison** – cross-model ranking performance comparison
- **Bias and Coverage** – novelty, coverage, and popularity-bias interpretation
- **Conclusion** – overall interpretation of the results and recommended model choice

## Outputs

The project writes outputs into dedicated folders so that training, evaluation, and presentation remain separated:

- `outputs/tables/` – model outputs, summary tables, and exported recommendation tables
- `outputs/figures/` – PNG and SVG figures
- `outputs/logs/` – JSON and markdown summaries from pipeline stages
- `outputs/saved_models/` – reusable model artefacts for dashboard inference
- `outputs/metrics/` – consolidated comparison artefacts built from saved model outputs
- `outputs/bias_coverage/` – dedicated bias and coverage reporting bundle

## Methodological notes

- **rating = 0** is treated as an observed but unrated interaction in the preprocessing logic
- the **explicit** modelling view excludes `rating = 0`
- the **implicit** modelling view preserves observed interactions for ranking-based recommendation
- chronological splitting is **per-user**, not random
- train-fitted mappings are reused for later splits to reduce leakage
- the comparison and bias/coverage scripts operate on **saved outputs** and do not retrain models

## Main dependencies

Core packages used in this repository include:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `joblib`
- `streamlit`
- `pyarrow`


