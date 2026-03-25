from __future__ import annotations

MODEL_DISPLAY_NAMES = {
    "popularity": "Popularity",
    "cf": "Collaborative Filtering",
    "svd": "SVD",
    "hybrid": "Hybrid",
    "bpr": "BPR",
}

MODEL_FILE_PATTERNS = {
    "popularity": [
        "07_popularity_metrics_dashboard.csv",
        "07_popularity_metrics.csv",
        "07_popularity_dashboard_summary.csv",
    ],
    "cf": [
        "08_cf_metrics_dashboard.csv",
        "08_cf_metrics.csv",
        "08_cf_dashboard_summary.csv",
    ],
    "svd": [
        "svd/09_svd_metrics_dashboard.csv",
        "svd/09_svd_metrics.csv",
        "svd/09_svd_dashboard_summary.csv",
        "09_svd_metrics_dashboard.csv",
        "09_svd_metrics.csv",
        "09_svd_dashboard_summary.csv",
    ],
    "hybrid": [
        "hybrid/10_hybrid_metrics_dashboard.csv",
        "hybrid/10_hybrid_metrics.csv",
        "hybrid/10_hybrid_dashboard_summary.csv",
        "10_hybrid_metrics_dashboard.csv",
        "10_hybrid_metrics.csv",
        "10_hybrid_dashboard_summary.csv",
    ],
    "bpr": [
        "bpr/11_bpr_metrics_dashboard.csv",
        "bpr/11_bpr_metrics.csv",
        "bpr/11_bpr_dashboard_summary.csv",
        "11_bpr_metrics_dashboard.csv",
        "11_bpr_metrics.csv",
        "11_bpr_dashboard_summary.csv",
    ],
}

RECOMMENDATION_FILE_CANDIDATES = {
    "popularity": {
        "valid": ["07_popularity_valid_recommendations_long.csv", "07_popularity_valid_recommendations_wide.csv"],
        "test": ["07_popularity_test_recommendations_long.csv", "07_popularity_test_recommendations_wide.csv"],
    },
    "cf": {
        "valid": ["08_cf_valid_recommendations_long.csv", "08_cf_valid_recommendations_wide.csv"],
        "test": ["08_cf_test_recommendations_long.csv", "08_cf_test_recommendations_wide.csv"],
    },
    "svd": {
        "valid": ["svd/09_svd_valid_recommendations_long.csv", "09_svd_valid_recommendations_long.csv", "09_svd_valid_recommendations.csv"],
        "test": ["svd/09_svd_test_recommendations_long.csv", "09_svd_test_recommendations_long.csv", "09_svd_test_recommendations.csv"],
    },
    "hybrid": {
        "valid": ["hybrid/10_hybrid_valid_recommendations.csv", "10_hybrid_valid_recommendations.csv"],
        "test": ["hybrid/10_hybrid_test_recommendations.csv", "10_hybrid_test_recommendations.csv"],
    },
    "bpr": {
        "valid": ["bpr/11_bpr_valid_recommendations_long.csv", "11_bpr_valid_recommendations_long.csv"],
        "test": ["bpr/11_bpr_test_recommendations_long.csv", "11_bpr_test_recommendations_long.csv"],
    },
}

DATASET_TABLES = {
    "dataset_stats": "01_dataset_stats.csv",
    "dataset_shapes": "01_dataset_shapes.csv",
    "rating_distribution": "01_dashboard_rating_distribution.csv",
    "user_activity": "01_user_activity.csv",
    "item_popularity": "01_item_popularity.csv",
    "year_counts": "01_year_counts.csv",
    "modelling_summary": "04_dashboard_modelling_summary.csv",
    "recipe_summary": "03_dashboard_recipe_summary.csv",
    "recipe_join": "03_dashboard_recipe_join_coverage.csv",
    "split_summary": "05_dashboard_split_summary.csv",
}

DATASET_FIGURES = [
    "01_rating_distribution.png",
    "01_user_activity_long_tail.png",
    "01_item_popularity_long_tail.png",
    "01_interactions_by_year.png",
]

BIAS_FIGURE_CANDIDATES = {
    "popularity": ["07_popularity_concentration_curve.png"],
    "svd": ["svd/09_svd_test_recommendation_concentration_curve.png", "svd/09_svd_valid_recommendation_concentration_curve.png"],
    "hybrid": ["hybrid/10_hybrid_test_recommendation_concentration_curve.png", "hybrid/10_hybrid_valid_recommendation_concentration_curve.png"],
    "bpr": ["bpr/11_bpr_test_recommendation_concentration_curve.png", "bpr/11_bpr_valid_recommendation_concentration_curve.png"],
}

BIAS_TABLE_CANDIDATES = {
    "popularity": ["07_item_popularity_table.csv", "01_item_popularity.csv"],
    "cf": ["08_cf_valid_recommendations_long.csv", "08_cf_test_recommendations_long.csv"],
    "svd": ["svd/09_svd_test_recommendation_popularity.csv", "svd/09_svd_valid_recommendation_popularity.csv"],
    "hybrid": ["hybrid/10_hybrid_test_recommendation_popularity.csv", "hybrid/10_hybrid_valid_recommendation_popularity.csv"],
    "bpr": ["bpr/11_bpr_test_recommendation_popularity.csv", "bpr/11_bpr_valid_recommendation_popularity.csv"],
}

POPULARITY_TABLE_CANDIDATES = [
    "07_item_popularity_table.csv",
    "01_item_popularity.csv",
    "svd/09_svd_item_popularity_table.csv",
    "hybrid/10_hybrid_item_popularity_table.csv",
    "bpr/11_bpr_item_popularity_table.csv",
]
