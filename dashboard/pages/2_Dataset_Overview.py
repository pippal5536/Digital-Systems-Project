from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from services.data_loader import load_generic_table
from utils.formatters import prettify_columns
from utils.paths import figures_dir


st.title("Dataset Overview")


def resolve_figure_path(stem: str) -> Path | None:
    """
    Return the first available figure path for a given stem.
    SVG is checked first because the audit pipeline saves both SVG and PNG.
    """
    base = figures_dir()
    for extension in (".svg", ".png"):
        candidate = base / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def render_table(
    title: str,
    file_name: str,
    description: str,
    *,
    optional: bool = False,
) -> None:
    """
    Render a table with academic-style contextual text.
    """
    df = load_generic_table(file_name)

    if df.empty:
        if not optional:
            st.warning(f"{title} could not be loaded from {file_name}.")
        return

    st.markdown(f"#### {title}")
    st.markdown(description)
    st.dataframe(
        prettify_columns(df),
        use_container_width=True,
        hide_index=True,
    )


def render_figure(
    title: str,
    stem: str,
    description: str,
    *,
    optional: bool = False,
) -> None:
    """
    Render a figure with academic-style contextual text.
    """
    figure_path = resolve_figure_path(stem)

    if figure_path is None:
        if not optional:
            st.warning(f"{title} could not be loaded from outputs/figures.")
        return

    st.markdown(f"#### {title}")
    st.markdown(description)
    st.image(str(figure_path), use_container_width=True)


# ============================================================
# Dataset Audit section
# ============================================================
st.header("Dataset Audit")

st.markdown(
    """
    This section presents the initial audit of the Genius Kitchen dataset prior to
    preprocessing and model development. The purpose of the audit is to establish
    the scale, completeness, consistency, and structural characteristics of the
    source data before any transformation steps are applied.

    The analysis is organised around five themes: dataset scale, temporal coverage,
    rating behaviour, metadata linkage, and interaction structure. Each table and
    figure is accompanied by brief interpretive commentary so that the section
    supports both dashboard-based inspection and academic reporting.
    """
)

st.divider()

# ============================================================
# 1. Dataset scale and temporal coverage
# ============================================================
st.subheader("1. Dataset Scale and Temporal Coverage")

st.markdown(
    """
    The first part of the audit establishes the overall size of the dataset and the
    temporal span of the available interactions. These results are important because
    they indicate whether the data volume is sufficient for recommender-system
    experimentation and whether chronological splitting can be justified.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Dataset Summary",
        "01_dashboard_dataset_summary.csv",
        """
        This table reports the headline dataset characteristics used throughout the
        project. It provides a concise summary of the user base, recipe catalogue,
        interaction count, and matrix sparsity.
        """,
    )

with col2:
    render_table(
        "Date Summary",
        "01_dashboard_date_summary.csv",
        """
        This table summarises the date parsing results and the temporal coverage of
        the interaction data. It is used to confirm that the dataset supports
        time-aware evaluation.
        """,
    )

render_figure(
    "Interactions by Year",
    "01_interactions_by_year",
    """
    This figure visualises the yearly distribution of interactions across the full
    observation window. It is intended to show whether interaction density is evenly
    distributed over time or concentrated in particular years, which is relevant
    when designing chronological train-validation-test splits.
    """,
)

st.divider()

# ============================================================
# 2. Rating behaviour
# ============================================================
st.subheader("2. Rating Behaviour")

st.markdown(
    """
    The rating structure is central to model design because it determines whether
    the data are more appropriate for explicit-feedback methods, implicit-feedback
    methods, or a comparative combination of both. The following evidence is used
    to inspect the balance and interpretability of the rating signal.
    """
)

render_table(
    "Rating Distribution Table",
    "01_dashboard_rating_distribution.csv",
    """
    This table reports the frequency and proportion of each rating value. It is
    useful for identifying class imbalance, the prevalence of high ratings, and the
    presence of non-standard values that may require explicit preprocessing
    decisions.
    """,
)

render_figure(
    "Rating Distribution",
    "01_rating_distribution",
    """
    This figure provides a visual summary of the rating pattern shown in the table
    above. It supports rapid interpretation of skewness in the feedback signal and
    helps explain why naive evaluation results may be inflated in strongly
    imbalanced rating environments.
    """,
)

st.divider()

# ============================================================
# 3. Metadata linkage and data integrity
# ============================================================
st.subheader("3. Metadata Linkage and Data Integrity")

st.markdown(
    """
    Recommendation experiments depend on reliable linkage between interaction data
    and recipe metadata. This part of the audit therefore examines whether recipe
    identifiers align consistently across the interaction file and the two recipe
    files, and whether basic integrity checks reveal any structural duplication
    concerns.
    """
)

render_table(
    "Join Coverage Table",
    "01_dashboard_join_coverage.csv",
    """
    This table compares recipe-ID coverage between the interaction data and the two
    recipe metadata files. It is especially important for determining which recipe
    source should be treated as the primary reference table during downstream
    feature engineering and model preparation.
    """,
)

render_figure(
    "Recipe ID Join Coverage",
    "01_join_coverage",
    """
    This figure presents the comparative join-coverage results in visual form. It
    is intended to make differences in metadata coverage immediately visible and to
    support the selection of an appropriate master recipe table for the project.
    """,
)

render_table(
    "Duplicate Diagnostics",
    "01_report_duplicates.csv",
    """
    This table summarises duplicate checks carried out during the audit. It provides
    supporting evidence on dataset consistency and confirms whether any duplicate
    patterns are likely to interfere with downstream preprocessing or evaluation.
    """,
    optional=True,
)

st.divider()

# ============================================================
# 4. Missingness assessment
# ============================================================
st.subheader("4. Missingness Assessment")

st.markdown(
    """
    Missing-data inspection is included to identify whether null values are broadly
    distributed across the datasets or concentrated in a small set of variables.
    This matters because uneven missingness may affect data preparation choices,
    especially where recipe attributes are later used in content-based or hybrid
    models.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_figure(
        "Missingness Overview: Interactions",
        "01_missingness_overview_interactions",
        """
        This figure highlights the columns with the greatest proportion of missing
        values in the interaction dataset. It is used to assess whether the core
        behavioural data are sufficiently complete for modelling.
        """,
    )

with col2:
    render_figure(
        "Missingness Overview: Raw Recipes",
        "01_missingness_overview_raw_recipes",
        """
        This figure highlights the columns with the greatest proportion of missing
        values in the raw recipe dataset. It supports evaluation of how suitable the
        raw metadata are for later descriptive analysis and feature extraction.
        """,
    )

# Optional, only if the figure exists later
render_figure(
    "Missingness Overview: Preprocessed Recipes",
    "01_missingness_overview_pp_recipes",
    """
    This optional figure can be included when the corresponding audit output is
    available. It serves the same purpose as the previous missingness summaries,
    but for the preprocessed recipe file.
    """,
    optional=True,
)

st.divider()

# ============================================================
# 5. Long-tail interaction structure
# ============================================================
st.subheader("5. Long-Tail Interaction Structure")

st.markdown(
    """
    Long-tail structure is a defining property of many real-world recommender
    datasets. The following figures examine concentration patterns on both the item
    side and the user side. These results help frame later discussions of sparsity,
    popularity bias, coverage, and cold-start difficulty.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_figure(
        "Item Popularity Long-Tail Distribution",
        "01_item_popularity_long_tail",
        """
        This figure ranks recipes by interaction frequency and displays the
        concentration of activity across the catalogue. It is intended to show
        whether a small number of recipes dominate observed engagement.
        """,
    )

with col2:
    render_figure(
        "User Activity Long-Tail Distribution",
        "01_user_activity_long_tail",
        """
        This figure ranks users by interaction frequency and shows the distribution
        of participation across the user base. It is useful for identifying whether
        the dataset contains a large proportion of minimally active users.
        """,
    )

st.divider()

st.markdown(
    """
    Taken together, the audit evidence provides a structured foundation for the
    subsequent preprocessing and modelling stages. The section demonstrates that the
    dataset is large and usable, while also identifying the main methodological
    issues that must be considered in later stages of the project, including
    sparsity, rating imbalance, metadata coverage differences, and long-tail
    behaviour.
    """
)
st.divider()

# ============================================================
# Interaction Cleaning section
# ============================================================
st.header("Interaction Cleaning")

st.markdown(
    """
    This section presents the interaction-cleaning stage that follows the initial
    dataset audit. Whereas the audit established the structural condition of the
    raw interaction data, the present stage formalises the preprocessing decisions
    required to produce modelling-ready interaction datasets.

    The purpose of this section is therefore not to repeat the full audit, but to
    demonstrate how the validated audit findings were translated into explicit
    cleaning rules. In particular, the section documents the standardisation of
    essential fields, the treatment of dates and duplicate records, the handling
    of review text, the modelling interpretation of rating = 0, and the effect of
    optional low-signal filtering on data retention.

    The discussion is organised around four themes: core cleaning outcomes,
    treatment of the rating signal, persistence of temporal and long-tail
    structure after cleaning, and the trade-off introduced by filtering
    thresholds.
    """
)

st.divider()

# ============================================================
# 1. Core cleaning outcomes
# ============================================================
st.subheader("1. Core Cleaning Outcomes")

st.markdown(
    """
    The first stage of interaction cleaning confirms whether the core behavioural
    records required any substantive correction before modelling. This is
    important because a dataset that is already structurally sound should be
    cleaned conservatively, so that preprocessing improves modelling readiness
    without unnecessarily altering the observed interaction history.
    """
)

render_table(
    "Interaction Cleaning Summary",
    "02_dashboard_interaction_cleaning_summary.csv",
    """
    This table summarises the principal outputs of the cleaning stage. It reports
    the retained row count after core validation, the size of the explicit and
    implicit interaction views, the number of zero-rated observations, the amount
    of preserved review text, the cleaned temporal range, and the size of the
    optional lightly filtered dataset.

    From an analytical perspective, the table is intended to demonstrate that the
    cleaning phase functioned primarily as a rule-enforcement and representation
    stage rather than a major corrective intervention. The absence of row loss in
    the core cleaned output indicates that the raw interaction file was already
    structurally suitable for downstream recommendation work.
    """,
)

render_table(
    "Review Text Summary",
    "02_dashboard_review_summary.csv",
    """
    This table reports the presence or absence of review text after cleaning. It
    is included because review coverage is useful descriptive context, even though
    review content is deliberately retained only as auxiliary information in this
    phase and is not transformed into modelling features.
    """,
    optional=True,
)

render_figure(
    "Review Text Coverage",
    "02_review_text_coverage",
    """
    This figure visualises the proportion of interactions that contain review
    text. The result shows that review availability remains extremely high after
    cleaning, which confirms that text coverage is not a substantive data-quality
    problem. However, the section also makes clear that preserving review text
    does not imply using it as a feature at this stage of the project.
    """,
)

st.divider()

# ============================================================
# 2. Rating treatment and modelling views
# ============================================================
st.subheader("2. Rating Treatment and Modelling Views")

st.markdown(
    """
    The rating signal is central to the methodological design of the recommender
    system experiments. The cleaning stage therefore does more than preserve the
    original ratings: it makes an explicit modelling distinction between rated and
    unrated observed behaviour. This is especially important because the dataset
    contains the non-standard value rating = 0, which cannot safely be interpreted
    as equivalent to the explicit scale from 1 to 5.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Cleaned Rating Distribution",
        "02_dashboard_rating_distribution.csv",
        """
        This table reports the cleaned rating frequencies and percentages. It is
        included to show whether the rating profile changed during cleaning and to
        support the modelling decision on how zero-valued observations should be
        treated.
        """,
    )

with col2:
    render_table(
        "Filtering Threshold Results",
        "02_dashboard_filtering_results.csv",
        """
        This table compares row retention, remaining users, and remaining items
        under increasingly strict iterative filtering thresholds. It supports the
        methodological discussion of whether low-signal filtering should be treated
        as a default preprocessing rule or as a model-specific experimental choice.
        """,
    )

render_figure(
    "Cleaned Rating Distribution",
    "02_rating_distribution",
    """
    This figure shows that the rating profile remains heavily concentrated at the
    upper end of the scale after cleaning. In substantive terms, the cleaning
    stage preserves the strong positive-feedback bias already identified in the
    audit. The figure therefore supports the decision to create a separate
    explicit-rating view while handling zero-valued observations as observed but
    unrated behaviour rather than as conventional negative feedback.
    """,
)

st.markdown(
    """
    In academic terms, this part of the cleaning stage justifies the creation of
    two modelling views. The explicit dataset retains only rows with ratings from
    1 to 5, whereas the implicit dataset preserves all observed interactions. This
    separation ensures that later models can be compared fairly without forcing a
    single interpretation onto the ambiguous zero-rating class.
    """
)

st.divider()

# ============================================================
# 3. Temporal continuity after cleaning
# ============================================================
st.subheader("3. Temporal Continuity After Cleaning")

st.markdown(
    """
    A useful cleaning stage should preserve the temporal structure of the original
    interaction history unless invalid records genuinely need to be removed. This
    subsection therefore examines whether the cleaned interaction history still
    reflects the same chronological pattern identified in the audit.
    """
)

render_figure(
    "Cleaned Interactions by Year",
    "02_interactions_by_year",
    """
    This figure presents the yearly interaction counts after cleaning. The shape
    of the distribution shows that the cleaned data still preserve the historical
    activity pattern observed in the raw dataset, with interaction volume rising
    through the early and middle years of the timeline before declining later.
    Because no rows were removed during essential type validation, date parsing,
    or exact deduplication, the temporal structure remains suitable for later
    chronological train-validation-test splitting.
    """,
)

st.divider()

# ============================================================
# 4. Long-tail structure after cleaning
# ============================================================
st.subheader("4. Long-Tail Structure After Cleaning")

st.markdown(
    """
    Beyond correctness, the cleaning stage must also preserve the behavioural
    structure that defines the recommendation problem itself. Two of the most
    important structural properties are item popularity concentration and uneven
    user participation. These are examined here to confirm that the cleaned
    outputs still represent a realistic sparse recommender environment.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_figure(
        "Item Popularity Long-Tail Distribution (Cleaned)",
        "02_item_popularity_long_tail",
        """
        This figure ranks recipes by interaction frequency after cleaning. The
        resulting long-tail shape indicates that a small minority of recipes
        continue to receive disproportionate attention, while most items remain
        weakly observed. This confirms that popularity concentration is a genuine
        structural property of the dataset rather than an artefact of the raw
        representation.
        """,
    )

with col2:
    render_figure(
        "User Activity Long-Tail Distribution (Cleaned)",
        "02_user_activity_long_tail",
        """
        This figure ranks users by interaction count after cleaning. It shows that
        the user side of the matrix remains highly uneven, with many minimally
        active users and a much smaller group of highly active contributors. This
        has direct implications for sparsity, robustness, and cold-start
        sensitivity in later recommender experiments.
        """,
    )

st.divider()

# ============================================================
# 5. Filtering threshold assessment
# ============================================================
st.subheader("5. Filtering Threshold Assessment")

st.markdown(
    """
    Optional low-signal filtering was evaluated in order to understand the trade-
    off between matrix robustness and information retention. In sparse
    recommender settings, filtering can reduce noise and remove extremely weakly
    connected users or items. However, aggressive filtering also risks discarding
    a large portion of the already limited behavioural evidence.
    """
)

render_figure(
    "Filtering Threshold Retention",
    "02_filtering_threshold_retention",
    """
    This figure compares the percentage of rows retained under progressively
    stricter minimum user-item interaction thresholds. The pattern demonstrates
    that even moderate threshold increases produce noticeable data loss, while
    stronger thresholds remove a substantial share of the interaction history.

    Methodologically, this result supports a conservative preprocessing stance.
    The full cleaned dataset remains the most appropriate default Phase 2 output,
    while lightly filtered variants can be reserved for later model-specific
    experiments where robustness gains may justify the associated loss of
    coverage.
    """,
)

st.markdown(
    """
    Taken together, the evidence from this section shows that interaction cleaning
    preserved the structural integrity of the behavioural dataset while making the
    modelling assumptions explicit. The cleaning process did not materially reduce
    the dataset through error correction, because the audit had already shown the
    source file to be highly consistent. Instead, its main contribution was to
    formalise the representation of interactions for downstream experimentation:
    dates were standardised for chronological evaluation, review text was retained
    as contextual information, rating = 0 was separated from the explicit target,
    and optional threshold filtering was evaluated without being imposed as a
    compulsory preprocessing rule.
    """
)
st.divider()

# ============================================================
# Recipe Preprocessing section
# ============================================================
st.header("Recipe Preprocessing")

st.markdown(
    """
    This section presents the recipe-preprocessing stage, which prepares the recipe
    metadata tables for downstream feature engineering, modelling, and dashboard
    exploration. The central objective of this phase is to preserve the full recipe
    catalogue while enriching it with structured features from the auxiliary
    preprocessed recipe table wherever such metadata are available.

    The preprocessing logic therefore treats the readable raw recipe table as the
    primary metadata source and uses the preprocessed recipe table as an enrichment
    layer rather than a replacement source. This design choice is important because
    the structured preprocessed table offers valuable token- and technique-based
    metadata, but does not cover the full set of recipe entities referenced in the
    wider dataset.

    The discussion is organised around five themes: recipe-source coverage and join
    strategy, feature completeness, structural recipe characteristics, temporal and
    tag-based descriptive patterns, and the overall suitability of the joined recipe
    table for later recommendation experiments.
    """
)

st.divider()

# ============================================================
# 1. Recipe-source coverage and join strategy
# ============================================================
st.subheader("1. Recipe-Source Coverage and Join Strategy")

st.markdown(
    """
    The first task in recipe preprocessing is to establish which recipe table should
    act as the primary source of truth. This is a crucial methodological decision,
    because later modelling and dashboard stages require a recipe table that is both
    complete and compatible with the cleaned interaction data.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Recipe Summary",
        "03_dashboard_recipe_summary.csv",
        """
        This table summarises the principal characteristics of the cleaned joined
        recipe table. It is intended to report the size of the recipe catalogue,
        the extent of preprocessed-feature availability, and compact descriptive
        statistics such as median preparation time, median number of steps, and
        median number of ingredients.
        """,
    )

with col2:
    render_table(
        "Recipe Table Join Coverage",
        "03_dashboard_recipe_join_coverage.csv",
        """
        This table compares recipe-ID coverage between the raw recipe table and the
        preprocessed recipe table. It provides the core evidence for determining why
        the raw recipe table should be treated as the base source and why a simple
        replacement strategy would be inappropriate.
        """,
    )

render_figure(
    "Recipe Table Join Coverage",
    "03_recipe_join_coverage",
    """
    This figure visualises the proportion of raw recipe IDs that are covered by the
    preprocessed recipe table. The result makes clear that the preprocessed table is
    a substantial but incomplete enrichment source. The chart therefore supports the
    decision to preserve all raw recipes and to attach structured preprocessed
    features only where a valid recipe_id match exists.
    """,
)

render_figure(
    "Interaction–Recipe Coverage",
    "03_interaction_recipe_coverage",
    """
    This figure reports the alignment between the cleaned interaction dataset and the
    joined recipe table. Its purpose is to confirm that the recipe-preprocessing
    strategy remains fully compatible with the behavioural data used in later
    recommendation experiments. A complete match here indicates that no recipe
    entities referenced by interactions were lost during recipe preprocessing.
    """,
)

st.divider()

# ============================================================
# 2. Feature completeness and missingness structure
# ============================================================
st.subheader("2. Feature Completeness and Missingness Structure")

st.markdown(
    """
    After the join is established, the next issue is feature completeness. In this
    phase, missingness is not interpreted solely as a defect. Instead, it must be
    separated into two distinct cases: missing core metadata in the base recipe
    catalogue, and structurally missing enrichment features for recipes that do not
    appear in the auxiliary preprocessed table.
    """
)

render_figure(
    "Top Missing Recipe Features",
    "03_recipe_feature_missingness",
    """
    This figure shows that the dominant source of missingness in the joined recipe
    table comes from the structured features inherited from the preprocessed recipe
    source. These columns share the same missingness rate because they are jointly
    absent whenever a raw recipe has no matching entry in the preprocessed table.
    By contrast, core metadata such as the recipe description exhibit only modest
    missingness, while most essential structural fields remain effectively complete.
    """,
)

st.markdown(
    """
    From an interpretive perspective, this means the joined recipe table remains
    analytically strong even though structured enrichment is incomplete. The
    missingness pattern reflects limited auxiliary coverage rather than widespread
    quality failure in the base recipe metadata.
    """
)

st.divider()

# ============================================================
# 3. Structural characteristics of recipes
# ============================================================
st.subheader("3. Structural Characteristics of Recipes")

st.markdown(
    """
    The joined recipe table also allows the structure of the recipe catalogue to be
    described in compact and modelling-relevant terms. The following figures focus
    on preparation time, ingredient count, step count, and calorie-level grouping.
    Together, they provide a descriptive profile of recipe complexity and content
    variation.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_figure(
        "Recipe Ingredient Count Distribution",
        "03_recipe_ingredient_distribution",
        """
        This figure shows the distribution of ingredient counts across the recipe
        catalogue. The pattern is right-skewed but concentrated around moderate
        values, indicating that most recipes contain neither extremely small nor
        extremely large ingredient lists. This is useful descriptive context for
        later content-based modelling and recipe-complexity interpretation.
        """,
    )

with col2:
    render_figure(
        "Recipe Step Count Distribution",
        "03_recipe_step_distribution",
        """
        This figure shows the distribution of recipe step counts. The distribution is
        again right-skewed, but the concentration of observations in lower-to-mid
        ranges suggests that the typical recipe is procedurally manageable rather
        than extremely long. This supports the use of step-derived compact features
        as informative but not excessively sparse recipe descriptors.
        """,
    )

col3, col4 = st.columns(2)

with col3:
    render_figure(
        "Recipe Preparation Time Distribution",
        "03_recipe_minutes_distribution",
        """
        This figure displays the distribution of recipe preparation times. The chart
        appears highly compressed because a small number of extreme values stretch the
        horizontal scale. For that reason, the visual should be interpreted alongside
        summary statistics, with the median preparation time providing a more stable
        description of the typical recipe than the full raw spread alone.
        """,
    )

with col4:
    render_figure(
        "Calorie Level Distribution",
        "03_calorie_level_distribution",
        """
        This figure shows the distribution of the calorie-level indicator available
        from the structured preprocessed recipe metadata. The categories are
        reasonably well represented, although lower coded calorie levels are more
        common than the highest category. This provides a simple but useful content
        attribute for later descriptive analysis and hybrid recommendation work.
        """,
    )

st.divider()

# ============================================================
# 4. Temporal profile and tag-based descriptors
# ============================================================
st.subheader("4. Temporal Profile and Tag-Based Descriptors")

st.markdown(
    """
    In addition to structural numeric features, the preprocessing stage derives a
    small set of readable binary indicators from recipe tags and retains the recipe
    submission date for temporal description. These variables support both dashboard
    explanation and later content-oriented modelling extensions.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Selected Tag Summary",
        "03_dashboard_tag_summary.csv",
        """
        This table summarises the frequency of selected exact-match recipe tags used
        as compact binary indicators. It is intended to provide a readable overview
        of broad recipe characteristics such as ease, dietary orientation, and meal
        context.
        """,
        optional=True,
    )

with col2:
    render_figure(
        "Selected Tag Frequency",
        "03_selected_tag_frequency",
        """
        This figure visualises the distribution of the selected tag indicators. The
        dominant presence of tags such as easy, healthy, and vegetarian suggests that
        broad accessibility and dietary labels are common descriptive signals in the
        recipe metadata. At the same time, the absence of some exact tags should be
        interpreted cautiously, since it may reflect source-vocabulary conventions
        rather than the true absence of those recipe types.
        """,
    )

render_figure(
    "Recipe Submissions by Year",
    "03_recipe_submissions_by_year",
    """
    This figure shows the yearly distribution of recipe submissions. It provides
    temporal context for the growth and later decline of recipe additions to the
    catalogue. This is useful for understanding how the metadata base evolved over
    time and for interpreting the historical concentration of recipe availability
    relative to later interaction behaviour.
    """,
)

st.divider()

# ============================================================
# 5. Section conclusion
# ============================================================
st.subheader("5. Interpretation")

st.markdown(
    """
    The evidence presented in this section shows that recipe preprocessing succeeded
    in producing a robust joined recipe representation suitable for downstream use.
    The raw recipe table was correctly retained as the base source because it
    preserves the complete catalogue, while structured preprocessed metadata were
    attached as an auxiliary enrichment layer wherever matching recipe identifiers
    were available.

    This strategy avoids unnecessary recipe loss, maintains full compatibility with
    the cleaned interaction dataset, and exposes a richer feature space for later
    hybrid and content-aware recommendation experiments. The main limitation is not
    instability in the base recipe catalogue, but incomplete coverage of auxiliary
    preprocessed features. That limitation is transparent, measurable, and can be
    handled explicitly through the retained missingness indicators.
    """
)
st.divider()

# ============================================================
# Modelling Dataset Construction section
# ============================================================
st.header("Modelling Dataset Construction")

st.markdown(
    """
    This section presents the construction of the modelling datasets used in later
    recommendation experiments. The purpose of this phase is to transform the
    cleaned behavioural data and the joined recipe metadata into modelling-ready
    representations that support explicit-rating, implicit-feedback, and
    metadata-enriched recommendation workflows.

    The logic of this stage follows directly from the earlier preprocessing
    decisions. The cleaned interaction table provides the behavioural base, the
    joined recipe table provides recipe-side descriptors, the non-standard
    rating = 0 class is excluded from the explicit-rating target, and the same
    observation is retained in the implicit-feedback representation. The result is
    a small family of related but distinct modelling datasets rather than a single
    universal table.

    The discussion is organised around four themes: explicit versus implicit
    target construction, dataset size and entity retention, join completeness, and
    recipe-side PP feature availability after joining.
    """
)

st.divider()

# ============================================================
# 1. Explicit and implicit modelling views
# ============================================================
st.subheader("1. Explicit and Implicit Modelling Views")

st.markdown(
    """
    The first task in modelling-dataset construction is to separate explicit-rating
    prediction from implicit-feedback learning. This distinction is methodologically
    necessary because the behavioural data contain both standard ratings from 1 to 5
    and observed but unrated interactions corresponding to rating = 0.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Explicit Rating Distribution",
        "04_dashboard_explicit_rating_distribution.csv",
        """
        This table reports the rating frequencies in the explicit modelling dataset
        after the unrated observation class has been removed. It is used to show the
        class balance of the explicit target that will later be passed to rating-
        based recommendation methods.
        """,
    )

with col2:
    render_table(
        "Implicit Observation Type Summary",
        "04_dashboard_implicit_unrated_distribution.csv",
        """
        This table summarises the composition of the implicit modelling dataset by
        distinguishing rated interactions from unrated observed interactions. It is
        included to show how much of the behavioural signal is preserved when the
        full interaction history is retained.
        """,
    )

render_figure(
    "Explicit Rating Distribution",
    "04_explicit_rating_distribution",
    """
    This figure shows the class structure of the explicit target after unrated
    observations have been removed. Compared with the earlier cleaned interaction
    distribution, the remaining explicit ratings are even more concentrated at the
    upper end of the scale. This indicates that explicit-rating models will still
    operate in a strongly positive and imbalanced prediction environment.
    """,
)

render_figure(
    "Implicit Dataset Observation Types",
    "04_implicit_observation_types",
    """
    This figure shows the composition of the implicit modelling dataset. The result
    demonstrates that the implicit view preserves the full behavioural history by
    retaining both rated interactions and unrated observed interactions. This is
    important because it prevents the modelling stage from discarding legitimate
    behavioural evidence simply because no explicit rating was recorded.
    """,
)

st.divider()

# ============================================================
# 2. Dataset size and entity retention
# ============================================================
st.subheader("2. Dataset Size and Entity Retention")

st.markdown(
    """
    After defining the explicit and implicit targets, the next question is how much
    data each modelling view retains. This matters because different modelling
    families do not merely operate on different labels; they can also differ in the
    number of rows, users, and recipes that remain available for learning and
    evaluation.
    """
)

render_table(
    "Modelling Dataset Summary",
    "04_dashboard_modelling_summary.csv",
    """
    This table reports the row counts, user counts, recipe counts, and temporal
    ranges of the principal modelling datasets. It is intended to show how much
    behavioural and entity coverage is preserved in each representation before any
    temporal splitting is applied.
    """,
)

col1, col2 = st.columns(2)

with col1:
    render_figure(
        "Rows in Modelling Datasets",
        "04_modelling_dataset_rows",
        """
        This figure compares the number of observations available in the explicit,
        implicit, and joined modelling datasets. It makes clear that the explicit
        dataset is smaller because unrated observations are removed, whereas the
        implicit and joined datasets preserve the full cleaned interaction count.
        """,
    )

with col2:
    render_figure(
        "Users and Recipes Across Modelling Datasets",
        "04_modelling_dataset_entities",
        """
        This figure compares the number of users and recipes retained in each
        modelling dataset. The key interpretation is that the explicit target does
        not only remove rows; it also reduces entity coverage, because some users
        and recipes are represented only through interactions that are not part of
        the explicit rating target.
        """,
    )

st.divider()

# ============================================================
# 3. Join completeness for modelling
# ============================================================
st.subheader("3. Join Completeness for Modelling")

st.markdown(
    """
    The joined modelling dataset is only useful if the alignment between interaction
    rows and recipe-side metadata remains sufficiently complete. This subsection
    therefore reports the practical outcome of the interaction-to-recipe join used
    for downstream modelling.
    """
)

render_table(
    "Join Summary",
    "04_dashboard_join_summary.csv",
    """
    This table summarises the interaction-side join outcome, including interaction
    volume, user and recipe coverage, and the residual level of missing recipe
    linkage after joining. It is intended to demonstrate whether the modelling
    pipeline retains effective compatibility between behavioural and recipe-side
    data.
    """,
)

st.markdown(
    """
    From an analytical perspective, this result should be interpreted as evidence
    that the modelling pipeline preserves essentially complete recipe alignment. In
    other words, the joined modelling table remains suitable for downstream hybrid
    recommendation experiments because the interaction history is not materially
    fragmented by failed recipe matches.
    """
)

st.divider()

# ============================================================
# 4. PP feature availability after joining
# ============================================================
st.subheader("4. PP Feature Availability After Joining")

st.markdown(
    """
    Although the joined modelling dataset preserves interaction-to-recipe alignment,
    it is still important to determine how much of that joined data carries the
    auxiliary PP features derived from the preprocessed recipe table. This affects
    the practical usefulness of the joined table for hybrid or metadata-enriched
    recommendation methods.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "PP Feature Coverage Summary",
        "04_dashboard_pp_feature_summary.csv",
        """
        This table reports how many joined modelling rows include PP-derived recipe
        features and how many do not. It provides a direct measure of the usable
        metadata coverage available for downstream feature-aware modelling.
        """,
    )

with col2:
    render_figure(
        "PP Feature Coverage After Join",
        "04_pp_feature_coverage",
        """
        This figure visualises the proportion of joined modelling rows that retain
        PP-derived recipe features. A notable point here is that interaction-level
        PP coverage can exceed recipe-level PP coverage, which suggests that recipes
        with PP features account for a relatively large share of observed user
        behaviour. This is favourable for later hybrid modelling, because the
        coverage of enriched rows is stronger than the recipe-table coverage alone
        might initially suggest.
        """,
    )

st.divider()

# ============================================================
# 5. Interpretation
# ============================================================
st.subheader("5. Interpretation")

st.markdown(
    """
    Overall, the modelling-dataset construction stage successfully converts the
    earlier preprocessing decisions into a coherent experimental data family. The
    explicit dataset isolates standard rating prediction, the implicit dataset
    preserves the full behavioural signal, and the joined dataset adds recipe-side
    descriptors while maintaining effective alignment with the interaction data.

    This design is methodologically important because it avoids forcing different
    recommendation paradigms into a single representation. At the same time, it
    preserves sufficient metadata coverage to support feature-aware and hybrid
    experiments in later stages. The remaining limitations are therefore not due to
    join failure or severe data loss, but to the already-known structural issues of
    the project: positive feedback imbalance, sparse behaviour, and incomplete PP
    coverage at the recipe level.
    """
)
st.divider()

# ============================================================
# Per-User Chronological Splitting section
# ============================================================
st.header("Per-User Chronological Splitting")

st.markdown(
    """
    This section presents the temporal split construction stage used to prepare the
    modelling datasets for downstream training and evaluation. The purpose of this
    phase is to ensure that recommendation models are assessed under time-aware
    conditions, so that later user behaviour is never used to train models that are
    evaluated on earlier interactions.

    The final split policy is per-user chronological rather than random. This means
    that each user's observed interaction history is ordered in time and then divided
    into train, validation, and test portions. The result is a more realistic
    evaluation setting for recommendation tasks, because models are trained on a
    user's earlier behaviour and evaluated on that same user's later behaviour.

    The section is organised around four themes: split policy and temporal coverage,
    row and entity retention across splits, train-fitted ID mappings, and the degree
    of unseen-user or unseen-recipe exposure in the holdout partitions.
    """
)

st.divider()

# ============================================================
# 1. Split policy and temporal coverage
# ============================================================
st.subheader("1. Split Policy and Temporal Coverage")

st.markdown(
    """
    The first task in temporal split construction is to define a leakage-resistant
    evaluation policy. In this project, the split is not performed by random
    sampling. Instead, each user's interaction history is ordered chronologically
    and then partitioned into train, validation, and test subsets using a fixed
    70 / 15 / 15 allocation.

    This approach should be interpreted carefully. Under a per-user chronological
    strategy, the aggregate calendar ranges of the split datasets can overlap.
    Such overlap does not indicate leakage. The important condition is that, within
    each user's own history, earlier interactions remain in train and later
    interactions are reserved for validation or test.
    """
)

render_table(
    "Per-User Split Summary",
    "05_dashboard_split_summary.csv",
    """
    This table reports the row counts, user counts, recipe counts, and date ranges
    for the explicit, implicit, and joined datasets after per-user temporal
    splitting. It is included to document the practical outcome of the split policy
    and to provide a compact audit trail for later modelling and evaluation stages.
    """,
)

col1, col2 = st.columns(2)

with col1:
    render_figure(
        "Rows Across Per-User Chronological Splits",
        "05_rows_across_splits",
        """
        This figure compares the number of observations retained in the train,
        validation, and test partitions across the modelling datasets. The chart
        confirms that the training partition remains the dominant component in each
        case, while still preserving substantial holdout data for model selection
        and final evaluation.
        """,
    )

with col2:
    render_figure(
        "Date Coverage Across Per-User Temporal Splits",
        "05_date_coverage_across_splits",
        """
        This figure visualises the temporal span of each split. The broad overlap
        between split date ranges is expected under a per-user chronological policy,
        because the split is applied within user histories rather than through a
        single global calendar cut. The figure is therefore useful for explaining
        why temporal realism can coexist with overlapping aggregate date coverage.
        """,
    )

st.divider()

# ============================================================
# 2. Row and entity retention across splits
# ============================================================
st.subheader("2. Row and Entity Retention Across Splits")

st.markdown(
    """
    The next issue is how much behavioural and entity coverage is retained after
    splitting. This matters because temporal evaluation should preserve a large and
    representative training base without leaving the validation and test partitions
    too small to support meaningful assessment.

    The split figures show that the explicit dataset remains smaller than the
    implicit and joined datasets, which is consistent with the earlier exclusion of
    unrated observations from the explicit target. By contrast, the implicit and
    joined datasets preserve the full behavioural record and therefore retain larger
    train, validation, and test partitions.
    """
)

render_figure(
    "Users and Recipes Across Per-User Splits",
    "05_entities_across_splits",
    """
    This figure compares user and recipe counts across all split partitions. The
    training split retains the broadest entity coverage, which is desirable because
    it maximises the catalogue and user history available for learning. Validation
    and test remain smaller but still substantial, which supports meaningful
    holdout-based evaluation without abandoning temporal realism.
    """,
)

st.divider()

# ============================================================
# 3. Train-fitted ID mappings
# ============================================================
st.subheader("3. Train-Fitted ID Mappings")

st.markdown(
    """
    After splitting, user and recipe ID mappings are fitted using the training data
    only. This is an important anti-leakage step, because it ensures that validation
    and test encodings do not incorporate information from entities that first
    appear outside the training partition.

    Separate mapping tables are created for the explicit and implicit modelling
    datasets. This keeps each recommendation setting self-contained and makes later
    experiments easier to reproduce. The joined dataset does not require a separate
    mapping family for this purpose because its interaction backbone follows the
    same temporal structure as the implicit interaction view.
    """
)

render_table(
    "Mapping Summary",
    "05_dashboard_mapping_summary.csv",
    """
    This table summarises the train-fitted mapping tables generated for the
    modelling pipeline. It documents the size of the user and recipe index spaces
    learned from training data and therefore provides a concise description of the
    train-side encoding scope available to downstream models.
    """,
)

st.divider()

# ============================================================
# 4. Unseen users and recipes outside train
# ============================================================
st.subheader("4. Unseen Users and Recipes Outside Train")

st.markdown(
    """
    Temporal splitting inevitably creates some degree of cold-start exposure outside
    the training partition. Validation and test may contain users or recipes that
    were not observed in train, and those cases produce missing mapped indices when
    train-fitted encoders are applied.

    These rows are not errors and should not be silently removed at this stage.
    Instead, they are retained and reported so that later modelling work can decide
    whether to apply fallback logic, filtered evaluation, or separate cold-start
    analysis.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Unseen Mapping Summary",
        "05_dashboard_unseen_summary.csv",
        """
        This table reports the number and percentage of validation and test rows
        that contain a user ID, recipe ID, or either index that was not observed in
        the training partition. It is included as a direct measure of holdout
        cold-start exposure.
        """,
    )

with col2:
    render_figure(
        "Rows with Unseen User or Recipe IDs",
        "05_unseen_mapping_rates",
        """
        This figure visualises unseen-ID exposure in the holdout partitions. The
        observed rates show that cold-start cases remain present, but do not
        dominate the validation and test data. This is a useful outcome because it
        preserves realism while still leaving most holdout rows connected to the
        train-fitted interaction space.
        """,
    )

st.divider()

# ============================================================
# 5. Interpretation
# ============================================================
st.subheader("5. Interpretation")

st.markdown(
    """
    Overall, the temporal split stage produces a coherent evaluation framework for
    the recommender experiments. The use of per-user chronological splitting aligns
    the data preparation process with the practical recommendation task: earlier
    behaviour is used to learn user preferences, while later behaviour is reserved
    for validation and test.

    The split outputs also provide a strong compromise between realism and usable
    coverage. Training remains large, holdout partitions remain substantial, and
    cold-start exposure is measured explicitly rather than hidden. The resulting
    artefacts therefore support later collaborative filtering, matrix factorisation,
    and hybrid modelling work without introducing temporal leakage.
    """
)

st.divider()

# ============================================================
# Feature Engineering section
# ============================================================
st.header("Feature Engineering")

st.markdown(
    """
    This section presents the feature-engineering stage used to convert the temporal
    split datasets into reusable modelling inputs. The purpose of this phase is not
    to train models directly, but to construct leakage-safe user-level and item-level
    feature tables, together with holdout feature datasets for validation and test.

    The design follows the chronological logic established in the previous section.
    Training-derived aggregates are treated as the only valid source for user and item
    summary features, and the holdout splits are mapped using those train-derived
    statistics rather than recomputed from future observations. This ensures that the
    feature pipeline remains temporally realistic and methodologically consistent with
    downstream recommendation experiments.

    The discussion is organised around four themes: the structure of the feature
    datasets, coverage of user and item training features, missingness in holdout
    splits, and the dominant sources of null values in the final feature outputs.
    """
)

st.divider()

# ============================================================
# 1. Feature dataset structure
# ============================================================
st.subheader("1. Feature Dataset Structure")

st.markdown(
    """
    The first task in this phase is to clarify what the feature outputs represent.
    The training feature tables are aggregate feature stores rather than direct
    interaction rows: one table summarises users from training history, and the
    other summarises items from training history combined with recipe-side metadata.
    The validation and test feature tables, by contrast, are row-level holdout
    datasets enriched with those train-derived aggregates.

    This distinction is important because it explains why the training feature
    tables and the holdout feature tables have different row-count meanings.
    User and item training features are keyed by entity, whereas validation and
    test features remain keyed by interaction rows.
    """
)

render_table(
    "Feature Dataset Summary",
    "06_dashboard_feature_dataset_summary.csv",
    """
    This table summarises the main Phase 6 outputs. It reports the size of the
    user feature training table, the item feature training table, and the
    validation and test holdout feature datasets used in later modelling.
    """,
)

render_figure(
    "Rows in Phase 6 Feature Datasets",
    "06_feature_dataset_rows",
    """
    This figure visualises the sizes of the principal Phase 6 datasets. It shows
    that the user and item training feature tables operate at entity level, while
    the validation and test outputs remain row-level holdout feature datasets.
    The chart therefore helps distinguish aggregate feature stores from model
    evaluation inputs.
    """,
)

st.divider()

# ============================================================
# 2. Training feature coverage
# ============================================================
st.subheader("2. Training Feature Coverage")

st.markdown(
    """
    The next question concerns the extent of feature coverage available from the
    training split. User-level and item-level features are both built from training
    history, but they do not necessarily cover the same number of entities. This
    matters because the breadth of train-side feature coverage shapes how much of
    the later holdout data can be represented without missing values.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "User Feature Summary",
        "06_dashboard_user_feature_summary.csv",
        """
        This table reports high-level descriptive statistics for the user feature
        training table. It is intended to summarise the scale of user coverage and
        the average behavioural profile captured by the train-derived user
        aggregates.
        """,
    )

with col2:
    render_table(
        "Item Feature Summary",
        "06_dashboard_item_feature_summary.csv",
        """
        This table reports high-level descriptive statistics for the item feature
        training table. It summarises the scale of item coverage and the average
        behavioural and recipe-linked characteristics captured by train-derived
        item aggregates.
        """,
    )

render_figure(
    "Training Feature Coverage Counts",
    "06_training_feature_coverage",
    """
    This figure compares the number of users and items covered by the training
    feature tables. The result shows that user-side coverage is broader than
    item-side coverage, which is expected because every user in the per-user
    temporal framework contributes some training history, whereas item coverage
    depends on whether a recipe has appeared in the training interaction set.
    """,
)

st.divider()

# ============================================================
# 3. Holdout feature mapping and missingness
# ============================================================
st.subheader("3. Holdout Feature Mapping and Missingness")

st.markdown(
    """
    Once train-side feature tables have been built, they are applied to the
    validation and test splits. This is the key leakage-safe step in the phase:
    holdout rows inherit train-derived user and item aggregates rather than
    receiving features recalculated from future observations.

    The resulting holdout missingness profile is methodologically informative.
    Missingness here does not mainly indicate feature-engineering failure. Instead,
    it reveals which holdout entities were not sufficiently represented in the
    training split to receive mapped feature values.
    """
)

col1, col2 = st.columns(2)

with col1:
    render_table(
        "Holdout Missingness Summary",
        "06_dashboard_holdout_missing_summary.csv",
        """
        This table reports the number and proportion of validation and test rows
        that are missing user-derived or item-derived features after train-based
        mapping. It provides a compact measure of how well the Phase 6 feature
        pipeline transfers into the holdout splits.
        """,
    )

with col2:
    render_figure(
        "Holdout Feature Missingness Rates",
        "06_holdout_feature_missingness",
        """
        This figure shows that missingness in the holdout splits is concentrated
        almost entirely on the item side, while user-derived feature coverage
        remains effectively complete. This is consistent with the per-user
        chronological split design, where holdout users are already represented in
        training history, but some holdout recipes do not have train-derived item
        aggregates available.
        """,
    )

st.divider()

# ============================================================
# 4. Dominant sources of null values
# ============================================================
st.subheader("4. Dominant Sources of Null Values")

st.markdown(
    """
    Beyond split-level missingness, it is also useful to inspect which individual
    feature columns contribute most strongly to null values across the Phase 6
    outputs. This makes it possible to distinguish between missingness caused by
    incomplete recipe enrichment and missingness caused by limited item history in
    the training data.
    """
)

render_figure(
    "Top Null-Count Feature Columns",
    "06_top_feature_null_columns",
    """
    This figure shows that the largest null concentrations occur in PP-related
    recipe attributes and in several item-history variables. The PP-side nulls are
    consistent with the earlier finding that not every recipe has preprocessed
    enrichment features. The item-history nulls are also expected, because holdout
    rows for recipes without sufficient training history cannot inherit complete
    train-derived item aggregates. Importantly, this pattern indicates that the
    main source of incompleteness lies in item-side representation rather than in
    user-side feature construction.
    """,
)

st.divider()

# ============================================================
# 5. Interpretation
# ============================================================
st.subheader("5. Interpretation")

st.markdown(
    """
    Overall, the feature-engineering stage produces a coherent and methodologically
    sound set of reusable outputs for later supervised and feature-aware modelling.
    The phase preserves temporal discipline by deriving aggregate features from the
    training split only, while still producing validation and test feature tables
    that can be used for downstream evaluation.

    The resulting feature pipeline is strongest on the user side, where training
    coverage remains broad and holdout missingness is negligible. The main
    limitation lies on the item side, where some holdout rows cannot inherit full
    train-derived aggregates and where PP-enriched recipe fields remain incomplete
    for part of the catalogue. Even so, the null structure is interpretable rather
    than arbitrary, and it reflects known properties of the earlier preprocessing
    stages rather than instability in.
    """
)