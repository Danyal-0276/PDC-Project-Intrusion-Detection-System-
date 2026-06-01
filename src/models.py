"""
Factory for PySpark MLlib classification models used in NIDS experiments.
"""

from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)

import config

# Model keys used in reports and ensemble joins
RANDOM_FOREST = "random_forest"
GRADIENT_BOOSTED_TREES = "gradient_boosted_trees"
LOGISTIC_REGRESSION = "logistic_regression"
NAIVE_BAYES = "naive_bayes"

SINGLE_MODELS = [
    RANDOM_FOREST,
    GRADIENT_BOOSTED_TREES,
    LOGISTIC_REGRESSION,
    NAIVE_BAYES,
]


# Train fastest models first, Random Forest last (slowest on weak VMs)
TRAINING_ORDER = [
    NAIVE_BAYES,
    LOGISTIC_REGRESSION,
    GRADIENT_BOOSTED_TREES,
    RANDOM_FOREST,
]


def get_models_to_train() -> list[str]:
    """Return models in training order: fast first, Random Forest last."""
    import config

    if config.MODELS_FILTER:
        allowed = {name.strip() for name in config.MODELS_FILTER.split(",") if name.strip()}
        aliases = {
            "gbt": GRADIENT_BOOSTED_TREES,
            "rf": RANDOM_FOREST,
            "nb": NAIVE_BAYES,
            "lr": LOGISTIC_REGRESSION,
        }
        resolved = set()
        for name in allowed:
            resolved.add(aliases.get(name, name))
        picked = [m for m in TRAINING_ORDER if m in resolved]
        if not picked:
            raise ValueError(
                f"Unknown NIDS_MODELS '{config.MODELS_FILTER}'. "
                f"Use: {', '.join(SINGLE_MODELS)} (aliases: gbt, rf, nb, lr)"
            )
        return picked

    if config.FAST_MODE:
        return [NAIVE_BAYES, RANDOM_FOREST]
    return list(TRAINING_ORDER)


def build_classifier(model_name: str):
    """
    Return an unfitted PySpark classifier for the given model name.

    All models use featuresCol='features' and labelCol='label_idx'.
    """
    weight_col = "weight" if config.USE_CLASS_WEIGHTS else None
    # Multinomial supports 2+ classes; binomial crashes if StringIndexer finds >2 labels.
    lr_family = "multinomial"

    if model_name == RANDOM_FOREST:
        return RandomForestClassifier(
            featuresCol="features",
            labelCol="label_idx",
            weightCol=weight_col,
            numTrees=config.RF_NUM_TREES,
            seed=config.RANDOM_SEED,
        )

    if model_name == GRADIENT_BOOSTED_TREES:
        return GBTClassifier(
            featuresCol="features",
            labelCol="label_idx",
            weightCol=weight_col,
            maxIter=config.GBT_MAX_ITER,
            seed=config.RANDOM_SEED,
        )

    if model_name == LOGISTIC_REGRESSION:
        return LogisticRegression(
            featuresCol="features",
            labelCol="label_idx",
            weightCol=weight_col,
            maxIter=config.LR_MAX_ITER,
            regParam=config.LR_REG_PARAM,
            family=lr_family,
        )

    if model_name == NAIVE_BAYES:
        return NaiveBayes(
            featuresCol="features",
            labelCol="label_idx",
        )

    raise ValueError(f"Unknown model: {model_name}. Choose from {SINGLE_MODELS}")
