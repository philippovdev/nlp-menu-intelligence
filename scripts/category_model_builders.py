from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class CategoryModelFamily:
    method: str
    classifier_label: str
    default_model_id: str
    build_pipeline: Callable[[], Pipeline]
    build_parameters: Callable[[], dict[str, object]]


def build_tfidf_logreg_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_tfidf_logreg_parameters() -> dict[str, object]:
    return {
        "vectorizer": {
            "type": "tfidf",
            "analyzer": "word",
            "ngram_range": [1, 2],
        },
        "classifier": {
            "type": "logistic_regression",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
        },
    }


def build_tfidf_linear_svm_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LinearSVC(
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_tfidf_linear_svm_parameters() -> dict[str, object]:
    return {
        "vectorizer": {
            "type": "tfidf",
            "analyzer": "word",
            "ngram_range": [1, 2],
        },
        "classifier": {
            "type": "linear_svc",
            "C": 1.0,
            "loss": "squared_hinge",
            "max_iter": 10000,
            "random_state": 42,
        },
    }


def build_tfidf_union_logreg_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                min_df=2,
                                max_df=0.95,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(4, 6),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_tfidf_union_logreg_parameters() -> dict[str, object]:
    return {
        "features": {
            "type": "feature_union",
            "word": {
                "type": "tfidf",
                "analyzer": "word",
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.95,
                "sublinear_tf": True,
            },
            "char": {
                "type": "tfidf",
                "analyzer": "char_wb",
                "ngram_range": [4, 6],
                "min_df": 2,
                "max_df": 1.0,
                "sublinear_tf": True,
            },
        },
        "classifier": {
            "type": "logistic_regression",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 4000,
            "random_state": 42,
        },
    }


def resolve_calibration_cv(labels: list[str], *, preferred_cv: int = 3) -> int:
    min_class_count = min(Counter(labels).values())
    return max(2, min(preferred_cv, min_class_count))


def build_tfidf_calibrated_linear_svm_pipeline(*, calibration_cv: int = 3) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        C=1.0,
                        max_iter=10000,
                        random_state=42,
                    ),
                    method="sigmoid",
                    cv=calibration_cv,
                ),
            ),
        ]
    )


def build_tfidf_calibrated_linear_svm_parameters(*, calibration_cv: int = 3) -> dict[str, object]:
    return {
        "vectorizer": {
            "type": "tfidf",
            "analyzer": "word",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
        "classifier": {
            "type": "calibrated_classifier_cv",
            "method": "sigmoid",
            "cv": calibration_cv,
            "estimator": {
                "type": "linear_svc",
                "C": 1.0,
                "loss": "squared_hinge",
                "max_iter": 10000,
                "random_state": 42,
            },
        },
    }


MODEL_FAMILIES: dict[str, CategoryModelFamily] = {
    "tfidf_logreg": CategoryModelFamily(
        method="tfidf_logistic_regression",
        classifier_label="LogisticRegression",
        default_model_id="tfidf-logreg-items-v2@1.0.0",
        build_pipeline=build_tfidf_logreg_pipeline,
        build_parameters=build_tfidf_logreg_parameters,
    ),
    "tfidf_union_logreg": CategoryModelFamily(
        method="tfidf_union_logistic_regression",
        classifier_label="LogisticRegression with word/char TF-IDF union",
        default_model_id="tfidf-union-logreg-items-v2@1.1.0",
        build_pipeline=build_tfidf_union_logreg_pipeline,
        build_parameters=build_tfidf_union_logreg_parameters,
    ),
    "tfidf_calibrated_linear_svm": CategoryModelFamily(
        method="tfidf_calibrated_linear_svm",
        classifier_label="Calibrated LinearSVC",
        default_model_id="tfidf-calibrated-linear-svm-items-v2@1.0.0",
        build_pipeline=build_tfidf_calibrated_linear_svm_pipeline,
        build_parameters=build_tfidf_calibrated_linear_svm_parameters,
    ),
}


def sparse_compute_backend_description() -> str:
    return "cpu (scikit-learn sparse text pipelines do not use Apple MPS)"
