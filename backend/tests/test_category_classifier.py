from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.category_classifier import load_category_classifier
from app.main import create_app


def build_test_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    pipeline.fit(
        ["Caesar Salad", "Tom Yum Soup", "Americano Coffee", "Pepperoni Pizza"],
        ["salads", "soups", "drinks_hot", "pizza"],
    )
    return pipeline


def test_load_category_classifier_returns_none_when_assets_are_missing(tmp_path: Path) -> None:
    classifier = load_category_classifier(
        classifier_path=tmp_path / "missing.pkl",
        metadata_path=tmp_path / "missing.json",
    )

    assert classifier is None


def test_load_category_classifier_rejects_incomplete_assets(tmp_path: Path) -> None:
    metadata_path = tmp_path / "classifier.json"
    metadata_path.write_text(
        json.dumps({"model_id": "test-model", "minimum_confidence": 0.35}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Category classifier assets are incomplete."):
        load_category_classifier(
            classifier_path=tmp_path / "missing.pkl",
            metadata_path=metadata_path,
        )


@pytest.mark.parametrize(
    ("metadata_payload", "expected_message"),
    [
        ("{not-json", "Category classifier metadata is invalid."),
        (json.dumps({"minimum_confidence": 0.35}), "Category classifier metadata is incomplete."),
    ],
)
def test_app_startup_fails_with_invalid_classifier_metadata(
    tmp_path: Path,
    metadata_payload: str,
    expected_message: str,
) -> None:
    classifier_path = tmp_path / "classifier.pkl"
    metadata_path = tmp_path / "classifier.json"
    with classifier_path.open("wb") as file_pointer:
        pickle.dump(build_test_pipeline(), file_pointer, protocol=5)
    metadata_path.write_text(metadata_payload, encoding="utf-8")

    app = create_app(
        category_classifier_loader=lambda: load_category_classifier(
            classifier_path=classifier_path,
            metadata_path=metadata_path,
        )
    )

    with pytest.raises(RuntimeError, match=expected_message):
        with TestClient(app):
            pass
