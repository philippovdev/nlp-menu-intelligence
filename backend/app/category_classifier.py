from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

CONFIGURED_CATEGORY_MODEL_ID = "tfidf-logreg-items-v2@1.0.0"
HEURISTIC_CATEGORY_MODEL_ID = "slice1-keyword-baseline@0.1.0"
DEFAULT_MINIMUM_CONFIDENCE = 0.35
MODEL_ASSET_DIR = Path(__file__).resolve().parent / "model_assets"
DEFAULT_CLASSIFIER_PATH = MODEL_ASSET_DIR / "category_classifier.pkl"
DEFAULT_METADATA_PATH = MODEL_ASSET_DIR / "category_classifier.json"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CategoryModelPrediction:
    raw_label: str
    label: str
    confidence: float


@dataclass(frozen=True)
class CategoryClassifier:
    pipeline: Pipeline
    model_id: str
    minimum_confidence: float
    classes: tuple[str, ...]

    def predict(
        self,
        *,
        text: str,
        allowed_labels: tuple[str, ...],
        reducer: Callable[[str | None, tuple[str, ...]], str | None],
    ) -> CategoryModelPrediction | None:
        probabilities = self.pipeline.predict_proba([text])[0]
        raw_probabilities = {
            label: float(probability)
            for label, probability in zip(self.classes, probabilities)
        }
        aggregated_probabilities: dict[str, float] = {}

        for raw_label, probability in raw_probabilities.items():
            reduced_label = reducer(raw_label, allowed_labels)
            if reduced_label is None:
                continue
            aggregated_probabilities[reduced_label] = (
                aggregated_probabilities.get(reduced_label, 0.0) + probability
            )

        if not aggregated_probabilities:
            return None

        label, confidence = max(
            sorted(aggregated_probabilities.items()),
            key=lambda item: item[1],
        )
        raw_label = max(
            sorted(raw_probabilities.items()),
            key=lambda item: item[1],
        )[0]
        return CategoryModelPrediction(
            raw_label=raw_label,
            label=label,
            confidence=round(confidence, 4),
        )

    def is_confident(self, prediction: CategoryModelPrediction) -> bool:
        return prediction.confidence >= self.minimum_confidence


def load_category_classifier(
    *,
    classifier_path: Path = DEFAULT_CLASSIFIER_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> CategoryClassifier | None:
    classifier_exists = classifier_path.is_file()
    metadata_exists = metadata_path.is_file()

    if not classifier_exists and not metadata_exists:
        LOGGER.warning(
            "Category classifier assets are missing at %s and %s; "
            "heuristic fallback remains active.",
            classifier_path,
            metadata_path,
        )
        return None

    if classifier_exists != metadata_exists:
        raise RuntimeError("Category classifier assets are incomplete.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with classifier_path.open("rb") as file_pointer:
        pipeline = pickle.load(file_pointer)

    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("Persisted category classifier does not support predict_proba().")

    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        raise RuntimeError("Persisted category classifier does not expose fitted classes_.")

    model_id = str(metadata.get("model_id") or CONFIGURED_CATEGORY_MODEL_ID)
    minimum_confidence = float(metadata.get("minimum_confidence") or DEFAULT_MINIMUM_CONFIDENCE)

    return CategoryClassifier(
        pipeline=pipeline,
        model_id=model_id,
        minimum_confidence=minimum_confidence,
        classes=tuple(str(label) for label in classes),
    )
