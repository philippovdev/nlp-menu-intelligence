from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from sklearn.pipeline import Pipeline

from app.category_model_features import (
    RECORD_INPUT_FORMAT,
    TEXT_INPUT_FORMAT,
    build_category_model_record,
)

CONFIGURED_CATEGORY_MODEL_ID = "tfidf-enriched-logreg-items-v2@1.2.0"
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
    input_format: str = TEXT_INPUT_FORMAT

    def predict(
        self,
        *,
        text: str,
        name: str | None = None,
        prices: list[object] | None = None,
        sizes: list[object] | None = None,
        allowed_labels: tuple[str, ...],
        reducer: Callable[[str | None, tuple[str, ...]], str | None],
    ) -> CategoryModelPrediction | None:
        model_input: object
        if self.input_format == RECORD_INPUT_FORMAT:
            model_input = build_category_model_record(
                text=text,
                name=name,
                prices=prices,
                sizes=sizes,
            )
        else:
            model_input = text

        probabilities = self.pipeline.predict_proba([model_input])[0]
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


def load_category_classifier_metadata(metadata_path: Path) -> tuple[str, float, str]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise RuntimeError("Category classifier metadata is invalid.") from exc

    if not isinstance(metadata, dict):
        raise RuntimeError("Category classifier metadata is invalid.")

    model_id = metadata.get("model_id")
    minimum_confidence = metadata.get("minimum_confidence")
    input_format = metadata.get("input_format", TEXT_INPUT_FORMAT)

    if not isinstance(model_id, str) or not model_id.strip():
        raise RuntimeError("Category classifier metadata is incomplete.")
    if not isinstance(minimum_confidence, int | float):
        raise RuntimeError("Category classifier metadata is incomplete.")
    if input_format not in {TEXT_INPUT_FORMAT, RECORD_INPUT_FORMAT}:
        raise RuntimeError("Category classifier metadata is invalid.")

    minimum_confidence = float(minimum_confidence)
    if minimum_confidence < 0 or minimum_confidence > 1:
        raise RuntimeError("Category classifier metadata is invalid.")

    return model_id.strip(), minimum_confidence, str(input_format)


def load_category_classifier_pipeline(classifier_path: Path) -> Pipeline:
    try:
        with classifier_path.open("rb") as file_pointer:
            pipeline = pickle.load(file_pointer)
    except (AttributeError, EOFError, ImportError, OSError, pickle.PickleError) as exc:
        raise RuntimeError("Category classifier artifact could not be loaded.") from exc

    if not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("Persisted category classifier does not support predict_proba().")
    if getattr(pipeline, "classes_", None) is None:
        raise RuntimeError("Persisted category classifier does not expose fitted classes_.")

    return pipeline


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

    model_id, minimum_confidence, input_format = load_category_classifier_metadata(metadata_path)
    pipeline = load_category_classifier_pipeline(classifier_path)
    classes = getattr(pipeline, "classes_", ())

    return CategoryClassifier(
        pipeline=pipeline,
        model_id=model_id,
        minimum_confidence=minimum_confidence,
        classes=tuple(str(label) for label in classes),
        input_format=input_format,
    )
