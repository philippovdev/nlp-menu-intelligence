from __future__ import annotations

import argparse
import pickle
from datetime import UTC, datetime
from pathlib import Path

from app.bio2_extraction import ACTIVE_ENTITY_LABELS, compute_entity_scores, compute_token_scores
from bio2_dataset_common import Bio2Record, load_bio2_records, split_bio2_records
from classification_baseline_common import (
    REPO_ROOT,
    detect_commit_sha,
    format_output_path,
    resolve_repo_path,
)
from pydantic import BaseModel, ConfigDict
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

DEFAULT_DATASET = REPO_ROOT / "data/interim/items.v2.bio2.jsonl"
DEFAULT_ARTIFACT = REPO_ROOT / "docs/course/artifacts/bio2-token-logreg-items-v2.json"
DEFAULT_MODEL = REPO_ROOT / "models/bio2-token-logreg-items-v2.pkl"
DEFAULT_RUN_ID = "bio2-token-logreg-v2-001"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractionSplitMetrics(StrictModel):
    item_count: int
    token_count: int
    entity_count: int
    token_micro_precision: float
    token_micro_recall: float
    token_micro_f1: float
    token_macro_f1: float
    token_per_label_f1: dict[str, float]
    entity_micro_precision: float
    entity_micro_recall: float
    entity_micro_f1: float
    entity_macro_f1: float
    entity_per_label_f1: dict[str, float]


class TokenClassifierArtifact(StrictModel):
    run_id: str
    run_date: str
    commit_sha: str | None = None
    scikit_learn_version: str
    dataset_file: str
    method: str
    train_item_count: int
    train_token_count: int
    tag_order: list[str]
    parameters: dict[str, object]
    model_file: str
    valid_metrics: ExtractionSplitMetrics
    test_metrics: ExtractionSplitMetrics
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--regularization-c", type=float, default=4.0)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--class-weight", default="balanced")
    return parser.parse_args()


def token_shape(token: str) -> str:
    pattern: list[str] = []
    for char in token:
        if char.isupper():
            pattern.append("X")
        elif char.islower():
            pattern.append("x")
        elif char.isdigit():
            pattern.append("d")
        else:
            pattern.append(char)

    compressed: list[str] = []
    for char in pattern:
        if not compressed or compressed[-1] != char:
            compressed.append(char)
    return "".join(compressed)


def build_token_features(
    tokens: list[str],
    index: int,
    *,
    context_window: int,
) -> dict[str, object]:
    token = tokens[index]
    lowered = token.lower()
    features: dict[str, object] = {
        "bias": 1.0,
        "token.lower": lowered,
        "token.shape": token_shape(token),
        "token.length": min(len(token), 12),
        "token.prefix1": lowered[:1],
        "token.prefix2": lowered[:2],
        "token.prefix3": lowered[:3],
        "token.suffix1": lowered[-1:],
        "token.suffix2": lowered[-2:],
        "token.suffix3": lowered[-3:],
        "token.is_digit": token.isdigit(),
        "token.has_digit": any(char.isdigit() for char in token),
        "token.is_alpha": token.isalpha(),
        "token.is_upper": token.isupper(),
        "token.is_title": token.istitle(),
        "token.has_slash": "/" in token,
        "token.has_dot": "." in token,
        "token.has_hyphen": "-" in token,
        "token.has_currency_symbol": any(char in token for char in "$EUR€£₽"),
        "bos": index == 0,
        "eos": index == len(tokens) - 1,
    }

    for offset in range(1, context_window + 1):
        for direction, neighbor_index in (
            (f"prev{offset}", index - offset),
            (f"next{offset}", index + offset),
        ):
            if 0 <= neighbor_index < len(tokens):
                neighbor = tokens[neighbor_index]
                features[f"{direction}.lower"] = neighbor.lower()
                features[f"{direction}.shape"] = token_shape(neighbor)
                features[f"{direction}.is_digit"] = neighbor.isdigit()
                features[f"{direction}.has_digit"] = any(
                    char.isdigit() for char in neighbor
                )
                features[f"{direction}.has_currency_symbol"] = any(
                    char in neighbor for char in "$EUR€£₽"
                )
            else:
                features[f"{direction}.boundary"] = True

    return features


def flatten_records(
    records: list[Bio2Record],
    *,
    context_window: int,
) -> tuple[list[dict[str, object]], list[str]]:
    features: list[dict[str, object]] = []
    tags: list[str] = []
    for record in records:
        for index, tag in enumerate(record.tags):
            features.append(
                build_token_features(
                    record.tokens,
                    index,
                    context_window=context_window,
                )
            )
            tags.append(tag)
    return features, tags


def repair_bio2_sequence(tags: list[str]) -> list[str]:
    repaired: list[str] = []
    previous = "O"
    for tag in tags:
        current = tag
        if current.startswith("I-"):
            label = current.split("-", 1)[1]
            if previous not in {f"B-{label}", f"I-{label}"}:
                current = f"B-{label}"
        repaired.append(current)
        previous = current
    return repaired


def build_split_metrics(
    records: list[Bio2Record],
    *,
    vectorizer: DictVectorizer,
    classifier: LogisticRegression,
    context_window: int,
) -> ExtractionSplitMetrics:
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []

    for record in records:
        feature_rows = [
            build_token_features(
                record.tokens,
                index,
                context_window=context_window,
            )
            for index in range(len(record.tokens))
        ]
        predicted_tags = classifier.predict(vectorizer.transform(feature_rows)).tolist()
        gold_sequences.append(record.tags)
        predicted_sequences.append(repair_bio2_sequence(predicted_tags))

    token_metrics = compute_token_scores(gold_sequences, predicted_sequences)
    entity_metrics = compute_entity_scores(gold_sequences, predicted_sequences)

    return ExtractionSplitMetrics(
        item_count=len(records),
        token_count=int(token_metrics["token_count"]),
        entity_count=int(entity_metrics["entity_count"]),
        token_micro_precision=float(token_metrics["micro_precision"]),
        token_micro_recall=float(token_metrics["micro_recall"]),
        token_micro_f1=float(token_metrics["micro_f1"]),
        token_macro_f1=float(token_metrics["macro_f1"]),
        token_per_label_f1=dict(token_metrics["per_label_f1"]),
        entity_micro_precision=float(entity_metrics["micro_precision"]),
        entity_micro_recall=float(entity_metrics["micro_recall"]),
        entity_micro_f1=float(entity_metrics["micro_f1"]),
        entity_macro_f1=float(entity_metrics["macro_f1"]),
        entity_per_label_f1=dict(entity_metrics["per_label_f1"]),
    )


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_model(
    path: Path,
    *,
    vectorizer: DictVectorizer,
    classifier: LogisticRegression,
    context_window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_pointer:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "context_window": context_window,
            },
            file_pointer,
            protocol=5,
        )


def build_artifact(
    *,
    dataset_path: Path,
    model_path: Path,
    run_id: str,
    run_date: str | None,
    commit_sha: str | None,
    tag_order: list[str],
    train_records: list[Bio2Record],
    parameters: dict[str, object],
    valid_metrics: ExtractionSplitMetrics,
    test_metrics: ExtractionSplitMetrics,
) -> TokenClassifierArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    resolved_model_path = resolve_repo_path(model_path)
    train_token_count = sum(len(record.tokens) for record in train_records)
    return TokenClassifierArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        scikit_learn_version=sklearn_version,
        dataset_file=relative_to_repo(resolved_dataset_path),
        method="contextual_logistic_regression_token_classifier",
        train_item_count=len(train_records),
        train_token_count=train_token_count,
        tag_order=tag_order,
        parameters=parameters,
        model_file=relative_to_repo(resolved_model_path),
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        notes=(
            "BIO2 token classifier trained on whitespace-tokenized item text using "
            "contextual token features and LogisticRegression. Predictions are repaired "
            "to valid BIO2 boundaries before entity scoring."
        ),
    )


def save_artifact(path: Path, artifact: TokenClassifierArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def print_summary(
    artifact: TokenClassifierArtifact,
    *,
    artifact_path: Path,
) -> None:
    print(f"Run ID: {artifact.run_id}")
    print(f"Dataset: {artifact.dataset_file}")
    print(f"Train items: {artifact.train_item_count}")
    print(f"Train tokens: {artifact.train_token_count}")
    print(
        "Valid: "
        f"token micro-F1={artifact.valid_metrics.token_micro_f1:.4f}, "
        f"entity micro-F1={artifact.valid_metrics.entity_micro_f1:.4f}"
    )
    print(
        "Test: "
        f"token micro-F1={artifact.test_metrics.token_micro_f1:.4f}, "
        f"entity micro-F1={artifact.test_metrics.entity_micro_f1:.4f}"
    )
    print(f"Artifact: {format_output_path(artifact_path)}")
    print(f"Model: {artifact.model_file}")


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    artifact_path = resolve_repo_path(args.artifact)
    model_path = resolve_repo_path(args.model_output)

    records = load_bio2_records(dataset_path)
    records_by_split = split_bio2_records(records)
    train_records = records_by_split["train"]
    valid_records = records_by_split["valid"]
    test_records = records_by_split["test"]

    train_features, train_tags = flatten_records(
        train_records,
        context_window=args.context_window,
    )
    vectorizer = DictVectorizer()
    classifier = LogisticRegression(
        max_iter=args.max_iter,
        C=args.regularization_c,
        class_weight=None if args.class_weight.lower() == "none" else args.class_weight,
        solver="lbfgs",
    )
    classifier.fit(vectorizer.fit_transform(train_features), train_tags)

    save_model(
        model_path,
        vectorizer=vectorizer,
        classifier=classifier,
        context_window=args.context_window,
    )

    artifact = build_artifact(
        dataset_path=dataset_path,
        model_path=model_path,
        run_id=args.run_id,
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        tag_order=classifier.classes_.tolist(),
        train_records=train_records,
        parameters={
            "context_window": args.context_window,
            "regularization_c": args.regularization_c,
            "max_iter": args.max_iter,
            "class_weight": args.class_weight,
            "entity_labels": list(ACTIVE_ENTITY_LABELS),
        },
        valid_metrics=build_split_metrics(
            valid_records,
            vectorizer=vectorizer,
            classifier=classifier,
            context_window=args.context_window,
        ),
        test_metrics=build_split_metrics(
            test_records,
            vectorizer=vectorizer,
            classifier=classifier,
            context_window=args.context_window,
        ),
    )

    save_artifact(artifact_path, artifact)
    print_summary(artifact, artifact_path=artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
