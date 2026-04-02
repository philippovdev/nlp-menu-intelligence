from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

from classification_baseline_common import (
    REPO_ROOT,
    build_label_order,
    build_split_metrics,
    evaluate_split,
    resolve_repo_path,
    split_items,
)
from dataset_common import AnnotatedItem, load_annotated_items
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from category_model_builders import sparse_compute_backend_description

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs/course/artifacts/sparse-search-items-v2.json"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    parameters: dict[str, object]

    def build_pipeline(self) -> Pipeline:
        if self.family == "word_logreg":
            return Pipeline(
                steps=[
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer=self.parameters.get("analyzer_override", "word"),
                            ngram_range=tuple(self.parameters["ngram_range"]),
                            min_df=self.parameters["min_df"],
                            max_df=self.parameters["max_df"],
                            sublinear_tf=self.parameters["sublinear_tf"],
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=self.parameters["C"],
                            max_iter=4000,
                            random_state=42,
                        ),
                    ),
                ]
            )

        if self.family == "union_logreg":
            return Pipeline(
                steps=[
                    (
                        "features",
                        FeatureUnion(
                            transformer_list=[
                                (
                                    "word",
                                    TfidfVectorizer(
                                        analyzer="word",
                                        ngram_range=tuple(self.parameters["word_ngram_range"]),
                                        min_df=self.parameters["word_min_df"],
                                        max_df=self.parameters["word_max_df"],
                                        sublinear_tf=True,
                                    ),
                                ),
                                (
                                    "char",
                                    TfidfVectorizer(
                                        analyzer="char_wb",
                                        ngram_range=tuple(self.parameters["char_ngram_range"]),
                                        min_df=self.parameters["char_min_df"],
                                        sublinear_tf=True,
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=self.parameters["C"],
                            max_iter=4000,
                            random_state=42,
                        ),
                    ),
                ]
            )

        raise ValueError(f"Unknown candidate family: {self.family}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--profile", choices=("quick", "wide"), default="quick")
    return parser.parse_args()


def build_candidates(*, profile: str) -> list[Candidate]:
    candidates: list[Candidate] = []

    if profile == "quick":
        word_ngrams = ((1, 2), (1, 3))
        word_min_dfs = (1, 2)
        word_max_dfs = (0.95, 1.0)
        word_cs = (0.3, 1.0, 3.0, 10.0)
        union_word_ngrams = ((1, 2), (1, 3))
        union_word_min_dfs = (1, 2)
        union_word_max_dfs = (0.95, 1.0)
        union_char_ngrams = ((3, 5), (4, 6))
        union_char_min_dfs = (1, 2)
        union_cs = (0.3, 1.0, 3.0)
        include_char_only = False
    else:
        word_ngrams = ((1, 1), (1, 2), (1, 3))
        word_min_dfs = (1, 2, 3)
        word_max_dfs = (0.9, 0.95, 1.0)
        word_cs = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
        union_word_ngrams = ((1, 2), (1, 3))
        union_word_min_dfs = (1, 2, 3)
        union_word_max_dfs = (0.9, 0.95, 1.0)
        union_char_ngrams = ((3, 5), (4, 6), (3, 6))
        union_char_min_dfs = (1, 2, 3)
        union_cs = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
        include_char_only = True

    for ngram_range, min_df, max_df, c_value in product(
        word_ngrams,
        word_min_dfs,
        word_max_dfs,
        word_cs,
    ):
        candidates.append(
            Candidate(
                name=(
                    "word-logreg"
                    f"-ngram={ngram_range[0]}_{ngram_range[1]}"
                    f"-min_df={min_df}"
                    f"-max_df={max_df}"
                    f"-C={c_value}"
                ),
                family="word_logreg",
                parameters={
                    "ngram_range": list(ngram_range),
                    "min_df": min_df,
                    "max_df": max_df,
                    "sublinear_tf": True,
                    "C": c_value,
                },
            )
        )

    if include_char_only:
        for char_ngram_range, char_min_df, c_value in product(
            ((3, 5), (4, 6), (3, 6)),
            (1, 2, 3),
            (0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
        ):
            candidates.append(
                Candidate(
                    name=(
                        "char-logreg"
                        f"-char={char_ngram_range[0]}_{char_ngram_range[1]}"
                        f"-cmin={char_min_df}"
                        f"-C={c_value}"
                    ),
                    family="word_logreg",
                    parameters={
                        "analyzer_override": "char_wb",
                        "ngram_range": list(char_ngram_range),
                        "min_df": char_min_df,
                        "max_df": 1.0,
                        "sublinear_tf": True,
                        "C": c_value,
                    },
                )
            )

    for word_ngram_range, word_min_df, word_max_df, char_ngram_range, char_min_df, c_value in product(
        union_word_ngrams,
        union_word_min_dfs,
        union_word_max_dfs,
        union_char_ngrams,
        union_char_min_dfs,
        union_cs,
    ):
        candidates.append(
            Candidate(
                name=(
                    "union-logreg"
                    f"-word={word_ngram_range[0]}_{word_ngram_range[1]}"
                    f"-wmin={word_min_df}"
                    f"-wmax={word_max_df}"
                    f"-char={char_ngram_range[0]}_{char_ngram_range[1]}"
                    f"-cmin={char_min_df}"
                    f"-C={c_value}"
                ),
                family="union_logreg",
                parameters={
                    "word_ngram_range": list(word_ngram_range),
                    "word_min_df": word_min_df,
                    "word_max_df": word_max_df,
                    "char_ngram_range": list(char_ngram_range),
                    "char_min_df": char_min_df,
                    "C": c_value,
                },
            )
        )

    return candidates


def evaluate_candidate_cv(
    *,
    candidate: Candidate,
    train_items: list[AnnotatedItem],
    label_order: list[str],
    cv_splits: int,
) -> tuple[float, list[dict[str, float]]]:
    texts = [item.text for item in train_items]
    labels = [item.category for item in train_items]
    groups = [item.restaurant_id for item in train_items]
    splitter = GroupKFold(n_splits=cv_splits)

    fold_results: list[dict[str, float]] = []
    for fold_index, (train_index, valid_index) in enumerate(splitter.split(texts, labels, groups), start=1):
        pipeline = candidate.build_pipeline()
        pipeline.fit(
            [texts[index] for index in train_index],
            [labels[index] for index in train_index],
        )
        predictions = pipeline.predict([texts[index] for index in valid_index]).tolist()
        metrics = build_split_metrics(
            gold=[labels[index] for index in valid_index],
            predicted=predictions,
            label_order=label_order,
        )
        fold_results.append(
            {
                "fold": fold_index,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
            }
        )

    mean_macro_f1 = round(
        sum(result["macro_f1"] for result in fold_results) / len(fold_results),
        4,
    )
    return mean_macro_f1, fold_results


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    items = load_annotated_items(dataset_path)
    items_by_split = split_items(items)
    train_items = items_by_split["train"]
    valid_items = items_by_split["valid"]
    test_items = items_by_split["test"]
    label_order = build_label_order(items)

    unique_groups = {item.restaurant_id for item in train_items}
    cv_splits = max(2, min(args.cv_splits, len(unique_groups)))
    candidates = build_candidates(profile=args.profile)

    best_valid_macro_f1 = -1.0
    best_result: dict[str, object] | None = None
    candidate_results: list[dict[str, object]] = []
    started_at = time.time()

    print(
        f"Starting sparse search with {len(candidates)} candidates, profile={args.profile}, "
        f"and {cv_splits}-fold GroupKFold.",
        flush=True,
    )
    print(f"Compute backend: {sparse_compute_backend_description()}", flush=True)

    for index, candidate in enumerate(candidates, start=1):
        cv_macro_f1, fold_results = evaluate_candidate_cv(
            candidate=candidate,
            train_items=train_items,
            label_order=label_order,
            cv_splits=cv_splits,
        )
        pipeline = candidate.build_pipeline()
        pipeline.fit(
            [item.text for item in train_items],
            [item.category for item in train_items],
        )
        valid_metrics = evaluate_split(
            pipeline=pipeline,
            items=valid_items,
            label_order=label_order,
        )
        result = {
            "rank_index": index,
            "name": candidate.name,
            "family": candidate.family,
            "parameters": candidate.parameters,
            "cv_macro_f1": cv_macro_f1,
            "cv_folds": fold_results,
            "valid_accuracy": valid_metrics.accuracy,
            "valid_macro_f1": valid_metrics.macro_f1,
        }
        candidate_results.append(result)

        if valid_metrics.macro_f1 > best_valid_macro_f1:
            best_valid_macro_f1 = valid_metrics.macro_f1
            best_result = result
            elapsed_seconds = int(time.time() - started_at)
            print(
                f"[{index}/{len(candidates)}] new best: {candidate.name} | "
                f"cv_macro_f1={cv_macro_f1:.4f} | "
                f"valid_macro_f1={valid_metrics.macro_f1:.4f} | "
                f"valid_accuracy={valid_metrics.accuracy:.4f} | "
                f"elapsed={elapsed_seconds}s",
                flush=True,
            )
        elif index % 10 == 0:
            elapsed_seconds = int(time.time() - started_at)
            print(
                f"[{index}/{len(candidates)}] progress | current={candidate.name} | "
                f"cv_macro_f1={cv_macro_f1:.4f} | valid_macro_f1={valid_metrics.macro_f1:.4f} | "
                f"best_valid_macro_f1={best_valid_macro_f1:.4f} | elapsed={elapsed_seconds}s",
                flush=True,
            )

    if best_result is None:
        raise RuntimeError("No candidates were evaluated.")

    best_candidate = Candidate(
        name=str(best_result["name"]),
        family=str(best_result["family"]),
        parameters=dict(best_result["parameters"]),
    )
    best_pipeline = best_candidate.build_pipeline()
    best_pipeline.fit(
        [item.text for item in train_items],
        [item.category for item in train_items],
    )
    final_valid_metrics = evaluate_split(
        pipeline=best_pipeline,
        items=valid_items,
        label_order=label_order,
    )
    final_test_metrics = evaluate_split(
        pipeline=best_pipeline,
        items=test_items,
        label_order=label_order,
    )

    payload = {
        "run_id": f"sparse-search-{dataset_path.stem.split('.')[-1]}-001",
        "run_date": datetime.now(UTC).date().isoformat(),
        "dataset_file": str(dataset_path.relative_to(REPO_ROOT)),
        "cv_splits": cv_splits,
        "candidate_count": len(candidates),
        "best_candidate": {
            "name": best_candidate.name,
            "family": best_candidate.family,
            "parameters": best_candidate.parameters,
            "valid_accuracy": final_valid_metrics.accuracy,
            "valid_macro_f1": final_valid_metrics.macro_f1,
            "test_accuracy": final_test_metrics.accuracy,
            "test_macro_f1": final_test_metrics.macro_f1,
        },
        "candidates": sorted(
            candidate_results,
            key=lambda item: (item["valid_macro_f1"], item["cv_macro_f1"]),
            reverse=True,
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "Search complete | "
        f"best={best_candidate.name} | "
        f"valid_macro_f1={final_valid_metrics.macro_f1:.4f} | "
        f"test_macro_f1={final_test_metrics.macro_f1:.4f} | "
        f"artifact={output_path.relative_to(REPO_ROOT)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
