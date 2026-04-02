from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from classification_baseline_common import (
    REPO_ROOT,
    SplitMetrics,
    build_label_order,
    build_split_metrics,
    default_output_path,
    default_run_id,
    resolve_repo_path,
    split_items,
)
from dataset_common import AnnotatedItem, load_annotated_items
from datasets import Dataset
from datasets import __version__ as datasets_version
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers import (
    __version__ as transformers_version,
)

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 64


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransformerArtifact(StrictModel):
    run_id: str
    run_date: str
    commit_sha: str | None = None
    dataset_file: str
    method: str
    model_name: str
    label_order: list[str]
    label_to_id: dict[str, int]
    train_item_count: int
    valid_item_count: int
    test_item_count: int
    transformers_version: str
    datasets_version: str
    torch_version: str
    device: str
    parameters: dict[str, object]
    model_dir: str
    best_checkpoint: str | None
    valid_metrics: SplitMetrics
    test_metrics: SplitMetrics
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=8.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def default_model_dir(dataset_path: Path) -> Path:
    dataset_tag = dataset_path.stem.split(".")[-1]
    return REPO_ROOT / f"models/transformer-classifier-items-{dataset_tag}"


def detect_commit_sha() -> str | None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        return None
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_label_to_id(items: list[AnnotatedItem]) -> dict[str, int]:
    return {label: index for index, label in enumerate(build_label_order(items))}


def create_split_dataset(items: list[AnnotatedItem], label_to_id: dict[str, int]) -> Dataset:
    return Dataset.from_dict(
        {
            "text": [item.text for item in items],
            "label": [label_to_id[item.category] for item in items],
        }
    )


def tokenize_split_dataset(
    *,
    dataset: Dataset,
    tokenizer,
    max_length: int,
) -> Dataset:
    tokenized = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )
    model_columns = [name for name in tokenizer.model_input_names if name in tokenized.column_names]
    tokenized.set_format(type="torch", columns=[*model_columns, "label"])
    return tokenized


def evaluate_with_trainer(
    *,
    trainer: Trainer,
    dataset: Dataset,
    label_order: list[str],
) -> SplitMetrics:
    predictions = trainer.predict(dataset)
    predicted_ids = np.asarray(predictions.predictions).argmax(axis=-1).tolist()
    gold_ids = predictions.label_ids.tolist()
    gold = [label_order[index] for index in gold_ids]
    predicted = [label_order[index] for index in predicted_ids]
    return build_split_metrics(gold=gold, predicted=predicted, label_order=label_order)


def compute_eval_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predicted = np.asarray(logits).argmax(axis=-1)
    accuracy = float(np.mean(predicted == labels))
    macro_f1 = float(f1_score(labels, predicted, average="macro", zero_division=0))
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
    }


def build_artifact(
    *,
    dataset_path: Path,
    run_id: str,
    run_date: str | None,
    commit_sha: str | None,
    model_name: str,
    label_order: list[str],
    label_to_id: dict[str, int],
    device: str,
    parameters: dict[str, object],
    model_dir: Path,
    best_checkpoint: str | None,
    valid_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    items_by_split: dict[str, list[AnnotatedItem]],
) -> TransformerArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    resolved_model_dir = resolve_repo_path(model_dir)
    return TransformerArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        method="transformer_sequence_classification",
        model_name=model_name,
        label_order=label_order,
        label_to_id=label_to_id,
        train_item_count=len(items_by_split["train"]),
        valid_item_count=len(items_by_split["valid"]),
        test_item_count=len(items_by_split["test"]),
        transformers_version=transformers_version,
        datasets_version=datasets_version,
        torch_version=torch.__version__,
        device=device,
        parameters=parameters,
        model_dir=relative_to_repo(resolved_model_dir),
        best_checkpoint=relative_to_repo(best_checkpoint),
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        notes=(
            f"Transformer sequence classifier fine-tuned on the fixed {dataset_path.name} split "
            f"using {model_name} and early stopping on validation Macro-F1."
        ),
    )


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_artifact(path: Path, artifact: TransformerArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def relative_to_repo(path: str | Path | None) -> str | None:
    if path is None:
        return None
    value = Path(path)
    try:
        return str(value.relative_to(REPO_ROOT))
    except ValueError:
        return str(value)


def format_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_summary(artifact: TransformerArtifact, artifact_path: Path) -> None:
    print(f"Run ID: {artifact.run_id}")
    print(f"Dataset: {artifact.dataset_file}")
    print(f"Model: {artifact.model_name}")
    print(f"Device: {artifact.device}")
    print(f"Train items: {artifact.train_item_count}")
    print(
        "Valid: "
        f"accuracy={artifact.valid_metrics.accuracy:.4f}, "
        f"macro-F1={artifact.valid_metrics.macro_f1:.4f}"
    )
    print(
        "Test: "
        f"accuracy={artifact.test_metrics.accuracy:.4f}, "
        f"macro-F1={artifact.test_metrics.macro_f1:.4f}"
    )
    print(f"Artifact: {format_output_path(artifact_path)}")
    print(f"Model dir: {artifact.model_dir}")


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_dir = resolve_repo_path(args.output_dir or default_model_dir(dataset_path))
    artifact_path = resolve_repo_path(
        args.artifact
        or default_output_path(
            prefix="transformer-classifier",
            dataset_path=dataset_path,
        )
    )
    run_id = args.run_id or default_run_id(
        prefix="transformer-classifier",
        dataset_path=dataset_path,
    )
    checkpoints_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best-model"
    device = detect_device()
    print(f"Detected accelerator: {device}", flush=True)

    set_random_seed(args.seed)
    items = load_annotated_items(dataset_path)
    items_by_split = split_items(items)
    label_to_id = build_label_to_id(items)
    label_order = sorted(label_to_id, key=label_to_id.get)
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = tokenize_split_dataset(
        dataset=create_split_dataset(items_by_split["train"], label_to_id),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    valid_dataset = tokenize_split_dataset(
        dataset=create_split_dataset(items_by_split["valid"], label_to_id),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = tokenize_split_dataset(
        dataset=create_split_dataset(items_by_split["test"], label_to_id),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_order),
        label2id=label_to_id,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        max_steps=args.max_steps,
        seed=args.seed,
        report_to="none",
        dataloader_pin_memory=device == "cuda",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_eval_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    save_json(
        best_model_dir / "label_mapping.json",
        {
            "label_order": label_order,
            "label_to_id": label_to_id,
            "id_to_label": {str(key): value for key, value in id_to_label.items()},
        },
    )

    valid_metrics = evaluate_with_trainer(
        trainer=trainer,
        dataset=valid_dataset,
        label_order=label_order,
    )
    test_metrics = evaluate_with_trainer(
        trainer=trainer,
        dataset=test_dataset,
        label_order=label_order,
    )
    artifact = build_artifact(
        dataset_path=dataset_path,
        run_id=run_id,
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        model_name=args.model_name,
        label_order=label_order,
        label_to_id=label_to_id,
        device=device,
        parameters={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "max_length": args.max_length,
            "early_stopping_patience": args.early_stopping_patience,
            "save_total_limit": args.save_total_limit,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        model_dir=best_model_dir,
        best_checkpoint=trainer.state.best_model_checkpoint,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        items_by_split=items_by_split,
    )

    save_artifact(artifact_path, artifact)
    print_summary(artifact, artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
