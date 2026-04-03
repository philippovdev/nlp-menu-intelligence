from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.menu_parser import extract_prices, extract_sizes
from app.schemas import Price, Size

EntityLabel = Literal["NAME", "DESC", "PRICE", "SIZE"]
ACTIVE_ENTITY_LABELS: tuple[EntityLabel, ...] = ("NAME", "PRICE", "SIZE")
TOKEN_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class EntitySpan:
    label: EntityLabel
    start_token: int
    end_token: int


def tokenize_with_offsets(text: str) -> list[TokenSpan]:
    return [
        TokenSpan(text=match.group(0), start=match.start(), end=match.end())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def build_gold_bio2_tags(
    text: str,
    *,
    name: str | None,
    prices: list[Price],
    sizes: list[Size],
    default_currency: str,
) -> tuple[list[str], list[TokenSpan]]:
    return build_bio2_tags(
        text,
        name=name,
        price_fragments=resolve_price_fragments(
            text,
            prices=prices,
            default_currency=default_currency,
        ),
        size_fragments=resolve_size_fragments(text, sizes=sizes),
    )


def build_predicted_bio2_tags(
    text: str,
    *,
    name: str | None,
    prices: list[Price],
    sizes: list[Size],
    default_currency: str,
) -> tuple[list[str], list[TokenSpan]]:
    return build_bio2_tags(
        text,
        name=name,
        price_fragments=resolve_price_fragments(
            text,
            prices=prices,
            default_currency=default_currency,
        ),
        size_fragments=resolve_size_fragments(text, sizes=sizes),
    )


def build_bio2_tags(
    text: str,
    *,
    name: str | None,
    price_fragments: list[str],
    size_fragments: list[str],
    description: str | None = None,
) -> tuple[list[str], list[TokenSpan]]:
    tokens = tokenize_with_offsets(text)
    tags = ["O"] * len(tokens)
    occupied: list[tuple[int, int]] = []

    spans: list[tuple[EntityLabel, int, int]] = []

    for label, fragment in (
        ("NAME", name),
        ("DESC", description),
    ):
        if fragment:
            match = find_non_overlapping_fragment_span(text, fragment, occupied)
            if match is not None:
                spans.append((label, *match))
                occupied.append(match)

    for fragment in size_fragments:
        match = find_non_overlapping_fragment_span(text, fragment, occupied)
        if match is not None:
            spans.append(("SIZE", *match))
            occupied.append(match)

    for fragment in price_fragments:
        match = find_non_overlapping_fragment_span(text, fragment, occupied)
        if match is not None:
            spans.append(("PRICE", *match))
            occupied.append(match)

    for label, start_char, end_char in sorted(spans, key=lambda item: (item[1], item[2])):
        assign_tags(tags, tokens=tokens, label=label, start_char=start_char, end_char=end_char)

    return tags, tokens


def find_non_overlapping_fragment_span(
    text: str,
    fragment: str,
    occupied: list[tuple[int, int]],
) -> tuple[int, int] | None:
    start = 0
    while True:
        index = text.find(fragment, start)
        if index < 0:
            return None
        span = (index, index + len(fragment))
        if not overlaps_any(span, occupied):
            return span
        start = index + 1


def overlaps_any(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(not (span[1] <= current[0] or span[0] >= current[1]) for current in occupied)


def assign_tags(
    tags: list[str],
    *,
    tokens: list[TokenSpan],
    label: EntityLabel,
    start_char: int,
    end_char: int,
) -> None:
    token_indexes = [
        index
        for index, token in enumerate(tokens)
        if token.start < end_char and token.end > start_char
    ]
    if not token_indexes:
        return

    for position, token_index in enumerate(token_indexes):
        prefix = "B" if position == 0 else "I"
        tags[token_index] = f"{prefix}-{label}"


def extract_entities(tags: list[str]) -> list[EntitySpan]:
    entities: list[EntitySpan] = []
    active_label: str | None = None
    start_index: int | None = None

    for index, tag in enumerate(tags):
        if tag == "O":
            if active_label is not None and start_index is not None:
                entities.append(
                    EntitySpan(
                        label=active_label,
                        start_token=start_index,
                        end_token=index - 1,
                    )
                )
            active_label = None
            start_index = None
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B" or active_label != label:
            if active_label is not None and start_index is not None:
                entities.append(
                    EntitySpan(
                        label=active_label,
                        start_token=start_index,
                        end_token=index - 1,
                    )
                )
            active_label = label
            start_index = index
        elif active_label is None:
            active_label = label
            start_index = index

    if active_label is not None and start_index is not None:
        entities.append(
            EntitySpan(
                label=active_label,
                start_token=start_index,
                end_token=len(tags) - 1,
            )
        )

    return entities


def compute_token_scores(
    gold_tags: list[list[str]],
    predicted_tags: list[list[str]],
) -> dict[str, object]:
    labels = [label for label in ACTIVE_ENTITY_LABELS]
    counts: Counter[tuple[str, str]] = Counter()
    per_label_true_positive = Counter[str]()
    per_label_predicted = Counter[str]()
    per_label_gold = Counter[str]()

    for gold_sequence, predicted_sequence in zip(gold_tags, predicted_tags):
        for gold_tag, predicted_tag in zip(gold_sequence, predicted_sequence):
            gold_label = normalize_tag_label(gold_tag)
            predicted_label = normalize_tag_label(predicted_tag)
            if gold_label is not None:
                per_label_gold[gold_label] += 1
            if predicted_label is not None:
                per_label_predicted[predicted_label] += 1
            if gold_label is not None and gold_label == predicted_label:
                per_label_true_positive[gold_label] += 1
            counts[(gold_tag, predicted_tag)] += 1

    return build_span_metric_payload(
        labels=labels,
        true_positive=per_label_true_positive,
        predicted_total=per_label_predicted,
        gold_total=per_label_gold,
        item_count=sum(len(sequence) for sequence in gold_tags),
        metric_name="token_count",
    )


def compute_entity_scores(
    gold_tags: list[list[str]],
    predicted_tags: list[list[str]],
) -> dict[str, object]:
    labels = [label for label in ACTIVE_ENTITY_LABELS]
    per_label_true_positive = Counter[str]()
    per_label_predicted = Counter[str]()
    per_label_gold = Counter[str]()

    for gold_sequence, predicted_sequence in zip(gold_tags, predicted_tags):
        gold_entities = Counter(
            (entity.label, entity.start_token, entity.end_token)
            for entity in extract_entities(gold_sequence)
        )
        predicted_entities = Counter(
            (entity.label, entity.start_token, entity.end_token)
            for entity in extract_entities(predicted_sequence)
        )
        overlap = gold_entities & predicted_entities

        for entity_key, count in gold_entities.items():
            per_label_gold[entity_key[0]] += count
        for entity_key, count in predicted_entities.items():
            per_label_predicted[entity_key[0]] += count
        for entity_key, count in overlap.items():
            per_label_true_positive[entity_key[0]] += count

    return build_span_metric_payload(
        labels=labels,
        true_positive=per_label_true_positive,
        predicted_total=per_label_predicted,
        gold_total=per_label_gold,
        item_count=sum(len(extract_entities(sequence)) for sequence in gold_tags),
        metric_name="entity_count",
    )


def normalize_tag_label(tag: str) -> str | None:
    if tag == "O":
        return None
    _, label = tag.split("-", 1)
    return label


def build_span_metric_payload(
    *,
    labels: list[str],
    true_positive: Counter[str],
    predicted_total: Counter[str],
    gold_total: Counter[str],
    item_count: int,
    metric_name: str,
) -> dict[str, object]:
    micro_true_positive = sum(true_positive.values())
    micro_predicted_total = sum(predicted_total.values())
    micro_gold_total = sum(gold_total.values())

    micro_precision = (
        round(micro_true_positive / micro_predicted_total, 4)
        if micro_predicted_total
        else 0.0
    )
    micro_recall = round(micro_true_positive / micro_gold_total, 4) if micro_gold_total else 0.0
    if micro_precision + micro_recall == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = round(2 * micro_precision * micro_recall / (micro_precision + micro_recall), 4)

    per_label_f1: dict[str, float] = {}
    for label in labels:
        precision = true_positive[label] / predicted_total[label] if predicted_total[label] else 0.0
        recall = true_positive[label] / gold_total[label] if gold_total[label] else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_label_f1[label] = round(f1, 4)

    macro_f1 = round(sum(per_label_f1.values()) / len(labels), 4) if labels else 0.0

    return {
        metric_name: item_count,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "per_label_f1": per_label_f1,
    }


def resolve_size_fragments(text: str, *, sizes: list[Size]) -> list[str]:
    if not sizes:
        return []

    candidate_sizes, candidate_fragments = extract_sizes(text)
    target_signatures = normalize_sizes(sizes)
    return resolve_candidate_fragments(
        candidate_fragments=candidate_fragments,
        target_signatures=target_signatures,
        parse_fragment=lambda fragment: normalize_sizes(extract_sizes(fragment)[0]),
        fallback_fragments=[size.raw for size in sizes if size.raw],
    )


def resolve_price_fragments(
    text: str,
    *,
    prices: list[Price],
    default_currency: str,
) -> list[str]:
    if not prices:
        return []

    size_fragments = extract_sizes(text)[1]
    _, candidate_fragments = extract_prices(text, size_fragments, default_currency=default_currency)
    target_signatures = normalize_prices(prices)
    return resolve_candidate_fragments(
        candidate_fragments=candidate_fragments,
        target_signatures=target_signatures,
        parse_fragment=lambda fragment: normalize_prices(
            extract_prices(fragment, [], default_currency=default_currency)[0]
        ),
        fallback_fragments=[price.raw for price in prices if price.raw],
    )


def resolve_candidate_fragments(
    *,
    candidate_fragments: list[str],
    target_signatures: list[tuple[object, ...]],
    parse_fragment: Callable[[str], list[tuple[object, ...]]],
    fallback_fragments: list[str],
) -> list[str]:
    remaining = Counter(target_signatures)
    matched_fragments: list[str] = []

    for fragment in candidate_fragments:
        fragment_signatures = parse_fragment(fragment)
        if not fragment_signatures:
            continue
        fragment_counter = Counter(fragment_signatures)
        if all(fragment_counter[key] <= remaining[key] for key in fragment_counter):
            matched_fragments.append(fragment)
            remaining.subtract(fragment_counter)
            remaining += Counter()

    if not remaining:
        return matched_fragments

    fallback_matches = [fragment for fragment in fallback_fragments if fragment]
    return dedupe_preserving_order([*matched_fragments, *fallback_matches])


def normalize_prices(prices: list[Price]) -> list[tuple[object, ...]]:
    return [(normalize_number(price.value), price.currency) for price in prices]


def normalize_sizes(sizes: list[Size]) -> list[tuple[object, ...]]:
    return [(normalize_number(size.value), size.unit) for size in sizes]


def normalize_number(value: int | float) -> int | float:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 4)


def dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
