from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SchemaVersion = Literal["v1"]
IssueLevel = Literal["info", "warning", "error"]
ItemKind = Literal["menu_item", "category_header", "noise"]
Number = int | float

DEFAULT_CATEGORY_LABELS = (
    "salads",
    "soups",
    "mains",
    "desserts",
    "drinks",
    "other",
)

DEFAULT_LANGUAGE = "ru"
DEFAULT_CURRENCY = "RUB"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Issue(StrictModel):
    code: str
    level: IssueLevel
    message: str
    path: str | None = None
    details: dict[str, Any] | None = None


class CategoryPrediction(StrictModel):
    label: str | None = None
    confidence: float | None = None


class Price(StrictModel):
    value: Number
    currency: str
    raw: str | None = None


class Size(StrictModel):
    value: Number
    unit: str
    raw: str | None = None


class ExtractedFields(StrictModel):
    name: str | None = None
    description: str | None = None
    prices: list[Price] = Field(default_factory=list)
    sizes: list[Size] = Field(default_factory=list)


class FieldConfidence(StrictModel):
    name: float | None = None
    description: float | None = None
    prices: float | None = None
    sizes: float | None = None


class Confidence(StrictModel):
    overall: float | None = None
    category: float | None = None
    fields: FieldConfidence = Field(default_factory=FieldConfidence)


class Source(StrictModel):
    line: int
    raw: str
    normalized: str


class MenuItem(StrictModel):
    id: str
    source: Source
    kind: ItemKind
    category: CategoryPrediction
    fields: ExtractedFields
    confidence: Confidence
    issues: list[Issue] = Field(default_factory=list)


class ModelVersion(StrictModel):
    category_model: str
    ner_model: str


class ParseMenuMeta(StrictModel):
    lang: str
    currency: str
    split_strategy: Literal["lines"] = "lines"


class MenuParseRequest(StrictModel):
    schema_version: SchemaVersion
    text: str = Field(min_length=1, max_length=50000)
    lang: str = DEFAULT_LANGUAGE
    currency_hint: str = DEFAULT_CURRENCY
    category_labels: list[str] | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("lang")
    @classmethod
    def normalize_lang(cls, value: str) -> str:
        normalized = value.strip().lower()
        return normalized or DEFAULT_LANGUAGE

    @field_validator("currency_hint")
    @classmethod
    def normalize_currency_hint(cls, value: str) -> str:
        normalized = value.strip().upper()
        return normalized or DEFAULT_CURRENCY

    @field_validator("category_labels")
    @classmethod
    def normalize_category_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        seen: set[str] = set()
        normalized_labels: list[str] = []

        for raw_label in value:
            label = normalize_label(raw_label)
            if not label or label in seen:
                continue
            seen.add(label)
            normalized_labels.append(label)

        return normalized_labels or None


class MenuParseResponse(StrictModel):
    schema_version: SchemaVersion = "v1"
    request_id: str
    meta: ParseMenuMeta
    model_version: ModelVersion
    items: list[MenuItem]
    issues: list[Issue] = Field(default_factory=list)


class ApiErrorBody(StrictModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiErrorResponse(StrictModel):
    schema_version: SchemaVersion = "v1"
    error: ApiErrorBody

    @classmethod
    def validation_error(cls, errors: list[dict[str, Any]]) -> ApiErrorResponse:
        return cls(
            error=ApiErrorBody(
                code="VALIDATION_ERROR",
                message="Request validation failed.",
                details={"errors": errors},
            )
        )


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")
