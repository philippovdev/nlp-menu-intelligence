from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from uuid import uuid4

from app.category_classifier import (
    CONFIGURED_CATEGORY_MODEL_ID,
    HEURISTIC_CATEGORY_MODEL_ID,
    CategoryClassifier,
)
from app.schemas import (
    DEFAULT_CATEGORY_LABELS,
    DEFAULT_CURRENCY,
    FULL_CATEGORY_LABELS,
    CategoryPrediction,
    Confidence,
    ExtractedFields,
    FieldConfidence,
    Issue,
    MenuItem,
    MenuParseRequest,
    MenuParseResponse,
    ModelVersion,
    ParseMenuMeta,
    Price,
    Size,
    Source,
)

CATEGORY_ORDER = FULL_CATEGORY_LABELS

CATEGORY_REDUCTION_PATHS: dict[str, tuple[str, ...]] = {
    "salads": ("salads", "other"),
    "soups": ("soups", "other"),
    "mains": ("mains", "other"),
    "pizza": ("pizza", "mains", "other"),
    "pasta": ("pasta", "mains", "other"),
    "burgers": ("burgers", "mains", "other"),
    "sides": ("sides", "mains", "other"),
    "desserts": ("desserts", "other"),
    "breakfast": ("breakfast", "mains", "other"),
    "drinks_hot": ("drinks_hot", "drinks", "other"),
    "drinks_cold": ("drinks_cold", "drinks", "other"),
    "other": ("other",),
}

HEADER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "salads": ("salad", "salads", "салат", "салаты"),
    "soups": ("soup", "soups", "суп", "супы"),
    "mains": (
        "mains",
        "main",
        "entree",
        "entrees",
        "main course",
        "hot dishes",
        "горячее",
        "горячие блюда",
    ),
    "pizza": ("pizza", "pizzas", "пицца", "пиццы"),
    "pasta": ("pasta", "паста"),
    "burgers": ("burger", "burgers", "бургер", "бургеры"),
    "sides": ("sides", "side dishes", "гарнир", "гарниры"),
    "desserts": ("dessert", "desserts", "десерт", "десерты"),
    "breakfast": ("breakfast", "brunch", "завтрак", "завтраки", "бранч"),
    "drinks_hot": (
        "hot drinks",
        "coffee",
        "tea",
        "горячие напитки",
        "кофе",
        "чай",
    ),
    "drinks_cold": (
        "cold drinks",
        "soft drinks",
        "лимонады",
        "холодные напитки",
        "прохладительные напитки",
    ),
}

ITEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "salads": (
        "caesar",
        "greek salad",
        "salad",
        "салат",
        "цезарь",
        "оливье",
    ),
    "soups": (
        "soup",
        "tom yum",
        "ramen",
        "borscht",
        "broth",
        "суп",
        "борщ",
        "том ям",
        "рамен",
    ),
    "mains": (
        "steak",
        "grill",
        "salmon",
        "chicken",
        "beef",
        "pork",
        "котлета",
        "лосось",
        "шашлык",
        "стейк",
    ),
    "pizza": (
        "pizza",
        "margherita",
        "pepperoni",
        "quattro formaggi",
        "пицца",
        "маргарита",
        "пепперони",
    ),
    "pasta": (
        "pasta",
        "spaghetti",
        "carbonara",
        "penne",
        "fettuccine",
        "паста",
        "спагетти",
        "карбонара",
    ),
    "burgers": (
        "burger",
        "cheeseburger",
        "hamburger",
        "бургер",
        "чизбургер",
    ),
    "sides": (
        "fries",
        "mashed potatoes",
        "rice",
        "гарнир",
        "картофель",
        "картошка",
        "рис",
    ),
    "desserts": (
        "dessert",
        "cake",
        "brownie",
        "ice cream",
        "cheesecake",
        "десерт",
        "торт",
        "мороженое",
        "чизкейк",
    ),
    "breakfast": (
        "omelette",
        "omelet",
        "porridge",
        "oatmeal",
        "syrniki",
        "pancakes",
        "омлет",
        "каша",
        "сырники",
        "блины",
    ),
    "drinks_hot": (
        "coffee",
        "tea",
        "latte",
        "cappuccino",
        "espresso",
        "americano",
        "раф",
        "чай",
        "кофе",
        "латте",
        "капучино",
        "эспрессо",
        "американо",
    ),
    "drinks_cold": (
        "lemonade",
        "juice",
        "smoothie",
        "water",
        "soda",
        "cola",
        "ice tea",
        "iced tea",
        "milkshake",
        "лимонад",
        "сок",
        "вода",
        "морс",
        "смузи",
        "милкшейк",
    ),
}

UNIT_MAP = {
    "g": "g",
    "gr": "g",
    "гр": "g",
    "г": "g",
    "kg": "kg",
    "кг": "kg",
    "ml": "ml",
    "мл": "ml",
    "l": "l",
    "л": "l",
    "cl": "cl",
    "oz": "oz",
    "cm": "cm",
    "см": "cm",
    "pcs": "pcs",
    "pc": "pcs",
    "шт": "pcs",
    "шт.": "pcs",
}

CURRENCY_MAP = {
    "₽": "RUB",
    "руб": "RUB",
    "руб.": "RUB",
    "р": "RUB",
    "р.": "RUB",
    "rub": "RUB",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

NUMBER_RE = r"(?:\d{1,3}(?:[ \u00A0]\d{3})+|\d+)(?:[.,]\d+)?"
SIZE_PATTERN = re.compile(
    rf"(?P<values>{NUMBER_RE}(?:\s*/\s*{NUMBER_RE})*)\s*(?P<unit>kg|g|gr|ml|l|cl|oz|cm|pc|pcs|шт\.?|кг|гр|г|мл|л|см)\b",
    re.IGNORECASE,
)
PRICE_PREFIX_PATTERN = re.compile(
    rf"(?<!\w)(?P<currency>[$€£₽])\s*(?P<value>{NUMBER_RE})(?!\s*(?:kg|g|gr|ml|l|cl|oz|cm|pc|pcs|шт\.?|кг|гр|г|мл|л|см)\b)",
    re.IGNORECASE,
)
PRICE_SUFFIX_PATTERN = re.compile(
    rf"(?<!\w)(?P<value>{NUMBER_RE})\s*(?P<currency>₽|руб\.?|р\.?\b|rub\b|usd\b|eur\b|gbp\b|[$€£])(?!\w)",
    re.IGNORECASE,
)
BARE_TRAILING_PRICES_PATTERN = re.compile(
    rf"(?P<group>{NUMBER_RE}(?:\s*/\s*{NUMBER_RE})*)\s*$",
    re.IGNORECASE,
)
NON_LETTER_PATTERN = re.compile(r"[^A-Za-zА-Яа-я]+")
LEADING_MARKER_PATTERN = re.compile(r"^\s*(?:\d+[.)]\s*)?(?:[-–—•·*]\s*)+")
LEADING_NUMBER_PATTERN = re.compile(r"^\s*\d+[.)]\s*")
SEPARATOR_ONLY_PATTERN = re.compile(r"^[\s\-–—•·*_=~.:|/\\]+$")
CONTACT_PATTERN = re.compile(r"(?:https?://|www\.|instagram|тел\.?|phone|@)", re.IGNORECASE)
DOT_LEADER_PATTERN = re.compile(r"[.·•]{2,}")
SPACED_SEPARATOR_PATTERN = re.compile(r"\s+[\-–—:|]+\s+")
MULTISPACE_PATTERN = re.compile(r"\s+")
PRICE_OR_SIZE_TAIL_PATTERN = re.compile(
    rf"(?:{NUMBER_RE}\s*(?:₽|руб\.?|р\.?\b|rub\b|usd\b|eur\b|gbp\b|[$€£]|kg|g|gr|ml|l|cl|oz|cm|pc|pcs|шт\.?|кг|гр|г|мл|л|см)?\s*/?\s*)+$",
    re.IGNORECASE,
)
DESCRIPTION_KEYWORD_PATTERN = re.compile(
    (
        r"\b(?:"
        r"белое|красное|розовое|сухое|полусухое|полусладкое|сладкое|брют|экстра\s+брют|"
        r"white|red|rose|ros[eé]|dry|semi[- ]dry|semi[- ]sweet|sweet|brut|extra\s+brut|"
        r"france|italy|spain|germany|chile|argentina|australia|new\s+zealand|"
        r"франция|италия|испания|германия|чили|аргентина|австралия|португалия|"
        r"бургундия|венето|пьемонт|тоскана|мальборо|шампань|риоха|мозель"
        r")\b"
    ),
    re.IGNORECASE,
)

ISSUE_DEFINITIONS = {
    "UNKNOWN_HEADER_CATEGORY": (
        "warning",
        "Header does not map to a known category.",
        "/items/{index}/category",
    ),
    "CATEGORY_CONFLICT_WITH_HEADER": (
        "warning",
        "Predicted item category conflicts with the active header category.",
        "/items/{index}/category",
    ),
    "CATEGORY_MODEL_LOW_CONFIDENCE": (
        "info",
        "Category model confidence was too low; heuristic fallback was used.",
        "/items/{index}/category",
    ),
    "MISSING_PRICE": (
        "warning",
        "Menu item does not contain a price.",
        "/items/{index}/fields/prices",
    ),
    "MULTIPLE_PRICES": (
        "warning",
        "Multiple prices were detected and need review.",
        "/items/{index}/fields/prices",
    ),
    "MULTIPLE_SIZES": (
        "warning",
        "Multiple sizes were detected and need review.",
        "/items/{index}/fields/sizes",
    ),
    "EMPTY_NAME": (
        "warning",
        "Menu item name could not be derived reliably.",
        "/items/{index}/fields/name",
    ),
    "UNCATEGORIZED": (
        "warning",
        "Category fallback ended in 'other'.",
        "/items/{index}/category",
    ),
    "WEAK_MENU_SIGNAL": (
        "info",
        "Line looks like a menu item but has weak extraction signals.",
        "/items/{index}",
    ),
}

NER_MODEL_ID = "slice1-deterministic-fields@0.1.0"
CATEGORY_MODEL_HEADER_OVERRIDE_CONFIDENCE = 0.6


def parse_menu_text(
    request: MenuParseRequest,
    *,
    category_classifier: CategoryClassifier | None = None,
) -> MenuParseResponse:
    indexed_lines, skipped_empty_lines = split_lines(request.text)
    active_header_category: str | None = None
    items: list[MenuItem] = []
    top_level_issues: list[Issue] = []
    allowed_labels = tuple(request.category_labels or DEFAULT_CATEGORY_LABELS)

    if category_classifier is None:
        top_level_issues.append(
            Issue(
                code="CATEGORY_MODEL_UNAVAILABLE",
                level="info",
                message=(
                    "Configured category classifier is unavailable; "
                    "heuristic fallback is active."
                ),
                details={"configured_model": CONFIGURED_CATEGORY_MODEL_ID},
            )
        )

    for item_index, (source_line_number, raw_line) in enumerate(indexed_lines):
        normalized = normalize_line(raw_line)
        sizes, size_fragments = extract_sizes(normalized)
        prices, price_fragments = extract_prices(
            normalized,
            size_fragments,
            default_currency=request.currency_hint or DEFAULT_CURRENCY,
        )

        kind, header_category = classify_line(
            normalized,
            prices,
            sizes,
            active_header_category=active_header_category,
        )
        if kind != "menu_item":
            prices = []
            sizes = []
            price_fragments = []
            size_fragments = []
        name = (
            derive_name(normalized, size_fragments, price_fragments)
            if kind == "menu_item"
            else None
        )

        issue_codes: list[str] = []
        internal_category: str | None = None
        category_source: str | None = None
        output_category: str | None = None
        keyword_category = guess_item_category(normalized)
        model_confidence: float | None = None

        if kind == "category_header":
            internal_category = header_category
            if internal_category is None:
                active_header_category = None
                issue_codes.append("UNKNOWN_HEADER_CATEGORY")
            else:
                active_header_category = internal_category
                category_source = "header_keyword"
        elif kind == "menu_item":
            model_prediction = None
            header_output_category = (
                reduce_category(active_header_category, allowed_labels)
                if active_header_category is not None
                else None
            )
            if category_classifier is not None:
                if contains_latin_letters(name or normalized):
                    model_prediction = category_classifier.predict(
                        text=normalized,
                        name=name,
                        prices=prices,
                        sizes=sizes,
                        allowed_labels=allowed_labels,
                        reducer=reduce_category,
                    )
                else:
                    issue_codes.append("CATEGORY_MODEL_LOW_CONFIDENCE")

            should_use_model = (
                model_prediction is not None and category_classifier.is_confident(model_prediction)
            )
            if (
                should_use_model
                and header_output_category is not None
                and model_prediction.label != header_output_category
                and model_prediction.confidence < CATEGORY_MODEL_HEADER_OVERRIDE_CONFIDENCE
            ):
                should_use_model = False
                issue_codes.append("CATEGORY_MODEL_LOW_CONFIDENCE")

            if should_use_model:
                output_category = model_prediction.label
                category_source = "model"
                model_confidence = model_prediction.confidence
                if header_output_category is not None and header_output_category != output_category:
                    issue_codes.append("CATEGORY_CONFLICT_WITH_HEADER")
            else:
                if (
                    model_prediction is not None
                    and "CATEGORY_MODEL_LOW_CONFIDENCE" not in issue_codes
                ):
                    issue_codes.append("CATEGORY_MODEL_LOW_CONFIDENCE")

            if output_category is not None:
                pass
            elif active_header_category is not None:
                internal_category = active_header_category
                category_source = "header_context"
                if keyword_category and keyword_category != active_header_category:
                    issue_codes.append("CATEGORY_CONFLICT_WITH_HEADER")
            elif keyword_category is not None:
                internal_category = keyword_category
                category_source = "keyword_fallback"
            else:
                internal_category = "other"
                category_source = "unknown"
                issue_codes.append("UNCATEGORIZED")
        else:
            pass

        if output_category is None:
            output_category = reduce_category(internal_category, allowed_labels)

        if kind == "menu_item":
            if not prices:
                issue_codes.append("MISSING_PRICE")
            if len(prices) > 1:
                issue_codes.append("MULTIPLE_PRICES")
            if len(sizes) > 1:
                issue_codes.append("MULTIPLE_SIZES")
            if not name:
                issue_codes.append("EMPTY_NAME")
            elif not prices and not sizes:
                issue_codes.append("WEAK_MENU_SIGNAL")

        category_confidence = calculate_category_confidence(
            kind=kind,
            category_source=category_source,
            category=output_category,
            issue_codes=issue_codes,
            model_confidence=model_confidence,
        )
        field_confidence = build_field_confidence(
            kind=kind,
            has_name=bool(name),
            prices=prices,
            sizes=sizes,
        )
        overall_confidence = calculate_overall_confidence(
            kind=kind,
            category_source=category_source,
            category_confidence=category_confidence,
            prices=prices,
            sizes=sizes,
            issue_codes=issue_codes,
            has_name=bool(name),
        )

        items.append(
            MenuItem(
                id=f"item_{item_index + 1}",
                source=Source(
                    line=source_line_number,
                    raw=raw_line,
                    normalized=normalized,
                ),
                kind=kind,
                category=CategoryPrediction(
                    label=output_category,
                    confidence=category_confidence,
                ),
                fields=ExtractedFields(
                    name=name,
                    description=None,
                    prices=prices if kind == "menu_item" else [],
                    sizes=sizes if kind == "menu_item" else [],
                ),
                confidence=Confidence(
                    overall=overall_confidence,
                    category=category_confidence,
                    fields=field_confidence,
                ),
                issues=build_item_issues(issue_codes, item_index),
            )
        )

    if skipped_empty_lines:
        top_level_issues.append(
            Issue(
                code="EMPTY_LINES_SKIPPED",
                level="info",
                message="Some empty lines were ignored.",
                details={"count": skipped_empty_lines},
            )
        )

    if not any(item.kind == "menu_item" for item in items):
        top_level_issues.append(
            Issue(
                code="NO_MENU_ITEMS_DETECTED",
                level="warning",
                message="Parser did not detect any menu items.",
            )
        )

    return MenuParseResponse(
        request_id=f"req_{uuid4().hex[:12]}",
        meta=ParseMenuMeta(lang=request.lang, currency=request.currency_hint),
        model_version=build_model_version(category_classifier),
        items=items,
        issues=top_level_issues,
    )


def build_model_version(category_classifier: CategoryClassifier | None) -> ModelVersion:
    return ModelVersion(
        category_model=(
            category_classifier.model_id
            if category_classifier is not None
            else HEURISTIC_CATEGORY_MODEL_ID
        ),
        ner_model=NER_MODEL_ID,
    )


def split_lines(text: str) -> tuple[list[tuple[int, str]], int]:
    normalized_text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[tuple[int, str]] = []
    skipped = 0

    for index, raw_line in enumerate(normalized_text.split("\n"), start=1):
        stripped = raw_line.strip()
        if stripped:
            lines.append((index, stripped))
        else:
            skipped += 1

    return lines, skipped


def normalize_line(line: str) -> str:
    normalized = unicodedata.normalize("NFKC", line)
    replacements = {
        "\u00A0": " ",
        "—": " - ",
        "–": " - ",
        "•": " • ",
        "·": " · ",
        "\t": " ",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = LEADING_MARKER_PATTERN.sub("", normalized)
    normalized = DOT_LEADER_PATTERN.sub(" ", normalized)
    normalized = MULTISPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def extract_sizes(line: str) -> tuple[list[Size], list[str]]:
    sizes: list[Size] = []
    fragments: list[str] = []

    for match in SIZE_PATTERN.finditer(line):
        raw_fragment = match.group(0).strip()
        unit = normalize_unit(match.group("unit"))
        values = split_numeric_values(match.group("values"))
        fragments.append(raw_fragment)
        for raw_value in values:
            parsed_value = parse_number(raw_value)
            if parsed_value is None:
                continue
            raw_unit = match.group("unit").strip()
            sizes.append(Size(value=parsed_value, unit=unit, raw=f"{raw_value} {raw_unit}"))

    return dedupe_models(sizes), dedupe_strings(fragments)


def extract_prices(
    line: str,
    size_fragments: list[str],
    *,
    default_currency: str,
) -> tuple[list[Price], list[str]]:
    prices: list[Price] = []
    fragments: list[str] = []

    for pattern in (PRICE_PREFIX_PATTERN, PRICE_SUFFIX_PATTERN):
        for match in pattern.finditer(line):
            raw_fragment = match.group(0).strip()
            parsed_value = parse_number(match.group("value"))
            if parsed_value is None:
                continue
            currency = normalize_currency(match.group("currency")) or default_currency
            prices.append(Price(value=parsed_value, currency=currency, raw=raw_fragment))
            fragments.append(raw_fragment)

    if prices:
        return dedupe_models(prices), dedupe_strings(fragments)

    cleaned = line
    for fragment in size_fragments:
        cleaned = cleaned.replace(fragment, " ", 1)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()

    match = BARE_TRAILING_PRICES_PATTERN.search(cleaned)
    if not match:
        return [], []

    raw_group = match.group("group").strip()
    for raw_value in split_numeric_values(raw_group):
        parsed_value = parse_number(raw_value)
        if parsed_value is None:
            continue
        prices.append(Price(value=parsed_value, currency=default_currency, raw=raw_value))
    fragments.append(raw_group)

    return dedupe_models(prices), dedupe_strings(fragments)


def classify_line(
    line: str,
    prices: list[Price],
    sizes: list[Size],
    *,
    active_header_category: str | None = None,
) -> tuple[str, str | None]:
    if is_noise(line):
        return "noise", None

    if looks_ocr_merged_row(line, prices=prices, sizes=sizes):
        return "noise", None

    header_category = guess_header_category(line)
    if header_category is not None:
        return "category_header", header_category

    if looks_generic_header(line, prices=prices, sizes=sizes):
        return "category_header", None

    if looks_description_line(line, prices=prices, sizes=sizes):
        return "noise", None

    if looks_menu_item(
        line,
        prices=prices,
        sizes=sizes,
        active_header_category=active_header_category,
    ):
        return "menu_item", None

    return "noise", None


def is_noise(line: str) -> bool:
    lowered = line.lower()
    letters = count_letters(line)

    if not line:
        return True
    if SEPARATOR_ONLY_PATTERN.fullmatch(line):
        return True
    if CONTACT_PATTERN.search(lowered) and not has_price_signal(line):
        return True
    if letters == 0 and not has_price_signal(line):
        return True
    if re.fullmatch(r"\d+(?:\s*/\s*\d+)?", line):
        return True
    return False


def looks_generic_header(line: str, prices: list[Price], sizes: list[Size]) -> bool:
    if prices or sizes:
        return False
    words = split_words(line)
    if not words or len(words) > 4:
        return False
    if line.endswith(":"):
        return True
    if line == line.upper() and count_letters(line) >= 3:
        return True
    return False


def looks_menu_item(
    line: str,
    *,
    prices: list[Price],
    sizes: list[Size],
    active_header_category: str | None = None,
) -> bool:
    words = split_words(line)
    if not words or count_letters(line) < 3:
        return False
    if len(words) > 16:
        return False
    if looks_ocr_merged_row(line, prices=prices, sizes=sizes):
        return False
    if looks_description_line(line, prices=prices, sizes=sizes):
        return False
    if prices or sizes:
        return True
    if active_header_category is not None and len(words) <= 10:
        return True
    return guess_item_category(line) is not None and len(words) <= 12


def looks_description_line(line: str, prices: list[Price], sizes: list[Size]) -> bool:
    if prices or sizes:
        return False
    if count_letters(line) < 6:
        return False

    slash_count = line.count("/")
    descriptor_keyword = DESCRIPTION_KEYWORD_PATTERN.search(line) is not None
    if slash_count < 2 and not descriptor_keyword:
        return False

    if guess_item_category(line) is not None:
        return False

    words = split_words(line)
    return len(words) >= 4


def looks_ocr_merged_row(line: str, prices: list[Price], sizes: list[Size]) -> bool:
    signal_count = len(prices) + len(sizes)
    if signal_count < 2:
        return False
    words = split_words(line)
    return len(line) > 160 or len(words) > 24


def guess_header_category(line: str) -> str | None:
    normalized = canonical_category_text(line)
    for category in CATEGORY_ORDER:
        for keyword in HEADER_KEYWORDS.get(category, ()):
            if normalized == canonical_category_text(keyword):
                return category
    return None


def guess_item_category(line: str) -> str | None:
    lowered = canonical_category_text(line)
    best_category: str | None = None
    best_score = 0

    for category in CATEGORY_ORDER:
        score = 0
        for keyword in ITEM_KEYWORDS.get(category, ()):
            candidate = canonical_category_text(keyword)
            if candidate and candidate in lowered:
                score = max(score, len(candidate.split()))
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def derive_name(line: str, size_fragments: list[str], price_fragments: list[str]) -> str | None:
    cleaned = f" {line} "
    for fragment in [*size_fragments, *price_fragments]:
        cleaned = cleaned.replace(fragment, " ", 1)

    cleaned = PRICE_OR_SIZE_TAIL_PATTERN.sub(" ", cleaned)
    cleaned = SPACED_SEPARATOR_PATTERN.sub(" ", cleaned)
    cleaned = LEADING_NUMBER_PATTERN.sub("", cleaned)
    cleaned = DOT_LEADER_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.strip(" -–—:|,.;/ ")

    return cleaned or None


def reduce_category(category: str | None, allowed_labels: tuple[str, ...]) -> str | None:
    if category is None:
        return None
    if category in allowed_labels:
        return category

    for candidate in CATEGORY_REDUCTION_PATHS.get(category, ()):
        if candidate in allowed_labels:
            return candidate

    if "other" in allowed_labels:
        return "other"

    return None


def calculate_category_confidence(
    *,
    kind: str,
    category_source: str | None,
    category: str | None,
    issue_codes: list[str],
    model_confidence: float | None = None,
) -> float | None:
    if kind == "noise" or category is None:
        return None

    if category_source == "model":
        if model_confidence is None:
            return None
        return round(max(0.01, min(model_confidence, 0.99)), 2)

    if kind == "category_header":
        score = 0.95 if category_source == "header_keyword" else 0.58
    elif category_source == "header_context":
        score = 0.92
    elif category_source == "keyword_fallback":
        score = 0.78
    else:
        score = 0.42

    if "CATEGORY_CONFLICT_WITH_HEADER" in issue_codes:
        score -= 0.18
    if "CATEGORY_MODEL_LOW_CONFIDENCE" in issue_codes:
        score -= 0.2
    if "UNCATEGORIZED" in issue_codes:
        score -= 0.1

    return round(max(0.05, min(score, 0.99)), 2)


def build_field_confidence(
    *,
    kind: str,
    has_name: bool,
    prices: list[Price],
    sizes: list[Size],
) -> FieldConfidence:
    if kind != "menu_item":
        return FieldConfidence()

    name_confidence = 0.96 if has_name else 0.12
    price_confidence = 0.9 if len(prices) == 1 else 0.72 if len(prices) > 1 else 0.16
    size_confidence = 0.88 if len(sizes) == 1 else 0.74 if len(sizes) > 1 else None

    return FieldConfidence(
        name=round(name_confidence, 2),
        prices=round(price_confidence, 2),
        sizes=None if size_confidence is None else round(size_confidence, 2),
    )


def calculate_overall_confidence(
    *,
    kind: str,
    category_source: str | None,
    category_confidence: float | None,
    prices: list[Price],
    sizes: list[Size],
    issue_codes: list[str],
    has_name: bool,
) -> float:
    if kind == "noise":
        score = 0.98
    elif kind == "category_header":
        score = 0.9
        if category_source == "header_keyword":
            score += 0.06
        else:
            score -= 0.14
    else:
        if category_source == "model":
            score = 0.2 + 0.5 * (category_confidence or 0.0)
        else:
            score = 0.45
        if has_name:
            score += 0.1
        if prices:
            score += 0.2
        if sizes:
            score += 0.08
        if category_source == "header_context":
            score += 0.18
        elif category_source == "keyword_fallback":
            score += 0.1
        elif category_source != "model":
            score -= 0.06

    penalty_map = {
        "UNKNOWN_HEADER_CATEGORY": 0.08,
        "CATEGORY_MODEL_LOW_CONFIDENCE": 0.08,
        "MISSING_PRICE": 0.12,
        "MULTIPLE_PRICES": 0.05,
        "MULTIPLE_SIZES": 0.04,
        "EMPTY_NAME": 0.25,
        "UNCATEGORIZED": 0.08,
        "WEAK_MENU_SIGNAL": 0.1,
        "CATEGORY_CONFLICT_WITH_HEADER": 0.12,
    }
    score -= sum(penalty_map.get(issue_code, 0.05) for issue_code in issue_codes)
    return round(max(0.05, min(score, 0.99)), 2)


def build_item_issues(issue_codes: list[str], item_index: int) -> list[Issue]:
    issues: list[Issue] = []

    for code in dedupe_strings(issue_codes):
        level, message, path_template = ISSUE_DEFINITIONS[code]
        issues.append(
            Issue(
                code=code,
                level=level,
                message=message,
                path=path_template.format(index=item_index),
            )
        )

    return issues


def canonical_category_text(text: str) -> str:
    lowered = unicodedata.normalize("NFKC", text).lower()
    lowered = NON_LETTER_PATTERN.sub(" ", lowered)
    lowered = MULTISPACE_PATTERN.sub(" ", lowered)
    return lowered.strip()


def contains_latin_letters(text: str) -> bool:
    return any(char.isascii() and char.isalpha() for char in text)


def has_price_signal(line: str) -> bool:
    return bool(PRICE_PREFIX_PATTERN.search(line) or PRICE_SUFFIX_PATTERN.search(line))


def count_letters(text: str) -> int:
    return sum(char.isalpha() for char in text)


def split_words(text: str) -> list[str]:
    return [part for part in NON_LETTER_PATTERN.split(text) if part]


def normalize_unit(raw_unit: str) -> str:
    return UNIT_MAP.get(raw_unit.lower(), raw_unit.lower())


def normalize_currency(raw_currency: str) -> str | None:
    return CURRENCY_MAP.get(raw_currency.lower())


def split_numeric_values(raw_values: str) -> list[str]:
    return [part.strip() for part in raw_values.split("/") if part.strip()]


def parse_number(raw_value: str) -> int | float | None:
    cleaned = raw_value.replace("\u00A0", " ").replace(" ", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value.is_integer():
        return int(value)
    return round(value, 2)


def dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def dedupe_models(models: Iterable[Price | Size]) -> list[Price | Size]:
    seen: set[tuple[tuple[str, object], ...]] = set()
    result: list[Price | Size] = []

    for model in models:
        signature = tuple(model.model_dump(mode="python").items())
        if signature in seen:
            continue
        seen.add(signature)
        result.append(model)

    return result
