from types import SimpleNamespace

from app.image_ocr import normalize_image_ocr_text, normalize_rapidocr_output


def build_box(x1: float, y1: float, x2: float, y2: float) -> list[list[float]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def make_result(
    texts: list[str],
    boxes: list[list[list[float]]],
) -> SimpleNamespace:
    return SimpleNamespace(txts=texts, boxes=boxes)


def test_normalize_rapidocr_output_reconstructs_columns_left_to_right() -> None:
    result = make_result(
        [
            "ВИНА",
            "ТЕКИЛА",
            "Шардоне",
            "300P",
            "Сауза",
            "280P",
        ],
        [
            build_box(40, 10, 180, 34),
            build_box(360, 10, 520, 34),
            build_box(40, 60, 180, 84),
            build_box(210, 60, 270, 84),
            build_box(360, 60, 500, 84),
            build_box(530, 60, 590, 84),
        ],
    )

    extracted = normalize_rapidocr_output(result)

    assert extracted.splitlines() == [
        "ВИНА",
        "Шардоне 300 ₽",
        "ТЕКИЛА",
        "Сауза 280 ₽",
    ]


def test_normalize_rapidocr_output_attaches_same_column_price_lines() -> None:
    result = make_result(
        ["Шардоне", "300P"],
        [
            build_box(40, 40, 220, 64),
            build_box(232, 82, 292, 106),
        ],
    )

    extracted = normalize_rapidocr_output(result)

    assert extracted.splitlines() == ["Шардоне 300 ₽"]


def test_normalize_rapidocr_output_pairs_split_text_and_price_subcolumns() -> None:
    result = make_result(
        ["Шардоне", "Совиньон Блан", "300P", "350P"],
        [
            build_box(40, 40, 220, 64),
            build_box(40, 102, 250, 126),
            build_box(248, 68, 308, 92),
            build_box(248, 132, 308, 156),
        ],
    )

    extracted = normalize_rapidocr_output(result)

    assert extracted.splitlines() == [
        "Шардоне 300 ₽",
        "Совиньон Блан 350 ₽",
    ]


def test_normalize_rapidocr_output_merges_descriptor_lines_without_swallowing_next_item() -> None:
    result = make_result(
        [
            "Шардоне",
            "Белое сухое / Италия / Венето",
            "300P",
            "Рислинг",
            "450P",
        ],
        [
            build_box(40, 40, 220, 64),
            build_box(46, 70, 308, 96),
            build_box(258, 100, 318, 124),
            build_box(40, 150, 190, 174),
            build_box(258, 182, 318, 206),
        ],
    )

    extracted = normalize_rapidocr_output(result)

    assert extracted.splitlines() == [
        "Шардоне Белое сухое / Италия / Венето 300 ₽",
        "Рислинг 450 ₽",
    ]


def test_normalize_rapidocr_output_does_not_attach_prices_to_headers() -> None:
    result = make_result(
        ["ВИНА", "300P", "Шардоне", "450P"],
        [
            build_box(40, 10, 180, 34),
            build_box(240, 38, 300, 62),
            build_box(40, 78, 220, 102),
            build_box(240, 110, 300, 134),
        ],
    )

    extracted = normalize_rapidocr_output(result)

    assert extracted.splitlines() == [
        "ВИНА",
        "Шардоне 450 ₽",
    ]


def test_normalize_rapidocr_output_suppresses_service_scope_lines() -> None:
    result = make_result(
        [
            "ВИНА",
            "по бокалам, 150 мл",
            "Шардоне",
            "300P",
            "по бутылкам, 750 мл",
        ],
        [
            build_box(40, 10, 180, 34),
            build_box(40, 42, 250, 66),
            build_box(40, 92, 220, 116),
            build_box(240, 122, 300, 146),
            build_box(40, 170, 260, 194),
        ],
    )

    extracted = normalize_rapidocr_output(result)

    assert extracted.splitlines() == [
        "ВИНА",
        "Шардоне 300 ₽",
    ]


def test_normalize_image_ocr_text_rewrites_common_ruble_suffix_noise_only_after_digits() -> None:
    normalized = normalize_image_ocr_text("Шардоне 300P\nПросекко 320 P\nАмаретто 2200Р\nPOM")

    assert normalized.splitlines() == [
        "Шардоне 300 ₽",
        "Просекко 320 ₽",
        "Амаретто 2200 ₽",
        "POM",
    ]
