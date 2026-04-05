from app.image_ocr import normalize_image_ocr_text
from app.menu_parser import classify_line, extract_prices, extract_sizes, parse_menu_text
from app.schemas import MenuParseRequest


def test_classify_line_treats_description_like_origin_lines_as_noise() -> None:
    line = "Белое сухое / Италия / Венето"
    kind, header_category = classify_line(
        line,
        prices=[],
        sizes=[],
        active_header_category="drinks_cold",
    )

    assert kind == "noise"
    assert header_category is None


def test_classify_line_filters_ocr_merged_rows() -> None:
    line = (
        "Просекко DOC Вилла дельи Олми 350P Биттер Кампари100 мл 280P "
        "Бифитер 50мл 300₽ Егермейстер 50 мл Гордонс 50 мл 300₽ "
        "Белое сухое / Италия / Венето Асти Перлино 2100P"
    )
    sizes, size_fragments = extract_sizes(line)
    prices, _ = extract_prices(line, size_fragments, default_currency="RUB")

    kind, header_category = classify_line(line, prices=prices, sizes=sizes)

    assert len(sizes) >= 2
    assert kind == "noise"
    assert header_category is None


def test_normalized_ocr_prices_parse_as_rubles() -> None:
    line = normalize_image_ocr_text("Граппа 300P")
    prices, _ = extract_prices(line, [], default_currency="RUB")

    assert [price.model_dump(mode="python") for price in prices] == [
        {"value": 300, "currency": "RUB", "raw": "300 ₽"}
    ]


def test_parse_menu_text_skips_collapsed_ocr_rows_and_keeps_cleaner_items() -> None:
    request = MenuParseRequest(
        schema_version="v1",
        text=normalize_image_ocr_text(
            "\n".join(
                [
                    "ВИНА",
                    "Белое сухое / Италия / Венето",
                    (
                        "Просекко DOC Вилла дельи Олми 350P Биттер Кампари100 мл 280P "
                        "Бифитер 50мл 300₽ Егермейстер 50 мл Гордонс 50 мл 300₽ "
                        "Белое сухое / Италия / Венето Асти Перлино 2100P"
                    ),
                    "Сауза золотая 50 мл 300P",
                ]
            )
        ),
        lang="ru",
        currency_hint="RUB",
    )

    response = parse_menu_text(request)

    assert [item.kind for item in response.items] == [
        "category_header",
        "noise",
        "noise",
        "menu_item",
    ]
    assert response.items[3].fields.name == "Сауза золотая"
    assert response.items[3].fields.prices[0].value == 300
