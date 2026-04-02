from fastapi.testclient import TestClient

from app.category_classifier import CONFIGURED_CATEGORY_MODEL_ID
from app.main import create_app


def test_health_and_version_endpoints(client: TestClient) -> None:
    health_response = client.get("/api/v1/health")
    version_response = client.get("/api/v1/version")
    status_response = client.get("/api/status")

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    assert version_response.status_code == 200
    assert version_response.json() == {
        "version": "0.0.1",
        "category_model": CONFIGURED_CATEGORY_MODEL_ID,
        "configured_category_model": CONFIGURED_CATEGORY_MODEL_ID,
        "category_model_ready": True,
    }

    assert status_response.status_code == 200
    assert status_response.json() == {
        "service": "menu-intelligence-api",
        "status": "ok",
        "version": "0.0.1",
    }


def test_parse_menu_returns_slice1_contract_and_header_context(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse",
        json={
            "schema_version": "v1",
            "text": "САЛАТЫ\n\nЦезарь с курицей 250 г - 390 ₽\nСУПЫ\nТом ям 300 мл - 450 ₽",
            "lang": "ru",
            "currency_hint": "RUB",
            "category_labels": ["salads", "soups", "mains", "desserts", "drinks", "other"],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["schema_version"] == "v1"
    assert payload["request_id"].startswith("req_")
    assert payload["meta"] == {
        "lang": "ru",
        "currency": "RUB",
        "split_strategy": "lines",
    }
    assert payload["model_version"] == {
        "category_model": CONFIGURED_CATEGORY_MODEL_ID,
        "ner_model": "slice1-deterministic-fields@0.1.0",
    }
    assert len(payload["items"]) == 4
    assert payload["issues"] == [
        {
            "code": "EMPTY_LINES_SKIPPED",
            "level": "info",
            "message": "Some empty lines were ignored.",
            "path": None,
            "details": {"count": 1},
        }
    ]

    header = payload["items"][0]
    assert header["kind"] == "category_header"
    assert header["source"] == {
        "line": 1,
        "raw": "САЛАТЫ",
        "normalized": "САЛАТЫ",
    }
    assert header["category"] == {"label": "salads", "confidence": 0.95}

    caesar = payload["items"][1]
    assert caesar["id"] == "item_2"
    assert caesar["kind"] == "menu_item"
    assert caesar["source"] == {
        "line": 3,
        "raw": "Цезарь с курицей 250 г - 390 ₽",
        "normalized": "Цезарь с курицей 250 г - 390 ₽",
    }
    assert caesar["category"] == {"label": "salads", "confidence": 0.72}
    assert caesar["fields"] == {
        "name": "Цезарь с курицей",
        "description": None,
        "prices": [{"value": 390, "currency": "RUB", "raw": "390 ₽"}],
        "sizes": [{"value": 250, "unit": "g", "raw": "250 г"}],
    }
    assert caesar["confidence"] == {
        "overall": 0.93,
        "category": 0.72,
        "fields": {
            "name": 0.96,
            "description": None,
            "prices": 0.9,
            "sizes": 0.88,
        },
    }
    assert caesar["issues"] == [
        {
            "code": "CATEGORY_MODEL_LOW_CONFIDENCE",
            "level": "info",
            "message": "Category model confidence was too low; heuristic fallback was used.",
            "path": "/items/1/category",
            "details": None,
        }
    ]


def test_parse_menu_uses_keyword_fallback_and_category_reduction(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse",
        json={
            "schema_version": "v1",
            "text": "Americano 300 ml 180",
            "currency_hint": "RUB",
            "category_labels": ["salads", "soups", "mains", "desserts", "drinks", "other"],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    line = payload["items"][0]
    assert line["kind"] == "menu_item"
    assert line["category"] == {"label": "drinks", "confidence": 0.5}
    assert line["fields"] == {
        "name": "Americano",
        "description": None,
        "prices": [{"value": 180, "currency": "RUB", "raw": "180"}],
        "sizes": [{"value": 300, "unit": "ml", "raw": "300 ml"}],
    }
    assert line["issues"] == []


def test_parse_menu_detects_noise_and_multiple_values(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse",
        json={
            "schema_version": "v1",
            "text": "www.example.com\nPepperoni Pizza 30/40 cm 590 / 790",
            "currency_hint": "RUB",
            "category_labels": ["salads", "soups", "mains", "desserts", "drinks", "other"],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    noise = payload["items"][0]
    assert noise["kind"] == "noise"
    assert noise["category"] == {"label": None, "confidence": None}
    assert noise["fields"] == {
        "name": None,
        "description": None,
        "prices": [],
        "sizes": [],
    }

    pizza = payload["items"][1]
    assert pizza["kind"] == "menu_item"
    assert pizza["category"] == {"label": "mains", "confidence": 0.83}
    assert pizza["fields"]["prices"] == [
        {"value": 590, "currency": "RUB", "raw": "590"},
        {"value": 790, "currency": "RUB", "raw": "790"},
    ]
    assert pizza["fields"]["sizes"] == [
        {"value": 30, "unit": "cm", "raw": "30 cm"},
        {"value": 40, "unit": "cm", "raw": "40 cm"},
    ]
    assert pizza["issues"] == [
        {
            "code": "MULTIPLE_PRICES",
            "level": "warning",
            "message": "Multiple prices were detected and need review.",
            "path": "/items/1/fields/prices",
            "details": None,
        },
        {
            "code": "MULTIPLE_SIZES",
            "level": "warning",
            "message": "Multiple sizes were detected and need review.",
            "path": "/items/1/fields/sizes",
            "details": None,
        },
    ]


def test_parse_menu_keeps_header_context_across_noise_lines(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse",
        json={
            "schema_version": "v1",
            "text": "САЛАТЫ\n-----\nГреческий 220 г 350 ₽",
            "currency_hint": "RUB",
            "category_labels": ["salads", "soups", "mains", "desserts", "drinks", "other"],
        },
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["items"][1]["kind"] == "noise"
    assert payload["items"][2]["category"] == {"label": "salads", "confidence": 0.72}
    assert payload["items"][2]["issues"] == [
        {
            "code": "CATEGORY_MODEL_LOW_CONFIDENCE",
            "level": "info",
            "message": "Category model confidence was too low; heuristic fallback was used.",
            "path": "/items/2/category",
            "details": None,
        }
    ]


def test_parse_menu_rejects_blank_text_with_api_error_shape(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse",
        json={"schema_version": "v1", "text": "   \n  "},
    )

    assert response.status_code == 422
    assert response.json()["schema_version"] == "v1"
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_parse_menu_uses_full_default_taxonomy_when_labels_are_omitted(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse",
        json={
            "schema_version": "v1",
            "text": "Pepperoni Pizza 30 cm 590",
            "currency_hint": "RUB",
        },
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["category"] == {
        "label": "pizza",
        "confidence": 0.72,
    }


def test_parse_menu_falls_back_to_heuristic_when_model_is_unavailable() -> None:
    app = create_app(category_classifier_loader=lambda: None)

    with TestClient(app) as client:
        version_response = client.get("/api/v1/version")
        parse_response = client.post(
            "/api/v1/menu/parse",
            json={
                "schema_version": "v1",
                "text": "Americano 300 ml 180",
                "currency_hint": "RUB",
                "category_labels": ["salads", "soups", "mains", "desserts", "drinks", "other"],
            },
        )

    assert version_response.status_code == 200
    assert version_response.json() == {
        "version": "0.0.1",
        "category_model": "slice1-keyword-baseline@0.1.0",
        "configured_category_model": CONFIGURED_CATEGORY_MODEL_ID,
        "category_model_ready": False,
    }

    assert parse_response.status_code == 200
    payload = parse_response.json()
    assert payload["model_version"] == {
        "category_model": "slice1-keyword-baseline@0.1.0",
        "ner_model": "slice1-deterministic-fields@0.1.0",
    }
    assert payload["items"][0]["category"] == {"label": "drinks", "confidence": 0.78}
    assert payload["issues"] == [
        {
            "code": "CATEGORY_MODEL_UNAVAILABLE",
            "level": "info",
            "message": (
            "Configured category classifier is unavailable; "
            "heuristic fallback is active."
        ),
        "path": None,
        "details": {"configured_model": CONFIGURED_CATEGORY_MODEL_ID},
    }
]
