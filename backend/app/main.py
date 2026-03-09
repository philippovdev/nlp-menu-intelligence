from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.document_parser import parse_menu_file
from app.menu_parser import parse_menu_text
from app.schemas import (
    ApiError,
    ApiErrorResponse,
    MenuParseRequest,
    MenuParseResponse,
    SchemaVersion,
)

APP_VERSION = "0.0.1"
APP_NAME = "menu-intelligence-api"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Menu Intelligence API",
        version=APP_VERSION,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _, exc: RequestValidationError
    ) -> JSONResponse:
        payload = ApiErrorResponse.validation_error(jsonable_encoder(exc.errors()))
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @app.exception_handler(ApiError)
    async def handle_api_error(_, exc: ApiError) -> JSONResponse:
        payload = ApiErrorResponse.from_api_error(exc)
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.get("/api/health")
    @app.get("/api/v1/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/version")
    @app.get("/api/v1/version")
    async def version() -> dict[str, str]:
        return {"version": APP_VERSION}

    @app.get("/api/status")
    async def status() -> dict[str, str]:
        return {"service": APP_NAME, "status": "ok", "version": APP_VERSION}

    @app.post("/api/v1/menu/parse", response_model=MenuParseResponse)
    async def parse_menu(request: MenuParseRequest) -> MenuParseResponse:
        return parse_menu_text(request)

    @app.post("/api/v1/menu/parse-file", response_model=MenuParseResponse)
    async def parse_menu_document(
        schema_version: SchemaVersion = Form("v1"),
        lang: str = Form("ru"),
        currency_hint: str = Form("RUB"),
        category_labels: list[str] | None = Form(default=None),
        file: UploadFile = File(description="A single menu document file"),
    ) -> MenuParseResponse:
        try:
            return await parse_menu_file(
                file,
                schema_version=schema_version,
                lang=lang,
                currency_hint=currency_hint,
                category_labels=category_labels,
            )
        finally:
            await file.close()

    return app


app = create_app()
