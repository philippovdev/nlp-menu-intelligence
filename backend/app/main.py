from fastapi import FastAPI

APP_VERSION = "0.0.1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Menu Intelligence API",
        version=APP_VERSION,
        openapi_url="/api/openapi.json",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    @app.get("/api/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/version")
    async def version() -> dict[str, str]:
        return {"version": APP_VERSION}

    return app


app = create_app()

