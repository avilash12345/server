"""TubeFlow download API package."""

from .app import app

__all__ = ["app", "main"]


def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
