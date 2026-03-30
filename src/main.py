"""Entry point for the SRE/DevOps Knowledge Copilot."""

import logging
import os

import uvicorn


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("APP_ENV", "production") == "development",
        log_level=log_level.lower(),
    )


def _configure_app_logging() -> None:
    """Configure src.* loggers — called at module import time so it runs before uvicorn overrides."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger("src").setLevel(log_level)


_configure_app_logging()


if __name__ == "__main__":
    main()
