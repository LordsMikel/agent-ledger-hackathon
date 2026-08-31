"""Launch Google ADK Web against the Firestore-backed invoice chat only.

This process never prepares images, submits Vertex Batch jobs, or reads batch
artifacts from Cloud Storage. Ingestion is intentionally owned by the two
separate batch scripts.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from loguru import logger

from config.settings import ConfigurationError, Settings


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_FIRESTORE_DATABASE = "agent-test1-100"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch ADK Web for the Firestore-backed invoice RAG chat."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used by the local ADK Web interface.",
    )
    return parser


def _configure_logging(settings: Settings) -> Path:
    log_directory = settings.output_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_path = log_directory / f"adk-web-{timestamp}.log"
    logger.add(
        log_path,
        level=settings.log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS ZZ} | {elapsed} | {level:<8} | "
            "{message}"
        ),
        encoding="utf-8",
    )
    return log_path


def _find_adk_executable() -> str:
    adjacent = Path(sys.executable).with_name("adk")
    if adjacent.is_file():
        return str(adjacent)
    discovered = shutil.which("adk")
    if discovered:
        return discovered
    raise RuntimeError(
        "The 'adk' command was not found. Install the project in the active "
        "virtual environment first."
    )


def _validate(settings: Settings, *, port: int) -> None:
    if not 1 <= port <= 65_535:
        raise ConfigurationError("ADK Web port must be between 1 and 65535.")
    settings.validate(require_cloud=True, require_input=False)
    if settings.app_root.resolve() != PROJECT_ROOT.resolve():
        raise ConfigurationError(
            f"run_adk.py requires AGENT_LEDGER_ROOT={PROJECT_ROOT.resolve()}; "
            f"received {settings.app_root.resolve()}."
        )
    if settings.firestore_database != EXPECTED_FIRESTORE_DATABASE:
        raise ConfigurationError(
            f"run_adk.py requires FIRESTORE_DATABASE={EXPECTED_FIRESTORE_DATABASE}; "
            f"received {settings.firestore_database!r}."
        )


def run(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    _validate(settings, port=arguments.port)
    log_path = _configure_logging(settings)
    url = f"http://127.0.0.1:{arguments.port}"
    summary = {
        "status": "starting",
        "mode": "chat_only",
        "url": url,
        "adk_app": "adk_invoice_chat",
        "agent": "invoice_chat",
        "rag_store": "firestore",
        "firestore_database": settings.firestore_database,
        "batch_bucket_access": False,
        "image_ingestion": False,
        "log_path": str(log_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    logger.info(
        f"Starting ADK Web at {url} against Firestore database "
        f"{settings.firestore_database}."
    )
    logger.info(
        "Chat-only mode: no images, batch manifests, or Cloud Storage outputs "
        "will be read by this launcher."
    )
    command = [_find_adk_executable(), "web", "--port", str(arguments.port)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def main() -> None:
    try:
        raise SystemExit(run())
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error(str(error))
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        logger.info("ADK Web stopped by the user.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
