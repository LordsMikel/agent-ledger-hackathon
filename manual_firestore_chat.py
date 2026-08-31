"""Index twenty additional invoice images, then start an interactive RAG chat.

This is an intentionally manual cloud integration test. It uses real Gemini and
Firestore services and can therefore incur usage costs.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

from config.settings import ConfigurationError, Settings, get_logger
from src.agents.chat.agent import build_chat_agent
from src.agents.chat.service import ChatExecutionError, ChatService
from src.agents.extractor.agent import discover_images
from src.agents.search.agent import InvoiceSearchAgent
from src.application.orchestrator import AgentLedgerOrchestrator
from src.main import build_orchestrator


ALREADY_INDEXED_IMAGE_COUNT = 3
TEST_IMAGE_COUNT = 20
EXIT_COMMANDS = frozenset({"exit", "quit", "salir"})
PROJECT_ROOT = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Skip the first three invoices, index the next twenty with Gemini and "
            "Firestore, then open an interactive agentic RAG chat."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract the twenty selected documents even when they are current.",
    )
    parser.add_argument("--user-id", default="manual-test-user")
    parser.add_argument("--session-id", default="manual-firestore-chat")
    parser.add_argument(
        "--interface",
        choices=("terminal", "adk-web"),
        default="terminal",
        help="Open the local terminal loop or Google's ADK development web interface.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used by ADK Web when --interface=adk-web.",
    )
    return parser


def _test_settings() -> Settings:
    """Enable one strict twenty-image batch without changing environment variables."""

    return replace(
        Settings.from_env(),
        insert_into_index=True,
        gemini_batch_enabled=True,
        gemini_batch_size=TEST_IMAGE_COUNT,
        gemini_batch_fallback=False,
    )


def _build_chat_service(
    settings: Settings,
    orchestrator: AgentLedgerOrchestrator,
) -> ChatService:
    """Reuse the indexing orchestrator and its loaded embedding model for chat."""

    search_agent = InvoiceSearchAgent(
        orchestrator,
        max_results=settings.chat_search_limit,
        max_list_results=settings.chat_list_limit,
    )
    agent = build_chat_agent(settings=settings, search_agent=search_agent)
    return ChatService(
        agent=agent,
        app_name=settings.chat_app_name,
        model_name=settings.gemini_model,
        max_output_tokens=settings.chat_max_output_tokens,
    )


async def _run_chat(
    service: ChatService,
    *,
    user_id: str,
    session_id: str,
) -> None:
    print("\nInteractive invoice chat is ready.")
    print("Ask about the indexed invoices. Type 'salir', 'exit', or 'quit' to finish.\n")
    while True:
        try:
            message = input("You> ").strip()
        except EOFError:
            print()
            return
        if message.casefold() in EXIT_COMMANDS:
            return
        if not message:
            continue
        try:
            response = await service.respond(
                message,
                user_id=user_id,
                session_id=session_id,
            )
        except ChatExecutionError as error:
            print(f"AgentLedger error> {error}\n")
            continue
        print(f"\nAgentLedger> {response.text}")
        metadata = {
            "finish_reason": response.finish_reason,
            "max_output_tokens": response.max_output_tokens,
            "token_usage": response.token_usage.to_dict(),
            "tools_used": list(response.tools_used),
        }
        print(f"Metadata> {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}\n")


def _find_adk_executable() -> str:
    """Find the ADK command beside the active Python or on the executable path."""

    adjacent_executable = Path(sys.executable).with_name("adk")
    if adjacent_executable.is_file():
        return str(adjacent_executable)
    discovered_executable = shutil.which("adk")
    if discovered_executable:
        return discovered_executable
    raise RuntimeError(
        "The 'adk' command was not found. Install the project dependencies and run "
        "this script with the project virtual environment."
    )


def _run_adk_web(*, port: int) -> int:
    """Launch Google's local ADK development interface for the discovered root agent."""

    if not 1 <= port <= 65_535:
        raise ValueError("ADK Web port must be between 1 and 65535.")
    command = [_find_adk_executable(), "web", "--port", str(port)]
    print(f"\nStarting Google ADK Web at http://127.0.0.1:{port}")
    print("Select 'adk_invoice_chat' in the agent menu. Press Ctrl-C here to stop it.\n")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def run(argv: Sequence[str] | None = None) -> int:
    """Index twenty images after the three existing records, then run the chat."""

    arguments = _build_parser().parse_args(argv)
    settings = _test_settings()
    settings.validate(require_cloud=True)

    required_image_count = ALREADY_INDEXED_IMAGE_COUNT + TEST_IMAGE_COUNT
    available_images = list(discover_images(settings.input_dir))[:required_image_count]
    selected_images = available_images[ALREADY_INDEXED_IMAGE_COUNT:]
    if len(selected_images) != TEST_IMAGE_COUNT:
        raise ConfigurationError(
            f"The manual test needs {required_image_count} available images to skip "
            f"{ALREADY_INDEXED_IMAGE_COUNT} and select {TEST_IMAGE_COUNT}; found "
            f"{len(available_images)} in {settings.input_dir}."
        )

    print(
        "Indexing exactly twenty additional images with one Gemini batch: "
        + ", ".join(path.name for path in selected_images)
    )
    orchestrator = build_orchestrator(settings)
    summary = orchestrator.index_images(
        force=arguments.force,
        limit=TEST_IMAGE_COUNT,
        offset=ALREADY_INDEXED_IMAGE_COUNT,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    if summary.failed:
        print("Indexing failed; the interactive chat was not started.")
        return 1
    if summary.processed == 0 and summary.skipped == TEST_IMAGE_COUNT:
        print("All twenty images were already current. Use --force to rebuild them.")

    if arguments.interface == "adk-web":
        return _run_adk_web(port=arguments.port)
    else:
        chat_service = _build_chat_service(settings, orchestrator)
        asyncio.run(
            _run_chat(
                chat_service,
                user_id=arguments.user_id,
                session_id=arguments.session_id,
            )
        )
    return 0


def main() -> None:
    """Translate expected integration errors into a concise process failure."""

    try:
        raise SystemExit(run())
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as error:
        get_logger().error(str(error))
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        print("\nManual integration test interrupted.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
