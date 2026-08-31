"""Command-line entry point for invoice indexing and semantic search.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Sequence

from config.settings import ConfigurationError, Settings, get_logger
from src.agents.chat.agent import build_chat_agent
from src.agents.chat.service import ChatService
from src.agents.extractor.agent import GeminiInvoiceExtractor, discover_images
from src.agents.search.agent import InvoiceSearchAgent
from src.application.orchestrator import AgentLedgerOrchestrator, IndexingSummary
from src.embeddings.vector_index import FirestoreVectorIndex, build_vector_index_command
from src.embeddings.vector_search import SentenceTransformerEmbeddingService


def build_orchestrator(settings: Settings) -> AgentLedgerOrchestrator:
    """Compose production adapters once for the requested command."""

    extractor = GeminiInvoiceExtractor(settings)
    embeddings = SentenceTransformerEmbeddingService(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        device=settings.device,
        normalize=settings.normalize_embeddings,
    )
    vector_index = FirestoreVectorIndex(settings)
    return AgentLedgerOrchestrator(
        settings=settings,
        extractor=extractor,
        embeddings=embeddings,
        vector_index=vector_index,
    )


def build_chat_agent_system(settings: Settings) -> Any:
    """Compose the RAG chat agent with its Firestore-backed search tool."""

    orchestrator = build_orchestrator(settings)
    search_agent = InvoiceSearchAgent(
        orchestrator,
        max_results=settings.chat_search_limit,
        max_list_results=settings.chat_list_limit,
    )
    return build_chat_agent(settings=settings, search_agent=search_agent)


def build_chat_service(settings: Settings) -> ChatService:
    """Compose the ADK runner and in-memory conversation session service."""

    agent = build_chat_agent_system(settings)
    return ChatService(
        agent=agent,
        app_name=settings.chat_app_name,
        model_name=settings.gemini_model,
        max_output_tokens=settings.chat_max_output_tokens,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-ledger",
        description="Extract invoice images, index their embeddings, and search them in Firestore.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate configuration and input data."
    )
    validate_parser.add_argument(
        "--cloud", action="store_true", help="Also require cloud project and Gemini credentials."
    )

    index_parser = subparsers.add_parser("index", help="Extract and index prepared invoice images.")
    index_parser.add_argument("--force", action="store_true", help="Reprocess current documents.")
    index_parser.add_argument("--limit", type=int, help="Process at most this many images.")

    search_parser = subparsers.add_parser("search", help="Search indexed invoices semantically.")
    search_parser.add_argument("query", help="Natural-language invoice query.")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum returned invoices.")
    search_parser.add_argument(
        "--distance-threshold", type=float, help="Optional Firestore distance threshold."
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Ask the agentic RAG chat a question about indexed invoices."
    )
    chat_parser.add_argument("message", help="User message sent to the invoice chat agent.")
    chat_parser.add_argument("--user-id", default="cli-user", help="Conversation user identifier.")
    chat_parser.add_argument(
        "--session-id", default="cli-session", help="Conversation session identifier."
    )

    subparsers.add_parser(
        "index-command", help="Print the gcloud command that creates the Firestore vector index."
    )
    return parser


def _print_progress(summary: IndexingSummary) -> None:
    print(
        f"discovered={summary.discovered} processed={summary.processed} "
        f"skipped={summary.skipped} failed={summary.failed}",
        file=sys.stderr,
    )


def _run_validate(settings: Settings, *, require_cloud: bool) -> int:
    settings.validate(require_cloud=require_cloud)
    image_count = sum(1 for _ in discover_images(settings.input_dir))
    output = {
        "status": "ok",
        "application_root": str(settings.app_root),
        "input_directory": str(settings.input_dir),
        "output_directory": str(settings.output_dir),
        "image_count": image_count,
        "insert_into_index": settings.insert_into_index,
        "gemini_backend": "vertex_ai",
        "gemini_model": settings.gemini_model,
        "chat_max_output_tokens": settings.chat_max_output_tokens,
        "chat_search_limit": settings.chat_search_limit,
        "chat_list_limit": settings.chat_list_limit,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "distance_measure": settings.distance_measure,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Run one CLI command and return its process exit status."""

    arguments = _build_parser().parse_args(argv)
    active_settings = settings or Settings.from_env()
    if arguments.command == "validate":
        return _run_validate(active_settings, require_cloud=arguments.cloud)
    if arguments.command == "index-command":
        print(build_vector_index_command(active_settings))
        return 0
    if arguments.command == "index":
        if not active_settings.insert_into_index:
            chat_agent = build_chat_agent_system(active_settings)
            output = {
                "status": "selected",
                "mode": "chat",
                "insert_into_index": False,
                "selected_agent": chat_agent.name,
            }
            get_logger().info(
                f"Image ingestion is disabled; selected the {chat_agent.name} agent."
            )
            print(json.dumps(output, indent=2, sort_keys=True))
            return 0
        active_settings.validate(require_cloud=True)
        get_logger().info("Image ingestion is enabled; starting the indexing pipeline.")
        orchestrator = build_orchestrator(active_settings)
        summary = orchestrator.index_images(
            force=arguments.force,
            limit=arguments.limit,
            progress=_print_progress,
        )
        output = {
            "status": "completed" if not summary.failed else "completed_with_errors",
            "mode": "ingestion",
            "insert_into_index": True,
            **summary.to_dict(),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 1 if summary.failed else 0
    if arguments.command == "search":
        active_settings.validate(require_cloud=True, require_input=False)
        orchestrator = build_orchestrator(active_settings)
        results = orchestrator.search(
            arguments.query,
            limit=arguments.limit,
            distance_threshold=arguments.distance_threshold,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True))
        return 0
    if arguments.command == "chat":
        active_settings.validate(require_cloud=True, require_input=False)
        chat_service = build_chat_service(active_settings)
        response = asyncio.run(
            chat_service.respond(
                arguments.message,
                user_id=arguments.user_id,
                session_id=arguments.session_id,
            )
        )
        print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


def main() -> None:
    """Translate expected application errors into concise CLI failures."""

    try:
        raise SystemExit(run())
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as error:
        get_logger().error(str(error))
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
