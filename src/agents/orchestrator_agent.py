"""Google ADK supervisor that routes work to specialized invoice agents.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from typing import Any

from config.settings import Settings
from src.agents.chat.agent import build_chat_agent
from src.agents.extractor.agent import InvoiceExtractor, build_adk_agent as build_extractor_agent
from src.agents.search.agent import InvoiceSearchAgent, build_adk_agent as build_search_agent


def build_adk_orchestrator(
    *,
    settings: Settings,
    extractor: InvoiceExtractor,
    search_agent: InvoiceSearchAgent,
) -> Any:
    """Build the supervisor and attach extraction, retrieval, and chat sub-agents."""

    try:
        from google.adk.agents import Agent
    except ImportError as error:
        raise RuntimeError("google-adk is required to build the agent supervisor.") from error
    extractor_agent = build_extractor_agent(extractor, model_name=settings.gemini_model)
    retrieval_agent = build_search_agent(search_agent, model_name=settings.gemini_model)
    chat_agent = build_chat_agent(settings=settings, search_agent=search_agent)
    return Agent(
        name="agent_ledger_coordinator",
        model=settings.gemini_model,
        description="Routes invoice ingestion, retrieval, and response tasks.",
        instruction=(
            "Delegate new invoice image analysis to invoice_extractor. Delegate historical or "
            "semantic invoice questions to invoice_search. Use invoice_chat to present already "
            "grounded results. Never calculate or invent values without retrieved evidence."
        ),
        sub_agents=[extractor_agent, retrieval_agent, chat_agent],
    )
