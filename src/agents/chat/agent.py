"""User-facing ADK agent that presents grounded financial answers.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from typing import Any

from config.settings import Settings, configure_vertex_ai_environment
from src.agents.chat.prompts import CHAT_INSTRUCTION
from src.agents.search.agent import InvoiceSearchAgent, build_adk_agent as build_search_agent


def build_chat_agent(*, settings: Settings, search_agent: InvoiceSearchAgent) -> Any:
    """Build the primary agentic RAG chat agent with Firestore search access."""

    configure_vertex_ai_environment()
    try:
        from google.adk.agents import Agent
        from google.adk.tools import AgentTool
        from google.genai import types
    except ImportError as error:
        raise RuntimeError(
            "google-adk and google-genai are required to build the chat agent."
        ) from error
    retrieval_agent = build_search_agent(
        search_agent,
        model_name=settings.gemini_model,
    )
    retrieval_tool = AgentTool(
        agent=retrieval_agent,
        skip_summarization=True,
        include_plugins=True,
    )
    return Agent(
        name="invoice_chat",
        model=settings.gemini_model,
        description=(
            "Answers invoice questions by retrieving grounded records from Firestore vector search."
        ),
        instruction=CHAT_INSTRUCTION,
        tools=[retrieval_tool],
        generate_content_config=types.GenerateContentConfig(
            temperature=settings.chat_temperature,
            top_p=settings.chat_top_p,
            max_output_tokens=settings.chat_max_output_tokens,
        ),
        mode="chat",
        output_key="chat_response",
    )
