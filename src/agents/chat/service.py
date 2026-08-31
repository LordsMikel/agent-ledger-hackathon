"""ADK session execution and response collection for invoice chat.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.chat.models import ChatResponse, TokenUsage


class ChatExecutionError(RuntimeError):
    """Raised when the ADK agent cannot produce a final chat response."""


class ChatService:
    """Run a stateful ADK chat agent and expose grounded text plus token usage."""

    def __init__(
        self,
        *,
        agent: Any,
        app_name: str,
        model_name: str,
        max_output_tokens: int,
        session_service: Any | None = None,
        runner: Any | None = None,
    ) -> None:
        try:
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService
        except ImportError as error:
            raise RuntimeError("google-adk is required to run the chat service.") from error
        self._agent = agent
        self._app_name = app_name
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens
        if session_service is not None:
            self._session_service = session_service
        elif runner is not None:
            self._session_service = runner.session_service
        else:
            self._session_service = InMemorySessionService()
        self._runner = runner or Runner(
            agent=agent,
            app_name=app_name,
            session_service=self._session_service,
        )
        self._session_lock = asyncio.Lock()

    @property
    def agent_name(self) -> str:
        """Return the selected ADK agent name."""

        return str(self._agent.name)

    async def respond(self, message: str, *, user_id: str, session_id: str) -> ChatResponse:
        """Send one user message through ADK and collect its final grounded response."""

        normalized_message = message.strip()
        normalized_user_id = user_id.strip()
        normalized_session_id = session_id.strip()
        if not normalized_message:
            raise ValueError("Chat message cannot be empty.")
        if not normalized_user_id:
            raise ValueError("Chat user ID cannot be empty.")
        if not normalized_session_id:
            raise ValueError("Chat session ID cannot be empty.")
        await self._ensure_session(normalized_user_id, normalized_session_id)
        try:
            from google.genai import types
        except ImportError as error:
            raise RuntimeError("google-genai is required to send chat messages.") from error

        content = types.Content(
            role="user",
            parts=[types.Part(text=normalized_message)],
        )
        usage = TokenUsage()
        tools_used: list[str] = []
        final_text: str | None = None
        finish_reason: str | None = None
        last_error: str | None = None

        async for event in self._runner.run_async(
            user_id=normalized_user_id,
            session_id=normalized_session_id,
            new_message=content,
        ):
            if event.usage_metadata:
                usage.add(event.usage_metadata)
            if event.error_message:
                last_error = str(event.error_message)
            if event.finish_reason:
                finish_reason = self._stringify_enum(event.finish_reason)
            for part in getattr(event.content, "parts", None) or []:
                function_call = getattr(part, "function_call", None)
                if function_call and function_call.name:
                    tools_used.append(str(function_call.name))
            if event.is_final_response() and event.content:
                text_parts = [
                    str(part.text)
                    for part in event.content.parts or []
                    if getattr(part, "text", None)
                ]
                if text_parts:
                    final_text = "".join(text_parts).strip()

        if not final_text:
            detail = last_error or "The agent completed without a text response."
            raise ChatExecutionError(detail)
        usage.ensure_total()
        return ChatResponse(
            text=final_text,
            user_id=normalized_user_id,
            session_id=normalized_session_id,
            model=self._model_name,
            max_output_tokens=self._max_output_tokens,
            token_usage=usage,
            tools_used=tuple(dict.fromkeys(tools_used)),
            finish_reason=finish_reason,
        )

    async def _ensure_session(self, user_id: str, session_id: str) -> None:
        async with self._session_lock:
            session = await self._session_service.get_session(
                app_name=self._app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if not session:
                await self._session_service.create_session(
                    app_name=self._app_name,
                    user_id=user_id,
                    session_id=session_id,
                )

    @staticmethod
    def _stringify_enum(value: Any) -> str:
        enum_name = getattr(value, "name", None)
        if enum_name:
            return str(enum_name)
        raw_value = getattr(value, "value", value)
        return str(raw_value)
