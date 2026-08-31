"""Structured chat responses and token-usage metadata.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TokenUsage:
    """Cumulative Gemini token usage for one agent turn."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    thought_tokens: int = 0
    tool_prompt_tokens: int = 0

    def add(self, metadata: Any) -> None:
        """Accumulate one ADK usage-metadata event."""

        self.prompt_tokens += int(getattr(metadata, "prompt_token_count", 0) or 0)
        self.output_tokens += int(getattr(metadata, "candidates_token_count", 0) or 0)
        self.total_tokens += int(getattr(metadata, "total_token_count", 0) or 0)
        self.cached_tokens += int(getattr(metadata, "cached_content_token_count", 0) or 0)
        self.thought_tokens += int(getattr(metadata, "thoughts_token_count", 0) or 0)
        self.tool_prompt_tokens += int(
            getattr(metadata, "tool_use_prompt_token_count", 0) or 0
        )

    def to_dict(self) -> dict[str, int]:
        """Return JSON-compatible token counters."""

        return asdict(self)

    def ensure_total(self) -> None:
        """Derive a total when an adapter omits the aggregate counter."""

        if not self.total_tokens:
            self.total_tokens = (
                self.prompt_tokens
                + self.output_tokens
                + self.thought_tokens
                + self.tool_prompt_tokens
            )


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Final grounded answer and execution metadata returned by the chat service."""

    text: str
    user_id: str
    session_id: str
    model: str
    max_output_tokens: int
    token_usage: TokenUsage
    tools_used: tuple[str, ...] = field(default_factory=tuple)
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible chat response."""

        return {
            "text": self.text,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "token_usage": self.token_usage.to_dict(),
            "tools_used": list(self.tools_used),
            "finish_reason": self.finish_reason,
        }
