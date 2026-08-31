"""RAG search agent backed by the deterministic application orchestrator.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.settings import configure_vertex_ai_environment, get_logger
from src.application.invoice_analysis import (
    AnalysisFilter,
    InvoiceAnalysisSpec,
    analyze_invoice_records,
    build_invoice_analysis_payload,
    render_invoice_analysis_evidence,
)
from src.application.invoice_analytics import (
    aggregate_supplier_totals,
    detect_language,
    render_supplier_totals,
)
from src.application.orchestrator import AgentLedgerOrchestrator
from src.embeddings.vector_index import InvoiceRecord


class AggregateInvoicesRequest(BaseModel):
    """Arguments for the deterministic supplier-total agent tool."""

    request: str = Field(
        description="The user's complete supplier-total or supplier-ranking request."
    )
    include_evidence: bool = Field(
        default=False,
        description=(
            "True only when the user asks to show, list, verify, or provide the invoices "
            "used in the calculation."
        ),
    )


class InvoiceAnalysisFilterRequest(BaseModel):
    """One structured predicate selected from a strict field/operator allowlist."""

    field: Literal[
        "supplier",
        "currency",
        "invoice_date",
        "year",
        "month",
        "total",
        "source",
    ]
    operator: Literal["eq", "neq", "in", "contains", "gt", "gte", "lt", "lte"] = "eq"
    value: str = Field(
        description=(
            "One comparison value. For operator='in', provide the explicitly requested values "
            "as one comma-separated string, for example '2002,2009'."
        )
    )


class InvoiceAnalysisHavingRequest(BaseModel):
    """A SQL-HAVING-like condition over the selected aggregate value."""

    operator: Literal["eq", "neq", "gt", "gte", "lt", "lte"]
    value: str


class AnalyzeInvoicesRequest(BaseModel):
    """Controlled generic invoice analysis arguments generated from one user request."""

    request: str = Field(description="The user's complete analysis request.")
    filters: list[InvoiceAnalysisFilterRequest] = Field(default_factory=list)
    group_by: list[
        Literal["supplier", "currency", "invoice_date", "year", "month"]
    ] = Field(default_factory=list)
    operation: Literal["count", "sum", "average", "minimum", "maximum"]
    having: InvoiceAnalysisHavingRequest | None = None
    sort_by: Literal["value", "invoice_count", "group"] = "value"
    sort_direction: Literal["asc", "desc"] = "desc"
    group_limit: int = Field(default=10, ge=1, le=200)
    requested_invoice_count: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description=(
            "The exact invoice quantity requested by the user for each selected group. "
            "Copy the user's number; never default it to 30. Leave null to show all evidence."
        ),
    )
    include_evidence: bool = Field(
        default=False,
        description=(
            "True when the user explicitly asks for matching invoices or when comparing explicit "
            "periods, because period comparisons include their supporting invoice tables."
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedInvoiceAnalysis:
    """Precomputed model context plus an optional deterministic evidence appendix."""

    model_input: str
    evidence_markdown: str
    record_count: int
    filtered_count: int
    group_count: int
    operation: str
    group_by: tuple[str, ...]


_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_COMPARISON_HINTS = (
    "compare",
    "comparison",
    "difference between",
    "versus",
    " vs ",
    "compara",
    "comparación",
    "diferencia entre",
    "frente a",
)
_EVIDENCE_PATTERNS = (
    re.compile(r"\b(?:show|list|display|provide|return|find|give me)\b.*\binvoices?\b"),
    re.compile(r"\binvoices?\s+used\b"),
    re.compile(r"\b(?:evidence|supporting records?)\b"),
    re.compile(r"\b(?:muestra|muéstrame|lista|encuentra|dame|devuelve)\b.*\bfacturas?\b"),
    re.compile(r"\bfacturas?\s+utilizadas?\b"),
    re.compile(r"\b(?:evidencia|registros de soporte)\b"),
)


def _normalize_analysis_arguments(arguments: AnalyzeInvoicesRequest) -> AnalyzeInvoicesRequest:
    """Make explicit comparison scope and evidence intent deterministic."""

    request = arguments.request.strip()
    normalized_request = request.casefold()
    is_comparison = any(
        hint in normalized_request for hint in _COMPARISON_HINTS
    )
    include_evidence = is_comparison or any(
        pattern.search(normalized_request) for pattern in _EVIDENCE_PATTERNS
    )
    updates: dict[str, Any] = {
        "include_evidence": include_evidence,
        "requested_invoice_count": (
            arguments.requested_invoice_count if include_evidence else None
        ),
    }
    years = tuple(dict.fromkeys(_YEAR_PATTERN.findall(request)))
    compares_years = len(years) >= 2 and is_comparison
    if compares_years:
        operation = _comparison_operation(normalized_request)
        retained_filters = [item for item in arguments.filters if item.field != "year"]
        retained_filters.append(
            InvoiceAnalysisFilterRequest(
                field="year",
                operator="in",
                value=",".join(years),
            )
        )
        updates.update(
            {
                "filters": retained_filters,
                "group_by": ["year"] if operation == "count" else ["year", "currency"],
                "operation": operation,
                "having": None,
                "sort_by": "group",
                "sort_direction": "asc",
                "group_limit": 200,
                "include_evidence": True,
            }
        )
    return arguments.model_copy(update=updates)


def _comparison_operation(request: str) -> str:
    if any(term in request for term in ("average", "mean", "promedio", "media")):
        return "average"
    if any(term in request for term in ("minimum", "lowest", "mínimo", "menor")):
        return "minimum"
    if any(term in request for term in ("maximum", "highest", "máximo", "mayor")):
        return "maximum"
    if any(
        term in request
        for term in ("how many", "invoice count", "number of invoices", "cuántas", "conteo")
    ):
        return "count"
    return "sum"


class InvoiceSearchAgent:
    """Expose invoice vector retrieval as a focused agent capability."""

    def __init__(
        self,
        orchestrator: AgentLedgerOrchestrator,
        *,
        max_results: int = 10,
        max_list_results: int = 200,
    ) -> None:
        self._orchestrator = orchestrator
        self._max_results = max_results
        self._max_list_results = max_list_results
        self._logger = get_logger()

    def semantic_search_invoices(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Find targeted invoices, including complete OCR-backed invoice details."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Invoice search query cannot be empty.")
        bounded_limit = min(max(limit, 1), self._max_results)
        results = self._orchestrator.search(normalized_query, limit=bounded_limit)
        normalized_query_casefold = normalized_query.casefold()
        exact_matches = []
        for result in results:
            invoice_number = str(result.invoice.get("invoice_number") or "").strip()
            if invoice_number and invoice_number.casefold() in normalized_query_casefold:
                exact_matches.append(result)
        if exact_matches:
            results = exact_matches
        return {
            "status": "success" if results else "empty",
            "query": normalized_query,
            "count": len(results),
            "exact_invoice_number_match": bool(exact_matches),
            "results": [item.to_dict() for item in results],
        }

    def list_all_invoices(self) -> dict[str, Any]:
        """List compact metadata once for global, statistical, or aggregate questions."""

        selected_records, truncated = self._load_invoice_records()
        invoices: list[dict[str, Any]] = []
        for record in selected_records:
            invoice = record.invoice
            invoices.append(
                {
                    "document_id": record.document_id,
                    "image_identifier": record.image_identifier,
                    "source_path": record.source_path,
                    "invoice_number": invoice.get("invoice_number"),
                    "supplier_name": invoice.get("supplier_name"),
                    "invoice_date": invoice.get("invoice_date"),
                    "currency": invoice.get("currency"),
                    "subtotal": invoice.get("subtotal"),
                    "tax": invoice.get("tax"),
                    "total": invoice.get("total"),
                }
            )
        return {
            "status": "success" if invoices else "empty",
            "count": len(invoices),
            "truncated": truncated,
            "invoices": invoices,
        }

    def aggregate_invoices(self, *, request: str, include_evidence: bool) -> str:
        """Calculate supplier totals and render evidence without asking Gemini to do arithmetic."""

        normalized_request = request.strip()
        if not normalized_request:
            raise ValueError("Invoice aggregation request cannot be empty.")
        operation_started = perf_counter()
        records, truncated = self._load_invoice_records()
        calculation_started = perf_counter()
        aggregation = aggregate_supplier_totals(records, truncated=truncated)
        calculation_seconds = perf_counter() - calculation_started
        render_started = perf_counter()
        markdown = render_supplier_totals(
            aggregation,
            include_evidence=include_evidence,
            language=detect_language(normalized_request),
        )
        render_seconds = perf_counter() - render_started
        self._logger.info(
            f"aggregate_invoices completed in {perf_counter() - operation_started:.3f}s | "
            f"records={len(records)} included={aggregation.included_invoice_count} "
            f"excluded={len(aggregation.exclusions)} groups={len(aggregation.groups)} "
            f"calculation={calculation_seconds:.3f}s render={render_seconds:.3f}s "
            f"include_evidence={include_evidence} response_chars={len(markdown)} "
            f"gemini_calculation_calls=0."
        )
        return markdown

    def analyze_invoices(self, arguments: AnalyzeInvoicesRequest) -> PreparedInvoiceAnalysis:
        """Precompute one generic analysis for a short grounded Gemini explanation."""

        arguments = _normalize_analysis_arguments(arguments)
        normalized_request = arguments.request.strip()
        if not normalized_request:
            raise ValueError("Invoice analysis request cannot be empty.")
        operation_started = perf_counter()
        records, truncated = self._load_invoice_records()
        having = arguments.having
        spec = InvoiceAnalysisSpec(
            group_by=tuple(arguments.group_by),
            operation=arguments.operation,
            filters=tuple(
                AnalysisFilter(field=item.field, operator=item.operator, value=item.value)
                for item in arguments.filters
            ),
            having_operator=having.operator if having else None,
            having_value=having.value if having else None,
            sort_by=arguments.sort_by,
            sort_direction=arguments.sort_direction,
            group_limit=arguments.group_limit,
            requested_invoice_count=arguments.requested_invoice_count,
            include_evidence=arguments.include_evidence,
        )
        calculation_started = perf_counter()
        result = analyze_invoice_records(records, spec=spec, truncated=truncated)
        calculation_seconds = perf_counter() - calculation_started
        preparation_started = perf_counter()
        payload = build_invoice_analysis_payload(result)
        evidence_markdown = render_invoice_analysis_evidence(
            result,
            language=detect_language(normalized_request),
        )
        model_input = json.dumps(
            {
                "user_request": normalized_request,
                "authoritative_precomputed_analysis": payload,
                "deterministic_evidence_appended_after_explanation": bool(
                    evidence_markdown
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        preparation_seconds = perf_counter() - preparation_started
        self._logger.info(
            f"analyze_invoices completed in {perf_counter() - operation_started:.3f}s | "
            f"records={len(records)} filtered={result.filtered_invoice_count} "
            f"groups={result.groups_before_having} selected={len(result.groups)} "
            f"fallback={len(result.fallback_groups)} operation={spec.operation} "
            f"group_by={spec.group_by} calculation={calculation_seconds:.3f}s "
            f"preparation={preparation_seconds:.3f}s requested_invoice_count="
            f"{spec.requested_invoice_count} model_input_chars={len(model_input)} "
            f"evidence_chars={len(evidence_markdown)} gemini_calculation_calls=0 "
            f"gemini_explanation_calls=1."
        )
        return PreparedInvoiceAnalysis(
            model_input=model_input,
            evidence_markdown=evidence_markdown,
            record_count=len(records),
            filtered_count=result.filtered_invoice_count,
            group_count=len(result.groups or result.fallback_groups),
            operation=spec.operation,
            group_by=spec.group_by,
        )

    def _load_invoice_records(self) -> tuple[list[InvoiceRecord], bool]:
        fetch_limit = min(self._max_list_results + 1, 1000)
        records = self._orchestrator.list_invoices(limit=fetch_limit)
        truncated = len(records) > self._max_list_results
        return records[: self._max_list_results], truncated


def build_adk_agent(search_agent: InvoiceSearchAgent, *, model_name: str) -> Any:
    """Build the specialized Google ADK retrieval agent."""

    configure_vertex_ai_environment()
    try:
        from google.adk.agents import Agent
        from google.adk.tools import AgentTool
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("google-adk is required to build the search agent.") from error

    async def run_deterministic_aggregation(callback_context):
        content = callback_context.user_content
        request_json = "".join(
            str(part.text)
            for part in (content.parts if content else []) or []
            if getattr(part, "text", None)
        )
        arguments = AggregateInvoicesRequest.model_validate_json(request_json)
        markdown = await asyncio.to_thread(
            search_agent.aggregate_invoices,
            request=arguments.request,
            include_evidence=arguments.include_evidence,
        )
        return types.Content(role="model", parts=[types.Part(text=markdown)])

    async def prepare_deterministic_analysis(callback_context, llm_request):
        content = callback_context.user_content
        request_json = "".join(
            str(part.text)
            for part in (content.parts if content else []) or []
            if getattr(part, "text", None)
        )
        arguments = AnalyzeInvoicesRequest.model_validate_json(request_json)
        prepared = await asyncio.to_thread(search_agent.analyze_invoices, arguments)
        callback_context.state["temp:invoice_analysis_evidence"] = (
            prepared.evidence_markdown
        )
        llm_request.contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prepared.model_input)],
            )
        ]
        return None

    def append_deterministic_evidence(callback_context):
        evidence = str(
            callback_context.state.get("temp:invoice_analysis_evidence", "") or ""
        ).strip()
        if not evidence:
            return None
        explanation = str(
            callback_context.state.get("temp:invoice_analysis_explanation", "") or ""
        ).strip()
        combined = f"{explanation}\n\n{evidence}" if explanation else evidence
        return types.Content(role="model", parts=[types.Part(text=combined)])

    aggregation_agent = Agent(
        name="aggregate_invoices",
        model=model_name,
        description=(
            "Deterministically groups invoice totals by supplier and normalized currency, "
            "optionally returning every invoice used as a ready-to-display Markdown table."
        ),
        instruction=(
            "This agent is implemented by deterministic Python. Its callback completes the "
            "request before any model generation occurs."
        ),
        input_schema=AggregateInvoicesRequest,
        before_agent_callback=run_deterministic_aggregation,
        include_contents="none",
        mode="chat",
    )
    aggregation_tool = AgentTool(
        agent=aggregation_agent,
        skip_summarization=True,
        include_plugins=True,
    )
    analysis_agent = Agent(
        name="analyze_invoices",
        model=model_name,
        description=(
            "Runs deterministic structured invoice filters, grouping, count/sum/average/min/max, "
            "HAVING-like conditions, comparisons, dynamic invoice limits, and evidence rendering, "
            "then explains only the precomputed facts."
        ),
        instruction=(
            "Answer the user from authoritative_precomputed_analysis only. Begin with a concise, "
            "plain-language conclusion that directly addresses the request. Explain meaningful "
            "differences using only the supplied precomputed_comparisons and statistics. Never "
            "perform arithmetic, infer a missing value, combine currencies, introduce unrequested "
            "years or groups, or mention implementation details such as Python, deterministic "
            "analysis, JSON, tools, or Gemini. Do not produce invoice-level rows: when the user "
            "requested invoices, a deterministic evidence table is appended after your answer. "
            "A small aggregate comparison table is allowed when it improves clarity. Preserve "
            "all amounts, signs, percentages, counts, dates, names, and currencies exactly as "
            "supplied. "
            "If condition_matched is false, clearly say that the requested condition was not met "
            "and explain the supplied closest result. Match the language of user_request and keep "
            "the explanation brief."
        ),
        input_schema=AnalyzeInvoicesRequest,
        before_model_callback=prepare_deterministic_analysis,
        after_agent_callback=append_deterministic_evidence,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.8,
            max_output_tokens=1200,
        ),
        include_contents="none",
        mode="chat",
        output_key="temp:invoice_analysis_explanation",
    )
    analysis_tool = AgentTool(
        agent=analysis_agent,
        skip_summarization=True,
        include_plugins=True,
    )
    return Agent(
        name="search_invoices",
        model=model_name,
        description=(
            "Classifies one invoice request and retrieves evidence through exactly one "
            "Firestore tool."
        ),
        instruction=(
            "You are Agent Ledger's invoice retrieval specialist. Classify each complete request "
            "before retrieving evidence and choose exactly one retrieval path.\n\n"
            "HIGHEST-PRIORITY ROUTING RULE: when the user supplies a concrete invoice number and "
            "asks what it contains, asks for its information, or requests its details, call "
            "semantic_search_invoices exactly once with the complete request and limit=10. Never "
            "send a concrete-invoice detail request to analyze_invoices, aggregate_invoices, or "
            "list_all_invoices. If exact_invoice_number_match=true, describe that invoice "
            "comprehensively from its structured fields and raw_text. Present the source, general "
            "invoice information, supplier and buyer/contact information, every line item in a "
            "table, financial totals, bank/payment details, and payment terms when those values "
            "are available. Use clear sections named General Invoice Information, Supplier & "
            "Buyer Information, Line Items, Financial Totals, and Payment & Bank Details when "
            "answering in English, with natural translated headings in other languages. Introduce "
            "the invoice and its source before those sections. Do not respond with an analytical "
            "count, dataset size, generic "
            "Conclusion/Analysis/Evidence template, or a one-row metadata table. Preserve unusual "
            "source values exactly and never fill a missing field.\n\n"
            "For a supplier-total, supplier-ranking, highest-total-supplier, or total-by-supplier "
            "question, call aggregate_invoices exactly once. Pass the user's complete request. "
            "Set include_evidence=true only when the user asks to show, list, verify, or provide "
            "the invoices used in the calculation. The tool performs currency normalization, "
            "exact Decimal SUM/COUNT calculations, and Markdown rendering in Python; its returned "
            "text is the complete final answer. Never call another tool before or after it.\n\n"
            "For other structured global analytics, call analyze_invoices exactly once. This "
            "includes filtering by structured fields; grouping by supplier, currency, date, year, "
            "or month; count, sum, average, minimum, or maximum; threshold conditions; sorting; "
            "and returning a user-requested quantity of evidence. Translate the user's wording "
            "into the tool schema without doing the calculation yourself. Copy every explicit "
            "invoice quantity into requested_invoice_count; never assume the number is 30. For a "
            "comparison between explicit values of one dimension, restrict the query to exactly "
            "those values with one comma-separated 'in' filter. For example, 'Compare 2002 with "
            "2009' uses year in '2002,2009', group_by=[year,currency], operation=sum, no HAVING, "
            "and include_evidence=true so Python appends only those years' invoice tables after "
            "the explanation. Do not return unrelated years. For 'N invoices "
            "from the same month and year', use group_by=[year, month], operation=count, HAVING "
            "count >= N, requested_invoice_count=N, include_evidence=true, and group_limit=1. For "
            "'N invoices from a specific date', apply an invoice_date equality filter, use "
            "operation=count with no grouping, requested_invoice_count=N, and "
            "include_evidence=true. For sum, average, minimum, or maximum, always include currency "
            "in group_by unless one exact currency equality filter is present; never combine "
            "different currencies. Apart from explicit period comparisons, set "
            "include_evidence=false unless the user asks to show, list, find, or provide invoice "
            "records. The child explains only values already "
            "calculated by Python and appends requested invoice evidence deterministically. Its "
            "returned response is final; never call another tool.\n\n"
            "Use list_all_invoices exactly once only for an explicit unfiltered request to inspect "
            "or list the full collection without structured analysis. Do not call another tool "
            "before or after list_all_invoices.\n\n"
            "For any other targeted supplier, date, content, or semantic match, call "
            "semantic_search_invoices exactly once. Do not call list_all_invoices before or after "
            "semantic_search_invoices, and do not call either deterministic analysis tool.\n\n"
            "Never chain retrieval calls for one request unless the selected tool explicitly "
            "returns an error. Use only fields returned by the selected tool as financial "
            "evidence. Preserve supplier names, invoice numbers, dates, currencies, and decimal "
            "precision exactly. For calculations, exclude missing or invalid amounts, explain "
            "exclusions, and never combine genuinely different currencies into one monetary total. "
            "For deterministic analytics, normalize EUR and the euro symbol to EUR, and normalize "
            "USD, US$, and the dollar symbol to USD. Explicitly disclose this alias normalization "
            "when relevant. Keep every other currency separate.\n\n"
            "Do not reproduce or recalculate aggregate_invoices or analyze_invoices output. Their "
            "child responses are already grounded, formatted, and final.\n\n"
            "If list_all_invoices reports truncated=true, disclose that the calculation covers "
            "only the returned records. If retrieval returns no records, say so. Never invent a "
            "record, amount, date, or calculation. Mention source_path when it helps verification. "
            "Match the user's language and return the complete evidence-based answer."
        ),
        tools=[
            aggregation_tool,
            analysis_tool,
            search_agent.semantic_search_invoices,
            search_agent.list_all_invoices,
        ],
        mode="chat",
        output_key="retrieval_result",
    )
