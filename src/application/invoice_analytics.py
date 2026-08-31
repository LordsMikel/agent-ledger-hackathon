"""Deterministic financial aggregation and presentation for invoice metadata.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Sequence

from src.embeddings.vector_index import InvoiceRecord


MONEY_QUANTUM = Decimal("0.01")
ZERO_MONEY = Decimal("0.00")
_CURRENCY_ALIASES = {
    "$": "USD",
    "US$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
}
_SPANISH_HINTS = {
    "calculo",
    "cálculo",
    "cual",
    "cuál",
    "factura",
    "facturas",
    "importe",
    "muestra",
    "proveedor",
    "total facturado",
    "utilizada",
    "utilizadas",
}


@dataclass(frozen=True, slots=True)
class InvoiceEvidence:
    """One normalized invoice included in a deterministic calculation."""

    invoice_number: str
    supplier: str
    invoice_date: str
    total: Decimal
    currency: str
    source_path: str


@dataclass(frozen=True, slots=True)
class AggregationExclusion:
    """One record excluded because a required financial field was invalid."""

    source_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class SupplierCurrencyTotal:
    """Exact total and evidence for one supplier and normalized currency."""

    supplier: str
    currency: str
    total: Decimal
    invoice_count: int
    evidence: tuple[InvoiceEvidence, ...]


@dataclass(frozen=True, slots=True)
class SupplierTotalsAggregation:
    """Complete supplier totals calculated without an LLM."""

    collection_invoice_count: int
    included_invoice_count: int
    groups: tuple[SupplierCurrencyTotal, ...]
    exclusions: tuple[AggregationExclusion, ...]
    truncated: bool


@dataclass(slots=True)
class _MutableGroup:
    total: Decimal
    evidence: list[InvoiceEvidence]


def normalize_currency(value: object) -> str | None:
    """Return a stable currency code while preserving unknown currencies separately."""

    normalized = str(value or "").strip().upper().replace(" ", "")
    if not normalized:
        return None
    return _CURRENCY_ALIASES.get(normalized, normalized)


def parse_money(value: object) -> Decimal | None:
    """Parse one extracted amount into an exact two-decimal value."""

    raw = str(value or "").strip()
    if not raw:
        return None
    parenthesized_negative = raw.startswith("(") and raw.endswith(")")
    cleaned = re.sub(r"[^0-9,\.\-()]", "", raw).replace("(", "").replace(")", "")
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        decimal_digits = len(cleaned.rsplit(",", 1)[-1])
        cleaned = cleaned.replace(",", ".") if decimal_digits == 2 else cleaned.replace(",", "")
    if parenthesized_negative and not cleaned.startswith("-"):
        cleaned = f"-{cleaned}"
    try:
        amount = Decimal(cleaned)
    except InvalidOperation:
        return None
    if not amount.is_finite():
        return None
    return amount.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def aggregate_supplier_totals(
    records: Sequence[InvoiceRecord],
    *,
    truncated: bool,
) -> SupplierTotalsAggregation:
    """Group structured Firestore values like SQL SUM/COUNT, using exact decimals."""

    grouped: dict[tuple[str, str], _MutableGroup] = {}
    exclusions: list[AggregationExclusion] = []
    for record in records:
        invoice = record.invoice
        supplier = str(invoice.get("supplier_name") or "").strip()
        currency = normalize_currency(invoice.get("currency"))
        amount = parse_money(invoice.get("total"))
        if not supplier:
            exclusions.append(
                AggregationExclusion(record.source_path, "missing supplier_name")
            )
            continue
        if currency is None:
            exclusions.append(AggregationExclusion(record.source_path, "missing currency"))
            continue
        if amount is None:
            exclusions.append(AggregationExclusion(record.source_path, "invalid total"))
            continue
        evidence = InvoiceEvidence(
            invoice_number=str(
                invoice.get("invoice_number") or record.image_identifier or record.document_id
            ),
            supplier=supplier,
            invoice_date=str(invoice.get("invoice_date") or "Unknown"),
            total=amount,
            currency=currency,
            source_path=record.source_path,
        )
        key = (supplier, currency)
        group = grouped.setdefault(key, _MutableGroup(total=ZERO_MONEY, evidence=[]))
        group.total += amount
        group.evidence.append(evidence)

    groups = tuple(
        SupplierCurrencyTotal(
            supplier=supplier,
            currency=currency,
            total=group.total.quantize(MONEY_QUANTUM),
            invoice_count=len(group.evidence),
            evidence=tuple(sorted(group.evidence, key=lambda item: item.source_path)),
        )
        for (supplier, currency), group in sorted(
            grouped.items(),
            key=lambda item: (-item[1].total, item[0][1], item[0][0].casefold()),
        )
    )
    return SupplierTotalsAggregation(
        collection_invoice_count=len(records),
        included_invoice_count=sum(group.invoice_count for group in groups),
        groups=groups,
        exclusions=tuple(exclusions),
        truncated=truncated,
    )


def detect_language(request: str) -> str:
    """Choose deterministic English or Spanish presentation for the direct response."""

    normalized = request.casefold()
    if any(hint in normalized for hint in _SPANISH_HINTS) or "¿" in request:
        return "es"
    return "en"


def render_supplier_totals(
    result: SupplierTotalsAggregation,
    *,
    include_evidence: bool,
    language: str,
) -> str:
    """Render the exact calculation as Markdown without consuming Gemini output tokens."""

    if not result.groups:
        if language == "es":
            return "No hay importes de factura válidos para calcular los totales por proveedor."
        return "No valid invoice amounts are available to calculate supplier totals."

    lines = _render_spanish_summary(result) if language == "es" else _render_english_summary(result)
    if include_evidence:
        lines.extend(_render_evidence(result, language=language))
    if result.exclusions:
        lines.extend(_render_exclusions(result, language=language))
    if result.truncated:
        lines.extend(
            [
                "",
                "**Advertencia:** la colección superó el límite configurado; los resultados "
                "solo cubren los registros devueltos."
                if language == "es"
                else "**Warning:** the collection exceeded the configured limit; results cover "
                "only the returned records.",
            ]
        )
    return "\n".join(lines)


def _render_english_summary(result: SupplierTotalsAggregation) -> list[str]:
    suppliers = {group.supplier for group in result.groups}
    leaders = _leaders_by_currency(result.groups)
    unique_leaders = {
        group.supplier
        for currency_leaders in leaders.values()
        for group in currency_leaders
    }
    if len(suppliers) == 1:
        supplier = next(iter(suppliers))
        conclusion = (
            f"**{_escape_markdown(supplier)}** is the only supplier in the returned collection "
            "and therefore has the highest invoiced total in every normalized currency."
        )
    elif len(unique_leaders) == 1 and all(len(items) == 1 for items in leaders.values()):
        supplier = next(iter(unique_leaders))
        conclusion = (
            f"**{_escape_markdown(supplier)}** has the highest invoiced total in every "
            "normalized currency."
        )
    else:
        conclusion = (
            "There is no single cross-currency winner without applying exchange rates; "
            "the deterministic leaders are reported separately by currency."
        )
    lines = [
        conclusion,
        "",
        f"The calculation includes **{result.included_invoice_count}** valid invoices. Currency "
        "aliases were normalized once by the tool (`$`/`US$` → `USD`, `€` → `EUR`).",
        "",
        "| Supplier | Currency | Invoice count | Exact total |",
        "| :--- | :---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {_escape_table(group.supplier)} | {group.currency} | {group.invoice_count} | "
        f"{group.currency} {_format_money(group.total)} |"
        for group in result.groups
    )
    lines.extend(["", "Totals and counts were calculated in Python with exact decimal arithmetic."])
    return lines


def _render_spanish_summary(result: SupplierTotalsAggregation) -> list[str]:
    suppliers = {group.supplier for group in result.groups}
    leaders = _leaders_by_currency(result.groups)
    unique_leaders = {
        group.supplier
        for currency_leaders in leaders.values()
        for group in currency_leaders
    }
    if len(suppliers) == 1:
        supplier = next(iter(suppliers))
        conclusion = (
            f"**{_escape_markdown(supplier)}** es el único proveedor de la colección devuelta "
            "y, por tanto, tiene el mayor total facturado en cada moneda normalizada."
        )
    elif len(unique_leaders) == 1 and all(len(items) == 1 for items in leaders.values()):
        supplier = next(iter(unique_leaders))
        conclusion = (
            f"**{_escape_markdown(supplier)}** tiene el mayor total facturado en todas las "
            "monedas normalizadas."
        )
    else:
        conclusion = (
            "No existe un único ganador entre monedas sin aplicar tipos de cambio; los líderes "
            "deterministas se muestran por separado para cada moneda."
        )
    lines = [
        conclusion,
        "",
        f"El cálculo incluye **{result.included_invoice_count}** facturas válidas. La tool "
        "normalizó una sola vez los alias de moneda (`$`/`US$` → `USD`, `€` → `EUR`).",
        "",
        "| Proveedor | Moneda | Número de facturas | Total exacto |",
        "| :--- | :---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {_escape_table(group.supplier)} | {group.currency} | {group.invoice_count} | "
        f"{group.currency} {_format_money(group.total)} |"
        for group in result.groups
    )
    lines.extend(
        ["", "Los totales y conteos se calcularon en Python con aritmética decimal exacta."]
    )
    return lines


def _render_evidence(result: SupplierTotalsAggregation, *, language: str) -> list[str]:
    lines = [
        "",
        "## Facturas utilizadas en el cálculo"
        if language == "es"
        else "## Invoices used in the calculation",
    ]
    for group in result.groups:
        lines.extend(
            [
                "",
                f"### {_escape_markdown(group.supplier)} — {group.currency}",
                "",
                "| Número de factura | Fecha | Importe | Fuente |"
                if language == "es"
                else "| Invoice number | Date | Amount | Source |",
                "| :--- | :---: | ---: | :--- |",
            ]
        )
        lines.extend(
            f"| {_escape_table(item.invoice_number)} | {_escape_table(item.invoice_date)} | "
            f"{item.currency} {_format_money(item.total)} | "
            f"{_escape_table(item.source_path)} |"
            for item in group.evidence
        )
        verification = (
            f"**Verificación determinista:** {group.invoice_count} facturas → "
            f"**{group.currency} {_format_money(group.total)}**"
            if language == "es"
            else f"**Deterministic verification:** {group.invoice_count} invoices → "
            f"**{group.currency} {_format_money(group.total)}**"
        )
        lines.extend(["", verification])
    return lines


def _render_exclusions(result: SupplierTotalsAggregation, *, language: str) -> list[str]:
    title = "## Registros excluidos" if language == "es" else "## Excluded records"
    explanation = (
        "Estos registros no se utilizaron porque faltaba un campo financiero obligatorio:"
        if language == "es"
        else "These records were not used because a required financial field was missing:"
    )
    lines = ["", title, "", explanation, ""]
    lines.extend(
        f"- `{_escape_markdown(item.source_path)}`: {_escape_markdown(item.reason)}"
        for item in result.exclusions
    )
    return lines


def _leaders_by_currency(
    groups: Sequence[SupplierCurrencyTotal],
) -> dict[str, tuple[SupplierCurrencyTotal, ...]]:
    by_currency: dict[str, list[SupplierCurrencyTotal]] = {}
    for group in groups:
        by_currency.setdefault(group.currency, []).append(group)
    leaders: dict[str, tuple[SupplierCurrencyTotal, ...]] = {}
    for currency, currency_groups in by_currency.items():
        maximum = max(group.total for group in currency_groups)
        leaders[currency] = tuple(
            group for group in currency_groups if group.total == maximum
        )
    return leaders


def _format_money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")


def _escape_table(value: str) -> str:
    return _escape_markdown(str(value).replace("\n", " ").replace("|", "\\|"))
