"""Generic deterministic analytics over existing structured invoice records.

Author: Miguel Medina Cantos
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

from src.application.invoice_analytics import (
    MONEY_QUANTUM,
    normalize_currency,
    parse_money,
)
from src.embeddings.vector_index import InvoiceRecord


GROUP_FIELDS = {"supplier", "currency", "year", "month", "invoice_date"}
FILTER_FIELDS = GROUP_FIELDS | {"total", "source"}
FILTER_OPERATORS = {"eq", "neq", "in", "contains", "gt", "gte", "lt", "lte"}
OPERATIONS = {"count", "sum", "average", "minimum", "maximum"}
HAVING_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte"}


@dataclass(frozen=True, slots=True)
class AnalysisFilter:
    """One whitelisted predicate applied before grouping."""

    field: str
    operator: str
    value: str


@dataclass(frozen=True, slots=True)
class InvoiceAnalysisSpec:
    """Controlled SQL-like analysis requested by the routing agent."""

    group_by: tuple[str, ...]
    operation: str
    filters: tuple[AnalysisFilter, ...] = ()
    having_operator: str | None = None
    having_value: str | None = None
    sort_by: str = "value"
    sort_direction: str = "desc"
    group_limit: int = 10
    requested_invoice_count: int | None = None
    include_evidence: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisInvoice:
    """Compact normalized invoice used by the in-memory analysis engine."""

    invoice_number: str
    supplier: str
    invoice_date: str
    parsed_date: date | None
    total: Decimal | None
    currency: str
    source_path: str


@dataclass(frozen=True, slots=True)
class AnalysisGroup:
    """One grouped result with its exact supporting records."""

    dimensions: tuple[tuple[str, str], ...]
    value: Decimal
    invoice_count: int
    evidence: tuple[AnalysisInvoice, ...]


@dataclass(frozen=True, slots=True)
class InvoiceAnalysisResult:
    """Selected and fallback groups for one deterministic analysis."""

    spec: InvoiceAnalysisSpec
    collection_invoice_count: int
    filtered_invoice_count: int
    groups_before_having: int
    groups: tuple[AnalysisGroup, ...]
    fallback_groups: tuple[AnalysisGroup, ...]
    invalid_record_count: int
    truncated: bool

    @property
    def condition_matched(self) -> bool:
        return bool(self.groups)


def analyze_invoice_records(
    records: Sequence[InvoiceRecord],
    *,
    spec: InvoiceAnalysisSpec,
    truncated: bool,
) -> InvoiceAnalysisResult:
    """Apply filters, grouping, aggregation, HAVING, sorting, and limits in Python."""

    _validate_spec(spec)
    invoices = tuple(_to_analysis_invoice(record) for record in records)
    filtered = tuple(
        invoice
        for invoice in invoices
        if all(_matches_filter(invoice, item) for item in spec.filters)
    )
    grouped: dict[tuple[str, ...], list[AnalysisInvoice]] = {}
    invalid_record_count = 0
    for invoice in filtered:
        dimensions = _group_values(invoice, spec.group_by)
        if dimensions is None:
            invalid_record_count += 1
            continue
        if spec.operation != "count" and invoice.total is None:
            invalid_record_count += 1
            continue
        grouped.setdefault(dimensions, []).append(invoice)

    all_groups = tuple(
        AnalysisGroup(
            dimensions=tuple(zip(spec.group_by, key, strict=True)),
            value=_calculate_value(items, operation=spec.operation),
            invoice_count=len(items),
            evidence=tuple(sorted(items, key=lambda invoice: invoice.source_path)),
        )
        for key, items in grouped.items()
        if items
    )
    sorted_groups = _sort_groups(all_groups, spec=spec)
    matching = tuple(
        group
        for group in sorted_groups
        if _matches_having(
            group.value,
            operator=spec.having_operator,
            target=spec.having_value,
        )
    )
    selected = matching[: spec.group_limit]
    fallback: tuple[AnalysisGroup, ...] = ()
    if spec.having_operator and not selected and sorted_groups:
        fallback = _closest_groups(sorted_groups, operator=spec.having_operator)
    return InvoiceAnalysisResult(
        spec=spec,
        collection_invoice_count=len(records),
        filtered_invoice_count=len(filtered),
        groups_before_having=len(all_groups),
        groups=selected,
        fallback_groups=fallback,
        invalid_record_count=invalid_record_count,
        truncated=truncated,
    )


def render_invoice_analysis(result: InvoiceAnalysisResult, *, language: str) -> str:
    """Render generic analysis and bounded evidence without another Gemini generation."""

    spanish = language == "es"
    spec = result.spec
    displayed_groups = result.groups or result.fallback_groups
    lines: list[str] = []
    if spec.having_operator and not result.groups:
        if displayed_groups:
            best = _format_metric(displayed_groups[0].value, operation=spec.operation)
            lines.append(
                (
                    f"No existe ningún grupo que cumpla **{_condition_label(spec)}**. "
                    f"El mejor valor disponible es **{best}**."
                )
                if spanish
                else (
                    f"No group satisfies **{_condition_label(spec)}**. "
                    f"The best available value is **{best}**."
                )
            )
        else:
            lines.append(
                "Ninguna factura coincide con los filtros indicados."
                if spanish
                else "No invoices match the requested filters."
            )
    elif result.groups:
        lines.append(
            (
                f"El análisis determinista encontró **{len(result.groups)}** grupo(s) que "
                f"cumplen **{_condition_label(spec)}**."
                if spec.having_operator
                else f"El análisis determinista produjo **{len(result.groups)}** grupo(s)."
            )
            if spanish
            else (
                f"The deterministic analysis found **{len(result.groups)}** group(s) satisfying "
                f"**{_condition_label(spec)}**."
                if spec.having_operator
                else f"The deterministic analysis produced **{len(result.groups)}** group(s)."
            )
        )
    else:
        lines.append(
            "Ninguna factura coincide con la consulta."
            if spanish
            else "No invoices match the analysis request."
        )

    if displayed_groups:
        lines.extend(_render_group_table(displayed_groups, spec=spec, language=language))
    lines.extend(
        [
            "",
            (
                f"Se inspeccionaron {result.collection_invoice_count} facturas y "
                f"{result.filtered_invoice_count} superaron los filtros. "
                "La operación se ejecutó en Python, no en Gemini."
            )
            if spanish
            else (
                f"The tool inspected {result.collection_invoice_count} invoices and "
                f"{result.filtered_invoice_count} passed the filters. "
                "Python performed the operation; Gemini did not calculate it."
            ),
        ]
    )
    if spec.include_evidence and displayed_groups:
        lines.extend(_render_analysis_evidence(displayed_groups, spec=spec, language=language))
    if result.invalid_record_count:
        lines.extend(
            [
                "",
                (
                    f"**Excluidas:** {result.invalid_record_count} facturas no tenían los campos "
                    "necesarios para esta operación."
                )
                if spanish
                else (
                    f"**Excluded:** {result.invalid_record_count} invoices lacked fields required "
                    "by this operation."
                ),
            ]
        )
    if result.truncated:
        lines.extend(
            [
                "",
                "**Advertencia:** el análisis cubre únicamente el límite de registros devuelto."
                if spanish
                else "**Warning:** the analysis covers only the configured record limit.",
            ]
        )
    return "\n".join(lines)


def build_invoice_analysis_payload(result: InvoiceAnalysisResult) -> dict[str, Any]:
    """Return compact precomputed facts that Gemini may explain but never calculate."""

    displayed_groups = result.groups or result.fallback_groups
    return {
        "status": "success" if displayed_groups else "empty",
        "condition_matched": result.condition_matched,
        "operation": result.spec.operation,
        "group_by": list(result.spec.group_by),
        "filters": [
            {"field": item.field, "operator": item.operator, "value": item.value}
            for item in result.spec.filters
        ],
        "having": (
            {
                "operator": result.spec.having_operator,
                "value": result.spec.having_value,
            }
            if result.spec.having_operator
            else None
        ),
        "collection_invoice_count": result.collection_invoice_count,
        "filtered_invoice_count": result.filtered_invoice_count,
        "groups_before_having": result.groups_before_having,
        "returned_group_count": len(displayed_groups),
        "used_closest_groups_fallback": bool(result.fallback_groups and not result.groups),
        "invalid_record_count": result.invalid_record_count,
        "truncated": result.truncated,
        "groups": [
            _serialize_analysis_group(group, spec=result.spec)
            for group in displayed_groups
        ],
        "precomputed_comparisons": _build_precomputed_comparisons(
            displayed_groups,
            spec=result.spec,
        ),
    }


def render_invoice_analysis_evidence(
    result: InvoiceAnalysisResult,
    *,
    language: str,
) -> str:
    """Render only requested evidence so Gemini never has to reproduce invoice rows."""

    if not result.spec.include_evidence:
        return ""
    displayed_groups = result.groups or result.fallback_groups
    if not displayed_groups:
        return ""
    lines = _render_analysis_evidence(
        displayed_groups,
        spec=result.spec,
        language=language,
    )
    if result.invalid_record_count:
        lines.extend(
            [
                "",
                (
                    f"**Excluidas:** {result.invalid_record_count} facturas no tenían los "
                    "campos necesarios para esta operación."
                    if language == "es"
                    else (
                        f"**Excluded:** {result.invalid_record_count} invoices lacked fields "
                        "required by this operation."
                    )
                ),
            ]
        )
    if result.truncated:
        lines.extend(
            [
                "",
                "**Advertencia:** las evidencias cubren únicamente el límite configurado."
                if language == "es"
                else "**Warning:** evidence covers only the configured record limit.",
            ]
        )
    return "\n".join(lines)


def _serialize_analysis_group(
    group: AnalysisGroup,
    *,
    spec: InvoiceAnalysisSpec,
) -> dict[str, Any]:
    statistics = _group_statistics(group, include_money=_money_is_comparable(spec))
    return {
        "dimensions": dict(group.dimensions),
        "selected_metric": {
            "operation": spec.operation,
            "value": _decimal_text(group.value, operation=spec.operation),
        },
        "statistics": statistics,
    }


def _group_statistics(group: AnalysisGroup, *, include_money: bool) -> dict[str, Any]:
    statistics: dict[str, Any] = {"invoice_count": group.invoice_count}
    if not include_money:
        return statistics
    totals = [item.total for item in group.evidence if item.total is not None]
    if not totals:
        return statistics
    total = sum(totals, Decimal("0")).quantize(MONEY_QUANTUM)
    statistics.update(
        {
            "amount_invoice_count": len(totals),
            "sum": _decimal_text(total),
            "average": _decimal_text(
                (total / Decimal(len(totals))).quantize(MONEY_QUANTUM)
            ),
            "minimum": _decimal_text(min(totals).quantize(MONEY_QUANTUM)),
            "maximum": _decimal_text(max(totals).quantize(MONEY_QUANTUM)),
        }
    )
    return statistics


def _money_is_comparable(spec: InvoiceAnalysisSpec) -> bool:
    if "currency" in spec.group_by:
        return True
    return any(
        item.field == "currency" and item.operator == "eq" for item in spec.filters
    )


def _build_precomputed_comparisons(
    groups: Sequence[AnalysisGroup],
    *,
    spec: InvoiceAnalysisSpec,
) -> list[dict[str, Any]]:
    comparison_filter = next(
        (
            item
            for item in spec.filters
            if item.operator == "in" and item.field in spec.group_by
        ),
        None,
    )
    if comparison_filter is None:
        return []
    requested_values = _split_filter_values(comparison_filter.value)
    if len(requested_values) < 2:
        return []

    comparison_field = comparison_filter.field
    keyed_groups: dict[tuple[tuple[str, str], ...], dict[str, AnalysisGroup]] = {}
    for group in groups:
        dimensions = dict(group.dimensions)
        comparison_value = dimensions.get(comparison_field)
        if comparison_value is None:
            continue
        shared_dimensions = tuple(
            (field, value)
            for field, value in group.dimensions
            if field != comparison_field
        )
        keyed_groups.setdefault(shared_dimensions, {})[comparison_value] = group

    comparisons: list[dict[str, Any]] = []
    baseline_value = requested_values[0]
    for compared_value in requested_values[1:]:
        for shared_dimensions, indexed in sorted(keyed_groups.items()):
            baseline = indexed.get(baseline_value)
            compared = indexed.get(compared_value)
            if baseline is None or compared is None:
                continue
            baseline_statistics = _group_statistics(
                baseline,
                include_money=_money_is_comparable(spec),
            )
            compared_statistics = _group_statistics(
                compared,
                include_money=_money_is_comparable(spec),
            )
            comparisons.append(
                {
                    "comparison_field": comparison_field,
                    "baseline": baseline_value,
                    "compared": compared_value,
                    "shared_dimensions": dict(shared_dimensions),
                    "direction": "compared_minus_baseline",
                    "selected_metric": spec.operation,
                    "baseline_value": _decimal_text(
                        baseline.value,
                        operation=spec.operation,
                    ),
                    "compared_value": _decimal_text(
                        compared.value,
                        operation=spec.operation,
                    ),
                    "difference": _decimal_text(
                        compared.value - baseline.value,
                        operation=spec.operation,
                    ),
                    "percentage_change": _percentage_change(
                        baseline.value,
                        compared.value,
                    ),
                    "statistics_differences": _statistics_differences(
                        baseline_statistics,
                        compared_statistics,
                    ),
                }
            )
    return comparisons


def _statistics_differences(
    baseline: dict[str, Any],
    compared: dict[str, Any],
) -> dict[str, Any]:
    differences: dict[str, Any] = {
        "invoice_count": int(compared["invoice_count"]) - int(baseline["invoice_count"])
    }
    for field in ("sum", "average", "minimum", "maximum"):
        if field not in baseline or field not in compared:
            continue
        baseline_value = Decimal(str(baseline[field]))
        compared_value = Decimal(str(compared[field]))
        differences[field] = _decimal_text(compared_value - baseline_value)
        differences[f"{field}_percentage_change"] = _percentage_change(
            baseline_value,
            compared_value,
        )
    return differences


def _percentage_change(baseline: Decimal, compared: Decimal) -> str | None:
    if baseline == 0:
        return None
    percentage = ((compared - baseline) / abs(baseline) * Decimal("100")).quantize(
        MONEY_QUANTUM
    )
    return _decimal_text(percentage)


def _decimal_text(value: Decimal, *, operation: str | None = None) -> str:
    if operation == "count":
        return str(int(value))
    return f"{value.quantize(MONEY_QUANTUM):.2f}"


def _split_filter_values(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_spec(spec: InvoiceAnalysisSpec) -> None:
    invalid_groups = set(spec.group_by) - GROUP_FIELDS
    if invalid_groups:
        raise ValueError(f"Unsupported group fields: {sorted(invalid_groups)}")
    if len(set(spec.group_by)) != len(spec.group_by):
        raise ValueError("group_by fields must be unique.")
    if spec.operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {spec.operation}")
    currency_is_fixed = any(
        item.field == "currency" and item.operator == "eq" for item in spec.filters
    )
    if spec.operation != "count" and "currency" not in spec.group_by and not currency_is_fixed:
        raise ValueError(
            "Monetary operations must group by currency or apply one exact currency filter."
        )
    if spec.having_operator and spec.having_operator not in HAVING_OPERATORS:
        raise ValueError(f"Unsupported HAVING operator: {spec.having_operator}")
    if bool(spec.having_operator) != (spec.having_value is not None):
        raise ValueError("HAVING operator and value must be provided together.")
    if spec.sort_by not in {"value", "invoice_count", "group"}:
        raise ValueError(f"Unsupported sort field: {spec.sort_by}")
    if spec.sort_direction not in {"asc", "desc"}:
        raise ValueError(f"Unsupported sort direction: {spec.sort_direction}")
    if not 1 <= spec.group_limit <= 200:
        raise ValueError("group_limit must be between 1 and 200.")
    if spec.requested_invoice_count is not None and not (
        1 <= spec.requested_invoice_count <= 200
    ):
        raise ValueError("requested_invoice_count must be between 1 and 200.")
    for item in spec.filters:
        if item.field not in FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field: {item.field}")
        if item.operator not in FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {item.operator}")


def _to_analysis_invoice(record: InvoiceRecord) -> AnalysisInvoice:
    invoice = record.invoice
    invoice_date = str(invoice.get("invoice_date") or "").strip()
    try:
        parsed_date = date.fromisoformat(invoice_date)
    except ValueError:
        parsed_date = None
    return AnalysisInvoice(
        invoice_number=str(
            invoice.get("invoice_number") or record.image_identifier or record.document_id
        ),
        supplier=str(invoice.get("supplier_name") or "Unknown supplier").strip(),
        invoice_date=invoice_date or "Unknown",
        parsed_date=parsed_date,
        total=parse_money(invoice.get("total")),
        currency=normalize_currency(invoice.get("currency")) or "UNKNOWN",
        source_path=record.source_path,
    )


def _group_values(invoice: AnalysisInvoice, fields: Sequence[str]) -> tuple[str, ...] | None:
    values: list[str] = []
    for field in fields:
        value = _field_value(invoice, field)
        if value is None:
            return None
        values.append(str(value))
    return tuple(values)


def _field_value(invoice: AnalysisInvoice, field: str):
    if field == "supplier":
        return invoice.supplier
    if field == "currency":
        return invoice.currency
    if field == "invoice_number":
        return invoice.invoice_number
    if field == "invoice_date":
        return invoice.invoice_date if invoice.parsed_date else None
    if field == "year":
        return invoice.parsed_date.year if invoice.parsed_date else None
    if field == "month":
        return f"{invoice.parsed_date.month:02d}" if invoice.parsed_date else None
    if field == "total":
        return invoice.total
    if field == "source":
        return invoice.source_path
    raise ValueError(f"Unsupported invoice field: {field}")


def _matches_filter(invoice: AnalysisInvoice, item: AnalysisFilter) -> bool:
    actual = _field_value(invoice, item.field)
    if actual is None:
        return False
    if item.operator == "in":
        expected_values = _split_filter_values(item.value)
        if not expected_values:
            raise ValueError("The 'in' filter requires at least one comma-separated value.")
        return any(
            _matches_filter(
                invoice,
                AnalysisFilter(field=item.field, operator="eq", value=expected),
            )
            for expected in expected_values
        )
    expected = item.value.strip()
    if item.field == "currency":
        expected = normalize_currency(expected) or expected.upper()
    if item.field == "total":
        expected_decimal = parse_money(expected)
        if expected_decimal is None:
            raise ValueError(f"Invalid monetary filter value: {item.value!r}")
        return _compare(Decimal(actual), expected_decimal, item.operator)
    if item.field in {"year", "month"}:
        try:
            return _compare(Decimal(str(actual)), Decimal(expected), item.operator)
        except InvalidOperation as error:
            raise ValueError(f"Invalid numeric filter value: {item.value!r}") from error
    actual_text = str(actual)
    if item.operator == "contains":
        return expected.casefold() in actual_text.casefold()
    if item.operator in {"gt", "gte", "lt", "lte"} and item.field == "invoice_date":
        try:
            return _compare(
                date.fromisoformat(actual_text),
                date.fromisoformat(expected),
                item.operator,
            )
        except ValueError as error:
            raise ValueError(f"Date filters must use YYYY-MM-DD: {item.value!r}") from error
    return _compare(actual_text.casefold(), expected.casefold(), item.operator)


def _calculate_value(items: Sequence[AnalysisInvoice], *, operation: str) -> Decimal:
    if operation == "count":
        return Decimal(len(items))
    totals = [invoice.total for invoice in items if invoice.total is not None]
    if not totals:
        return Decimal("0")
    if operation == "sum":
        return sum(totals, Decimal("0")).quantize(MONEY_QUANTUM)
    if operation == "average":
        return (sum(totals, Decimal("0")) / Decimal(len(totals))).quantize(MONEY_QUANTUM)
    if operation == "minimum":
        return min(totals).quantize(MONEY_QUANTUM)
    if operation == "maximum":
        return max(totals).quantize(MONEY_QUANTUM)
    raise ValueError(f"Unsupported operation: {operation}")


def _matches_having(value: Decimal, *, operator: str | None, target: str | None) -> bool:
    if operator is None:
        return True
    try:
        expected = Decimal(str(target))
    except InvalidOperation as error:
        raise ValueError(f"Invalid HAVING value: {target!r}") from error
    return _compare(value, expected, operator)


def _compare(actual, expected, operator: str) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "neq":
        return actual != expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    raise ValueError(f"Unsupported comparison operator: {operator}")


def _sort_groups(
    groups: Sequence[AnalysisGroup],
    *,
    spec: InvoiceAnalysisSpec,
) -> tuple[AnalysisGroup, ...]:
    if spec.sort_by == "value":
        key = lambda group: (group.value, group.dimensions)
    elif spec.sort_by == "invoice_count":
        key = lambda group: (group.invoice_count, group.dimensions)
    else:
        key = lambda group: group.dimensions
    return tuple(sorted(groups, key=key, reverse=spec.sort_direction == "desc"))


def _closest_groups(
    groups: Sequence[AnalysisGroup],
    *,
    operator: str,
) -> tuple[AnalysisGroup, ...]:
    if operator in {"lt", "lte"}:
        closest_value = min(group.value for group in groups)
    else:
        closest_value = max(group.value for group in groups)
    return tuple(group for group in groups if group.value == closest_value)


def _render_group_table(
    groups: Sequence[AnalysisGroup],
    *,
    spec: InvoiceAnalysisSpec,
    language: str,
) -> list[str]:
    dimension_headers = [field.replace("_", " ").title() for field in spec.group_by]
    metric = _operation_label(spec.operation, language=language)
    headers = [*dimension_headers, metric, "Invoices" if language != "es" else "Facturas"]
    alignments = [":---" for _ in dimension_headers] + ["---:", "---:"]
    lines = ["", f"| {' | '.join(headers)} |", f"| {' | '.join(alignments)} |"]
    for group in groups:
        dimensions = [_escape_table(value) for _, value in group.dimensions]
        values = [
            *dimensions,
            _format_metric(group.value, operation=spec.operation),
            str(group.invoice_count),
        ]
        lines.append(f"| {' | '.join(values)} |")
    return lines


def _render_analysis_evidence(
    groups: Sequence[AnalysisGroup],
    *,
    spec: InvoiceAnalysisSpec,
    language: str,
) -> list[str]:
    spanish = language == "es"
    lines = ["", "## Facturas utilizadas" if spanish else "## Supporting invoices"]
    for group in groups:
        title = " — ".join(value for _, value in group.dimensions)
        evidence_title = title or ("Facturas filtradas" if spanish else "Filtered invoices")
        lines.extend(
            [
                "",
                f"### {_escape_table(evidence_title)}",
                "",
                "| Factura | Fecha | Proveedor | Importe | Moneda | Fuente |"
                if spanish
                else "| Invoice | Date | Supplier | Amount | Currency | Source |",
                "| :--- | :---: | :--- | ---: | :---: | :--- |",
            ]
        )
        requested = spec.requested_invoice_count or len(group.evidence)
        selected = group.evidence[:requested]
        for invoice in selected:
            amount = f"{invoice.total:,.2f}" if invoice.total is not None else "—"
            lines.append(
                f"| {_escape_table(invoice.invoice_number)} | "
                f"{_escape_table(invoice.invoice_date)} | {_escape_table(invoice.supplier)} | "
                f"{amount} | {invoice.currency} | {_escape_table(invoice.source_path)} |"
            )
        if len(selected) < group.invoice_count:
            lines.extend(
                [
                    "",
                    (
                        f"Se muestran {len(selected)} de {group.invoice_count} facturas del grupo."
                        if spanish
                        else (
                            f"Showing {len(selected)} of {group.invoice_count} invoices "
                            "in this group."
                        )
                    ),
                ]
            )
        elif spec.requested_invoice_count and group.invoice_count < spec.requested_invoice_count:
            lines.extend(
                [
                    "",
                    (
                        f"Se solicitaron {spec.requested_invoice_count} facturas, pero este grupo "
                        f"solo contiene {group.invoice_count}."
                        if spanish
                        else (
                            f"The user requested {spec.requested_invoice_count} invoices, but "
                            f"this group contains only {group.invoice_count}."
                        )
                    ),
                ]
            )
    return lines


def _condition_label(spec: InvoiceAnalysisSpec) -> str:
    if not spec.having_operator:
        return spec.operation
    symbols = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    return f"{spec.operation} {symbols[spec.having_operator]} {spec.having_value}"


def _operation_label(operation: str, *, language: str) -> str:
    if language == "es":
        return {
            "count": "Conteo",
            "sum": "Suma",
            "average": "Media",
            "minimum": "Mínimo",
            "maximum": "Máximo",
        }[operation]
    return operation.title()


def _format_metric(value: Decimal, *, operation: str) -> str:
    return str(int(value)) if operation == "count" else f"{value:,.2f}"


def _escape_table(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
