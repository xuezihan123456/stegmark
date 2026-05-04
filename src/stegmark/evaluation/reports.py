from __future__ import annotations

import csv
import html
import json
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from stegmark.evaluation.types import BenchmarkComparisonResult, BenchmarkResult


def render_benchmark_report(result: BenchmarkResult, report_format: str) -> str:
    if report_format == "json":
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if report_format == "csv":
        return benchmark_csv_text(result)
    if report_format == "html":
        return benchmark_html(result)
    raise ValueError(f"unsupported report format: {report_format}")


def render_benchmark_comparison_report(result: BenchmarkComparisonResult, report_format: str) -> str:
    if report_format == "json":
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if report_format == "csv":
        return comparison_csv_text(result)
    if report_format == "html":
        return comparison_html(result)
    raise ValueError(f"unsupported report format: {report_format}")


def benchmark_csv_text(result: BenchmarkResult) -> str:
    buffer = StringIO()
    fieldnames = [
        "attack",
        "message_match",
        "extracted_message",
        "found",
        "bit_accuracy",
        "psnr",
        "elapsed",
        "error",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in result.to_rows():
        writer.writerow(row)
    return buffer.getvalue()


def benchmark_html(result: BenchmarkResult) -> str:
    headers = [
        "attack",
        "message_match",
        "extracted_message",
        "found",
        "bit_accuracy",
        "psnr",
        "elapsed",
        "error",
    ]
    parts = [
        "<html><head><meta charset='utf-8'><title>StegMark Benchmark</title></head><body>",
        f"<h1>StegMark Benchmark ({html.escape(result.engine)})</h1>",
        "<table>",
        "<thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead>",
        "<tbody>",
    ]
    for row in result.to_rows():
        parts.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
            + "</tr>"
        )
    parts.extend(["</tbody>", "</table>", "</body></html>"])
    return "".join(parts)


def benchmark_table(result: BenchmarkResult) -> Table:
    table = Table(title="StegMark Benchmark")
    table.add_column("Attack")
    table.add_column("Match")
    table.add_column("Found")
    table.add_column("Bit Accuracy", justify="right")
    table.add_column("PSNR", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("Error")
    for attack_result in result.attack_results.values():
        table.add_row(
            attack_result.attack,
            "yes" if attack_result.message_match else "no",
            "yes" if attack_result.found else "no",
            f"{attack_result.bit_accuracy:.3f}",
            f"{attack_result.psnr:.2f}",
            f"{attack_result.elapsed:.3f}s",
            attack_result.error or "",
        )
    return table


def benchmark_terminal_text(result: BenchmarkResult) -> str:
    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False)
    console.print(benchmark_table(result))
    summary = result.summary
    console.print(
        "\nSummary: "
        f"attacks={summary.attack_count} "
        f"matches={summary.message_match_count} "
        f"found={summary.found_count} "
        f"avg_bit_accuracy={summary.average_bit_accuracy:.3f} "
        f"avg_psnr={summary.average_psnr:.2f} "
        f"worst_attack={summary.worst_attack or 'n/a'}"
    )
    if result.output_report is not None:
        console.print(f"report={result.output_report}")
    return buffer.getvalue()


def comparison_csv_text(result: BenchmarkComparisonResult) -> str:
    buffer = StringIO()
    fieldnames = [
        "engine",
        "available",
        "average_bit_accuracy",
        "average_psnr",
        "worst_attack",
        "error",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for engine_result in result.engines.values():
        if engine_result.result is not None:
            summary = engine_result.result.summary
            writer.writerow(
                {
                    "engine": engine_result.engine,
                    "available": engine_result.available,
                    "average_bit_accuracy": summary.average_bit_accuracy,
                    "average_psnr": summary.average_psnr,
                    "worst_attack": summary.worst_attack,
                    "error": "",
                }
            )
        else:
            writer.writerow(
                {
                    "engine": engine_result.engine,
                    "available": engine_result.available,
                    "average_bit_accuracy": "",
                    "average_psnr": "",
                    "worst_attack": "",
                    "error": engine_result.error or "",
                }
            )
    return buffer.getvalue()


def comparison_html(result: BenchmarkComparisonResult) -> str:
    parts = [
        "<html><head><meta charset='utf-8'><title>StegMark Benchmark Compare</title></head><body>",
        "<h1>StegMark Benchmark Compare</h1>",
        "<table>",
        "<thead><tr><th>engine</th><th>available</th><th>average_bit_accuracy</th><th>average_psnr</th><th>worst_attack</th><th>error</th></tr></thead>",
        "<tbody>",
    ]
    for engine_result in result.engines.values():
        if engine_result.result is not None:
            summary = engine_result.result.summary
            parts.append(
                "<tr>"
                f"<td>{html.escape(engine_result.engine)}</td>"
                "<td>True</td>"
                f"<td>{summary.average_bit_accuracy:.3f}</td>"
                f"<td>{summary.average_psnr:.2f}</td>"
                f"<td>{html.escape(summary.worst_attack or '')}</td>"
                "<td></td>"
                "</tr>"
            )
        else:
            parts.append(
                "<tr>"
                f"<td>{html.escape(engine_result.engine)}</td>"
                "<td>False</td>"
                "<td></td><td></td><td></td>"
                f"<td>{html.escape(engine_result.error or '')}</td>"
                "</tr>"
            )
    parts.extend(["</tbody>", "</table>", "</body></html>"])
    return "".join(parts)


def comparison_table(result: BenchmarkComparisonResult) -> Table:
    table = Table(title="StegMark Benchmark Compare")
    table.add_column("Engine")
    table.add_column("Available")
    table.add_column("Avg Bit Accuracy", justify="right")
    table.add_column("Avg PSNR", justify="right")
    table.add_column("Best Attack", justify="left")
    table.add_column("Error")
    for engine_result in result.engines.values():
        if engine_result.result is not None:
            summary = engine_result.result.summary
            table.add_row(
                engine_result.engine,
                "yes",
                f"{summary.average_bit_accuracy:.3f}",
                f"{summary.average_psnr:.2f}",
                summary.worst_attack or "",
                "",
            )
        else:
            table.add_row(
                engine_result.engine,
                "no",
                "-",
                "-",
                "-",
                engine_result.error or "",
            )
    return table


def comparison_terminal_text(result: BenchmarkComparisonResult) -> str:
    buffer = StringIO()
    console = Console(file=buffer, width=120, force_terminal=False)
    console.print(comparison_table(result))
    summary = result.summary
    console.print(
        "\nSummary: "
        f"total={summary.total_engines} "
        f"available={summary.available_engines} "
        f"unavailable={summary.unavailable_engines} "
        f"best_engine={summary.best_engine_by_average_bit_accuracy or 'n/a'}"
    )
    if result.output_report is not None:
        console.print(f"report={result.output_report}")
    return buffer.getvalue()
