from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import click

import stegmark
from stegmark.config import default_config_path, load_config, reset_config_file, save_config_value
from stegmark.evaluation.benchmark import list_attacks
from stegmark.types import BatchItemResult, ExtractResult, InfoResult, VerifyResult

T = TypeVar("T")


@click.group()
@click.version_option(version=stegmark.__version__)
def main() -> None:
    """Embed, extract, verify, and inspect invisible image watermarks."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--message", "-m", help="Watermark text to embed.")
@click.option("--bits", "-b", help="Hex payload to embed instead of text.")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output image path.",
)
@click.option("--overwrite", "-y", is_flag=True, help="Overwrite existing outputs.")
@click.option("--format", "-f", "output_format", help="Force output format: png, jpeg, or webp.")
@click.option("--quality", "-Q", default=95, show_default=True, type=int, help="Output quality for JPEG/WebP.")
@click.option("--compare", is_flag=True, help="Generate a compare report for single-file embed output.")
@click.option("--engine", "-e", default=None, help="Engine backend to use.")
@click.option(
    "--strength",
    "-s",
    default=None,
    type=float,
    help="Embedding strength.",
)
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories when input is a directory.")
@click.option("--workers", "-w", default=None, type=int, help="Number of worker threads for batch directory input.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def embed(
    input_path: Path,
    message: str | None,
    bits: str | None,
    output: Path | None,
    overwrite: bool,
    output_format: str | None,
    quality: int,
    compare: bool,
    engine: str | None,
    strength: float | None,
    recursive: bool,
    workers: int | None,
    as_json: bool,
) -> None:
    config = load_config()
    if (message is None) == (bits is None):
        raise click.ClickException("Provide exactly one of --message or --bits.")
    if input_path.is_dir():
        if compare:
            raise click.ClickException("Directory embed does not support --compare.")
        batch_result = stegmark.embed_directory(
            input_path,
            message=message,
            bits=bits,
            engine=engine or config.engine,
            strength=_coalesce_option(strength, config.strength),
            recursive=recursive,
            output_dir=output or input_path,
            workers=_coalesce_option(workers, config.workers),
            overwrite=overwrite,
            output_format=output_format,
            quality=quality,
        )
        payload = {
            "total": batch_result.total,
            "succeeded": batch_result.succeeded,
            "failed": batch_result.failed,
            "items": [
                {
                    "input": str(item.input_path),
                    "output": str(item.output_path) if item.output_path else None,
                    "success": item.success,
                    "error": item.error,
                }
                for item in batch_result.items
            ],
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False))
            return
        click.echo(
            f"Processed {batch_result.total} file(s): {batch_result.succeeded} succeeded, {batch_result.failed} failed"
        )
        return
    embed_result = stegmark.embed(
        input_path,
        message,
        bits=bits,
        output=output,
        overwrite=overwrite,
        output_format=output_format,
        quality=quality,
        compare=compare,
        engine=engine or config.engine,
        strength=_coalesce_option(strength, config.strength),
    )
    payload = {
        "output": str(embed_result.output_path),
        "engine": embed_result.engine,
        "message": embed_result.message,
        "psnr": embed_result.psnr,
        "compare_report": str(embed_result.compare_report) if embed_result.compare_report else None,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False))
        return
    click.echo(f"Saved watermark image to {embed_result.output_path}")
    if compare and embed_result.compare_report is not None:
        click.echo(f"compare={embed_result.compare_report}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--engine", "-e", default=None, help="Engine backend to use.")
@click.option("--mode", type=click.Choice(["text", "bits"]), default="text", show_default=True)
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories when input is a directory.")
@click.option("--workers", "-w", default=None, type=int, help="Number of worker threads for batch directory input.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def extract(
    input_path: Path,
    engine: str | None,
    mode: str,
    recursive: bool,
    workers: int | None,
    as_json: bool,
) -> None:
    config = load_config()
    if input_path.is_dir():
        batch_result = stegmark.extract_directory(
            input_path,
            engine=engine or config.engine,
            recursive=recursive,
            workers=_coalesce_option(workers, config.workers),
        )
        payload = {
            "total": batch_result.total,
            "succeeded": batch_result.succeeded,
            "failed": batch_result.failed,
            "items": [_extract_item_payload(item, mode=mode) for item in batch_result.items],
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False))
            return
        click.echo(
            f"Processed {batch_result.total} file(s): {batch_result.succeeded} succeeded, {batch_result.failed} failed"
        )
        return
    extract_result = stegmark.extract(input_path, engine=engine or config.engine)
    payload = {
        "found": extract_result.found,
        "engine": extract_result.engine,
        "message": extract_result.message,
        "payload_hex": extract_result.payload_hex,
        "confidence": extract_result.confidence,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False))
        return
    if not extract_result.found:
        raise click.ClickException("Watermark not found.")
    if mode == "bits":
        if extract_result.payload_hex is None:
            raise click.ClickException("Bits mode is not available for this engine output.")
        click.echo(extract_result.payload_hex)
        return
    click.echo(extract_result.message or "")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--message", "-m", required=True, help="Expected watermark text.")
@click.option("--engine", "-e", default=None, help="Engine backend to use.")
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories when input is a directory.")
@click.option("--workers", "-w", default=None, type=int, help="Number of worker threads for batch directory input.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def verify(
    input_path: Path,
    message: str,
    engine: str | None,
    recursive: bool,
    workers: int | None,
    as_json: bool,
) -> None:
    config = load_config()
    if input_path.is_dir():
        batch_result = stegmark.verify_directory(
            input_path,
            message,
            engine=engine or config.engine,
            recursive=recursive,
            workers=_coalesce_option(workers, config.workers),
        )
        payload = {
            "total": batch_result.total,
            "succeeded": batch_result.succeeded,
            "failed": batch_result.failed,
            "items": [_verify_item_payload(item) for item in batch_result.items],
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False))
            _exit(0 if batch_result.failed == 0 else 1)
        click.echo(
            f"Processed {batch_result.total} file(s): {batch_result.succeeded} succeeded, {batch_result.failed} failed"
        )
        _exit(0 if batch_result.failed == 0 else 1)
    result = stegmark.verify(input_path, message, engine=engine or config.engine)
    payload = {
        "matched": result.matched,
        "engine": result.engine,
        "expected": result.expected,
        "actual": result.actual,
        "confidence": result.confidence,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False))
    else:
        click.echo("matched" if result.matched else "not matched")
    _exit(0 if result.matched else 1)


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.option("--engine", "-e", default=None, help="Engine backend to use.")
@click.option("--recursive", "-r", is_flag=True, help="Recurse into subdirectories when input is a directory.")
@click.option("--workers", "-w", default=None, type=int, help="Number of worker threads for batch directory input.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def info(input_path: Path, engine: str | None, recursive: bool, workers: int | None, as_json: bool) -> None:
    config = load_config()
    if input_path.is_dir():
        batch_result = stegmark.info_directory(
            input_path,
            engine=engine or config.engine,
            recursive=recursive,
            workers=_coalesce_option(workers, config.workers),
        )
        payload = {
            "total": batch_result.total,
            "succeeded": batch_result.succeeded,
            "failed": batch_result.failed,
            "items": [_info_item_payload(item) for item in batch_result.items],
        }
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False))
            return
        click.echo(
            f"Processed {batch_result.total} file(s): {batch_result.succeeded} succeeded, {batch_result.failed} failed"
        )
        return
    result = stegmark.info(input_path, engine=engine or config.engine)
    payload = {
        "found": result.found,
        "engine": result.engine,
        "width": result.width,
        "height": result.height,
        "format": result.format,
        "confidence": result.confidence,
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False))
        return
    click.echo(
        f"found={result.found} engine={result.engine} size={result.width}x{result.height} format={result.format}"
    )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--message", "-m", required=True, help="Expected watermark text to benchmark.")
@click.option("--engine", "-e", default=None, help="Engine backend to benchmark.")
@click.option(
    "--engines",
    default="",
    help="Comma-separated engine names to compare. When set, compare mode is used instead of --engine.",
)
@click.option(
    "--attacks",
    default="",
    help="Comma-separated attack names. Leave empty to use the default benchmark subset.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for benchmark.json output.",
)
@click.option(
    "--report-format",
    type=click.Choice(["json", "csv", "html"]),
    default="json",
    show_default=True,
    help="Format for the saved benchmark report.",
)
@click.option(
    "--min-average-bit-accuracy",
    type=float,
    default=None,
    help="Fail if the average bit accuracy falls below this threshold.",
)
@click.option(
    "--min-average-psnr",
    type=float,
    default=None,
    help="Fail if the average PSNR falls below this threshold.",
)
@click.option(
    "--require-all-matches",
    is_flag=True,
    help="Fail unless every attack result reproduces the expected message.",
)
@click.option(
    "--require-all-found",
    is_flag=True,
    help="Fail unless every attack result still contains a detectable watermark.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def benchmark(
    input_path: Path,
    message: str,
    engine: str | None,
    engines: str,
    attacks: str,
    output_dir: Path | None,
    report_format: str,
    min_average_bit_accuracy: float | None,
    min_average_psnr: float | None,
    require_all_matches: bool,
    require_all_found: bool,
    as_json: bool,
) -> None:
    config = load_config()
    attack_names = [item.strip() for item in attacks.split(",") if item.strip()] or None
    engine_names = [item.strip() for item in engines.split(",") if item.strip()]
    if engine_names:
        comparison_result = stegmark.benchmark_compare(
            input_path,
            message,
            engines=engine_names,
            attacks=attack_names,
            output_dir=output_dir,
            report_format=report_format,
        )
        gate = comparison_result.evaluate_gate(
            min_average_bit_accuracy=min_average_bit_accuracy,
            min_average_psnr=min_average_psnr,
            require_all_matches=require_all_matches,
            require_all_found=require_all_found,
        )
        payload = comparison_result.to_dict()
        if (
            min_average_bit_accuracy is not None
            or min_average_psnr is not None
            or require_all_matches
            or require_all_found
        ):
            payload["gate"] = {"passed": gate.passed, "failures": list(gate.failures)}
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False))
            _exit(0 if gate.passed else 1)
        click.echo(comparison_result.to_terminal_text())
        if not gate.passed:
            click.echo(f"Gate failed: {'; '.join(gate.failures)}")
            _exit(1)
    else:
        benchmark_result = stegmark.benchmark(
            input_path,
            message,
            engine=engine or config.engine or "native",
            attacks=attack_names,
            output_dir=output_dir,
            report_format=report_format,
        )
        gate = benchmark_result.evaluate_gate(
            min_average_bit_accuracy=min_average_bit_accuracy,
            min_average_psnr=min_average_psnr,
            require_all_matches=require_all_matches,
            require_all_found=require_all_found,
        )
        payload = benchmark_result.to_dict()
        if (
            min_average_bit_accuracy is not None
            or min_average_psnr is not None
            or require_all_matches
            or require_all_found
        ):
            payload["gate"] = {"passed": gate.passed, "failures": list(gate.failures)}
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False))
            _exit(0 if gate.passed else 1)
        click.echo(benchmark_result.to_terminal_text())
        if not gate.passed:
            click.echo(f"Gate failed: {'; '.join(gate.failures)}")
            _exit(1)
    click.echo(f"available_attacks={','.join(list_attacks())}")


@main.group(name="config")
def config_group() -> None:
    """Show, set, and reset persistent CLI defaults."""


@config_group.command(name="show")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def config_show(as_json: bool) -> None:
    config = load_config()
    payload = config.to_dict()
    payload["config_path"] = str(default_config_path())
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False))
        return
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    config_path = default_config_path()
    save_config_value(config_path, key, value)
    click.echo(f"Saved {key} to {config_path}")


@config_group.command(name="reset")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def config_reset(yes: bool) -> None:
    config_path = default_config_path()
    if not yes and not click.confirm(f"Delete config file at {config_path}?", default=False):
        _exit(1)
    reset_config_file(config_path)
    click.echo(f"Reset config at {config_path}")


def _coalesce_option(value: T | None, fallback: T) -> T:
    return fallback if value is None else value


def _exit(code: int) -> None:
    click.get_current_context().exit(code)


def _extract_item_payload(item: BatchItemResult, *, mode: str) -> dict[str, object]:
    message = item.result.message if isinstance(item.result, ExtractResult) else None
    payload_hex = item.result.payload_hex if isinstance(item.result, ExtractResult) else None
    return {
        "input": str(item.input_path),
        "success": item.success,
        "message": message if mode == "text" else None,
        "payload_hex": payload_hex if mode == "bits" else None,
        "error": item.error,
    }


def _verify_item_payload(item: BatchItemResult) -> dict[str, object]:
    matched = item.result.matched if isinstance(item.result, VerifyResult) else False
    return {
        "input": str(item.input_path),
        "success": item.success,
        "matched": matched,
        "error": item.error,
    }


def _info_item_payload(item: BatchItemResult) -> dict[str, object]:
    found = item.result.found if isinstance(item.result, InfoResult) else False
    return {
        "input": str(item.input_path),
        "success": item.success,
        "found": found,
        "error": item.error,
    }
