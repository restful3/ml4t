#!/usr/bin/env python3
"""Score executed Chan chapter notebooks with a transparent static rubric."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


CATEGORY_MAXIMUMS = {
    "execution": 20,
    "reproducibility": 20,
    "rigor": 25,
    "pedagogy": 20,
    "trading_context": 15,
}

def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def find_project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").exists() or (
            candidate / "requirements.txt"
        ).exists():
            return candidate
    return path.parent


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def project_math_is_clean(project_root: Path, text: str) -> bool:
    common_path = project_root / "book3_common.py"
    if not common_path.exists():
        return not any(
            ord(character) < 32 and character != "\n" for character in text
        )
    spec = importlib.util.spec_from_file_location("chan_book_common", common_path)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.assert_clean_markdown_math(text, "audited Markdown")
    except AssertionError:
        return False
    return True


def audit_notebook(notebook_path: Path) -> dict[str, Any]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown = "\n".join(cell_source(cell) for cell in markdown_cells)
    code = "\n".join(cell_source(cell) for cell in code_cells)
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]

    script_candidates = sorted(notebook_path.parent.glob("run_chapter*_analysis.py"))
    report_candidates = sorted(
        path
        for path in (notebook_path.parent / "reports").glob("*.md")
        if "quality_benchmark" not in path.name
    )
    report = "\n".join(
        path.read_text(encoding="utf-8") for path in report_candidates
    )
    all_text = "\n".join((markdown, code, report))
    project_root = find_project_root(notebook_path)

    executed_count = sum(
        cell.get("execution_count") is not None for cell in code_cells
    )
    error_count = sum(output.get("output_type") == "error" for output in outputs)
    embedded_png_count = sum(
        "image/png" in output.get("data", {}) for output in outputs
    )
    pre_generated_figure_read = has(
        r"Image\s*\(\s*filename|imread\s*\(|reports[/\\]figures|open\([^\n]+\.png",
        code,
    )
    korean_character_count = len(re.findall(r"[가-힣]", markdown))
    heading_count = sum(
        line.lstrip().startswith("#") for line in markdown.splitlines()
    )

    features = {
        "all_code_executed": bool(code_cells) and executed_count == len(code_cells),
        "zero_error_outputs": error_count == 0,
        "two_or_more_inline_figures": embedded_png_count >= 2,
        "no_pre_generated_figure_reads": not pre_generated_figure_read,
        "analysis_entrypoint": bool(script_candidates),
        "pyproject": (project_root / "pyproject.toml").exists(),
        "lock_file": (project_root / "uv.lock").exists(),
        "environment_spec": any(
            (project_root / name).exists()
            for name in ("pyproject.toml", "requirements.txt", "environment.yml")
        ),
        "checksum_provenance": has(r"SHA-?256|checksum|체크섬", all_text),
        "source_url": has(r"https?://", all_text),
        "environment_versions": has(r"uv\.lock.*SHA|environment.*version|패키지.*버전", all_text),
        "automated_checks": has(r"assert|verification|자동\s*검증|검증\s*통과", all_text),
        "clean_latex_math": project_math_is_clean(
            project_root, "\n".join((markdown, report))
        ),
        "data_diagnostics": has(
            r"missing|결측|nonpositive|비양수|describe\s*\(|기술통계|데이터\s*구조",
            all_text,
        ),
        "formula_to_code": has(r"수식.*코드|formula.to.code|구현\s*함수", all_text),
        "book_comparison": has(r"책.*비교|book.*compar|MATLAB.*비교|원본.*비교", all_text),
        "limitations": has(
            r"한계|주의사항|limitation|look-ahead|survivorship|생존\s*편향", all_text
        ),
        "deterministic_randomness": has(r"PCG64|random.?seed|rng\(|seed\s*[=:]", all_text),
        "reproduction_classification": has(
            r"정확.*재현|근사.*재현|개념.*재현|output-only|재현\s*상태", all_text
        ),
        "report": bool(report_candidates),
        "korean_narrative": korean_character_count >= 500,
        "latex_equations": has(r"\$\$|\\\(|\\\[", markdown),
        "substantial_narrative": len(markdown) >= 4_000,
        "moderate_narrative": len(markdown) >= 2_500,
        "structured_headings": heading_count >= 12,
        "learning_questions": has(r"질문|문제\s*정의|learning question", markdown),
        "coverage_matrix": has(r"coverage|커버리지|구현\s*범위|재현\s*상태", markdown),
        "conclusion": has(r"결론|conclusion", markdown),
        "bias_warnings": has(
            r"look-ahead|survivorship|selection bias|선택.*편향|생존.*편향", all_text
        ),
        "costs": has(r"거래.?비용|transaction.?cost|slippage|슬리피지", all_text),
        "out_of_sample": has(r"out.of.sample|표본\s*외|walk.forward|롤링\s*창", all_text),
        "risk_metrics": has(r"Sharpe|샤프|drawdown|낙폭|Calmar|MAR", all_text),
        "backtest_scope": has(r"백테스트가\s*아님|backtest|백테스트", all_text),
    }

    scores = {
        "execution": (
            8 * features["all_code_executed"]
            + 6 * features["zero_error_outputs"]
            + 4 * features["two_or_more_inline_figures"]
            + 2 * features["no_pre_generated_figure_reads"]
        ),
        "reproducibility": (
            3 * features["analysis_entrypoint"]
            + 2 * features["pyproject"]
            + 6 * features["lock_file"]
            + 2 * features["environment_spec"]
            + 4 * features["checksum_provenance"]
            + 1 * features["source_url"]
            + 2 * features["environment_versions"]
        ),
        "rigor": (
            5 * features["automated_checks"] * features["clean_latex_math"]
            + 4 * features["data_diagnostics"]
            + 3 * features["formula_to_code"]
            + 4 * features["book_comparison"]
            + 3 * features["limitations"]
            + 2 * features["deterministic_randomness"]
            + 2 * features["reproduction_classification"]
            + 2 * features["report"]
        ),
        "pedagogy": (
            3 * features["korean_narrative"]
            + 3 * features["latex_equations"]
            + (4 if features["substantial_narrative"] else 2 * features["moderate_narrative"])
            + 3 * features["structured_headings"]
            + 2 * features["learning_questions"]
            + 3 * features["coverage_matrix"]
            + 2 * features["conclusion"]
        ),
        "trading_context": (
            3 * features["bias_warnings"]
            + 3 * features["costs"]
            + 3 * features["out_of_sample"]
            + 2 * features["risk_metrics"]
            + 2 * features["backtest_scope"]
            + 2 * features["two_or_more_inline_figures"]
        ),
    }

    return {
        "notebook": str(notebook_path),
        "project_root": str(project_root),
        "counts": {
            "cells": len(cells),
            "markdown_cells": len(markdown_cells),
            "code_cells": len(code_cells),
            "executed_code_cells": executed_count,
            "error_outputs": error_count,
            "embedded_png_outputs": embedded_png_count,
            "markdown_characters": len(markdown),
            "korean_characters": korean_character_count,
            "headings": heading_count,
        },
        "features": features,
        "scores": scores,
        "total": sum(scores.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [audit_notebook(path.resolve()) for path in args.notebooks]
    if args.json:
        print(json.dumps(audits, ensure_ascii=False, indent=2))
        return

    headers = ["notebook", *CATEGORY_MAXIMUMS, "total"]
    print("| " + " | ".join(headers) + " |")
    print("|---" * len(headers) + "|")
    for audit in audits:
        values = [
            Path(audit["notebook"]).name,
            *(f"{audit['scores'][name]}/{maximum}" for name, maximum in CATEGORY_MAXIMUMS.items()),
            f"{audit['total']}/100",
        ]
        print("| " + " | ".join(values) + " |")


if __name__ == "__main__":
    main()
