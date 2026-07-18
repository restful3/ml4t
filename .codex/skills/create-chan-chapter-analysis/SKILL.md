---
name: create-chan-chapter-analysis
description: Build, repair, or review reproducible Python chapter experiments for Ernest P. Chan book projects, including run_chapterX_analysis.py, executed Jupyter notebooks, official data and source provenance, Markdown reports, figures, and book-result comparisons. Use when Codex is asked to implement or improve a chapter analysis, experiment notebook, backtest report, or execution environment for Chan's Quantitative Trading, Algorithmic Trading, or Machine Trading directories.
---

# Create Chan Chapter Analysis

Build an auditable chapter study that a reader can execute from a locked environment and compare with the book. Treat prose, formulas, source code, data, numerical output, figures, and limitations as one deliverable.

## Establish Scope

1. Locate the actual book and chapter directories; do not assume `source/` or `archive/` or hard-code an old absolute path.
2. Read applicable `AGENTS.md` files and inspect `git status` before editing. Preserve unrelated user changes.
3. Inspect the English and Korean chapter material, original MATLAB or other companion source, existing Python and notebook files, reports, and data.
4. Derive the chapter's concepts and experiments from those materials. Do not reuse a chapter map from a different edition without verification.
5. Classify every example as exactly reproducible, approximately comparable, or output-only. State the reason for anything that cannot be reproduced.

Create a chapter coverage matrix before implementation. Include every major prose section, box, example, and formula, with one of these states: executed, numerically compared, conceptual explanation, historical context, unavailable data, or out of scope with reason. Do not call a notebook a full chapter report when it silently covers only the code-backed examples.

## Benchmark Existing Chapter Work

Before creating or substantially revising a chapter, inspect the strongest existing chapter artifacts in the other Chan book directories. Compare at least:

- the most pedagogically detailed notebook;
- the broadest multi-example notebook;
- the strongest executable backtest or numerical report;
- the target chapter's previous version, if any.

Run `scripts/audit_chapter_artifacts.py` on the target and candidates. Treat its scores as a repeatable baseline, not a substitute for judgment. Deep-read the leading candidates and manually check numerical claims, result-to-conclusion consistency, truncated experiments, chart legibility, and actual isolated execution.

Set the target so the new artifact is not lower than the best relevant benchmark overall, and is not lower in execution, reproducibility, or rigor. Match the benchmark's useful teaching depth and experiment coverage without copying its errors.

## Record Source and Data Provenance

Prefer the publisher or author's official companion archive when one exists. Record:

- source page and direct archive URL;
- archive and input-file SHA-256 values;
- original filenames and relevant archive members;
- symbols, date range, shape, missing values, and nonpositive values;
- transformations, exclusions, and their reasons;
- any license or redistribution constraint visible in the source.

Never silently replace unavailable licensed data with a convenient public dataset. If a substitution is necessary, label the result as a conceptual reproduction rather than a numerical replication.

When a chapter has no dedicated companion archive, say so explicitly. Reuse an official file from another chapter or edition only after recording its checksum and proving the exact overlap and value identity relevant to the claim. Limit comparisons to that proven scope; do not imply that an extended period, different symbol pair, or prose-only example is book output.

Preserve extracted companion source byte-for-byte. Do not normalize CRLF, encodings, or trailing whitespace merely to satisfy repository formatters. When exact official files trigger text-only checks, use a narrowly scoped `.gitattributes` rule such as `original_matlab/*.m -diff -text`, keep the manifest hashes authoritative, and continue running normal whitespace checks on authored files.

Keep downloaded raw data out of version control when appropriate. Provide a checksum-backed manifest or downloader, and make offline validation possible after the first download.

## Create a Reproducible Environment

Use the book project environment, normally `pyproject.toml` plus `uv.lock`. Use relative paths based on `__file__`; do not depend on a stale virtual-environment activation path. Pin random seeds and name the random-number generator when simulations are involved.

Make generated notebooks byte-reproducible when practical: assign stable cell IDs and disable execution-timing metadata before writing the executed notebook. Verify this by rebuilding twice and comparing hashes; matching numerical outputs alone is not enough.

Implement current library idioms, including positional indexing with `iloc` and row assembly with `concat` rather than removed pandas APIs. Keep charts portable by using English chart labels, or explicitly configure a Korean font.

Do not assume a pandas datetime array has nanosecond storage. Convert explicitly with `to_numpy(dtype="datetime64[ns]").astype(np.int64)` before integer time-window arithmetic, and test an exact window boundary. Validate every auxiliary input array even when the original script does not use it: record shape mismatches, non-finite values, and invalid or crossed quotes instead of silently cleaning them.

## Implement the Analysis

Separate reusable calculation and figure-construction functions from orchestration. Use `run_chapterX_analysis.py` as the repeatable entry point when that convention fits the project.

For each experiment:

1. State the question and the book formula.
2. Map the formula to the exact function or code block.
3. Show the input data and assumptions.
4. Compute the result rather than copying a printed number.
5. Compare Python output with the book using an explicit tolerance.
6. Explain differences caused by solver tolerances, random generators, APIs, precision, or data revisions.

Generate conclusions from computed metrics where possible. Audit every qualitative claim against the displayed number. Never describe a nonsignificant test as significant, a shortened run as full-period evidence, or an in-sample illustration as out-of-sample performance.

Preserve the semantics of the original state machine. If a source emits a continuous position, probability, or target weight, do not reduce it to `{-1, 0, 1}` without algebraically deriving the equivalent thresholds. Test hold bands, equality boundaries, and transitions independently of aggregate P&L.

Distinguish estimator families rather than treating close-looking implementations as equivalent. For example, Gaussian maximum-likelihood BIC, conditional OLS, and Yule-Walker AR selection can choose different lags. Replay the source's fixed-order result separately from an approximate modern estimator and label both clearly.

Trust executed branches over comments, variable names, and printed prose when they disagree. Preserve a source-faithful path, add a corrected interpretation when useful, and encode irreconcilable comparisons as `compared: false` with a reason rather than manufacturing agreement.

Save generated reports and figures under the chapter's established report directory. Make the command idempotent and expose an offline mode when downloads are involved.

## Build an Executed Teaching Notebook

Write the narrative in Korean unless the user requests another language. The notebook must run cleanly from the locked environment and contain, in order:

1. scope, learning questions, chapter coverage matrix, and a clear distinction between a numerical illustration and a trading backtest;
2. formulas and a formula-to-code map;
3. source URL, hashes, environment versions, and lock-file hash;
4. data preview, shape, period, symbols, missingness, positivity checks, and useful descriptive statistics;
5. normalized prices or another appropriate input diagnostic;
6. each book experiment, its result, and book-versus-Python tables;
7. figures calculated and embedded inline during notebook execution;
8. interpretations connected to trading decisions;
9. look-ahead, survivorship, selection, adjustment, transaction-cost, and out-of-sample limitations as applicable;
10. executable assertions or a final verification table.

Cover prose-only chapter concepts at teaching depth even when they have no companion code. Label vendor lists, platform descriptions, and market conventions from older books as historical context rather than current recommendations. Add an executable diagnostic when it materially illuminates a conceptual claim, but do not invent a backtest merely to increase experiment count.

Do not make the notebook display only pre-generated PNG files. It may import reusable calculation and figure functions, but it must construct and embed the figures during execution so its outputs remain self-contained.

Treat LaTeX inside generated Python strings as code, not decorative prose. Prefer raw strings or explicitly escaped backslashes. Reject tabs and other control characters in generated Markdown, and scan math spans for bare command words such as `phi`, `theta`, `sum`, `epsilon`, `Phi`, and `cdots`; these usually mean a backslash was consumed by the host language. Exercise this guard on both the report and every notebook Markdown cell.

## Write the Chapter Report

Include these sections, adapting names to the chapter:

1. overview, problem, and key equations;
2. data table and data-selection rationale;
3. analysis 1: purpose, method, result, and interpretation;
4. analysis 2: purpose, method, result, and interpretation;
5. backtest method and metrics only when a true time-ordered backtest exists; otherwise label it a portfolio or numerical illustration;
6. conclusion, recommendations, numerical comparison, biases, costs, and applicability limits;
7. reproducibility and provenance details.

Use result tables and concise interpretation rather than a stream of console output.

Represent an inactive strategy segment as `null`, not as a fabricated zero-return metric, and include a machine-readable reason such as `train_segment: no positions ...` for every such strategy.

State the performance clock. If the source removes flat or inactive days before annualization, report both active-day and calendar-day APR/Sharpe and treat calendar time as the economic default unless capital redeployment is explicitly modeled. Define exactly which return index pays turnover costs relative to lagged positions; test that convention for look-ahead safety.

For simulations or operational examples, use a named deterministic generator such as PCG64 and ensure decisions use only information available through the prior observation. Label a deterministic lifecycle demonstration as conceptual rather than a backtest. Treat historical broker, vendor, regulatory, tax, and legal statements as dated context unless current primary sources are verified.

## Verify Before Claiming Completion

Run the relevant equivalents of:

```bash
uv lock --check
uv sync --locked
uv run python path/to/run_chapterX_analysis.py --offline
```

If a downloader exists, also run its online or validation-only path and confirm a second run is idempotent. Execute every notebook cell with the locked kernel. Confirm:

- every code cell has an execution count;
- there are zero error outputs;
- numerical checks and constraints pass;
- book comparisons use documented tolerances;
- reports and figures are regenerated by the entry point;
- charts are visually legible and not clipped;
- notebook cells do not read pre-generated figures merely to display them;
- generated report and notebook Markdown contain no tabs, control characters, or bare LaTeX command words inside math spans;
- two consecutive full builds produce identical hashes for deterministic reports, metrics, figures, and notebooks;
- `git diff --check` reports no whitespace errors.

Do not say that code or a notebook was executed unless it was actually run. Report the command and the decisive output or metric.

When the user requires chapter-by-chapter peer consensus, keep the workflow sequential: implement and verify one chapter, obtain an explicit approval verdict, apply requested fixes, and only then begin the next chapter. Preserve the verdict and its evidence in the work log.

## Iterate Against the Benchmark

After each material revision:

1. regenerate the script outputs, report, figures, and executed notebook;
2. rerun the audit script against the same benchmark set;
3. inspect any category below the target and identify the concrete missing behavior;
4. update this skill when the gap reflects a reusable process failure rather than a chapter-specific omission;
5. regenerate and reassess until the target is met or exceeded;
6. preserve the final before-and-after evidence.

Do not raise the score by adding empty headings, copied prose, decorative charts, or checks that do not cover the relevant claim. A higher score is acceptable only when the underlying artifact is materially better.

## Quality Gate

Reject the deliverable as incomplete if any of these are missing:

- official-source provenance or an explicit substitution warning;
- deterministic, locked execution;
- an executed notebook with inline-generated figures;
- formula-to-code mapping and data diagnostics;
- book-versus-Python comparison;
- explanation of material numerical differences;
- trading interpretation and relevant bias or cost warnings;
- automated checks plus a clean full execution.
- a complete chapter coverage matrix and no unsupported result-to-conclusion claims;
- benchmark evidence showing the result is at least as strong as the best relevant existing artifact.
