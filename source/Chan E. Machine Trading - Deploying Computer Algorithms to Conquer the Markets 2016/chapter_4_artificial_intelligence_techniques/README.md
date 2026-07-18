# Chapter 4 Python experiments

Run from the Machine Trading project directory:

```bash
uv run python chapter_4_artificial_intelligence_techniques/src/run_chapter4_analysis.py
uv run python chapter_4_artificial_intelligence_techniques/src/run_chapter4_analysis.py --offline
```

The first command verifies the official author ZIP and materializes source/data. The offline
command revalidates every SHA-256 without network access, regenerates metrics, five figures,
the Korean report, and the executed notebook. Raw `.mat` and `.xls` inputs stay ignored;
the tracked `SOURCE_MANIFEST.json` pins their hashes.

Source-faithful numerical replays preserve and disclose the MATLAB one-row target-boundary
leakage; corrected linear/stepwise and all scikit-learn adaptations exclude that row. Extreme
tree rules and published HMM parameters are replayed separately. Non-comparable MATLAB outputs
are output-only references, including an ambiguous comment duplicated across five scripts.
Cross-sectional stock examples are output-only because the
official archive does not include the author's private `fundamentalData` input.
