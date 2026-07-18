# Chapter 5 — Options Strategies

```bash
uv run python chapter_5_options_strategies/src/run_chapter5_analysis.py
uv run python chapter_5_options_strategies/src/run_chapter5_analysis.py --offline
```

첫 명령은 공식 저자 ZIP의 SHA-256을 검증하고 MATLAB/MAT 멤버를 분리한다. 두 번째는 체크섬으로 고정된 로컬 자산만 사용한다. 실행 시 보고서, strict JSON metrics, 그림, 실행 완료 노트북과 품질 감사를 다시 만든다.

- Exact replay: `compareVolWithVXX.m`, `XIV_SPY_rollreturn_lagged.m`
- Independent experiment: Black–Scholes/Greeks/IV, synthetic gamma scalping
- Output-only: missing-version or licensed option data paths, each with `compared:false + reason`
