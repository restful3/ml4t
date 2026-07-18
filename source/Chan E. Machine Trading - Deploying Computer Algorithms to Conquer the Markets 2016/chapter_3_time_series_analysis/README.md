# Chapter 3 Time Series Analysis 실험 환경

공식 Book 3 ZIP의 MATLAB/FIG 소스 28개를 `original_matlab/`에 보존하고, 5개
연구 데이터는 Git에서 제외된 `data/raw/book3/chapter_3/`에 둔다. AR/ARMA 책
수치 재현, 원본 AR(10) look-ahead 진단과 train-only 수정, 공식 close 기반
VAR/VEC·Kalman 적응, EWA–EWC published-parameter replay를 실행한다.

```bash
uv run python chapter_3_time_series_analysis/src/run_chapter3_analysis.py
uv run python chapter_3_time_series_analysis/src/run_chapter3_analysis.py --offline
uv run pytest tests/test_chapter3_numerics.py
```

`--skip-notebook`은 계산·검증·리포트·그림만 갱신한다. 저장소 `.codex` 감사 도구가
없으면 핵심 실행은 유지하고 품질 벤치마크만 경고와 함께 생략한다.
