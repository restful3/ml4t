# Chapter 2 Factor Models 실험 환경

공식 Book 3 ZIP의 MATLAB 소스 전체를 `original_matlab/`에 보존하고 데이터는
Git에서 제외된 `data/raw/book3/chapter_2/`에 둔다. 예제 2.1·2.2·2.3은 책 수치와
비교하고, 예제 2.5는 21일 재학습 적응으로 명확히 구분한다.

```bash
uv run python chapter_2_factor_models/src/run_chapter2_analysis.py
uv run python chapter_2_factor_models/src/run_chapter2_analysis.py --offline
```

`--skip-notebook`은 계산·리포트·그림만 갱신한다. 저장소 `.codex` 감사 도구가
없으면 핵심 실행은 유지하고 품질 벤치마크만 경고와 함께 생략한다.
