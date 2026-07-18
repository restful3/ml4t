# Chapter 1 실험 환경

*Machine Trading* 1장의 시스템·데이터·성과지표·포트폴리오 주제를 한국어로
연결하고 Box 1.1~1.3을 실행하는 Python 프로젝트다. 공식 코드가 있는 Box
1.1·1.2는 책과 수치 비교하고, Box 1.3은 공식 ETF 데이터에 분석해를 적용한다.
포트폴리오 결과는 전략 성과 백테스트가 아니라 정적 수치 실험이다.

## 실행

책 프로젝트 디렉터리에서 실행한다.

```bash
uv sync --locked
uv run python chapter_1_the_basics_of_algorithmic_trading/src/run_chapter1_analysis.py --offline
```

최초 다운로드가 필요하면 `--offline`을 빼고 실행한다. 공식 ZIP과 추출 파일의
SHA-256을 확인하며, 원시 데이터는 Git에서 제외된 `data/raw/` 아래에 둔다.
다운로드 상태만 검증하려면 다음을 실행한다.

```bash
uv run python chapter_1_the_basics_of_algorithmic_trading/src/run_chapter1_analysis.py --download-only
```

## 산출물

- `src/chapter1_full_report.ipynb`: 실행 결과가 내장된 한국어 학습 노트북
- `src/reports/chapter1_report.md`: 공식 데이터·수식·책 비교·한계 리포트
- `src/reports/metrics.json`: 해시와 자동 검증을 포함한 기계 판독 결과
- `src/reports/figures/`: 재생성되는 도표
- `original_matlab/`: 체크섬으로 검증한 공식 원본 소스와 그림

노트북은 CLI와 같은 계산 함수를 쓰되, 도표는 실행 중 직접 만들어 셀 출력에
내장한다. 따라서 미리 생성된 PNG가 없어도 노트북 내용 자체는 재실행 가능하다.
장 전체 coverage 표는 코드가 없는 주제, 역사적 제품 정보, 실행한 실험을
구분하며, 수치 결론은 `metrics.json`의 자동 검증과 연결한다.

전체 저장소에는 `.codex/skills/create-chan-chapter-analysis/` 품질 감사 도구가
함께 버전관리된다. 전체 실행은 이 도구가 있을 때 `quality_benchmark.md`도
갱신한다. 책 디렉터리만 독립 복사해 감사 도구가 없으면 경고 후 벤치마크
갱신만 건너뛰며, 핵심 리포트·metrics·노트북·그림 생성은 실패하지 않는다.
`--skip-notebook`은 노트북과 품질 벤치마크를 모두 갱신하지 않아 서로 다른
실행 시점의 계측과 수치를 한 문서에 섞지 않는다.
