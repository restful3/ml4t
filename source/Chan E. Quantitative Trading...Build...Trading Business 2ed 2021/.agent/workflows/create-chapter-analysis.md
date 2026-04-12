---
description: Create run_chapterX_analysis.py script for Chan's Quantitative Trading (2nd Ed.) book chapters
---

# 챕터 분석 스크립트 생성 가이드

이 workflow는 Ernest Chan의 "Quantitative Trading: How to Build Your Own Algorithmic Trading Business" (2nd Edition, 2021) 책의 각 챕터에 대한 종합 분석 스크립트와 마크다운 리포트를 생성하는 방법을 안내합니다.

> **참고**: 이 workflow는 "Algorithmic Trading" (2013) 프로젝트의 동일 워크플로우를 기반으로 하되, QT 책의 구조적 차이(듀얼 트랙, 공유 데이터 폴더, XLS 포맷 등)를 반영합니다.

---

## 듀얼 트랙 시스템

QT 책은 AT 책과 달리 **코드가 없는 개념 챕터** 가 존재합니다. 따라서 두 가지 트랙으로 나누어 처리합니다.

| 트랙 | 대상 챕터 | 스크립트 | 클래스 |
|------|-----------|----------|--------|
| **Track 1: 개념 분석** | Ch1, Ch2, Ch4, Ch5 | `run_chapterX_concept_review.py` | `ChapterXConceptReviewer` |
| **Track 2: 계산 분석** | Ch3, Ch6, Ch7 | `run_chapterX_analysis.py` | `ChapterXAnalyzer` |

---

## 작업 전 준비사항

1. **챕터 한글 문서 읽기**: `chapter_X_.../XX_chapter_X_..._ko.md` 파일을 읽어 핵심 개념과 문제 정의 파악
2. **설명판 문서 확인** (있는 경우): `_ko_explained.md` 파일이 Ch1, Ch2, Ch4, Ch5, Ch6에 존재
3. **기존 소스 코드 분석**: `chapter_X_.../src/` 폴더의 Python 파일들 분석 (Track 2만)
4. **공유 데이터 확인**: `../../data/` 폴더의 XLS/TXT 파일 목록 확인
5. **Chapter 3 참조**: Track 2의 경우 Chapter 3을 템플릿으로 활용 (가장 풍부한 예제)

---

## 프로젝트 데이터 구조

### 공유 데이터 (`data/`)

AT 책과 달리 데이터가 챕터별이 아닌 **프로젝트 루트의 `data/` 폴더에 중앙 집중** 되어 있습니다.

| 파일명 | 포맷 | 내용 | 사용 챕터 |
|--------|------|------|-----------|
| `GLD.xls` | XLS | SPDR Gold Shares ETF | Ch3, Ch7 |
| `GDX.xls` | XLS | VanEck Gold Miners ETF | Ch3, Ch7 |
| `IGE.xls` | XLS | iShares Global Energy ETF | Ch3 |
| `SPY.xls` | XLS | SPDR S&P 500 ETF | Ch3 |
| `KO.xls` | XLS | Coca-Cola 주식 | Ch7 |
| `PEP.xls` | XLS | PepsiCo 주식 | Ch7 |
| `OIH.xls` | XLS | VanEck Oil Services ETF | Ch6 |
| `RKH.xls` | XLS | VanEck Regional Banks ETF | Ch6 |
| `RTH.xls` | XLS | VanEck Retail ETF | Ch6 |
| `SPX_20071123.txt` | TXT (탭) | S&P 500 일별 가격 (2007.11.23 기준) | Ch7 |
| `SPX_op_20071123.txt` | TXT (탭) | S&P 500 시가 데이터 | Ch3 |
| `IJR_20080114.txt` | TXT (탭) | iShares Small-Cap 구성종목 (2008.01.14) | Ch7 |
| `IJR_20080131.txt` | TXT (탭) | iShares Small-Cap 구성종목 (2008.01.31) | Ch7 |

### 공유 유틸리티 (`src/`)

| 파일명 | 기능 |
|--------|------|
| `calculateMaxDD.py` | 최대 낙폭(MDD) 및 낙폭 지속기간 계산 |
| `RetrieveYahooFinancialData.py` | Yahoo Finance 데이터 다운로드 |
| `calculateMaxDD_UnitTest.py` | MDD 함수 단위 테스트 |

### 데이터 경로 처리

기존 예제들은 `pd.read_excel("miscFiles/X.xls")` 경로를 사용합니다. 분석 스크립트에서는 다음과 같이 통일합니다:

```python
from pathlib import Path

# 프로젝트 루트 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UTIL_DIR = PROJECT_ROOT / "src"

# 데이터 로드
import sys
sys.path.insert(0, str(UTIL_DIR))
from calculateMaxDD import calculateMaxDD

df = pd.read_excel(DATA_DIR / "GLD.xls")
```

---

## Track 1: 개념 분석 (Concept Review)

### 대상 챕터

- Chapter 1: The Whats, Whos, and Whys of Quantitative Trading
- Chapter 2: Fishing for Ideas
- Chapter 4: Setting Up Your Business
- Chapter 5: Execution Systems

### 스크립트 생성 프롬프트

```
[Chapter X] 개념 리뷰 스크립트 생성

## 목표
`chapter_X_[챕터명]/src/run_chapterX_concept_review.py` 스크립트를 생성해주세요.

## 참조 파일
1. 한글 문서: `chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko.md`
2. 설명판 문서: `chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko_explained.md` (있는 경우)

## 스크립트 요구사항

### 1. 구조
- `ChapterXConceptReviewer` 클래스 생성
- `extract_key_concepts()`: 핵심 개념 추출 및 정리
- `build_decision_framework()`: 의사결정 프레임워크/체크리스트 구성
- `generate_report()`: 마크다운 리포트 생성
- `run()`: 전체 오케스트레이션

### 2. 리포트 형식

1. **개요 및 핵심 질문**
   - 이 챕터에서 답하려는 핵심 질문들
   - 챕터의 위치 (전체 책 구조에서의 역할)

2. **핵심 개념 정리**
   - 주요 개념을 표로 정리 (개념, 설명, 트레이딩 함의)
   - 핵심 용어 정의

3. **의사결정 프레임워크**
   - 챕터에서 제시하는 판단 기준/체크리스트
   - 비교표 (예: 리테일 vs 프롭, 반자동 vs 완전자동)

4. **실무 체크리스트**
   - 실행 가능한 액션 아이템

5. **AT 책과의 연결점**
   - "Algorithmic Trading" (2013)의 관련 챕터와의 교차 참조

6. **결론 및 권고사항**
   - 핵심 요약

### 3. 출력
- 리포트: `reports/chapterX_concept_report.md`
- 콘솔: 진행 상황 출력 (이모지와 구분선 사용)
```

---

## Track 2: 계산 분석 (Computational Analysis)

### 대상 챕터

- Chapter 3: Backtesting (6개 예제)
- Chapter 6: Money and Risk Management (1개 예제)
- Chapter 7: Special Topics in Quantitative Trading (6개 예제)

### 스크립트 생성 프롬프트

```
[Chapter X] 분석 스크립트 생성

## 목표
`chapter_X_[챕터명]/src/run_chapterX_analysis.py` 스크립트를 생성해주세요.

## 참조 파일
1. 한글 문서: `chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko.md`
2. 설명판 문서: `_ko_explained.md` (있는 경우)
3. 기존 소스: `chapter_X_[챕터명]/src/Example*.py`
4. 공유 데이터: `../../data/` (XLS, TXT 파일)
5. 유틸리티: `../../src/calculateMaxDD.py`

## 스크립트 요구사항

### 1. 구조
- `ChapterXAnalyzer` 클래스 생성
- `load_data()`: `../../data/`에서 데이터 로드
- `analyze_[핵심개념1]()`: 첫 번째 핵심 분석
- `analyze_[핵심개념2]()`: 두 번째 핵심 분석
- `backtest_strategy()`: 전략 백테스트 (해당시)
- `generate_report()`: 마크다운 리포트 생성
- `run()`: 전체 분석 오케스트레이션

### 2. 리포트 형식

1. **개요 및 문제 정의**
   - 이 챕터에서 해결하려는 핵심 질문
   - 수학적 개념과 수식 ($LaTeX$ 형식)

2. **사용 데이터**
   - 데이터 파일명, 내용, 기간, 용도를 테이블로 정리
   - 데이터 선정 이유

3. **분석 1: [핵심개념1]**
   - 분석 목적
   - 방법론 설명 (수식 포함)
   - 결과 테이블
   - 해석 및 결론

4. **분석 2: [핵심개념2]**
   - (위와 동일한 형식)

5. **전략 백테스트**
   - 전략 원리 (Python 코드 블록)
   - 성과 지표 (APR, 샤프 비율, MDD 등)
   - 차트 (figures/ 폴더에 저장)

6. **결론 및 권고사항**
   - 핵심 발견 요약 테이블
   - 트레이딩 권고사항
   - 주의사항 (bias, 거래비용 등)

### 3. 출력
- 리포트: `reports/chapterX_report.md`
- 차트: `reports/figures/*.png`
- 콘솔: 진행 상황 출력 (이모지와 구분선 사용)

## 기술적 주의사항
- pandas 2.0+ 호환: `df.fillna(method='ffill')` → `df.ffill()` 사용
- XLS 읽기: `pd.read_excel(path)` (xlrd<2.0 필요)
- TXT 읽기: `pd.read_table(path)` (탭 구분)
- 데이터 경로: `Path(__file__).parent.parent.parent / "data"`
- 유틸리티 import: `sys.path.insert(0, str(PROJECT_ROOT / "src"))`
- 코드 주석은 한글로 작성
```

---

## 챕터별 핵심 분석 내용 가이드

### Chapter 1: 퀀트 트레이딩의 무엇, 누가, 왜 (Track 1 -- 개념)

- **핵심 개념**: 퀀트 트레이딩 정의, 확장성(scalability), 시간 투자, 지적 재산권
- **의사결정 프레임워크**: 개인 적합성 평가 (자본, 시간, 기술, 성향)
- **실무 체크리스트**: 퀀트 트레이더로서의 준비도 자가 진단
- **AT 책 연결**: AT 책 전체의 서론적 역할

### Chapter 2: 아이디어 찾기 (Track 1 -- 개념)

- **핵심 개념**: 전략 소싱 (학술논문, 블로그, 포럼), 전략 평가 기준 (Sharpe > 1, 파라미터 최소화, 짧은 보유기간)
- **기존 자산**: `src/Table-2_1.md` (트레이딩 아이디어 소스), `src/Table-2_2.md` (자본 규모별 전략 선택)
- **의사결정 프레임워크**: Table 2.1/2.2를 활용한 전략 선택 매트릭스
- **AT 책 연결**: AT Ch2 (평균 회귀 기초)에서 구체적 전략 사례 참조

### Chapter 3: 백테스팅 (Track 2 -- 계산)

이 챕터가 **가장 풍부한 예제** 를 보유하고 있으며, 6개 Python 예제를 모두 통합합니다.

- **분석 항목**:
  1. `analyze_data_retrieval()`: Example 3.1 -- Yahoo Finance 데이터 다운로드 검증
  2. `analyze_sharpe_ratio()`: Example 3.4 -- IGE 롱온리 vs IGE-SPY 시장중립 샤프 비율 비교
  3. `analyze_max_drawdown()`: Example 3.5 -- `calculateMaxDD()` 활용 MDD 계산
  4. `analyze_pair_trading()`: Example 3.6 -- GLD-GDX 페어 트레이딩 (훈련/테스트 분할, z-score 진입/청산)
  5. `analyze_cross_sectional_mean_reversion()`: Example 3.7 -- SPX 횡단면 평균회귀 (거래비용 포함/미포함)
  6. `analyze_open_price_strategy()`: Example 3.8 -- 시가 기반 동일 전략
- **주요 데이터**: `IGE.xls`, `SPY.xls`, `GLD.xls`, `GDX.xls`, `SPX_20071123.txt`, `SPX_op_20071123.txt`
- **핵심 지표**: 샤프 비율 (훈련/테스트), MaxDD, MaxDD Duration, 거래비용 영향
- **차트**: 누적 수익률, 스프레드 z-score, GLD-GDX 산포도, 낙폭 곡선
- **주의**: Example 3.2, 3.3은 MATLAB/Excel 전용으로 Python 소스 없음

### Chapter 4: 사업체 설정 (Track 1 -- 개념)

- **핵심 개념**: 리테일 vs 프로프라이어터리 트레이딩, LLC/S-Corp 설립, 브로커 선택 기준
- **의사결정 프레임워크**: 리테일 vs 프롭 비교표, 브로커 선택 체크리스트 (수수료, 다크풀, API, 실행속도)
- **실무 체크리스트**: 트레이딩 사업 설립 단계별 가이드
- **AT 책 연결**: AT Ch5 (실행)에서 실제 구현 세부사항 참조

### Chapter 5: 실행 시스템 (Track 1 -- 개념)

- **핵심 개념**: ATS 아키텍처, 반자동 vs 완전자동, 주문 관리, 슬리피지, 실행 알고리즘
- **의사결정 프레임워크**: ATS 아키텍처 선택 (Excel DDE vs Python API vs QuantConnect/Blueshift)
- **실무 체크리스트**: 실행 시스템 구축 단계, 거래비용 최소화 전략
- **AT 책 연결**: AT 책 전반의 전략 실행 패턴

### Chapter 6: 자금 및 리스크 관리 (Track 2 -- 계산)

- **분석 항목**:
  1. `analyze_kelly_optimal_allocation()`: Example 6.3 -- OIH/RKH/RTH 3자산 포트폴리오 Kelly 최적 배분
     - 초과수익률 벡터 $M$, 공분산 행렬 $C$ 계산
     - 최적 레버리지 $F^* = C^{-1}M$
     - 최적 성장률 $g = r + S^2/2$
     - 포트폴리오 샤프 비율 $S = \sqrt{F^T C F}$
  2. `analyze_kelly_intuition()`: 기하 랜덤워크 퍼즐 ($g = m - s^2/2$) 시뮬레이션
  3. `analyze_half_kelly()`: Half-Kelly 배팅의 강건성 분석
  4. `analyze_leverage_constraints()`: Regulation T 제약 (야간 2x, 장중 4x)
- **주요 데이터**: `OIH.xls`, `RKH.xls`, `RTH.xls`
- **핵심 수식**: $F^* = C^{-1}M$, $f_i = m_i / s_i^2$, $g = r + S^2/2$
- **차트**: 최적 레버리지 곡선, 성장률 vs 레버리지, 낙폭 민감도

### Chapter 7: 특수 주제 (Track 2 -- 계산)

이 챕터는 6개 예제가 **다양한 주제** 를 다루므로 주제별로 그룹핑합니다.

- **분석 항목**:
  1. `analyze_cointegration_gld_gdx()`: Example 7.2 -- GLD-GDX Engle-Granger 공적분 검정, 헤지 비율
  2. `analyze_correlation_vs_cointegration()`: Example 7.3 -- KO-PEP 높은 상관관계지만 공적분 없음 (중요한 구분)
  3. `analyze_pca_factor_model()`: Example 7.4 -- IJR 구성종목 PCA 팩터모델, 롱/숏 포트폴리오 구성 (lookback=252, numFactors=5, topN=50)
  4. `analyze_half_life()`: Example 7.5 -- GLD-GDX 스프레드 반감기 (Ornstein-Uhlenbeck)
  5. `analyze_january_effect()`: Example 7.6 -- IJR 구성종목 1월 효과 백테스트
  6. `analyze_seasonal_momentum()`: Example 7.7 -- SPX 구성종목 연간 계절성 트렌딩 전략
- **주요 데이터**: `GLD.xls`, `GDX.xls`, `KO.xls`, `PEP.xls`, `IJR_20080114.txt`, `IJR_20080131.txt`, `SPX_20071123.txt`
- **차트**: 공적분 스프레드, PCA 분산 설명력, 반감기 회귀, 계절성 성과 캘린더
- **주의**: Example 7.1 (CPO with PredictNow.ai)은 외부 API 필요로 제외

---

## 환경 설정

QT 프로젝트에는 AT 책과 달리 `.venv`와 `requirements.txt`가 아직 없습니다. 먼저 설정이 필요합니다.

### Step 1: requirements.txt 생성

프로젝트 루트에 생성:

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
statsmodels>=0.14
scipy>=1.10
scikit-learn>=1.3
xlrd>=1.2,<2.0
openpyxl>=3.1
```

> **중요**: `xlrd>=1.2,<2.0`은 레거시 `.xls` 파일 읽기에 필수입니다. `xlrd` 2.0+는 `.xls` 포맷 지원을 중단했습니다.

### Step 2: 가상환경 생성

```bash
cd "Chan E. Quantitative Trading...Build...Trading Business 2ed 2021/"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: 데이터 접근 검증

```bash
python3 -c "import pandas as pd; df = pd.read_excel('data/GLD.xls'); print(f'GLD: {len(df)} rows, columns: {list(df.columns)}')"
```

---

## 실행 단계

### Step 1: 챕터 문서 분석

```bash
# 한글 문서 확인
cat chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko.md

# 설명판 문서 확인 (있는 경우)
cat chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko_explained.md

# 기존 소스 코드 확인 (Track 2만)
ls -la chapter_X_[챕터명]/src/
```

### Step 2: 트랙 결정 및 프롬프트 사용

- Ch1, Ch2, Ch4, Ch5 → Track 1 (개념 분석) 프롬프트 사용
- Ch3, Ch6, Ch7 → Track 2 (계산 분석) 프롬프트 사용

### Step 3: 스크립트 실행

```bash
cd chapter_X_[챕터명]/src
source ../../.venv/bin/activate
python run_chapterX_analysis.py       # Track 2
# 또는
python run_chapterX_concept_review.py # Track 1
```

### Step 4: 결과 확인

```bash
# 리포트 확인
cat reports/chapterX_report.md        # Track 2
cat reports/chapterX_concept_report.md # Track 1

# 차트 확인 (Track 2만)
ls reports/figures/
```

---

## 체크리스트

스크립트 생성 후 다음 항목을 확인하세요:

### 공통
- [ ] 한글 문서의 핵심 개념이 리포트에 반영되었는가?
- [ ] 스크립트가 오류 없이 실행되는가?
- [ ] 콘솔 출력이 진행 상황을 명확히 보여주는가?

### Track 2 전용
- [ ] 수학 공식이 LaTeX 형식으로 올바르게 표현되었는가?
- [ ] 데이터 설명과 선정 이유가 포함되었는가?
- [ ] 각 분석 결과에 해석과 트레이딩 함의가 있는가?
- [ ] 백테스트 결과에 주의사항(bias, 거래비용 등)이 명시되었는가?
- [ ] 차트가 `reports/figures/`에 저장되는가?
- [ ] pandas 2.0+ 호환성이 유지되는가? (`df.ffill()` 등)
- [ ] 데이터 경로가 `../../data/`로 올바르게 해결되는가?
- [ ] 유틸리티 import가 `../../src/`에서 정상 작동하는가?

### Track 1 전용
- [ ] 의사결정 프레임워크와 체크리스트가 실행 가능한가?
- [ ] AT 책과의 교차 참조가 포함되었는가?

---

## 최종 디렉토리 구조

### Track 1 (개념 챕터)

```
chapter_X_[챕터명]/
├── XX_chapter_X_[챕터명]_ko.md
├── XX_chapter_X_[챕터명]_ko_explained.md  (Ch1, Ch2, Ch4, Ch5, Ch6)
├── ch*_audio.mp3, ch*_video.mp4, ...      (스튜디오 아티팩트)
└── src/
    ├── Table-*.md                          (기존 테이블, Ch2만)
    ├── run_chapterX_concept_review.py      (생성할 스크립트)
    └── reports/
        └── chapterX_concept_report.md      (생성될 리포트)
```

### Track 2 (계산 챕터)

```
chapter_X_[챕터명]/
├── XX_chapter_X_[챕터명]_ko.md
├── XX_chapter_X_[챕터명]_ko_explained.md  (Ch6만)
├── ch*_audio.mp3, ch*_video.mp4, ...      (스튜디오 아티팩트)
└── src/
    ├── Example*.py                         (기존 예제 코드)
    ├── run_chapterX_analysis.py            (생성할 스크립트)
    └── reports/
        ├── chapterX_report.md              (생성될 리포트)
        └── figures/
            └── *.png                       (생성될 차트)
```

### 프로젝트 전체

```
Chan E. Quantitative Trading...Build...Trading Business 2ed 2021/
├── .agent/workflows/create-chapter-analysis.md  (이 워크플로우)
├── .venv/                                        (생성 필요)
├── requirements.txt                              (생성 필요)
├── data/                                         (기존 공유 데이터)
│   ├── GLD.xls, GDX.xls, IGE.xls, ...
│   ├── SPX_20071123.txt, SPX_op_20071123.txt
│   └── IJR_20080114.txt, IJR_20080131.txt
├── src/                                          (기존 공유 유틸리티)
│   ├── calculateMaxDD.py
│   ├── calculateMaxDD_UnitTest.py
│   └── RetrieveYahooFinancialData.py
├── chapter_1_.../                                (Track 1)
├── chapter_2_.../                                (Track 1)
├── chapter_3_.../                                (Track 2)
├── chapter_4_.../                                (Track 1)
├── chapter_5_.../                                (Track 1)
├── chapter_6_.../                                (Track 2)
└── chapter_7_.../                                (Track 2)
```

---

## 권장 실행 순서

1. **환경 설정**: `requirements.txt` 생성 + `.venv` 구축
2. **Chapter 3** (Track 2): 가장 풍부한 예제, 템플릿 검증용
3. **Chapter 7** (Track 2): 두 번째로 풍부, 다양한 주제
4. **Chapter 6** (Track 2): 단일 예제이지만 핵심 Kelly 공식
5. **Chapter 1** (Track 1): 개념 리뷰 템플릿 확립
6. **Chapter 2** (Track 1): 기존 Table 자산 활용
7. **Chapter 4, 5** (Track 1): 나머지 개념 챕터

---

*이 workflow는 "Algorithmic Trading" (2013) 프로젝트의 Chapter 2 분석 스크립트 개발 경험과 워크플로우를 기반으로 작성되었습니다.*
