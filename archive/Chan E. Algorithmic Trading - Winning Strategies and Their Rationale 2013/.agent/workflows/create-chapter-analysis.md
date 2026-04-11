---
description: Create run_chapterX_analysis.py script for Chan's Algorithmic Trading book chapters
---

# 챕터 분석 스크립트 생성 가이드

이 workflow는 Ernest Chan의 "Algorithmic Trading" 책의 각 챕터에 대한 종합 분석 스크립트(`run_chapterX_analysis.py`)와 마크다운 리포트를 생성하는 방법을 안내합니다.

---

## 📋 작업 전 준비사항

1. **챕터 한글 문서 읽기**: `chapter_X_.../XX_chapter_X_..._ko.md` 파일을 읽어 핵심 개념과 문제 정의를 파악
2. **기존 소스 코드 분석**: `chapter_X_.../src/` 폴더의 Python 파일들 분석
3. **데이터 파일 확인**: `inputData_*.csv` 등 사용할 데이터 파일 목록 확인
4. **Chapter 2 참조**: `/chapter_2_the_basics_of_mean_reversion/src/run_chapter2_analysis.py`를 템플릿으로 활용

---

## 🎯 스크립트 생성 요청 프롬프트

다음 프롬프트를 사용하여 각 챕터의 분석 스크립트를 생성하세요:

```
[Chapter X] 분석 스크립트 생성

## 목표
`chapter_X_[챕터명]/src/run_chapterX_analysis.py` 스크립트를 생성해주세요.

## 참조 파일
1. 한글 문서: `chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko.md`
2. 기존 소스: `chapter_X_[챕터명]/src/*.py`
3. 템플릿: `chapter_2_the_basics_of_mean_reversion/src/run_chapter2_analysis.py`

## 스크립트 요구사항

### 1. 구조
- `ChapterXAnalyzer` 클래스 생성
- `load_data()`: 데이터 로드
- `analyze_[핵심개념1]()`: 첫 번째 핵심 분석
- `analyze_[핵심개념2]()`: 두 번째 핵심 분석
- `backtest_strategy()`: 전략 백테스트
- `generate_report()`: 마크다운 리포트 생성
- `run()`: 전체 분석 오케스트레이션

### 2. 리포트 형식
리포트는 다음 섹션을 포함해야 합니다:

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
   - 해석 및 결론 (✅/⚠️/❌ 이모지 사용)

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

## 품질 기준
- 한글 문서의 이론적 배경을 리포트에 반영
- 모든 수학 공식은 LaTeX 형식 사용
- 각 분석 결과에 대한 해석과 트레이딩 함의 포함
- pandas 2.0+ 호환성 유지
- 코드 주석은 한글로 작성
```

---

## 📚 챕터별 핵심 분석 내용 가이드

### Chapter 3: 평균 회귀 전략 구현 (Implementing Mean Reversion Strategies)
- **핵심 개념**: 볼린저 밴드, 칼만 필터, 스케일링 인/아웃
- **분석 항목**:
  - 볼린저 밴드 진입/청산 전략
  - 칼만 필터 기반 동적 헤지 비율
  - 스케일링 인(추가 매수) 전략 비교
- **주요 데이터**: EWA/EWC/IGE (Chapter 2와 동일)

### Chapter 4: 주식 및 ETF 평균 회귀 (Mean Reversion of Stocks and ETFs)
- **핵심 개념**: 횡단면 평균 회귀, 주식 페어 트레이딩
- **분석 항목**:
  - 횡단면 평균 회귀 vs 시계열 평균 회귀
  - 주식 페어 선택 기준
  - ETF 내 종목간 공적분
- **주요 데이터**: 개별 주식, 섹터 ETF

### Chapter 5: 통화 및 선물 평균 회귀 (Mean Reversion of Currencies and Futures)
- **핵심 개념**: 롤오버, 캐리 트레이드, 계절성
- **분석 항목**:
  - 통화쌍 캐리 전략
  - 선물 롤오버 비용 분석
  - 계절성 기반 전략
- **주요 데이터**: 통화 선물, 상품 선물

### Chapter 6: 일간 모멘텀 전략 (Interday Momentum Strategies)
- **핵심 개념**: 시계열 모멘텀, 횡단면 모멘텀
- **분석 항목**:
  - 모멘텀 팩터 계산
  - 포트폴리오 구성 (롱/숏)
  - 모멘텀 크래시 분석
- **주요 데이터**: 주식 지수, 상품 ETF

### Chapter 7: 일중 모멘텀 전략 (Intraday Momentum Strategies)
- **핵심 개념**: 오프닝 갭, 일중 모멘텀
- **분석 항목**:
  - 갭 트레이딩 전략
  - 일중 가격 패턴
  - 거래량 분석
- **주요 데이터**: 분봉/틱 데이터

### Chapter 8: 리스크 관리 (Risk Management)
- **핵심 개념**: 켈리 공식, VaR, 드로다운 관리
- **분석 항목**:
  - 최적 레버리지 계산 (켈리 공식)
  - VaR/CVaR 계산
  - 최대 드로다운 기반 청산 규칙
- **주요 데이터**: Chapter 2-7 전략 결과 활용

---

## 🔧 실행 단계

### Step 1: 챕터 문서 분석
```bash
# 한글 문서 확인
cat chapter_X_[챕터명]/XX_chapter_X_[챕터명]_ko.md

# 기존 소스 코드 확인
ls -la chapter_X_[챕터명]/src/
```

### Step 2: 스크립트 생성 요청
위의 프롬프트 템플릿에 챕터별 정보를 채워서 요청

### Step 3: 스크립트 실행
```bash
cd chapter_X_[챕터명]/src
source ../../.venv/bin/activate
python run_chapterX_analysis.py
```

### Step 4: 결과 확인
```bash
# 리포트 확인
cat reports/chapterX_report.md

# 차트 확인
ls reports/figures/
```

---

## ✅ 체크리스트

스크립트 생성 후 다음 항목을 확인하세요:

- [ ] 한글 문서의 핵심 개념이 리포트에 반영되었는가?
- [ ] 수학 공식이 LaTeX 형식으로 올바르게 표현되었는가?
- [ ] 데이터 설명과 선정 이유가 포함되었는가?
- [ ] 각 분석 결과에 해석과 트레이딩 함의가 있는가?
- [ ] 백테스트 결과에 주의사항(bias, 거래비용 등)이 명시되었는가?
- [ ] 차트가 `reports/figures/`에 저장되는가?
- [ ] pandas 2.0+ 호환성이 유지되는가?
- [ ] 스크립트가 오류 없이 실행되는가?

---

## 📁 최종 디렉토리 구조

```
chapter_X_[챕터명]/
├── XX_chapter_X_[챕터명]_ko.md    # 한글 이론 문서
├── src/
│   ├── *.py                        # 기존 예제 코드
│   ├── inputData_*.csv             # 데이터 파일
│   ├── run_chapterX_analysis.py    # ✨ 생성할 스크립트
│   └── reports/
│       ├── chapterX_report.md      # ✨ 생성될 리포트
│       └── figures/
│           └── *.png               # ✨ 생성될 차트
```

---

*이 workflow는 Chapter 2 분석 스크립트 개발 경험을 바탕으로 작성되었습니다.*
