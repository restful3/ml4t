# 알고리즘 트레이딩 전략 및 실무 지침서: 브리핑 문서

### **[발표자료 PDF 다운로드](./Building_the_Statistical_Trading_Edge.pdf)**

![Algorithmic Trading Book Cover](./unnamed.png)

## 챕터별 요약 목차 (Table of Contents)

각 챕터의 상세 내용은 아래 링크를 클릭하여 확인할 수 있습니다.

*   [**Chapter 1: Backtesting and Automated Execution**](./chapter_1_backtesting_and_automated_execution/README.md) ([PDF](./chapter_1_backtesting_and_automated_execution/Quant_Trading_System_Blueprint.pdf))
*   [**Chapter 2: The Basics of Mean Reversion**](./chapter_2_the_basics_of_mean_reversion/README.md) ([PDF](./chapter_2_the_basics_of_mean_reversion/Quant_Alpha_From_Chaos.pdf)) | [분석 리포트](./chapter_2_the_basics_of_mean_reversion/src/reports/chapter2_report.md)
*   [**Chapter 3: Implementing Mean Reversion Strategies**](./chapter_3_implementing_mean_reversion_strategies/README.md) ([PDF](./chapter_3_implementing_mean_reversion_strategies/Adaptive_Mean_Reversion_Algorithms.pdf)) | [분석 리포트](./chapter_3_implementing_mean_reversion_strategies/src/reports/chapter3_report.md)
*   [**Chapter 4: Mean Reversion of Stocks and ETFs**](./chapter_4_mean_reversion_of_stocks_and_etfs/README.md) ([PDF](./chapter_4_mean_reversion_of_stocks_and_etfs/Mean_Reversion_Alpha_Evolution.pdf)) | [분석 리포트](./chapter_4_mean_reversion_of_stocks_and_etfs/src/reports/chapter4_report.md)
*   [**Chapter 5: Mean Reversion of Currencies and Futures**](./chapter_5_mean_reversion_of_currencies_and_futures/README.md) ([PDF](./chapter_5_mean_reversion_of_currencies_and_futures/Structural_Alpha_in_Momentum_Markets.pdf)) | [분석 리포트](./chapter_5_mean_reversion_of_currencies_and_futures/src/reports/chapter5_report.md)
*   [**Chapter 6: Interday Momentum Strategies**](./chapter_6_interday_momentum_strategies/README.md) ([PDF](./chapter_6_interday_momentum_strategies/Momentum_Strategy_Dissected.pdf)) | [분석 리포트](./chapter_6_interday_momentum_strategies/src/reports/chapter6_report.md)
*   [**Chapter 7: Intraday Momentum Strategies**](./chapter_7_intraday_momentum_strategies/README.md) ([PDF](./chapter_7_intraday_momentum_strategies/Intraday_Alpha_Strategies.pdf)) | [분석 리포트](./chapter_7_intraday_momentum_strategies/src/reports/chapter7_report.md)
*   [**Chapter 8: Risk Management**](./chapter_8_risk_management/README.md) ([PDF](./chapter_8_risk_management/Engineering_Portfolio_Survival.pdf)) | [분석 리포트](./chapter_8_risk_management/src/reports/chapter8_report.md)
*   [**Conclusion**](./conclusion/README.md)

---

## 코드 구조 (Code Structure)

각 챕터별로 `src/` 폴더에 관련 Python 코드와 데이터 파일이 정리되어 있습니다.

```
├── chapter_2_the_basics_of_mean_reversion/src/
│   ├── run_chapter2_analysis.py    # [종합 분석] ADF, 허스트, 공적분, 반감기
│   ├── stationaryTests.py          # ADF 검정, 허스트 지수
│   ├── cointegrationTests.py       # CADF, Johansen 공적분 검정
│   ├── genhurst.py                 # 허스트 지수 계산 함수
│   ├── inputData_*.csv             # EWA/EWC/IGE, USDCAD 데이터
│   └── reports/                    # 분석 리포트 및 차트
│
├── chapter_3_implementing_mean_reversion_strategies/src/
│   ├── run_chapter3_analysis.py    # [종합 분석] 볼린저 밴드, 칼만 필터
│   ├── bollinger.py                # 볼린저 밴드 전략
│   ├── KF_beta_EWA_EWC.py          # 칼만 필터 전략
│   ├── PriceSpread.py              # 가격 스프레드 전략
│   ├── LogPriceSpread.py           # 로그 가격 스프레드 전략
│   ├── Ratio.py                    # 비율 전략
│   └── reports/                    # 분석 리포트 및 차트
│
├── chapter_4_mean_reversion_of_stocks_and_etfs/src/
│   ├── run_chapter4_analysis.py    # [종합 분석] 인덱스 차익거래, PEAD, Buy-on-Gap, 횡단면
│   ├── indexArb.py                 # SPY 인덱스 차익거래
│   ├── pead.py                     # 실적 발표 후 드리프트 (PEAD)
│   ├── bog.py                      # Buy-on-Gap 전략
│   ├── andrewlo_2007_2012.py       # 선형 롱숏 모델
│   └── reports/                    # 분석 리포트 및 차트
│
├── chapter_5_mean_reversion_of_currencies_and_futures/src/
│   ├── run_chapter5_analysis.py    # [종합 분석] AUD/CAD 페어, 롤오버, 캘린더 스프레드
│   ├── AUDCAD_daily.py             # AUD/CAD 페어 트레이딩
│   ├── AUDCAD_unequal.py           # Johansen 기반 통화쌍 트레이딩
│   ├── calendarSpdsMeanReversion.py # 캘린더 스프레드 평균회귀
│   ├── estimateFuturesReturns.py   # 선물 수익률 추정
│   └── reports/                    # 분석 리포트 및 차트
│
├── chapter_6_interday_momentum_strategies/src/
│   ├── run_chapter6_analysis.py    # [종합 분석] TU 모멘텀, 가설 검정, 횡단면 모멘텀
│   ├── TU_mom.py                   # 채권선물 모멘텀
│   ├── TU_mom_hypothesisTest.py    # 모멘텀 가설 검정
│   ├── kentdaniel.py               # 횡단면 모멘텀 전략
│   └── reports/                    # 분석 리포트 및 차트
│
├── chapter_7_intraday_momentum_strategies/src/
│   ├── run_chapter7_analysis.py    # [종합 분석] FSTX 시가 갭, VX-ES 롤 수익률
│   ├── gapFutures_FSTX.py          # 갭 오프닝 전략
│   ├── VX_ES_rollreturn.py         # VIX 선물 vs ES 롤 수익률
│   └── reports/                    # 분석 리포트 및 차트
│
└── chapter_8_risk_management/src/
    ├── run_chapter8_analysis.py    # [종합 분석] 켈리 공식, 몬테카를로, CPPI
    ├── monteCarloOptimLeverage.py  # 몬테카를로 레버리지 최적화
    ├── calculateMaxDD.py           # 최대 낙폭(MDD) 계산
    └── reports/                    # 분석 리포트 및 차트
```

### 가상환경 설정

```bash
cd "/home/restful3/workspace/ml4t/source/Chan E. Algorithmic Trading - Winning Strategies and Their Rationale 2013"

# 가상환경 활성화
source .venv/bin/activate

# 예시: Chapter 2 코드 실행
cd chapter_2_the_basics_of_mean_reversion/src
python stationaryTests.py
```

### 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `numpy`, `pandas` | 수치 연산, 데이터 분석 |
| `statsmodels` | ADF 검정, 공적분 검정, 회귀 분석 |
| `arch` | 분산비 검정, GARCH 모델 |
| `matplotlib` | 시각화 |
| `scipy` | 과학 계산, 최적화 |

### 종합 분석 스크립트 (Analysis Scripts)

각 챕터별 `run_chapterX_analysis.py` 스크립트는 책의 예제를 재현하고 백테스팅 결과를 리포트로 생성합니다. 실행 방법:

```bash
source .venv/bin/activate
cd chapter_X_*/src
python run_chapterX_analysis.py
```

| 챕터 | 대표 전략/분석 | 핵심 지표 | 결과 | 책 기대값 |
|------|--------------|-----------|------|----------|
| Ch2 | EWA-EWC 공적분 검정 | CADF p-value | 0.0284 | < 0.05 |
| Ch3 | 칼만 필터 EWA-EWC | Sharpe Ratio | 3.31 | 2.4 |
| Ch4 | 횡단면 평균 회귀 (O2C) | Sharpe Ratio | 4.94 | 4.7 |
| Ch5 | AUD/CAD 페어 트레이딩 | Sharpe Ratio | 1.36 | 1.36 |
| Ch6 | 가우시안 가설 검정 | 검정 통계량 | 2.77 | 2.77 |
| Ch7 | FSTX 시가 갭 전략 | APR / Sharpe | 7.49% / 0.49 | 7.5% / 0.49 |
| Ch8 | 몬테카를로 최적 레버리지 | f* / g(f*) | 25.51 / 0.0058 | 25.51 / 0.0058 |

---

## 1. 개요 (Executive Summary)

본 문서는 **어네스트 찬(Ernest P. Chan)** 의 저서 **"Algorithmic Trading: Winning Strategies and Their Rationale"** 의 내용을 바탕으로 알고리즘 트레이딩의 핵심 전략, 백테스팅 실무, 그리고 통계적 검증 방법을 요약한 브리핑 문서이다.

핵심 요지는 다음과 같다:

*   **백테스팅의 중요성 및 위험성**: 백테스팅은 전략의 과거 성과를 확인하는 필수 과정이나, **생존 편향(Survivorship Bias)**, **데이터 스누핑(Data-snooping)**, **룩어헤드 편향(Look-ahead Bias)** 등 수많은 함정이 존재하며 이를 제거하지 못하면 실전에서 막대한 손실을 초래할 수 있다.
*   **평균 회귀(Mean Reversion) 전략**: 금융 시계열의 **정상성(Stationarity)** 과 **공적분(Cointegration)** 을 통계적으로 검정(ADF 테스트, 허스트 지수, 요한슨 테스트 등)하여 수익성 있는 포트폴리오를 구축할 수 있다. 특히 **반감기(Half-life)** 개념은 포트폴리오의 보유 기간과 룩백 기간을 설정하는 데 중요한 척도가 된다.
*   **시스템 구축**: 백테스팅 플랫폼과 자동 매매 실행 플랫폼의 일치 여부는 실행 효율성과 데이터 일관성 측면에서 매우 중요하다. MATLAB, R, Python과 같은 스크립트 언어와 전용 IDE(Marketcetera, TradeLink 등)의 특성을 이해하고 트레이더의 역량에 맞는 도구를 선택해야 한다.

---

## 2. 백테스팅의 실무와 일반적인 함정

백테스팅은 과거 데이터를 통해 전략의 미래 수익성을 예측하는 과정이지만, 실제 거래 결과가 백테스트와 일치하지 않는 경우가 많다. 이를 방지하기 위해 다음의 위험 요소를 관리해야 한다.

### 2.1 주요 백테스팅 함정 (Pitfalls)

| 항목 | 상세 내용 |
| :--- | :--- |
| **룩어헤드 편향 (Look-ahead Bias)** | 미래의 정보를 현재의 매매 신호 결정에 사용하는 오류. 백테스팅과 라이브 실행 코드를 동일하게 유지함으로써 예방 가능함. |
| **데이터 스누핑 편향 (Data-snooping Bias)** | 너무 많은 매개변수를 과거 데이터에 과도하게 최적화(Overfitting)하는 현상. 모델을 단순화하고 워크포워드(Walk-forward) 테스트를 통해 검증해야 함. |
| **생존 편향 (Survivorship Bias)** | 현재 상장되어 있는 종목만을 대상으로 백테스트를 수행하는 오류. 상장 폐지된 종목을 포함한 데이터베이스를 사용해야 함. |
| **수정 주가 오류** | 주식 분할 및 배당금 조정을 반영하지 않을 경우, 가격 급락을 매매 신호로 오인할 수 있음. |
| **체결 가능성 및 거래소 차이** | 통합 주가(Consolidated Quotes)와 기본 거래소(Primary Exchange) 주가의 차이, 공매도 제한(Uptick Rule, 주식 대차 불가) 등을 고려하지 않을 경우 수익률이 부풀려질 수 있음. |

### 2.2 통계적 유의성 검정

백테스트 결과가 단순한 운에 의한 것인지 판단하기 위해 **가설 검정(Hypothesis Testing)** 을 수행한다.

*   **귀무 가설 (Null Hypothesis)**: 전략의 진정한 평균 일일 수익률이 0이라는 가정.
*   **p-value**: 귀무 가설 하에서 관찰된 수익률 이상의 결과가 나타날 확률. 일반적으로 **0.01 미만**일 때 통계적으로 유의하다고 간주함.
*   **몬테카를로 시뮬레이션**: 역사적 데이터를 무작위로 섞거나 시뮬레이션된 데이터를 사용하여 전략의 성과가 무작위성보다 우수한지 검증함.

---

## 3. 평균 회귀(Mean Reversion) 전략의 기초

평균 회귀는 가격이 평균에서 벗어났을 때 다시 평균으로 돌아오려는 성질을 이용한다.

### 3.1 정상성(Stationarity) 검정 도구

가격 시계열이 평균 회귀적 인지를 확인하기 위해 다음의 통계적 도구를 사용한다.

*   **ADF(Augmented Dickey-Fuller) 테스트**: 시계열에 단위근이 존재하는지(랜덤 워크인지) 확인하는 테스트. 검정 통계량이 임계값보다 작아야(더 음수여야) 정상성을 인정받음.
*   **허스트 지수(Hurst Exponent, H)**: 시계열의 확산 속도를 측정. **H < 0.5**이면 평균 회귀, **H > 0.5**이면 추세 추종, **H = 0.5**이면 랜덤 워크를 의미함.
*   **반감기(Half-life)**: 가격이 평균으로부터 벗어난 거리가 절반으로 회복되는 데 걸리는 시간. 이는 이동 평균의 룩백 기간이나 포지션 보유 기간을 설정하는 기준이 됨.

### 3.2 공적분(Cointegration)과 쌍 트레이딩(Pairs Trading)

개별 종목은 정상성을 띠지 않더라도, 두 종목 이상의 선형 결합(포트폴리오)은 정상성을 가질 수 있다.

*   **CADF 테스트**: 두 종목 간의 최적 헤지 비율을 구한 후, 잔차(Residuals)의 정상성을 ADF 테스트로 검정함.
*   **요한슨(Johansen) 테스트**: 세 종목 이상의 자산군에서 공적분 관계의 수와 헤지 비율(고유벡터)을 찾는 데 적합함.

---

## 4. 트레이딩 플랫폼 선택 가이드

성공적인 알고리즘 트레이딩을 위해서는 연구(백테스팅)와 실행(자동 매매) 간의 간극을 좁히는 플랫폼 선택이 중요하다.

### 4.1 오픈 소스 통합 개발 환경(IDE) 비교

| IDE | 언어 | 자산군 | 특징 |
| :--- | :--- | :--- | :--- |
| **ActiveQuant** | Java, MATLAB, R | 다양함 | 틱(Tick) 기반 실행 지원 |
| **Algo-Trader** | Java | 다양함 | 복합 이벤트 처리(CEP) 지원 |
| **Marketcetera** | Java, Python, Ruby | 다양함 | 다양한 브로커 및 FIX 프로토콜 지원 |
| **TradeLink** | .NET, Java, Python 등 | 다양함 | 틱 기반 및 오픈 소스 기반 |

### 4.2 실행 및 자동화 고려사항

*   **코로케이션(Colocation)**: 체결 지연(Latency)을 줄이기 위해 브로커나 거래소의 데이터 센터와 물리적으로 가까운 곳에 서버를 배치함.
*   **복합 이벤트 처리(CEP)**: 단순한 바(Bar) 기반 처리가 아닌, 틱 데이터나 뉴스 도착 등의 이벤트를 실시간으로 처리하는 기술.
*   **이벤트 기반(Event-driven) 시스템**: 뉴스 피드나 개별 틱의 변화에 즉각적으로 반응하는 고빈도 트레이딩(HFT)의 필수 요소.

---

## 5. 결론 및 통찰 (Insights)

*   **단순함의 미학**: 복잡한 비선형 모델이나 많은 매개변수를 가진 모델보다, 물리적/경제적 근거가 있는 **단순한 선형 모델**이 데이터 스누핑 편향을 피하고 실전에서 더 견고한 성과를 내는 경우가 많다.
*   **심리적 요인**: 평균 회귀 전략은 높은 승률을 보이지만, 전략이 실패할 때(Regime Shift)의 손실이 매우 클 수 있다. 따라서 손절매보다는 적절한 레버리지 관리와 **켈리 공식(Kelly Formula)** 기반의 자산 배분이 필수적이다.
*   **지속적인 시장 감시**: 2001년 소수점 호가 도입, 2008년 금융 위기, 2010년 플래시 크래시와 같은 시장 구조 변화는 과거의 백테스트 결과를 무용지물로 만들 수 있으므로 시장 환경 변화에 기민하게 대응해야 한다.

> "알고리즘 트레이딩은 단순히 수학과 프로그래밍의 결합이 아니라, 시장 구조에 대한 깊은 이해와 통계적 엄밀함이 결합된 예술이다."

---