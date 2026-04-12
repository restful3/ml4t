# 퀀트 트레이딩(Quantitative Trading) 종합 학습 가이드

> **Ernest P. Chan, *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*, 2nd Edition (2021)**

이 학습 가이드는 제공된 소스 자료를 바탕으로 퀀트 트레이딩의 정의, 비즈니스 구조, 실행 시스템, 그리고 주요 전략적 개념을 체계적으로 정리한 문서입니다.

![퀀트 트레이딩 성공 로드맵](qt2ed_infographic.png)

---

## 학습 자료

| 자료 | 파일 | 설명 |
|------|------|------|
| Audio | [qt2ed_audio.mp3](qt2ed_audio.mp3) | Deep Dive 팟캐스트 (한국어) |
| Video | [qt2ed_video.mp4](qt2ed_video.mp4) | Explainer 비디오 (한국어) |
| Slides | [qt2ed_slides.pptx](qt2ed_slides.pptx) | Quant Systems Blueprint 프레젠테이션 |
| Flashcards | [qt2ed_flashcards.html](qt2ed_flashcards.html) | 핵심 개념 플래시카드 (9장) |
| Quiz | [qt2ed_quiz.html](qt2ed_quiz.html) | 복습 퀴즈 (9문항) |
| Data Table | [qt2ed_data_table.csv](qt2ed_data_table.csv) | 챕터별 핵심 개념 및 전략 정리 |
| Mind Map | [qt2ed_mindmap.json](qt2ed_mindmap.json) | 전체 구조 마인드맵 |

### 챕터별 분석 노트북

| 챕터 | 노트북 | 리포트 |
|------|--------|--------|
| Ch3. Backtesting | [chapter3_full_report.ipynb](chapter_3_backtesting/src/chapter3_full_report.ipynb) | [chapter3_report.md](chapter_3_backtesting/src/reports/chapter3_report.md) |
| Ch6. Money & Risk Management | [chapter6_full_report.ipynb](chapter_6_money_and_risk_management/src/chapter6_full_report.ipynb) | [chapter6_report.md](chapter_6_money_and_risk_management/src/reports/chapter6_report.md) |
| Ch7. Special Topics | [chapter7_full_report.ipynb](chapter_7_special_topics_in_quantitative_trading/src/chapter7_full_report.ipynb) | [chapter7_report.md](chapter_7_special_topics_in_quantitative_trading/src/reports/chapter7_analysis_report.md) |

---

## 1. 퀀트 트레이딩의 기초

### 퀀트 트레이딩의 정의
**퀀트 트레이딩(Quantitative Trading)** 또는 알고리즘 트레이딩은 컴퓨터 알고리즘이 내리는 매수/매도 결정에 전적으로 의존하여 유가증권을 거래하는 방식입니다. 이러한 알고리즘은 과거의 금융 데이터를 바탕으로 테스트된 전략을 프로그래밍하여 구축됩니다.

*   **기술적 분석과의 차이:** 퀀트 트레이딩은 기술적 분석을 포함할 수 있지만, "헤드 앤 숄더 패턴 찾기"와 같은 주관적이고 수치화하기 어려운 기법은 배제됩니다. 모든 정보는 비트와 바이트로 변환되어 컴퓨터가 이해할 수 있어야 합니다.
*   **데이터 활용:** 단순히 가격 데이터뿐만 아니라 매출, 현금 흐름, 부채 비율과 같은 근본적(Fundamental) 데이터와 뉴스 리포트 등의 텍스트 정보도 수치화하여 입력값으로 사용할 수 있습니다.

### 누가 퀀트 트레이더가 되는가?
전통적으로 물리학자, 수학자, 엔지니어 등 고등 학위 소지자들이 주를 이루었으나, **통계적 차익거래(Statistical Arbitrage)**와 같은 분야는 복잡한 파생상품이 아닌 주식, 선물, 외환 등을 다루기 때문에 고등학교 수준의 수학, 통계, 프로그래밍 지식만으로도 접근 가능합니다.

*   **독립 트레이더의 자격:** 금융이나 프로그래밍 경험이 있고, 손실을 견딜 수 있는 충분한 저축(Nest egg)이 있으며, 공포와 탐욕 사이의 감정적 균형을 유지할 수 있는 사람이 이상적입니다.

---

## 2. 비즈니스 구조 및 인프라 설정

독립 트레이더로서 비즈니스를 시작할 때 가장 먼저 결정해야 할 사항은 리테일 계좌를 개설할 것인지, 아니면 프랍 트레이딩(Proprietary Trading) 회사에 합류할 것인지입니다.

### 리테일 트레이딩 vs. 프랍 트레이딩 비교

| 구분 | 리테일 트레이딩 (Retail) | 프랍 트레이딩 (Proprietary) |
| :--- | :--- | :--- |
| **법적 요구사항** | 없음 | FINRA Series 7 시험 합격 필요 |
| **자본 요구량** | 상당함 | 적음 |
| **레버리지** | SEC Reg T에 의해 제한 (보통 2배~4배) | 회사의 재량 (20배 이상 가능) |
| **손실 책임** | 무한대 (S-Corp/LLC 미설정 시) | 초기 투자금으로 제한 |
| **수수료 및 비용** | 낮은 수수료, 최소한의 데이터 비용 | 높은 수수료 및 상당한 월간 비용 |
| **교육 및 멘토링** | 없음 | 제공될 수 있음 (유료 가능) |
| **영업 비밀 노출** | 위험 낮음 | 관리자가 수익성 전략을 모방할 위험 있음 |
| **위험 관리** | 주로 자기 통제 | 관리자에 의해 강제되는 포괄적 관리 |

### 물리적 인프라
초기 단계의 인프라는 간단하게 구축할 수 있습니다.
*   **하드웨어:** 개인용 컴퓨터, 고속 인터넷, 무정전 전원 장치(UPS).
*   **소프트웨어/데이터:** 실시간 뉴스 피드(Thomson Reuters, Bloomberg 등), API 지원 트레이딩 플랫폼.
*   **확장 단계:** 실행 지연(Slippage)을 줄이기 위해 거래소 서버와 인접한 가상 사설 서버(VPS)를 활용할 수 있습니다.

---

## 3. 실행 시스템 (Execution Systems)

자동 매매 시스템(ATS)은 시장 데이터를 검색하고, 알고리즘을 실행하여 주문을 생성 및 전송하는 역할을 합니다.

### 시스템의 종류
1.  **반자동 시스템 (Semi-automated):** Excel, MATLAB, Python 등으로 주문 목록을 생성한 뒤, 브로커가 제공하는 '바스켓 트레이더'나 '스프레드 트레이더' 도구를 통해 수동으로 전송합니다. 주문 전 최종 확인이 가능하다는 장점이 있습니다.
2.  **완전 자동 시스템 (Fully automated):** 하루 종일 루프를 돌며 데이터를 스캔하고 주문을 즉각 전송합니다. 고빈도 매매(HFT)에 필수적이며, Java, C#, C++ 또는 전용 API를 사용합니다.

### 거래 비용 최소화 전략
*   **수수료 절감:** 주당 가격이 낮은 주식(보통 $5 미만)은 수수료와 호가 스프레드 비중이 높으므로 피하는 것이 좋습니다.
*   **시장 충격 완화:** 주문 크기는 해당 주식 일평균 거래량의 1%를 넘지 않아야 합니다. 또한, 시가 총액의 네제곱근(Fourth root)에 비례하여 자본 비중을 조절하는 방식이 권장됩니다.

---

## 4. 주요 전략 테마 및 특수 토픽

### 평균 회귀(Mean-Reverting) vs. 모멘텀(Momentum)
*   **평균 회귀:** 가격이 일시적 편차에서 벗어나 평균으로 돌아올 것이라는 믿음에 기반합니다. 개별 주식보다는 두 주식의 스프레드를 이용한 '교차 섹션 평균 회귀'가 더 흔합니다.
*   **모멘텀:** 가격이 일정 방향으로 계속 움직일 것이라는 믿음에 기반합니다. 정보의 느린 확산, 대규모 주문의 분할 실행, 투자자의 군집 행동(Herding) 등에 의해 발생합니다.

### 정체성(Stationarity)과 공적분(Cointegration)
*   **정체성:** 시계열 데이터가 초기 값에서 멀리 벗어나지 않는 성질입니다.
*   **공적분:** 두 개 이상의 비정체성 시계열(주가 등)을 선형 결합했을 때 정체성을 가진 시계열이 형성되는 상태입니다. 이는 단순한 '상관관계(Correlation)'와 다르며, 장기적인 가격 수렴을 보장합니다.

### 조건부 매개변수 최적화 (Conditional Parameter Optimization, CPO)
머신러닝(랜덤 포레스트 등)을 사용하여 변화하는 시장 상황에 따라 전략의 매개변수를 실시간으로 조정하는 기법입니다. 과거 데이터를 고정하여 최적화하는 기존 방식보다 시장 변화에 훨씬 더 민감하게 대응할 수 있습니다.

### 요인 모델 (Factor Models)
주식의 초과 수익을 시장 요인(Market), 기업 규모(SMB), 가치(HML) 등 여러 공통 동인으로 설명하는 프레임워크입니다. (예: 파마-프렌치 3요인 모델)

---

## 5. 복습 및 실전 연습

### 단답형 연습 문제

1.  **Q:** 리테일 브로커 계좌에서 레버리지 사용 시 발생하는 무한 책임을 피하기 위한 법적 조치는 무엇인가?
    *   **A:** LLC(유한책임회사)나 S-Corp과 같은 법인을 설립하여 계좌를 개설한다.
2.  **Q:** 퀀트 전략 실행 전 소프트웨어 버그나 룩어헤드 편향(Look-ahead bias)을 확인하기 위해 반드시 거쳐야 하는 과정은?
    *   **A:** 페이퍼 트레이딩(Paper Trading).
3.  **Q:** 평균 회귀 시계열에서 가격이 평균으로 돌아가는 데 걸리는 시간을 측정하는 수식은?
    *   **A:** 오른슈타인-울렌벡(Ornstein-Uhlenbeck) 공식을 활용한 반감기(Half-life) 계산.
4.  **Q:** 고빈도 매매(HFT)가 높은 샤프 지수(Sharpe Ratio)를 가질 수 있는 통계적 근거는?
    *   **A:** 대수의 법칙(Law of Large Numbers). 많은 횟수의 베팅을 통해 평균 수익으로부터의 편차를 최소화할 수 있기 때문이다.

### 에세이 연습 문제

1.  **시장 구조의 변화와 퀀트 전략:** 2001년 미 주식 시장의 소수점 호가 도입(Decimalization)과 2007년 공매도 플러스 틱 룰(Plus-tick rule) 폐지가 통계적 차익거래 전략의 수익성에 어떤 영향을 미쳤는지 논하시오.
2.  **독립 트레이더의 전략 선택:** 자본력과 인프라가 제한적인 독립 트레이더가 기관 투자자와의 경쟁에서 승리하기 위해 왜 복잡한 파생상품보다 단순한 통계적 차익거래나 특정 원자재(가솔린, 천연가스 등)의 계절적 패턴에 주목해야 하는지 서술하시오.

---

## 6. 핵심 용어 사전 (Glossary)

*   **Algorithmic Trading (알고리즘 트레이딩):** 컴퓨터 프로그램을 통해 사전에 정의된 규칙에 따라 거래를 실행하는 방식.
*   **Statistical Arbitrage (통계적 차익거래):** 주식, 선물 등의 단순한 금융상품을 대상으로 통계적 불균형을 이용해 수익을 내는 퀀트 트레이딩의 한 종류.
*   **Slippage (슬리피지):** 주문을 실행하기로 한 시점의 가격과 실제 체결된 가격 사이의 차이.
*   **Dark Pool (다크 풀):** 거래소 밖에서 기관 투자자들의 대량 주문을 처리하는 익명성 유동성 공급처.
*   **API (Application Programming Interface):** 트레이딩 소프트웨어가 브로커의 시스템과 통신하여 데이터를 받고 주문을 보낼 수 있게 하는 연결 통로.
*   **Regime Shift (레짐 시프트):** 금융 시장의 구조나 경제 환경이 급격하게 변하여 기존 전략의 유효성이 떨어지는 현상.
*   **Cointegration (공적분):** 비정체적인 데이터들이 장기적으로 일정한 관계를 유지하며 함께 움직이는 상태.
*   **SMB (Small Minus Big):** 시가총액이 작은 주식이 큰 주식보다 높은 수익을 내는 요인을 측정하는 지표.
*   **HML (High Minus Low):** 장부가치 대 가격 비율이 높은(저평가된) 주식이 낮은 주식보다 높은 수익을 내는 요인을 측정하는 지표.