# ch02 The Basics Of Mean Reversion

### **[발표자료 PDF 다운로드](./Quant_Alpha_From_Chaos.pdf)**
### **[원본 도서 내용 (한글 번역)](./05_chapter_2_the_basics_of_mean_reversion_ko.md)**

## 요약 (Executive Summary)

본 문서는 금융 시계열 데이터에서 **평균 회귀(Mean Reversion)** 속성을 식별하고 활용하는 방법에 대해 다룬다. 대부분의 금융 가격 데이터는 기하학적 랜덤 워크(Geometric Random Walk)를 따르지만, 통계적 검정을 통해 평균으로 회귀하는 **정상성(Stationarity)** 을 지닌 시계열이나 포트폴리오를 찾아낼 수 있다. 이를 위해 ADF 검정, Hurst 지수, 공적분(Cointegration) 등의 수학적 도구를 활용하며, 이러한 분석을 바탕으로 파라미터 최적화 없이도 유효한 선형 매매 전략을 수립하는 방법을 제시한다.

![Figure](unnamed.png)

---

## 1. 평균 회귀와 정상성 (Mean Reversion and Stationarity)

평균 회귀와 정상성은 같은 현상을 바라보는 두 가지 관점이며, 각각 다른 통계적 검정 방법을 사용한다.

*   **정상성(Stationarity)의 의미**: 가격이 일정 범위 내에 머무른다는 것이 아니라, 가격의 분산(Variance)이 랜덤 워크보다 느리게 증가함(Sublinear function of time)을 의미한다.
*   **ADF 검정 (Augmented Dickey-Fuller Test)**: 시계열의 다음 변화가 현재 가격 수준에 의존하는지($\lambda \neq 0$)를 검정한다. 기각 시 평균 회귀 성향이 있다고 본다.
*   **Hurst 지수 (Hurst Exponent)**: 시계열의 성질을 $H$ 값으로 판단한다.
    *   $H = 0.5$: 기하학적 랜덤 워크
    *   $H < 0.5$: 평균 회귀 (Mean Reverting)
    *   $H > 0.5$: 추세 추종 (Trending)

---

## 2. 반감기 (Half-Life)와 선형 매매 전략

통계적으로 유의미한(90% 신뢰수준 이상) 평균 회귀가 아니더라도, **반감기** 분석을 통해 실전 매매의 가능성을 타진할 수 있다.

*   **반감기 산출**: Ornstein-Uhlenbeck 공식을 통해 평균으로 회귀하는 속도를 측정한다. 반감기 $= -log(2) / \lambda$.
*   **전략적 활용**: 반감기는 이동평균이나 표준편차 계산을 위한 **룩백(Look-back) 기간**을 설정하는 자연스러운 기준이 된다. 이를 통해 과최적화(Overfitting)를 방지할 수 있다.
*   **선형 평균 회귀 전략**: 가격이 이동평균에서 벗어난 정도(Z-Score)에 비례하여 포지션 규모를 조절한다. 즉, 가격이 평균보다 높으면 매도(Short), 낮으면 매수(Long) 포지션을 취한다.

---

## 3. 공적분 (Cointegration)과 포트폴리오 구성

개별 자산이 정상이 아니더라도(Non-stationary), 이들의 선형 결합은 정상성을 띨 수 있다. 이를 **공적분(Cointegration)** 이라 한다.

### 3.1. 공적분 검정 방법
| 검정 방식 | 설명 및 특징 |
| --- | --- |
| **CADF 검정** | 두 자산 간의 공적분을 검정한다. 변수의 순서(어떤 자산을 종속변수로 두느냐)에 따라 결과가 달라질 수 있다. |
| **Johansen 검정** | 3개 이상의 자산에 대해 공적분을 검정할 수 있다. 변수의 순서에 영향을 받지 않으며, 고유벡터(Eigenvector)를 통해 최적의 헤지 비율을 도출한다. |

### 3.2. 페어 트레이딩과 포트폴리오
*   **페어 트레이딩**: 유사한 움직임을 보이는 두 자산(예: EWA와 EWC, GDX와 GLD)을 매수/매도하여 시장 중립적인 수익을 추구한다.
*   **트리플렛 및 그 이상**: Johansen 검정의 고유벡터 중 고유값(Eigenvalue)이 가장 큰 벡터를 선택하면 반감기가 가장 짧은(가장 강력한 평균 회귀를 보이는) 포트폴리오를 구성할 수 있다.

---

## 4. 장단점 및 핵심 요약 (Key Points)

*   **장점**:
    *   내재적으로 정상인 자산뿐만 아니라, 인위적으로 구성한 포트폴리오를 통해 다양한 기회를 창출할 수 있다.
    *   경제적/펀더멘털 연관성에 기반하므로 전략의 유효성을 이해하기 쉽다.
    *   초단타(HFT)부터 장기 투자까지 다양한 시간 프레임에 적용 가능하다.
*   **위험**:
    *   높은 승률로 인한 과신은 과도한 레버리지 사용을 유발할 수 있다.
    *   구조적 변화(Regime Shift) 시 예기치 못한 대규모 손실(Tail Risk)이 발생할 수 있으므로 리스크 관리가 필수적이다.

### 핵심 요약
*   평균 회귀는 가격 변화가 평균과의 차이에 비례한다는 것을 의미한다.
*   공적분은 비정상 시계열들을 결합하여 정상 포트폴리오를 만드는 개념이다.
*   반감기는 전략의 타임 스케일을 결정하는 중요한 지표다.
*   Johansen 검정을 통해 최적의 헤지 비율과 포트폴리오 구성을 도출할 수 있다.

---

<div align="center">

[< Previous](../chapter_1_backtesting_and_automated_execution/README.md) | [Table of Contents](../README.md) | [Next >](../chapter_3_implementing_mean_reversion_strategies/README.md)

</div>
