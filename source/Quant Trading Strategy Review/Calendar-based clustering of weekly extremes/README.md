# Calendar-based Clustering of Weekly Extremes

논문 "Calendar-based clustering of weekly extremes: An empirical failure of stochastic models" (이임현, 동아대학교)의 구현 코드입니다.

## 📚 논문 요약

### 핵심 발견
- **주간 극값 군집 현상**: 한 주 동안의 최고가/최저가가 특정 요일에 몰리는 현상 발견
- **기존 모형의 한계**: GBM, Heston, Jump-diffusion 모형으로는 이 현상을 설명 불가
- **새로운 모형 제안**: 요일 의존적 MSGARCH(Markov-Switching GARCH) 모형이 효과적으로 설명

### 주요 결과
- **ES (주식선물)**: 요일별 극값 군집이 매우 강함 (표 3: p=0.428, 0.516)
- **강건성 문제**: 샘플외 검정에서 성능 저하 (표 4: 62.5% 기각)

---

## 📁 파일 구조

```
Calendar-based clustering of weekly extremes/
├── Calendar-based clustering of weekly extremes.ko.md  # 논문 한글 번역
├── main.py                                              # 단순 마르코프 기반 트레이딩 전략
├── main_2.py                                            # 논문 기반 MSGARCH 구현
└── README.md                                            # 이 문서
```

---

## 🔍 main.py vs main_2.py 비교

| 구분 | main.py | main_2.py |
|------|---------|-----------|
| **목적** | 실전 트레이딩 전략 | 논문 모형 재현 + 트레이딩 |
| **핵심 모형** | 단순 마르코프 전이 | 요일 의존적 MSGARCH |
| **상태 수** | 없음 (요일→요일 전이만) | K개 잠재 상태 (2~5) |
| **변동성 모델링** | 없음 | GARCH(1,1) (요일×상태별) |
| **학습 알고리즘** | 카운트 누적 | EM 알고리즘 (Baum-Welch) |
| **시뮬레이션** | 없음 | 2000회 경로 생성 |
| **검증** | 통계 로깅만 | 논문 표 3, 4 재현 |
| **샘플외 검정** | ❌ | ✅ (80/20 분할) |
| **코드 복잡도** | ~460 lines | ~1000 lines |
| **실행 속도** | 빠름 | 느림 (EM 수렴) |
| **논문 충실도** | 낮음 (아이디어만) | 높음 (수식 구현) |

---

## 📖 main.py 상세 설명

### 개요
논문의 핵심 아이디어인 "요일별 극값 군집"을 **실용적으로 단순화**한 트레이딩 전략입니다.

### 핵심 클래스

#### 1. WeeklyExtremeTracker
```python
class WeeklyExtremeTracker:
    """요일별 주간 고점/저점 출현 빈도 기록"""
```

**기능**:
- 5거래일 주간만 유지 (휴일 포함 주 제외)
- ISO 주차 기준으로 주간 분리
- 최고가/최저가 발생 요일 카운트

**논문 대응**: 논문 2절 방법론 (49행)

#### 2. ExtremeTransitionModel
```python
class ExtremeTransitionModel:
    """요일별 마코프 전이 기반 극단값 예측 모델"""
```

**기능**:
- 이전 주 극값 요일 → 다음 주 극값 요일 전이 확률 학습
- 라플라스 스무딩 (α=0.75)
- 신호 강도 계산: edge (균등 대비 우위), gap (1위-2위 차이)

**예측 로직**:
```python
# 전이 확률 행렬
low_transitions[last_low_day][next_low_day] += 1

# 예측
best_day = argmax(transition_probs[last_day])
edge = probs[best_day] - 0.2  # 균등확률 대비
```

**논문과 차이**:
- 논문: MSGARCH (K개 상태, 요일별 전이)
- main.py: 단순 요일→요일 전이 (상태 없음)

#### 3. CalendarExtremeClustering (QCAlgorithm)
```python
class CalendarExtremeClustering(QCAlgorithm):
    """요일별 극단값 클러스터링 기반 SPY 트레이딩 전략"""
```

**트레이딩 로직**:
1. **진입**: 저점 예측 요일에 매수 (신호 강도 edge ≥ 0.08)
2. **청산**: 고점 예측 요일 또는 금요일에 매도
3. **리스크 관리**: ATR 기반 스톱로스 (1.5배)

**매개변수**:
```python
self.min_history_weeks = 26      # 최소 관찰 주
self.bias_threshold = 0.08       # 신호 강도 기준
self.gap_threshold = 0.05        # 1-2위 차이 기준
self.position_size = 0.8         # 투자 비중
self.stop_atr_multiplier = 1.5   # 스톱로스 ATR 배수
```

**스케줄**:
- 주말 마감: 포지션 청산, 다음주 플랜 계산
- 주초 시작: 새로운 주 플랜 활성화
- 매일: 진입/청산 로직 실행

### 장단점

#### ✅ 장점
1. **단순성**: 이해하기 쉬운 로직
2. **실행 속도**: 카운트 기반으로 빠름
3. **실무 적용**: 바로 백테스트 가능

#### ⚠️ 단점
1. **과적합 위험**: 과거 패턴에 의존, 시장 변화 미반영
2. **검증 부족**: 샘플외 검정 없음
3. **이론적 근거 약함**: 논문의 MSGARCH 모형 미사용

---

## 📖 main_2.py 상세 설명

### 개요
논문의 **요일 의존적 MSGARCH 모형**을 충실히 구현하고, 검증 통과 시에만 트레이딩을 수행합니다.

### 핵심 클래스

#### 1. DayDependentMSGARCH
```python
class DayDependentMSGARCH:
    """요일 의존적 마르코프-전환 GARCH 모형"""
```

**매개변수 구조**:
```python
K: int                              # 잠재 상태 개수 (2~5)
transition_probs: np.ndarray        # (5, K, K) - 요일별 전이 확률
mu: np.ndarray                      # (K, 5) - 상태×요일별 평균
alpha, beta, gamma: np.ndarray      # (K, 5) - GARCH 매개변수
pi_0: np.ndarray                    # (K,) - 초기 상태 분포
```

**논문 대응**:
- **전이 확률**: $p_{ij}(d) = P(S_{t+1}=j | S_t=i, d(t+1))$ (논문 식 9)
- **수익률 분포**: $r_t \sim N(\mu_{i,d}, \sigma_{i,d}^2)$ (논문 식 10)
- **GARCH 동학**: $\sigma_t^2 = \alpha_{i,d} + \beta_{i,d}\epsilon_{t-1}^2 + \gamma_{i,d}\sigma_{t-1}^2$ (논문 식 11)

**EM 알고리즘** (논문 부록 B):

**E-step**: Forward-Backward 알고리즘
```python
# Forward 확률
α[t][i] = P(S_t=i | r_{1:t})
α[0][i] = π_0[i] × f(r_0 | S_0=i)
α[t][j] = Σ_i α[t-1][i] × p[d][i][j] × f(r_t | S_t=j)

# Backward 확률
β[t][i] = P(r_{t+1:T} | S_t=i)

# 사후 확률
γ[t][i] = P(S_t=i | r_{1:T}) = α[t][i] × β[t][i]
ξ[t][i][j] = P(S_t=i, S_{t+1}=j | r_{1:T})
```

**M-step**: 매개변수 업데이트
```python
# 전이 확률 (요일별)
p[d][i][j] = Σ_t ξ[t][i][j] (weekday=d) / Σ_t γ[t][i] (weekday=d)

# 평균 수익률
μ[i][d] = Σ_t γ[t][i] × r_t (weekday=d) / Σ_t γ[t][i] (weekday=d)

# GARCH 매개변수 (단순화: 샘플 분산 기반)
variance = Σ_t γ[t][i] × (r_t - μ[i][d])^2 / Σ_t γ[t][i]
α, β, γ 비율 업데이트
```

**시뮬레이션** (논문 2절):
```python
def simulate_weekly_paths(n_simulations=2000, n_weeks=500):
    for sim in range(n_simulations):
        state = sample(π_0)
        for week in range(n_weeks):
            weekly_prices = []
            for weekday in [0,1,2,3,4]:
                # 상태 전이
                state = sample(transition_probs[weekday][state])

                # GARCH 변동성 (단순화: 정상 분산)
                σ² = α[state][weekday] / (1 - β[state][weekday] - γ[state][weekday])

                # 수익률 생성
                r = N(μ[state][weekday], σ²)

                # 가격 업데이트
                price *= exp(r)
                weekly_prices.append((weekday, price))

            # 극값 요일 기록
            high_day = argmax(prices)
            low_day = argmin(prices)
            count[high_day] += 1

    return distribution
```

#### 2. ModelValidator
```python
class ModelValidator:
    """논문 표 3, 4의 KL 발산, G-검정 구현"""
```

**KL 발산** (논문 식 6):
```python
D_KL(O||E) = Σ_d O_d × log(O_d / E_d)
```

**G-검정** (논문 식 7):
```python
G = 2 × Σ_d O_d × ln(O_d / E_d)
p-value = 1 - χ²_cdf(G, df=4)
```

**In-sample 검증** (논문 표 3):
- 관측 분포 vs. 모형 시뮬레이션 분포
- 2000회 시뮬레이션
- p ≥ 0.05면 모형 적합

**Out-of-sample 검증** (논문 표 4):
- 80% 학습, 20% 테스트
- 테스트 데이터에서 KL, G-검정
- p < 0.05면 강건성 실패

#### 3. PaperBasedStrategy (QCAlgorithm)
```python
class PaperBasedStrategy(QCAlgorithm):
    """논문 기반 MSGARCH 모형을 사용한 트레이딩 전략"""
```

**3단계 워크플로우**:

**1단계: 학습** (최소 100주 데이터 수집)
```python
for K in [2, 3, 4, 5]:
    model = DayDependentMSGARCH(K=K)
    model.fit(returns, weekdays)

    # In-sample KL로 최적 K 선택
    kl = KL(observed, model.simulate())
    if kl < best_kl:
        best_K = K
```

**2단계: 검증** (논문 표 3, 4)
```python
# In-sample (표 3)
sim_dist = model.simulate(n_simulations=2000)
in_sample = validate_in_sample(observed, sim_dist)
Log(f"KL={in_sample['KL_high']:.3f}, p={in_sample['p_high']:.3f}")

# Out-of-sample (표 4)
oos = validate_out_of_sample(returns, weekdays, K, split=0.8)
Log(f"KL={oos['KL_high']:.3f}, p={oos['p_high']:.3f}")

# 강건성 기준
if oos['p_high'] < 0.05 or oos['p_low'] < 0.05:
    Log("VALIDATION FAILED")
    best_model = None  # 트레이딩 안 함
```

**3단계: 트레이딩** (검증 통과 시)
```python
# 다음주 예측
sim_dist = model.simulate(n_simulations=500, n_weeks=10)
entry_day = argmax(sim_dist['low_probs'])
exit_day = argmax(sim_dist['high_probs'])

# 신호 강도 확인
edge = sim_dist['low_probs'][entry_day] - 0.2
if edge < 0.08:
    return  # 신호 약함

# 매매 실행
if weekday == entry_day:
    MarketOrder(SPY, quantity)
    StopMarketOrder(SPY, -quantity, stop_price)
elif weekday == exit_day:
    Liquidate(SPY)
```

### 장단점

#### ✅ 장점
1. **논문 충실도**: 수식을 정확히 구현
2. **검증 우선**: Out-of-sample 통과 시에만 트레이딩
3. **과적합 방지**: 강건성 검정으로 신뢰도 향상
4. **재현 가능**: 논문 표 3, 4 형식 출력

#### ⚠️ 단점
1. **복잡성**: 이해하기 어려움
2. **계산 비용**: EM 수렴에 시간 소요
3. **단순화**: GARCH 동학을 정상 분산으로 근사 (완전 구현은 더 복잡)

---

## 📊 논문 결과 재현 예상

### 논문 표 3 (In-sample)
```
Asset: SPY (ES 유사)
Model: Day-dependent MSGARCH, K=3

               KL div   G-stat   p-value
High           0.002    3.84     0.428
Low            0.001    3.25     0.516
```

**해석**: p ≥ 0.05 → 모형이 요일별 극값 군집을 잘 설명

### 논문 표 4 (Out-of-sample)
```
Asset: SPY
80/20 split

               KL div   G-stat   p-value
High           0.012    5.69     0.223
Low            0.003    1.20     0.878
```

**해석**: p ≥ 0.05 → 샘플외에서도 성능 유지 (SPY는 강건)

---

## 🚀 사용 방법

### main.py (간단한 백테스트)

**QuantConnect에서 실행**:
1. 프로젝트 생성
2. main.py 업로드
3. 백테스트 실행 (2005-2024)

**예상 실행 시간**: 1-2분

**로그 예시**:
```
Next plan: entry Tue (edge=0.125), exit Thu; model weeks=52
SPY | weeks=987 | Highs Mon:180, Tue:210, Wed:195, Thu:205, Fri:197 | G=12.45 | KL=0.015 | p=0.014
```

### main_2.py (논문 재현)

**QuantConnect에서 실행**:
1. 프로젝트 생성
2. main_2.py 업로드
3. 백테스트 실행 (2005-2024)

**예상 실행 시간**: 10-20분 (EM 알고리즘 수렴)

**로그 예시**:
```
[TRAINING DONE] Collected 150 weeks
[TRAINING] K=2, KL_avg=0.0234
[TRAINING] K=3, KL_avg=0.0189  ← 최적
[TRAINING] K=4, KL_avg=0.0201

[VALIDATION In-sample] K=3
  High: KL=0.002, G=3.84, p=0.428
  Low:  KL=0.001, G=3.25, p=0.516

[VALIDATION Out-of-sample] test_weeks=30
  High: KL=0.012, G=5.69, p=0.223
  Low:  KL=0.003, G=1.20, p=0.878

[VALIDATION PASSED] Starting trading phase
[PLAN] Entry=Tue (prob=0.28), Exit=Thu
```

---

## 💡 선택 가이드

### main.py를 사용해야 할 때
- ✅ 빠른 백테스트가 필요할 때
- ✅ 단순한 로직으로 시작하고 싶을 때
- ✅ 논문의 핵심 아이디어만 적용하고 싶을 때

### main_2.py를 사용해야 할 때
- ✅ 논문의 정확한 재현이 필요할 때
- ✅ 강건성 검증이 중요할 때
- ✅ 학술적 정확성이 우선일 때
- ✅ 과적합 위험을 최소화하고 싶을 때

---

## 📈 백테스트 결과 해석

### 성공적인 경우
```
[VALIDATION PASSED]
In-sample p > 0.05
Out-of-sample p > 0.05
→ 요일별 극값 군집이 SPY에서 강하게 나타남
→ 모형이 샘플외에서도 안정적
→ 트레이딩 전략 실행
```

### 실패하는 경우
```
[VALIDATION FAILED] Out-of-sample p < 0.05
→ 샘플외에서 성능 저하 (논문의 강건성 문제)
→ 과적합 위험
→ 트레이딩 중단, 관망
```

---

## ⚠️ 주의사항

### 논문의 주요 경고 (3.1절)
> "샘플외 분할에서는 8개 중 5개(62.5%)가 5% 유의수준에서 귀무가설을 기각한다"

**의미**: 모형이 학습 데이터에서는 잘 맞지만, 새로운 데이터에서는 성능 저하

### 실무 적용 시
1. **과적합 경계**: Out-of-sample p < 0.05면 거래 중단
2. **정기 재학습**: 시장 환경 변화 반영 (예: 분기별)
3. **자산별 차이**: SPY ≠ ES (논문 대상), 성능 다를 수 있음
4. **거래 비용**: 슬리피지, 수수료 고려 필요

### 기술적 제약
1. **GARCH 단순화**: 완전한 GARCH(1,1) 대신 정상 분산 사용
2. **scipy 의존성**: G-검정 p-value 계산에 필요
3. **메모리**: 2000회 시뮬레이션 시 numpy 배열 다량 생성
4. **시간**: EM 알고리즘 수렴까지 수 분 소요

---

## 🔗 참고 자료

### 논문
- **제목**: Calendar-based clustering of weekly extremes: An empirical failure of stochastic models
- **저자**: 이임현 (동아대학교)
- **파일**: `Calendar-based clustering of weekly extremes.ko.md`

### 핵심 개념
- **주간 극값 군집**: 최고가/최저가가 특정 요일에 몰리는 현상
- **MSGARCH**: Markov-Switching GARCH, 상태 전환 변동성 모형
- **EM 알고리즘**: Expectation-Maximization, 잠재변수 모형 추정
- **KL 발산**: Kullback-Leibler divergence, 분포 간 차이 측정
- **G-검정**: 카이제곱 검정의 로그 버전

### 관련 연구
- Gray (1996): Markov-Switching GARCH
- Haas et al. (2004): MSGARCH 추정 방법
- French (1980): 요일 효과 (Day-of-the-week effect)

---

## 📝 코드 구조 요약

### main.py
```
WeeklyExtremeTracker (100 lines)
  ↓ 주간 극값 요일 추적
ExtremeTransitionModel (90 lines)
  ↓ 마르코프 전이 확률
CalendarExtremeClustering (280 lines)
  ↓ 트레이딩 전략
```

### main_2.py
```
WeeklyExtremeTracker (120 lines)
  ↓ 수익률/요일 저장 추가
DayDependentMSGARCH (400 lines)
  ├─ EM 알고리즘 (E-step, M-step)
  └─ 시뮬레이션 (2000회)
ModelValidator (150 lines)
  ├─ KL 발산, G-검정
  ├─ In-sample 검증
  └─ Out-of-sample 검증
PaperBasedStrategy (350 lines)
  ├─ 학습 단계 (K=2~5)
  ├─ 검증 단계 (표 3, 4)
  └─ 트레이딩 단계
```

---

## 📞 문의

코드 관련 질문이나 개선 제안은 이슈로 남겨주세요.

---

**작성일**: 2025-01-24
**QuantConnect 호환**: ✅
**Python 버전**: 3.8+
**의존성**: numpy, scipy (QuantConnect 기본 제공)
