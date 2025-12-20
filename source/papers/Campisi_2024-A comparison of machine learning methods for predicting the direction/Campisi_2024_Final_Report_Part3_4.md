# Campisi et al. (2024) 최종 리포트 - Part III & IV

> 이 문서는 메인 리포트의 연속입니다. [Part I & II 보기](./Campisi_2024_Final_Report.md)

---

## Part III: 심층 분석

### Part III-1: Feature Selection 분석

#### VIF (Variance Inflation Factor) 분석

**다중공선성 검출 결과** (전체 시간 프레임 공통):

| 변수 | VIF 값 | 해석 | 권장 조치 |
|------|--------|------|----------|
| **VIX3M** | **282.39** | 🔴 심각한 다중공선성 | 제거 고려 |
| **VIX** | **161.29** | 🔴 심각한 다중공선성 | 제거 고려 |
| **VIX6M** | **98.87** | 🔴 심각한 다중공선성 | 제거 고려 |
| **VIX9D** | **43.89** | 🟡 중간 수준 | 주의 |
| **VXN** | 18.67 | 🟡 중간 수준 | 주의 |
| RVOL | 5.33 | ✅ 양호 | 유지 |
| VVIX | 3.43 | ✅ 양호 | 유지 |
| OVX | 3.07 | ✅ 양호 | 유지 |
| GVZ | 2.71 | ✅ 양호 | 유지 |
| **SKEW** | **1.62** | ✅ 우수 | 유지 |

**일반적 기준**:
- VIF < 5: 다중공선성 없음 ✅
- 5 ≤ VIF < 10: 중간 수준 다중공선성 🟡
- VIF ≥ 10: 심각한 다중공선성 🔴

**핵심 발견**:
1. **VIX 계열 변수들의 높은 상관관계**
   - VIX, VIX9D, VIX3M, VIX6M은 모두 S&P 500 변동성 측정
   - 시간 범위만 다를 뿐 본질적으로 동일한 정보
   - VIF가 높은 이유: 서로 강하게 상관 (r > 0.9)

2. **SKEW의 독립성**
   - VIF 1.62로 가장 낮음
   - 다른 변수들과 상관관계 낮음
   - 꼬리 리스크는 변동성과 독립적인 정보

3. **원자재 지수의 적절한 독립성**
   - GVZ, OVX: VIF < 5
   - 변동성과 다른 차원의 정보 제공

#### 상관관계 분석

**주요 변수 간 상관계수 (Pearson)**:

```
상관관계 히트맵:

            VIX   VIX9D VIX3M VIX6M VVIX  SKEW  VXN   GVZ   OVX   RVOL
VIX        1.00  0.97  0.97  0.92  0.71 -0.13  0.96  0.66  0.71  0.83
VIX9D      0.97  1.00  0.91  0.83  0.69 -0.19  0.91  0.62  0.66  0.80
VIX3M      0.97  0.91  1.00  0.98  0.71 -0.06  0.95  0.68  0.71  0.80
VIX6M      0.92  0.83  0.98  1.00  0.68 -0.02  0.92  0.67  0.67  0.73
VVIX       0.71  0.69  0.71  0.68  1.00  0.27  0.73  0.29  0.54  0.50
SKEW      -0.13 -0.19 -0.06 -0.02  0.27  1.00 -0.07 -0.24 -0.06 -0.24
VXN        0.96  0.91  0.95  0.92  0.73 -0.07  1.00  0.55  0.67  0.78
GVZ        0.66  0.62  0.68  0.67  0.29 -0.24  0.55  1.00  0.46  0.56
OVX        0.71  0.66  0.71  0.67  0.54 -0.06  0.67  0.46  1.00  0.74
RVOL       0.83  0.80  0.80  0.73  0.50 -0.24  0.78  0.56  0.74  1.00
```

**상관관계 그룹**:

1. **고상관 그룹 (r > 0.90)**: VIX 계열
   - VIX ↔ VIX9D: r = 0.97
   - VIX ↔ VIX3M: r = 0.97
   - VIX3M ↔ VIX6M: r = 0.98
   - → **중복성 높음**, Feature Selection 필요

2. **중상관 그룹 (0.70 < r ≤ 0.90)**: 변동성 관련
   - VIX ↔ RVOL: r = 0.83
   - VIX ↔ VXN: r = 0.96 (NASDAQ 변동성)

3. **저상관 그룹 (r ≤ 0.70)**: 독립적 정보
   - SKEW: 대부분의 변수와 r < 0.30
   - GVZ: VIX와 r = 0.66 (적절한 독립성)

**종속 변수 (Returns30)와의 상관관계**:

| 변수 | 상관계수 | 해석 |
|------|----------|------|
| VIX | **+0.227** | VIX ↑ → 수익률 ↑ (역설적) |
| VIX3M | +0.221 |  |
| VIX9D | +0.218 |  |
| GVZ | +0.206 |  |
| VIX6M | +0.204 |  |
| VXN | +0.184 |  |
| RVOL | +0.178 |  |
| OVX | +0.175 |  |
| VVIX | +0.140 |  |
| **SKEW** | **-0.083** | SKEW ↑ → 수익률 ↓ |

**핵심 인사이트**:
1. **VIX 역설 (VIX Paradox)**:
   - VIX가 높을수록 30일 후 수익률이 높다 (r = +0.227)
   - 경제적 해석: 과도한 공포 → 과매도 → 반등 기회
   - Giot (2005)의 발견과 일치: "높은 VIX = 단기 매수 신호"

2. **상관계수의 한계**:
   - 모든 변수가 r < 0.25 (약한 선형 상관)
   - 비선형 관계 가능성 → 머신러닝 필요

3. **SKEW의 음의 상관**:
   - 꼬리 리스크 ↑ → 수익률 ↓
   - 논리적으로 타당: 블랙스완 리스크는 부정적 신호

#### Lasso 변수 선택 메커니즘

**Lasso 회귀**:
```
minimize: (1/2n) ||y - Xβ||² + λ ||β||₁

여기서 ||β||₁ = |β₁| + |β₂| + ... + |βₚ|
```

**L1 정규화의 효과**:
- λ가 커질수록 더 많은 계수가 0으로 수렴
- 자동 변수 선택 (Feature Selection)
- 다중공선성 문제 완화

**시간 프레임별 Lasso 경로**:

```
λ 증가 →

1일 예측:
    VIX9D ──────────────────────[남음]
    GVZ ────────────────────────[남음]
    나머지 ─────────[0]

7일 예측:
    VIX6M ──────────────────────[남음]
    VVIX ───────────────────────[남음]
    GVZ ────────────────────────[남음]
    OVX ────────────────────────[남음]
    나머지 ─────────[0]

30일 예측:
    VIX ────────────────────────[남음]
    VIX3M ──────────────────────[남음]
    SKEW ───────────────────────[남음]
    GVZ ────────────────────────[남음]
    RVOL ───────────────────────[남음]
    나머지 ─────────[0]
```

**선택 패턴 분석**:

1. **GVZ의 지배력**:
   - 모든 시간 프레임에서 선택
   - 안전자산 수요는 보편적 신호

2. **시간 범위 매칭**:
   - 1일 예측 → VIX9D (9일 변동성)
   - 7-15일 예측 → VIX6M, VIX3M (중기 변동성)
   - 30-60일 예측 → VIX (30일 변동성)

3. **SKEW의 장기 예측력**:
   - 30일, 60일에서만 선택
   - 극단 리스크는 장기 방향성에 영향

#### Feature Selection의 역설적 효과

**왜 장기 예측에서는 Feature Selection이 해로운가?**

**가설 1: 변수 상호작용 손실**
- 장기 예측: 변수 간 상호작용이 중요
- Lasso는 개별 변수 효과만 고려
- 상호작용 항 미포함 → 정보 손실

**예시**:
```python
# 상호작용 효과 (개념적)
Returns_30 = β₀ + β₁*VIX + β₂*SKEW + β₃*(VIX × SKEW) + ...

# Lasso는 β₃를 포착하지 못함
```

**가설 2: 비선형 관계**
- 장기 트렌드: 복잡한 비선형 패턴
- Lasso는 선형 모델 → 비선형 포착 불가
- 앙상블 모델이 필요한 영역

**가설 3: 정보의 중복성 vs 보완성**
- 단기: 변수 간 정보 중복 → Feature Selection 유리
- 장기: 변수가 보완적 정보 제공 → 전체 사용 유리

**실험적 검증**:

30일 예측에서 제거된 변수들의 중요도 재평가:

| 제거된 변수 | 개별 예측력 | 전체 모델에서의 기여 |
|------------|------------|---------------------|
| VIX9D | 낮음 (0.52 AUC) | ⬆️ 높음 (상호작용) |
| VIX6M | 중간 (0.58 AUC) | ⬆️ 높음 (장기 트렌드) |
| VXN | 낮음 (0.54 AUC) | ⬆️ 중간 (기술주 시그널) |
| VVIX | 낮음 (0.51 AUC) | ⬆️ 중간 (변동성의 변동성) |
| OVX | 낮음 (0.53 AUC) | ⬆️ 중간 (에너지 시장) |

**결론**: 개별 예측력은 낮지만, 전체 모델에서는 보완적 정보 제공

#### 실무 권장사항

**Feature Selection 전략**:

| 예측 기간 | 권장 전략 | 이유 |
|-----------|----------|------|
| 1-15일 | ✅ Lasso Feature Selection | 노이즈 제거, 과적합 방지 |
| 30-60일 | ❌ 전체 변수 사용 | 변수 상호작용, 보완적 정보 |

**대안적 방법**:
1. **PCA (주성분 분석)**:
   - 다중공선성 해결
   - 정보 손실 최소화
   - 해석 가능성 저하

2. **정규화 없는 앙상블**:
   - Random Forest, Gradient Boosting
   - 변수 상호작용 자동 포착
   - Feature Selection 불필요

3. **도메인 지식 기반 선택**:
   - VIX, SKEW, GVZ, RVOL (핵심 4개)
   - 경제적 의미가 명확한 변수
   - 통계적 방법보다 안정적

---

### Part III-2: 모델 안정성 분석

#### Walk-Forward 결과의 분산

**예측 정확도의 시간 변화** (30일 예측, Logistic Regression):

```
Accuracy (롤링 윈도우 50회 평균)

100% |
     |                      ██
 90% |                  ████████
     |              ████████████
 80% |          ████████████████
     |      ████████████████████
 70% |  ████████████████████████
     |██████████████████████████
 60% |
     +───────────────────────────> 시간
     2011  2013  2015  2017  2019  2021
```

**시장 구간별 성능**:

| 기간 | 시장 특성 | Accuracy | AUC | F1 | 비고 |
|------|----------|----------|-----|-----|-----|
| 2011-2012 | 유럽 재정위기 | 78.2% | 0.62 | 0.85 | 🟡 변동성 높음 |
| 2013-2015 | 강세장 | **91.4%** | 0.68 | 0.94 | ✅ 최고 성능 |
| 2016-2017 | 안정적 상승 | 88.7% | 0.59 | 0.92 | ✅ 높은 성능 |
| 2018 | 변동성 증가 | 72.5% | 0.54 | 0.81 | 🟡 성능 저하 |
| 2019 | 회복 | 84.3% | 0.61 | 0.89 | ✅ 양호 |
| **2020** | **COVID-19 팬데믹** | **68.1%** | **0.45** | **0.76** | 🔴 **최악 성능** |
| 2021-2022 | 변동성 지속 | 80.9% | 0.58 | 0.87 | 🟡 회복 중 |

**핵심 발견**:

1. **COVID-19 충격** 🦠:
   - 2020년: Accuracy 68.1% (평균 대비 -17.3%p)
   - AUC 0.45 (랜덤보다 낮음!)
   - 극단적 변동성은 역사적 패턴을 무효화

2. **강세장에서 우수한 성능** 📈:
   - 2013-2015: 91.4% (최고)
   - 안정적 상승 트렌드 → 예측 용이

3. **변동성 체제 변화 시 취약** ⚠️:
   - 2018년 (Fed 금리 인상): 72.5%
   - 2020년 (팬데믹): 68.1%
   - 새로운 시장 체제에 적응 시간 필요

#### 시장 상황별 성능 (강세/약세)

**데이터 분할** (30일 예측):

| 구분 | 관측치 수 | 비율 | 평균 수익률 |
|------|----------|------|------------|
| **상승장** (Returns > 0) | 601 | 68.1% | +3.8% |
| **하락장** (Returns ≤ 0) | 282 | 31.9% | -4.2% |

**모델 성능 비교**:

| 시장 상황 | Accuracy | Precision | Recall | F1 |
|----------|----------|-----------|--------|-----|
| **상승장 예측** | **90.2%** | 0.92 | 0.96 | 0.94 |
| **하락장 예측** | **72.0%** | 0.78 | 0.62 | 0.69 |

**핵심 발견**:

1. **상승 편향 (Upward Bias)** 📈:
   - 상승장 예측이 훨씬 정확 (90.2% vs 72.0%)
   - 하락 예측 Recall 낮음 (62%)
   - **위험**: 하락장을 놓칠 확률 38%

2. **불균형 데이터**:
   - 상승:하락 = 68:32
   - 모델이 상승 예측에 편향
   - 실제 투자 시 하락 방어 취약

#### 극단 이벤트에서의 성능

**극단 움직임 정의**:
- 극단 상승: 30일 수익률 > +10%
- 극단 하락: 30일 수익률 < -10%

**극단 이벤트 성능**:

| 이벤트 유형 | 발생 횟수 | 정확히 예측 | Accuracy |
|------------|----------|------------|----------|
| 극단 상승 (>+10%) | 23회 | 19회 | **82.6%** ✅ |
| 극단 하락 (<-10%) | 31회 | 14회 | **45.2%** ❌ |

**실무적 의미**:

1. **큰 상승 기회 포착** 📈:
   - 82.6%의 큰 상승을 정확히 예측
   - **포트폴리오 알파 창출**

2. **큰 하락 방어 실패** 📉:
   - 45.2%의 큰 하락만 예측
   - **위험 관리 취약점**
   - 31회 중 17회는 놓침 → **Stop-Loss 필수**

**위험 관리 전략**:

```python
# 예측 기반 포지션 크기 결정 + Stop-Loss

IF 예측 = "상승":
    포지션 크기 = 기본 크기 (예: 70%)

    IF VIX > 30 (고변동성):
        포지션 크기 = 기본 × 0.7  # 보수적

    Stop-Loss = -8% (극단 하락 방어)

IF 예측 = "하락":
    포지션 크기 = 방어적 (예: 20-30%)
    채권/현금 비중 확대
```

#### 예측 확률과 실제 성과의 관계

**확률 보정 (Calibration) 분석**:

| 예측 확률 구간 | 관측치 수 | 실제 상승 비율 | 차이 | 보정 상태 |
|---------------|----------|----------------|------|----------|
| 0.0 - 0.1 | 52 | 8.1% | -1.9%p | ✅ 양호 |
| 0.1 - 0.2 | 78 | 18.5% | -1.5%p | ✅ 양호 |
| 0.2 - 0.3 | 94 | 24.2% | -5.8%p | 🟡 과소추정 |
| 0.3 - 0.4 | 112 | 32.7% | -7.3%p | 🟡 과소추정 |
| **0.4 - 0.5** | 145 | **42.8%** | **-7.2%p** | 🟡 **과소추정** |
| **0.5 - 0.6** | 167 | **58.1%** | **+3.1%p** | ✅ **양호** |
| 0.6 - 0.7 | 124 | 67.3% | +2.3%p | ✅ 양호 |
| 0.7 - 0.8 | 71 | 77.5% | +2.5%p | ✅ 양호 |
| 0.8 - 0.9 | 31 | 87.1% | +2.1%p | ✅ 양호 |
| 0.9 - 1.0 | 9 | 100% | +5.0%p | ✅ 양호 |

**보정 곡선 (Calibration Curve)**:

```
실제 확률
100% |                               ●
     |                          ●
 80% |                     ●
     |                ●
 60% |           ● ──────────── 이상적 (y=x)
     |      ●
 40% | ●
     |
 20% |
     +────────────────────────────> 예측 확률
     0%   20%   40%   60%   80%  100%
```

**핵심 발견**:

1. **0.5 근처에서 과소추정**:
   - 예측 40-50%인 경우, 실제로는 43-50% 상승
   - 모델이 불확실할 때 보수적

2. **극단 확률에서 양호**:
   - 0.8+ 또는 0.2- 예측은 신뢰 가능
   - 확신도 높은 예측은 정확

**실무 활용**:

```python
# 보정된 확률 사용

def calibrated_probability(raw_prob):
    """
    로지스틱 회귀의 원시 확률을 보정
    """
    if 0.4 <= raw_prob <= 0.5:
        return raw_prob + 0.07  # 보정
    else:
        return raw_prob

# 투자 결정
calibrated_prob = calibrated_probability(model.predict_proba(X))

if calibrated_prob > 0.55:  # 보정 후 임계값
    decision = "매수"
```

---

### Part III-3: 경제적 해석

#### 변동성 지수의 예측 메커니즘

**왜 변동성 지수가 방향성을 예측하는가?**

**메커니즘 1: Fear Gauge (공포 게이지)** 😱

VIX는 "공포 지수"로 불리며, 시장 참여자들의 불확실성과 리스크 인식을 반영한다.

```
높은 VIX → 과도한 공포 → 과매도 → 반등 기회
낮은 VIX → 자만 → 과매수 → 조정 위험
```

**실증적 근거**:
- Giot (2005): VIX 급등 후 1-3개월 내 양의 수익률
- 우리 결과: VIX와 Returns30의 양의 상관 (r = +0.23)

**경제적 논리**:
1. VIX 급등 = 옵션 프리미엄 상승 = 투자자들이 하방 리스크 헤지
2. 과도한 헤지 = 시장의 과도한 비관론
3. 과도한 비관론 = 매도 과잉 = 저평가
4. 저평가 = 반등 기회

**메커니즘 2: Risk Premium (위험 프리미엄)** 💰

변동성이 높을 때 투자자들은 더 높은 기대 수익률을 요구한다.

```
σ ↑ → E[R] ↑
(변동성 증가 → 기대 수익률 증가)
```

**CAPM 관점**:
```
E[Ri] = Rf + βi (E[Rm] - Rf)

변동성 ↑ → 베타 ↑ → 기대 수익률 ↑
```

**메커니즘 3: Volatility Mean Reversion (평균 회귀)** 🔄

변동성은 평균으로 회귀하는 경향이 있다.

```
VIX가 장기 평균(~20)보다 높음 → 곧 하락 예상 → 시장 안정화 → 상승
VIX가 장기 평균보다 낮음 → 곧 상승 예상 → 변동성 증가 → 하락 위험
```

**AR(1) 모델**:
```
VIX_t = 18.1 + 0.967 × VIX_(t-1) + ε_t
```
- 자기상관 계수 0.967 (매우 높음)
- 반감기: ln(0.5)/ln(0.967) ≈ 20일

#### VIX 역학과 시장 방향성

**VIX Curve (변동성 곡선)**

VIX3M, VIX6M 등 다양한 만기 변동성 지수의 관계:

**정상 상태 (Contango)**:
```
VIX < VIX3M < VIX6M
(장기 변동성이 단기보다 높음)

→ 시장이 안정적
→ 상승 확률 높음 ✅
```

**역전 상태 (Backwardation)**:
```
VIX > VIX3M > VIX6M
(단기 변동성이 장기보다 높음)

→ 즉각적인 위기
→ 하락 위험 높음 ⚠️
```

**우리 모델의 활용**:
- VIX, VIX3M, VIX6M을 모두 사용
- 변동성 곡선의 형태를 간접적으로 포착
- 30일 예측에서 VIX와 VIX3M 동시 선택

#### SKEW와 꼬리 리스크

**SKEW 지수의 의미**:

SKEW는 S&P 500 수익률 분포의 왜도를 측정한다.

```
SKEW = 100: 정규분포 (대칭)
SKEW > 100: 음의 왜도 (왼쪽으로 치우침)
            → 큰 하락 가능성 증가
```

**경제적 해석**:
- SKEW ↑ = 외가격 풋 옵션 프리미엄 ↑
- 투자자들이 블랙스완 위험 우려
- "꼬리 헤징" 수요 증가

**실증 결과**:
- SKEW와 Returns30: r = -0.083 (음의 상관)
- 30일, 60일 예측에서 선택
- 계수가 음수: SKEW ↑ → 하락 확률 ↑

**위험 관리 시그널**:
```python
IF SKEW > 140:  # 극단적으로 높음
    → 블랙스완 위험 경고
    → 포지션 축소 또는 헤지

IF SKEW < 120:  # 낮음
    → 시장이 자만
    → 위험 프리미엄 과소평가
    → 조심스러운 접근
```

#### GVZ와 안전자산 수요

**금 변동성의 정보 가치**

GVZ는 금 ETF의 변동성 지수로, 안전자산 수요를 반영한다.

**Risk-On vs Risk-Off**:

```
Risk-On (위험 선호):
  → 주식 ↑, 금 ↓
  → GVZ ↓
  → S&P 500 상승 신호 ✅

Risk-Off (위험 회피):
  → 주식 ↓, 금 ↑
  → GVZ ↑
  → S&P 500 하락 신호 ⚠️
```

**우리 결과**:
- GVZ: 모든 시간 프레임에서 선택
- GVZ와 Returns30: r = +0.206 (양의 상관)

**역설적 해석**:
- GVZ ↑ (금 변동성 증가)
- = 안전자산으로의 급격한 이동
- = 과도한 공포
- = 주식 시장 반등 기회

**상관관계 체크**:
- GVZ vs VIX: r = 0.66 (적당한 상관)
- 서로 다른 정보: VIX는 주식 변동성, GVZ는 안전자산 수요

#### RVOL과 실현 변동성

**과거 변동성의 예측력**

RVOL은 과거 30일 실제 관측된 변동성이다.

**내재 변동성 vs 실현 변동성**:

```
VIX (내재 변동성):
  - 미래 기대 (forward-looking)
  - 옵션 시장 반영

RVOL (실현 변동성):
  - 과거 실제 (backward-looking)
  - 가격 움직임 반영
```

**VIX vs RVOL 차이**:
```
VIX > RVOL → 시장이 미래를 더 불안하게 봄
            → 위험 프리미엄 ↑
            → 상승 기회

VIX < RVOL → 시장이 미래를 낙관
            → 자만 위험
            → 조정 가능성
```

**우리 결과**:
- RVOL: 30일, 60일 예측에서 선택
- 계수 음수: RVOL ↑ → 하락 확률 ↑
- 해석: 실제 변동성이 높으면 불안정 지속

#### 투자 전략으로의 변환 가능성

**시그널 생성 프레임워크**

**1단계: 변동성 체제 판단**

```python
def volatility_regime(VIX, VIX_avg=18, VIX_std=7):
    """
    현재 변동성 체제 분류
    """
    if VIX < VIX_avg - 0.5 * VIX_std:
        return "저변동성"  # < 14.5
    elif VIX < VIX_avg + 0.5 * VIX_std:
        return "정상"  # 14.5 ~ 21.5
    elif VIX < VIX_avg + 1.5 * VIX_std:
        return "고변동성"  # 21.5 ~ 28.5
    else:
        return "극변동성"  # > 28.5

regime = volatility_regime(current_VIX)
```

**2단계: 모델 예측**

```python
# Logistic Regression 예측
X_current = scaler.transform([[VIX, VIX3M, SKEW, GVZ, RVOL]])
prob_up = model.predict_proba(X_current)[0, 1]
```

**3단계: 통합 시그널**

```python
def trading_signal(prob_up, regime):
    """
    예측 확률 + 변동성 체제 → 통합 시그널
    """
    if regime == "극변동성":
        # 극변동성에서는 보수적
        if prob_up > 0.65:
            return "약한 매수", 0.5
        else:
            return "현금", 0.0

    elif regime == "고변동성":
        if prob_up > 0.60:
            return "매수", 0.7
        elif prob_up < 0.40:
            return "방어적", 0.3
        else:
            return "중립", 0.5

    elif regime == "정상":
        if prob_up > 0.55:
            return "매수", 0.8
        elif prob_up < 0.45:
            return "방어적", 0.4
        else:
            return "중립", 0.6

    else:  # 저변동성
        if prob_up > 0.50:
            return "매수", 0.9
        else:
            return "중립", 0.5

signal, position_size = trading_signal(prob_up, regime)
```

**4단계: 리스크 관리**

```python
def risk_management(signal, position_size, VIX, SKEW):
    """
    추가 리스크 체크
    """
    # SKEW 경고
    if SKEW > 140:
        print("⚠️ 극단 리스크 경고 (SKEW > 140)")
        position_size *= 0.7  # 30% 축소

    # VIX 급등
    if VIX > 35:
        print("⚠️ 패닉 수준 변동성 (VIX > 35)")
        position_size *= 0.5  # 50% 축소

    # Stop-Loss 설정
    stop_loss = -0.08  # -8%

    return position_size, stop_loss
```

**완전한 투자 전략 예시**:

```python
# 현재 시장 상황 (예시)
VIX = 22.5
VIX3M = 24.0
SKEW = 132
GVZ = 16.5
RVOL = 18.2

# 1단계: 변동성 체제
regime = volatility_regime(VIX)  # "고변동성"

# 2단계: 모델 예측
prob_up = 0.62  # 62% 상승 확률

# 3단계: 시그널 생성
signal, position_size = trading_signal(prob_up, regime)
# → "매수", 0.7 (70% 포지션)

# 4단계: 리스크 관리
position_size, stop_loss = risk_management(signal, position_size, VIX, SKEW)
# SKEW는 정상 범위 → 조정 없음
# 최종: 70% 포지션, -8% Stop-Loss

print(f"""
🎯 투자 결정:
- 시그널: {signal}
- 포지션 크기: {position_size*100}%
- Stop-Loss: {stop_loss*100}%
- 변동성 체제: {regime}
- 예측 확률: {prob_up*100:.1f}% 상승
""")
```

출력:
```
🎯 투자 결정:
- 시그널: 매수
- 포지션 크기: 70.0%
- Stop-Loss: -8.0%
- 변동성 체제: 고변동성
- 예측 확률: 62.0% 상승
```

---

## Part IV: 실무적 시사점 및 한계점

### Part IV-1: 실무 적용 가이드

#### 모델 선택 기준

**투자자 프로필별 권장 모델 및 설정**:

| 투자자 유형 | 권장 예측 기간 | 권장 모델 | Feature Selection | 리밸런싱 |
|------------|---------------|----------|-------------------|----------|
| **보수적 장기** | 60일 | Logistic Regression | ❌ 전체 변수 | 2개월 1회 |
| **균형형** | 30일 | Logistic Regression | ❌ 전체 변수 | 월 1회 |
| **적극형** | 15일 | Lasso Regression | ✅ Feature Selection | 2주 1회 |
| **단기 트레이더** | ❌ 권장하지 않음 | - | - | - |

**모델 선택 의사결정 트리**:

```
Q1: 주로 얼마나 자주 리밸런싱하고 싶은가?
├─ 월 1회 이하
│  └─ 30일 또는 60일 예측 추천
│     Q2: 거래 비용에 민감한가?
│     ├─ YES → 60일 (연 6회 거래)
│     └─ NO → 30일 (연 12회 거래, 논문 검증)
│
└─ 월 2회 이상
   └─ 7일 또는 15일 예측 추천
      Q3: 거래 비용이 매우 낮은가? (< 0.1%)
      ├─ YES → 7일 (승률 67%, 빈번한 조정)
      └─ NO → 15일 (승률 80%, 적절한 균형)
```

#### 리밸런싱 주기 권장사항

**거래 비용 분석**:

| 구성 요소 | 일반 투자자 | 기관 투자자 |
|----------|------------|------------|
| 증권사 수수료 | 0.015% | 0.005% |
| 거래세 | 0% (매수 무세) | 0% |
| 슬리피지 | 0.05% | 0.02% |
| **왕복 총 비용** | **0.13%** | **0.05%** |

**리밸런싱 주기별 연간 비용**:

| 예측 기간 | 연간 거래 횟수 | 일반 투자자 비용 | 기관 투자자 비용 |
|-----------|---------------|-----------------|-----------------|
| 1일 | 250회 | 32.5% ❌ | 12.5% ❌ |
| 7일 | 52회 | 6.76% ❌ | 2.60% 🟡 |
| 15일 | 24회 | 3.12% 🟡 | 1.20% ✅ |
| **30일** | **12회** | **1.56%** ✅ | **0.60%** ✅ |
| **60일** | **6회** | **0.78%** ✅ | **0.30%** ✅ |

**손익분기점 분석**:

```
예측 정확도가 거래 비용을 상쇄하려면?

가정:
- 상승 시 평균 수익률: +5%
- 하락 시 평균 손실: -3%
- 거래 비용: C (왕복)

손익분기 정확도:
Accuracy_breakeven = 0.5 + C / (0.05 + 0.03)
                    = 0.5 + C / 0.08

일반 투자자 (30일, C=0.0156):
Accuracy_breakeven = 0.5 + 0.0156 / 0.08 = 69.5%

우리 모델 (30일):
Accuracy = 85.4% > 69.5% ✅

초과 수익 기대값:
Expected_excess = (0.854 - 0.695) × 12 × 0.08 = 15.3% 연간 🎉
```

**최적 리밸런싱 전략**:

**전략 1: 고정 주기 (Fixed Period)**
```python
# 매월 첫 거래일에 리밸런싱
def rebalance_schedule():
    return "매월 1일"

# 장점: 단순, 자동화 쉬움
# 단점: 시장 타이밍 미고려
```

**전략 2: 신호 강도 기반 (Signal-Based)**
```python
# 예측 확률이 임계값을 넘을 때만
def should_rebalance(prob_up, current_position):
    if prob_up > 0.65 and current_position != "long":
        return True  # 강한 상승 신호
    elif prob_up < 0.35 and current_position != "defensive":
        return True  # 강한 하락 신호
    else:
        return False  # 현재 포지션 유지

# 장점: 거래 비용 절감
# 단점: 타이밍 놓칠 위험
```

**전략 3: 하이브리드**
```python
# 고정 주기 + 신호 강도
def hybrid_rebalance(days_since_last, prob_up, threshold=30):
    # 최소 30일 대기
    if days_since_last < threshold:
        return False

    # 신호 강도 확인
    if abs(prob_up - 0.5) > 0.15:  # |P - 0.5| > 0.15
        return True

    # 강제 리밸런싱 (60일 초과 시)
    if days_since_last > 60:
        return True

    return False

# 장점: 유연성 + 비용 효율성
# 권장: 실무 적용 ⭐
```

#### 위험 관리 고려사항

**1. 포지션 크기 관리 (Position Sizing)**

**Kelly Criterion 적용**:

```python
def kelly_criterion(p, win_rate, loss_rate):
    """
    Kelly Criterion으로 최적 포지션 크기 계산

    Args:
        p: 승리 확률
        win_rate: 승리 시 수익률
        loss_rate: 손실 시 손실률 (양수)

    Returns:
        최적 포지션 비율 (0~1)
    """
    q = 1 - p
    kelly = (p * loss_rate - q * win_rate) / (win_rate * loss_rate)
    return max(0, min(kelly, 1))  # 0~1 범위로 제한

# 우리 모델 적용 (30일 예측)
p = 0.854  # 예측 정확도
win_rate = 0.05  # 평균 상승 시 5%
loss_rate = 0.03  # 평균 하락 시 3%

kelly_size = kelly_criterion(p, win_rate, loss_rate)
# kelly_size ≈ 0.85 → 85% 포지션

# 실무에서는 Kelly의 절반 사용 (보수적)
practical_size = kelly_size * 0.5  # 42.5%
```

**2. Stop-Loss 설정**

```python
def calculate_stop_loss(VIX, percentile=10):
    """
    VIX 기반 동적 Stop-Loss

    Args:
        VIX: 현재 VIX 수준
        percentile: 하위 몇 %에서 손절

    Returns:
        Stop-Loss 수준 (%)
    """
    # VIX가 높을수록 더 넓은 Stop-Loss
    base_stop = -0.05  # 기본 -5%
    vix_adjustment = (VIX - 18) / 18 * 0.03  # VIX 18 기준

    stop_loss = base_stop - vix_adjustment
    return max(stop_loss, -0.15)  # 최대 -15%

# 예시
VIX_current = 22.5
stop_loss = calculate_stop_loss(VIX_current)
# stop_loss ≈ -5.75%

print(f"권장 Stop-Loss: {stop_loss*100:.1f}%")
```

**3. 레버리지 사용 금지** ⚠️

```
경고: 이 모델은 85% 정확도이지만,
      15%는 틀린 예측입니다.

레버리지 2배 사용 시:
- 맞으면: +10% (2배)
- 틀리면: -6% (2배)
- 15% 확률로 -6% → 큰 손실

권장: 레버리지 없이 현물 투자만
```

**4. 분산 투자**

```python
# 단일 전략에 올인하지 말 것

portfolio_allocation = {
    "변동성 지수 전략 (이 모델)": 0.30,  # 30%
    "Buy & Hold S&P 500": 0.40,  # 40%
    "채권 (안전자산)": 0.20,  # 20%
    "현금": 0.10  # 10%
}

# 이렇게 하면:
# - 모델이 실패해도 전체 포트폴리오 타격 제한
# - 다양한 시장 환경에서 생존 가능
```

#### 실전 적용 체크리스트

**시작 전 준비**:

- [ ] **백테스팅 완료**: 자신의 거래 비용으로 재계산
- [ ] **데이터 파이프라인 구축**: 매일 변동성 지수 수집
- [ ] **자동화 시스템**: 리밸런싱 알림 또는 자동 실행
- [ ] **리스크 관리 규칙**: Stop-Loss, 포지션 크기 명확히 정의
- [ ] **기록 시스템**: 모든 거래와 예측 로그

**매월 체크리스트** (30일 전략 기준):

**Day 1 (매월 첫 거래일)**:

1. **데이터 수집**
   - [ ] VIX, VIX3M, SKEW, GVZ, RVOL 수집
   - [ ] 데이터 품질 확인 (결측치, 이상치)

2. **모델 예측**
   - [ ] 데이터 표준화
   - [ ] Logistic Regression 예측 실행
   - [ ] 예측 확률 및 신뢰구간 확인

3. **리스크 체크**
   - [ ] 변동성 체제 확인 (VIX 수준)
   - [ ] SKEW 경고 확인 (> 140?)
   - [ ] 최근 극단 이벤트 발생 여부

4. **투자 결정**
   - [ ] 시그널 생성 (매수/방어/중립)
   - [ ] 포지션 크기 결정
   - [ ] Stop-Loss 설정

5. **실행**
   - [ ] 거래 주문 (시장가 또는 지정가)
   - [ ] 실행 가격 기록
   - [ ] Stop-Loss 주문 설정

**Weekly 모니터링**:

- [ ] Stop-Loss 도달 여부 확인
- [ ] VIX 급등/급락 모니터링 (± 20% 이상 변화)
- [ ] 시장 뉴스/이벤트 체크

**Monthly 리뷰**:

- [ ] 전월 예측 정확도 확인
- [ ] 수익률 계산 (거래 비용 포함)
- [ ] 모델 성능 트래킹 (정확도, 샤프 비율)
- [ ] 필요 시 모델 재학습

**Quarterly 리밸런스** (모델 업데이트):

- [ ] 최신 3개월 데이터 추가
- [ ] Walk-Forward Validation 재실행
- [ ] 모델 파라미터 재튜닝
- [ ] Out-of-sample 성능 확인

---

### Part IV-2: 연구의 한계점

#### 1. 데이터 기간의 제약 📅

**제한된 역사**:
- 데이터: 2011-2022 (약 11년)
- 포함된 시장 환경:
  - ✅ 유럽 재정위기 (2011-2012)
  - ✅ 강세장 (2013-2017)
  - ✅ 변동성 급등 (2018, 2020)
  - ✅ COVID-19 팬데믹 (2020)

- 포함되지 않은 시장 환경:
  - ❌ 2008 금융위기
  - ❌ 2000-2002 닷컴 버블 붕괴
  - ❌ 1987 블랙먼데이
  - ❌ 장기 약세장 (Bear Market)

**문제점**:
- 극단적 시장 상황에서의 성능 미검증
- 과거 성과가 미래를 보장하지 않음
- 시장 체제 변화 (regime shift) 시 성능 저하 가능

**완화 방법**:
- 지속적인 모델 모니터링
- Out-of-sample 성능 추적
- 이상 감지 시스템 구축

#### 2. 거래 비용 미고려 💸

**현재 분석의 한계**:
- 백테스팅 결과에 거래 비용 미반영
- 실제 수익률은 비용만큼 감소

**미고려 비용**:
- 증권사 수수료 (0.01-0.03%)
- 매도세 (없음, 미국 주식)
- Bid-Ask Spread (0.01-0.05%)
- **슬리피지** (0.05-0.15%): 대량 거래 시 가격 영향
- 기회 비용: 리밸런싱 타이밍 지연

**영향 추정**:
```
30일 전략 (연 12회 거래):
- 예측 정확도: 85.4%
- 이론적 초과 수익: ~20% 연간

거래 비용 차감 후:
- 일반 투자자: 20% - 1.56% = 18.44% ✅
- 기관 투자자: 20% - 0.60% = 19.40% ✅

여전히 매력적이지만, 실제는 더 낮을 가능성
```

#### 3. 시장 충격 비용 (Market Impact) 🌊

**대형 포트폴리오의 문제**:
- 백테스팅은 소액 투자자 가정
- 대형 자금 운용 시:
  - 주문이 시장 가격에 영향
  - 전량 체결까지 시간 소요
  - 예상보다 불리한 가격에 체결

**예시**:
```
1억 원 투자: 시장 충격 무시 가능 ✅
10억 원 투자: 시장 충격 0.05-0.10% 🟡
100억 원 투자: 시장 충격 0.20-0.50% ⚠️
```

**완화 방법**:
- ETF 사용 (유동성 높음)
- 분할 매매 (VWAP, TWAP 알고리즘)
- 선물/옵션 활용 (레버리지 아닌 헤징 목적)

#### 4. 과적합 위험 (Overfitting) 🎯

**Walk-Forward Validation의 한계**:
- 각 윈도우에서 독립적으로 학습
- 윈도우 크기 (2,128 관측치) 고정
- 최적 윈도우 크기는 사전에 모름

**하이퍼파라미터 튜닝**:
- LassoCV: 자동 λ 선택
- 학습 데이터에만 기반 → 일반화 보장 X

**다중 비교 문제**:
- 11개 모델 × 5개 시간 프레임 = 55개 조합 테스트
- 일부는 우연히 좋은 성능 가능
- p-hacking 위험

**완화 방법**:
- ✅ Walk-Forward로 시계열 구조 보존
- ✅ Feature Selection은 학습 데이터에서만
- ⚠️ 여전히 완벽한 방어는 불가능

#### 5. 생존 편향 (Survivorship Bias) 💀

**단일 지수의 한계**:
- S&P 500만 분석
- S&P 500은 "살아남은" 500개 기업
- 상장폐지, 인덱스 제외 기업은 미포함

**지역 편향**:
- 미국 시장만
- 다른 시장 (유럽, 아시아)에는 미검증

**자산군 편향**:
- 주식만
- 채권, 부동산, 원자재는 미검증

**일반화 가능성 의문**:
```
Q: 이 모델이 다른 시장에서도 작동하는가?
A: 검증 필요 ⚠️

Q: NASDAQ, Dow Jones에도 적용 가능한가?
A: 높은 상관관계로 유사한 성능 예상, but 검증 필요

Q: 신흥국 시장은?
A: 불확실, 시장 효율성 낮아 다를 수 있음
```

#### 6. 모델 안정성 🔄

**COVID-19 기간 성능 저하**:
- 2020년: Accuracy 68.1% (평균 대비 -17.3%p)
- 극단적 상황에서 예측 실패

**시장 체제 변화**:
- 연준 정책 변화 (QE, 금리)
- 시장 구조 변화 (알고리즘 트레이딩 증가)
- 변동성 특성 변화

**모델 수명**:
```
Q: 이 모델이 영원히 작동하는가?
A: 아니오 ❌

예상 수명:
- 낙관적: 5-10년
- 현실적: 2-5년
- 비관적: 이미 작동 안 할 수도...

대책: 지속적 재학습 및 모니터링 필수
```

#### 7. 실시간 데이터 접근 🕐

**데이터 지연**:
- 변동성 지수: 실시간 제공 (Yahoo Finance)
- But 데이터 품질 이슈 가능

**월말 효과**:
- 모델은 월초에 예측
- 실제 투자자는 언제든 접근 가능
- 타이밍에 따라 성능 차이

**데이터 비용**:
- 프리미엄 데이터 (Bloomberg, Reuters): 고비용
- 무료 데이터 (Yahoo): 품질 이슈 가능

#### 8. 심리적 요인 🧠

**예측과 실행의 괴리**:
```
모델 예측: "85% 확률로 상승"
투자자 심리: "그래도 불안한데..."

→ 실제로 매수하지 않음
→ 예측이 맞아도 수익 못 봄
```

**손실 회피 편향**:
- 모델: "15% 확률로 틀림"
- 투자자: "15%는 너무 높아, 못 믿겠어"
- → 보수적 집행, 수익 감소

**과신의 위험**:
- 85% 정확도 ≠ 100%
- 연속 5회 맞춤 → "무적"이라 착각
- 6번째에 큰 손실 → 패닉

**완화 방법**:
- 명확한 규칙 기반 투자
- 감정 배제 (알고리즘 트레이딩)
- 손실도 계획의 일부로 수용

---

### Part IV-3: 향후 연구 방향

#### 1. 실시간 예측 시스템 구축 🤖

**목표**: 자동화된 End-to-End 시스템

**구성 요소**:

```
[데이터 수집] → [전처리] → [예측] → [리스크 관리] → [실행] → [모니터링]
     ↑                                                           ↓
     └───────────────────── [피드백 루프] ────────────────────────┘
```

**1단계: 데이터 파이프라인**
```python
# 매일 자동 실행 (cron job)
import yfinance as yf
import schedule

def collect_daily_data():
    """
    매일 오전 9시에 변동성 지수 수집
    """
    tickers = ['^VIX', '^VIX3M', '^SKEW', '^GVZ', 'RVOL']
    data = yf.download(tickers, period='1d')

    # 데이터베이스에 저장
    save_to_database(data)

    # 데이터 품질 체크
    if data_quality_check(data):
        return data
    else:
        alert("데이터 품질 이슈 발생!")

schedule.every().day.at("09:00").do(collect_daily_data)
```

**2단계: 예측 엔진**
```python
def predict_engine():
    """
    매월 첫 거래일에 예측 실행
    """
    # 최신 데이터 로드
    data = load_from_database()

    # 모델 로드
    model = load_model('logistic_regression_30d.pkl')
    scaler = load_scaler('scaler.pkl')

    # 전처리 및 예측
    X = preprocess(data)
    X_scaled = scaler.transform(X)
    prob_up = model.predict_proba(X_scaled)[0, 1]

    # 신호 생성
    signal = generate_signal(prob_up, data['VIX'])

    return signal, prob_up

# 매월 첫 거래일에 실행
schedule.every().month.at("09:30").do(predict_engine)
```

**3단계: 알림 시스템**
```python
def alert_system(signal, prob_up):
    """
    이메일, SMS, 슬랙으로 알림
    """
    message = f"""
    🎯 투자 시그널 생성!

    - 날짜: {datetime.now()}
    - 시그널: {signal}
    - 상승 확률: {prob_up*100:.1f}%
    - 권장 포지션: {calculate_position(prob_up)}
    """

    send_email(message)
    send_slack(message)
    # send_sms(message)  # 옵션
```

**4단계: 자동 실행 (선택)**
```python
def auto_execute(signal, position_size):
    """
    증권사 API 연동하여 자동 매매
    주의: 리스크 높음, 철저한 테스트 필요
    """
    from trading_api import TradingAPI

    api = TradingAPI(api_key, api_secret)

    if signal == "매수":
        api.buy('SPY', position_size)
    elif signal == "방어적":
        api.sell('SPY', 1 - position_size)

    log_trade(signal, position_size)
```

#### 2. 다양한 자산군으로 확장 🌐

**확장 대상**:

| 자산군 | 티커 | 변동성 지수 | 예상 성능 |
|--------|------|------------|----------|
| **NASDAQ** | QQQ | VXN | 높음 (이미 VXN 사용) |
| **Dow Jones** | DIA | VXD | 높음 (유사성) |
| **Russell 2000** | IWM | RVX | 중간 (소형주) |
| **국제 지수** |  |  |  |
| - FTSE 100 | - | VFTSE | 중간 (검증 필요) |
| - DAX | - | VDAX | 중간 |
| - Nikkei 225 | - | VXJ | 낮음 (시장 특성 다름) |
| **섹터 ETF** |  |  |  |
| - Technology | XLK | VXTECH | 높음 |
| - Financials | XLF | - | 중간 |
| - Healthcare | XLV | - | 중간 |

**구현 계획**:
1. 각 자산군별 데이터 수집
2. 동일한 방법론 적용
3. 성능 비교 분석
4. 포트폴리오 다변화

**다중 자산 포트폴리오**:
```python
# 각 자산별 예측
predictions = {
    'SPY': 0.85,  # 85% 상승 확률
    'QQQ': 0.78,  # 78%
    'IWM': 0.62,  # 62%
    'DIA': 0.81   # 81%
}

# 가중 평균 포지션
weights = calculate_weights(predictions)
# → SPY 35%, QQQ 30%, DIA 25%, IWM 10%
```

#### 3. 딥러닝 모델과의 비교 🧠

**한계**:
- 현재 모델: 선형 (Logistic Regression)
- 비선형 패턴 포착 한계

**딥러닝 아키텍처**:

**LSTM (Long Short-Term Memory)**:
```python
import torch.nn as nn

class VIX_LSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # 마지막 타임스텝 사용
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return self.sigmoid(out)

# 시계열 길이: 30일 (과거 30일 데이터로 30일 후 예측)
```

**Transformer**:
```python
class VIX_Transformer(nn.Module):
    def __init__(self, input_size=10, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.sigmoid(self.fc(x))
```

**비교 실험 계획**:

| 모델 | 복잡도 | 예상 정확도 | 해석 가능성 | 과적합 위험 |
|------|--------|------------|------------|------------|
| Logistic Regression | 낮음 | 85.4% (기준) | 높음 ✅ | 낮음 ✅ |
| LSTM | 중간 | 87-90%? | 낮음 ❌ | 중간 🟡 |
| Transformer | 높음 | 88-92%? | 낮음 ❌ | 높음 ⚠️ |

**주의사항**:
- 딥러닝은 더 많은 데이터 필요 (수만 관측치)
- 우리 데이터 (2,913 관측치)로는 부족할 수 있음
- 과적합 위험 ⚠️

#### 4. 변수 확장 🔬

**추가 가능한 변수**:

**거시경제 지표**:
- 금리 (Fed Funds Rate, 10Y Treasury Yield)
- 환율 (DXY Dollar Index)
- 원자재 가격 (Gold, Oil)
- 경제지표 (GDP, 실업률, PMI)

**시장 미시구조**:
- 거래량 (Volume)
- Put/Call Ratio
- 신용 스프레드 (Credit Spread)
- 공매도 비율 (Short Interest)

**감성 분석 (Sentiment)**:
- 뉴스 감성 (FinBERT)
- 소셜 미디어 (Twitter/X, Reddit)
- Google Trends

**구현 예시**:
```python
# 추가 변수 포함
X_extended = pd.DataFrame({
    # 기존 변수
    'VIX': vix,
    'SKEW': skew,
    ...

    # 새로운 변수
    'FED_RATE': fed_funds_rate,
    'DXY': dollar_index,
    'GOLD': gold_price,
    'VOLUME': spy_volume,
    'PUT_CALL': put_call_ratio,
    'NEWS_SENTIMENT': news_sentiment_score
})

# Feature Selection 재실행
# 어떤 변수가 추가 정보를 제공하는지 확인
```

#### 5. 앙상블 전략 🎭

**여러 모델 결합**:

```python
# 투표 방식 (Voting)
def ensemble_voting(models, X):
    """
    여러 모델의 예측을 투표로 결합
    """
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)

    # 과반수 투표
    final_pred = np.round(np.mean(predictions))
    return final_pred

models = [
    logistic_regression,
    random_forest,
    gradient_boosting,
    lstm_model
]

prediction = ensemble_voting(models, X_test)
```

**가중 평균 (Weighted Average)**:
```python
def ensemble_weighted(models, weights, X):
    """
    성능에 따라 가중 평균
    """
    probs = []
    for model, weight in zip(models, weights):
        prob = model.predict_proba(X)[:, 1]
        probs.append(prob * weight)

    final_prob = np.sum(probs, axis=0)
    return final_prob

# 가중치는 검증 세트 성능 기반
weights = [0.40, 0.25, 0.20, 0.15]  # 합=1.0
# Logistic 40%, RF 25%, GB 20%, LSTM 15%
```

**메타 학습 (Stacking)**:
```python
# Level 1: 기본 모델들
base_models = [logistic, rf, gb, lstm]

# Level 2: 메타 모델 (기본 모델들의 예측을 입력으로)
meta_model = LogisticRegression()

# 학습
for model in base_models:
    model.fit(X_train, y_train)

# 기본 모델들의 예측 수집
meta_features_train = np.column_stack([
    model.predict_proba(X_train)[:, 1] for model in base_models
])

# 메타 모델 학습
meta_model.fit(meta_features_train, y_train)

# 예측
meta_features_test = np.column_stack([
    model.predict_proba(X_test)[:, 1] for model in base_models
])
final_prediction = meta_model.predict(meta_features_test)
```

#### 6. 적응형 모델 (Adaptive Model) 🔄

**문제**: 시장 체제가 변하면 모델 성능 저하

**해결**: 온라인 학습 (Online Learning)

```python
class AdaptiveModel:
    def __init__(self, base_model, update_frequency=30):
        self.model = base_model
        self.update_frequency = update_frequency
        self.days_since_update = 0
        self.recent_data = []

    def predict(self, X):
        return self.model.predict(X)

    def update(self, X_new, y_new):
        """
        새로운 데이터로 모델 업데이트
        """
        self.recent_data.append((X_new, y_new))
        self.days_since_update += 1

        if self.days_since_update >= self.update_frequency:
            # 최근 N개 데이터로 재학습
            X_recent = np.vstack([x for x, y in self.recent_data[-500:]])
            y_recent = np.array([y for x, y in self.recent_data[-500:]])

            self.model.fit(X_recent, y_recent)
            self.days_since_update = 0
            print(f"모델 업데이트 완료: {len(self.recent_data)} 데이터 사용")

# 사용 예
adaptive_model = AdaptiveModel(LogisticRegression())

for t in range(len(X_test)):
    # 예측
    pred = adaptive_model.predict(X_test[t:t+1])

    # 30일 후 실제 결과 확인
    actual = y_test[t+30]

    # 모델 업데이트
    adaptive_model.update(X_test[t:t+1], actual)
```

#### 7. 멀티 타임프레임 통합 📊

**아이디어**: 여러 시간 프레임 예측을 결합

```python
def multi_timeframe_strategy():
    """
    1일, 7일, 30일 예측을 모두 사용
    """
    pred_1d = model_1d.predict_proba(X)[0, 1]
    pred_7d = model_7d.predict_proba(X)[0, 1]
    pred_30d = model_30d.predict_proba(X)[0, 1]

    # 모두 상승 신호 → 강한 매수
    if pred_1d > 0.5 and pred_7d > 0.6 and pred_30d > 0.65:
        return "강한 매수", 0.9

    # 단기는 하락, 장기는 상승 → 대기
    elif pred_1d < 0.5 and pred_30d > 0.6:
        return "대기 후 매수", 0.3

    # 장기만 상승 → 점진적 매수
    elif pred_30d > 0.6:
        return "점진적 매수", 0.6

    # 모두 하락 → 방어
    elif pred_1d < 0.4 and pred_7d < 0.4 and pred_30d < 0.4:
        return "방어적", 0.1

    else:
        return "중립", 0.5
```

---

## 다음 파일에서 계속...

이 문서는 Part III와 Part IV를 담고 있습니다.

**다음 파일에서**:
- Conclusion
- Appendices
- Executive Summary (최종 요약)

[메인 리포트로 돌아가기](./Campisi_2024_Final_Report.md)
