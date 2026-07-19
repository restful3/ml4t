---
lang: ko
format:
  html:
    toc: true
    embed-resources: true
    theme: cosmo
---

# 시계열 분석


경제학자와 전기공학자들은 오랫동안 시계열의 다음 신호를 예측하려고 노력해 왔으며, 이는 트레이더들이 하려는 일과 정확히 같습니다. 이 장은 계량경제학 (econometrics)과 신호 처리 (signal processing)에서 잘 알려져 있고 계량 투자 커뮤니티에서 널리 받아들여진 도구들에 대한 입문입니다.

여러분은 이미 제 이전 저서들(Chan, 2009 and 2013)에서 가격 시계열의 정상성 (stationarity) 또는 공적분 (cointegration)을 검정하는 방법으로 일부 시계열 분석 기법이 실제로 사용되는 것을 보았을 수도 있습니다. 그러나 이것들은 ARIMA, VAR, VEC와 같은 약어로 불리는 선형 모델링 기법의 일반적인 패키지 중 일부에 불과합니다. 마찬가지로, 거의 모든 기술적 트레이더는 가격 시계열의 잡음을 걸러내는 방법으로 이동평균을 시도해 보았습니다. 그러나 그들은 칼만 필터 (Kalman filter)와 같은 고급 신호 처리 필터를 많이 시도해 보았을까요?

시계열 기법은 펀더멘털 정보와 직관이 부족하거나 단기 예측에 특히 유용하지 않은 시장에서 가장 유용합니다. 통화와 비트코인이 여기에 해당합니다. Lyons 교수(2001)는 “ … 우리의 교과서적 모델이 설명할 수 있는 월간 환율 변화의 비율은 본질적으로 0이다.”라고 썼습니다. 이 장에서는 통화 수익률을 예측하기 위해 시계열 기법을 사용하는 몇 가지 예를 언급하고, 비트코인 예시는 7장으로 남겨 두겠습니다. 그러나 펀더멘털 정보가 풍부한 주식 거래에서도 기술적 분석이 유용할 수 있듯이, 시계열 분석을 주식에 적용할 수 있는 예를 설명하겠습니다.

시계열 분석에 관한 다른 책들과 달리, 우리는 이러한 기법의 내부 작동 원리를 논의하지 않고, 예측을 하기 위해 기성 소프트웨어 패키지를 어떻게 사용할 수 있는지에만 초점을 맞출 것입니다. 대부분의 예는 MATLAB Econometrics Toolbox를 사용하여 구현되었지만, R 사용자는 forecast, vars, dlm 패키지에서 유사한 함수를 찾을 수 있습니다.

$$
\operatorname{AR}(\mathfrak{p})
$$

시계열 분석에서 가장 단순한 모델은 $\mathrm{AR}(1)$입니다. 이는 한 막대의 가격을 다음 막대의 가격과 연결하는 선형 회귀 모델일 뿐입니다.

$$
Y(t)-\mu=\phi(Y(t-1)-\mu)+\varepsilon(t)\tag{3.1}
$$

여기서 $Y(t)$는 시간 $t$에서의 가격이고, $\Phi$는 (자기)회귀 계수이며, ε는 평균이 0인 가우스 잡음으로, 때로는 혁신 (innovation)이라고 불립니다. 따라서 자기회귀 과정 (auto-regressive process)이라는 이름이 붙습니다. 시계열은 평균과 분산이 시간에 따라 일정하면 약정상 (weakly stationary)<sup>1</sup>이라고 하며, $\operatorname{AR}(1)$은 $|\Phi|<1$이면 약정상입니다(증명은 연습문제로 남깁니다). 약정상 시계열은 또한 평균회귀 (mean reverting)합니다(Chan, 2013). $|\Phi|>1$이면 시계열은 추세를 보일 것입니다. $\Phi=1$이면 랜덤 워크가 됩니다. $\Phi$를 추정하기 위해 Econometrics Toolbox의 arima 및 estimate 함수를 사용합니다.

model\_ar1=arima(1, 0, 0) % 미지의 모수를 가진 AR(1)을 가정함   
model\_ar1\_estimates=estimate(model\_ar1, cl);

함수 arima $(p,d,q)$는 $p=1$ 및 $d=0$으로 설정하면 AR(1) 모델로 축소됩니다(더 일반적인 버전은 다음 절에서 논의하겠습니다). estimate 함수는 입력 가격 시계열을 기반으로 AR(1) 모델의 모수를 찾기 위해 최대가능도 추정 (maximum likelihood estimation)을 적용할 뿐입니다. 이를 2007년 7월 24일부터 2015년 8월 3일까지의 AUD.USD 1분 중간가격 막대에 적용하면 $\Phi=0.99997$의 추정값이 반환되며, 표준오차는 0.00001입니다.2 우리는 AUD.USD가 매우 약하게 정상적이기는 하지만, 랜덤 워크에 매우 가깝다고 결론 내립니다. 매수–매도 호가 반등 (bid–ask bounce)을 줄이기 위해 거래 가격 대신 중간가격으로 검정했다는 점에 유의하십시오. 매수–매도 호가 반등은 실제로 거래할 수 없는 허상적 평균회귀를 만들어내는 경향이 있습니다.

AR(1)을 약간 일반화하면, 다음과 같이 표현되는 $\operatorname{AR}(p)$를 고려할 수 있습니다.

$$
Y(t) = \mu + \phi_{1} Y(t - 1) + \phi_{2} Y(t - 2) + \cdot \cdot \cdot + \phi_{p} Y(t - p) + \varepsilon(t) .\tag{3.2}
$$

이는 시간 t의 가격을 종속변수(반응변수)로 하고, 최대 $P$개 봉의 지연까지 과거 가격을 독립변수(예측변수)로 하는 다중 회귀 모델일 뿐임을 알 수 있습니다. 그러나 p를 추가 매개변수로 도입하면, $\operatorname{AR}(p)$ 모델이 우리 데이터에 가장 잘 적합되도록 하는 최적의 $P$를 찾을 수 있습니다. 많은 통계 모델에서와 마찬가지로, 우리는 다음을 사용할 것입니다.

```matlab
model=arima(pMin, 0, 0) % assumes an AR(pMin) with unknown
parameters
```

베이지안 정보 기준(BIC)은 모델의 음의 로그 가능도에 비례하지만, 복잡도에 벌점을 부과하는 p에 비례하는 추가 항을 포함합니다. 우리의 목적은 BIC를 최소화하는 것이며, 이를 무차별 대입식 전수 탐색으로 수행합니다:3

```matlab
LOGL=zeros(60, 1); % log likelihood for up to 60 lags (1 hour)
P=zeros(size(LOGL)); % p values
for p=1:length(P)
model=arima(p, 0, 0);
[∼,∼,logL] = estimate(model, mid(trainset),'print',false);
LOGL(p) = logL;
P(p) = p;
```

위 코드 조각에서 mid는 중간가격을 포함하는 배열입니다. p의 최선 추정치를 결정하면, estimate 함수를 적용하여 계수 μ, ϕ1,ϕ2, … ,ϕp를 찾을 수 있습니다.

```javascript
fit=estimate(model, mid);
```

이 함수들을 2007년 7월 24일부터 2014년 8월 12일까지의 1분 중간가격 봉에 대한 AUD.USD에 적용하면, 표 3.1에 제시된 계수와 함께 p = 10이 최적값으로 산출됩니다.

이제 이 AR(10) 모델을 2014년 8월 12일부터 2015년 8월 3일까지의 표본 외 데이터셋에 대한 예측에 사용할 수 있습니다.

```matlab
yF=NaN(size(mid));
for t=testset(1):size(mid, 1)
[y, ∼]=forecast(fit, 1, 'Y0', mid(t-pMin+1:t)); % Need only
most recent pMin data points for prediction
yF(t)=y(end);
```

yF(t)는 시간 t까지의 데이터로 만든 예측값이라는 점에 유의하십시오. 따라서 이는 실제로 시간 t + 1의 예측 가격입니다. 다음 봉 예측이 이루어지면, 이를 사용하여 거래 신호를 생성할 수 있습니다. 예측 가격이 현재 가격보다 높으면 단순히 매수하고, 낮으면 매도합니다.

표 3.1 AUD.USD에 적용한 AR(10) 모델의 계수
<table><tr><td>계수</td><td>값</td><td>표준오차</td></tr><tr><td> $\mu$ </td><td>1.37196e-06</td><td>8.65314e-07</td></tr><tr><td> $\Phi_{1}$ </td><td>0.993434</td><td>0.000187164</td></tr><tr><td> $\Phi_{2}$ </td><td>-0.00121205</td><td>0.000293356</td></tr><tr><td> $\Phi_{3}$ </td><td>-0.000352717</td><td>0.000305831</td></tr><tr><td> $\Phi_{4}$ </td><td>0.000753222</td><td>0.000354121</td></tr><tr><td> $\Phi_{5}$ </td><td>0.00662641</td><td>0.000358673</td></tr><tr><td> $\Phi_{6}$ </td><td>-0.00224118</td><td>0.000330092</td></tr><tr><td> $\Phi_{7}$ </td><td>-0.00305157</td><td>0.000365348</td></tr><tr><td> $\Phi_{8}$ </td><td>0.00351317</td><td>0.000394538</td></tr><tr><td> $\Phi_{9}$ </td><td>-0.00154844</td><td>0.000398956</td></tr><tr><td> $\Phi_{10}$ </td><td>0.00407798</td><td>0.000281821</td></tr></table>

```javascript
deltaYF=yF-mid;
```

```javascript
pos=zeros(size(mid));
pos(deltaYF > 0)=1;
pos(deltaYF < 0)=-1;
```

이 전략은 표본 외 집합에서 연율 158퍼센트의 수익률을 산출합니다. 그 자산 곡선은 그림 3.1을 참조하십시오. 이러한 놀라운 수익률을 실현하려면 중간가격에서 체결할 수 있어야 하므로, 지정가 주문을 관리하는 저지연 체결 프로그램이 필요합니다.

![](images/bcb8103809e5729fc38478fa4f208f11fde75383d89de2817c9c2f51d2719356.jpg)  
그림 3.1 AUD.USD에 적용된 AR(10) 거래 전략

### ARMA(p, q)


AUD.USD에 $\operatorname{AR}(p)$를 적용한 결과에서, 최적 적합에는 10개의 시차 (lag)가 필요함을 알 수 있습니다. 이렇게 많은 시차 수는 $\operatorname{AR}(p)$ 모델에서 매우 흔합니다. 이는 더 많은 항을 사용하여 모델 구조의 단순성을 보완하려 하기 때문입니다. AR 모델을 약간 확장하여 $q$개의 지연 잡음 항을 포함하면 필요한 시차 수가 줄어드는 경우가 많습니다. 이를 $\mathrm{ARMA}(p,q)$ 모델, 또는 자기회귀 이동평균 과정 (auto-regressive moving average process)이라고 하며, 여기서 $q$개의 지연 잡음 항은 이동평균으로 설명됩니다.

$$
\begin{array}{c}
{{Y(t)=\mu+\phi_{1}Y(t-1)+\phi_{2}Y(t-2)+\cdots+\phi_{p}Y(t-p)+\varepsilon(t)}} \\
{{\nonumber}} \\
{{+~\theta_{1}\varepsilon(t-1)+\cdots+\theta_{q}\varepsilon(t-q)}}
\end{array}{}{}\nonumber\tag{3.3}
$$

식 3.3에서 $p$와 $q$의 최적값 및 각 항의 계수를 찾는 것은 $\operatorname{AR}(p)$에서 수행한 절차와 유사하지만, 이제 두 변수에 대해 전수 탐색을 수행하므로 중첩된 for-루프가 필요합니다.

```matlab
LOGL=-Inf(10, 9); % log likelihood for up to 10 p and 9 q
(10 minutes)
PQ=zeros(size(LOGL)); % p and q values
for p=1:size(PQ, 1)
for q=1:size(PQ, 2)
model=arima(p, 0, q);
[∼,∼,logL] = estimate(model, mid(trainset),'print',false);
LOGL(p, q) = logL;
PQ(p, q) = p+q;
```

각 $p$와 $q$에 대해 로그 가능도를 $LOGL(p,q)$에 저장하고, $p+q$를 $PQ(p,q)$에 저장합니다. 후자는 BIC를 최소화할 때 벌점 항으로 사용되기 때문입니다. LOGL 및 PQ 행렬로부터 BIC를 최소화하는 최적의 $p$와 $q$를 어떻게 식별할 수 있을까요? 이를 1차원 벡터로 변환하고, aicbic 함수를 적용한 다음, min 함수를 사용해야 합니다.

```matlab
% Has p+q+1 parameters, including constant
LOGL_vector = reshape(LOGL, size(LOGL, 1)*size(LOGL, 2), 1);
PQ_vector = reshape(PQ, size(LOGL, 1)*size(LOGL, 2), 1);
```

[∼, bic]=aicbic(LOGL\_vector, PQ\_vector+1, length(mid(trainset)));   
[bicMin, pMin]=min(bic)

마지막으로, 해당 셀의 행 번호($p$에 해당)와 열 번호($q$에 해당)를 쉽게 시각적으로 식별할 수 있도록, 1차원 BIC 벡터를 다시 2차원 배열로 변환해야 합니다. 다만 최소값에 해당하는 셀만 값이 채워지도록 합니다.

```matlab
bic(:)=NaN;
bic(pMin)=bicMin;
bic=reshape(bic,size(LOGL))
```

이 모든 절차는 buildARMA\_findPQ\_ AUDUSD.m 프로그램에 포함되어 있습니다. AUD.USD에 대한 출력은 다음과 같습니다.

열 1부터 4까지

여기서 최소 BIC를 갖는 셀이 $p=2$ 및 $q=5$에 해당함을 쉽게 확인할 수 있습니다. 이는 실제로 $\operatorname{AR}(p)$ 모델에서 사용한 $p=10$보다 더 짧은 시차입니다. 이러한 값을 arima 함수에 대입한 다음, $\operatorname{AR}(p)$ 절에서 수행한 것처럼 ARMA(2, 5) 모델에 estimate 함수를 적용하면 표 3.2에 제시된 계수가 산출됩니다.

$|\Phi_{1}|$가 이제 확실히 1보다 작아졌으며, 이는 강한 평균 회귀 (mean reversion)를 나타낸다는 점에 유의해야 합니다. 그러나 이전과 같이 예측 함수를 사용하여 거래 신호를 생성하면 실제로 표본외 (out-of-sample) 연율화 수익률이 158퍼센트에서 60퍼센트로 감소합니다. 이 경우 이동평균 (moving average)을 사용하는 추가적인 복잡성은 성과로 이어지지 않았습니다. 자산 곡선 (equity curve)은 그림 3.2에 제시되어 있습니다. 백테스트 프로그램은 buildARMA\_AUDUSD.m으로 제공됩니다.

우리가 $\mathrm{AR}(p)$ 및 ARMA $(p,q)$ 모델에 사용한 함수가 왜 arima라고 불리는지 궁금할 수 있습니다. 또한 왜 수익률이 아니라 가격을 예측하는 데 초점을 맞추는지 궁금할 수도 있습니다. 두 질문에 대한 답은 $\mathrm{ARIMA}(p,d,q)$ 모델을 연구함으로써 이해할 수 있습니다.

표 3.2 AUD.USD에 적용된 ARMA(2, 5) 모델의 계수
<table><tr><td>계수</td><td>값</td><td>표준오차</td></tr><tr><td> $\mu$ </td><td>2.80383e-06</td><td>4.58975e-06</td></tr><tr><td> $\Phi_{1}$ </td><td>0.649011</td><td>0.000249771</td></tr><tr><td> $\Phi_{2}$ </td><td>0.350986</td><td>0.000249775</td></tr><tr><td> $\theta_{1}$ </td><td>0.345806</td><td>0.000499929</td></tr><tr><td> $\theta_{2}$ </td><td>−0.00906282</td><td>0.000874713</td></tr><tr><td> $\boldsymbol{\theta}_{3}$ </td><td>-0.0106082</td><td>0.000896239</td></tr><tr><td> $\theta_{4}$ </td><td>-0.0102606</td><td>0.0010664</td></tr><tr><td> $\theta_{5}$ </td><td>-0.00251154</td><td>0.000910359</td></tr></table>

AUD.USD에 대한 ARMA(2, 5) 모델  
![](images/bbed2b106645edf899c47db0e2fc1b8f5f4c7b7e06369035131be5e1196293f6.jpg)  
그림 3.2 AUD.USD에 적용된 ARMA(2, 5) 거래 전략

$\mathrm{ARIMA}(p,d,q)$는 자기회귀 누적 이동평균 (autoregressive integrated moving average)을 의미합니다. 여기서는 금융에서 가장 단순하고 가장 일반적인 경우인 $d=1$에만 관심을 두겠습니다. $Y(t)$가 $\mathrm{ARIMA}(p,\ 1,\ q)$ 모델이라면, 이는 $\Delta Y(t)$가 $\mathrm{ARMA}(p,q)$임을 의미하며, 여기서 $\Delta Y(t)=Y(t)-Y(t-1)$입니다. $Y(t)$가 가격이 아니라 로그 가격을 나타낸다면 이를 훨씬 더 잘 이해할 수 있습니다. 이 경우 로그 수익률을 모델링하기 위해 $\mathrm{ARMA}(p,q)$를 사용하는 것은 로그 가격을 모델링하기 위해 ARIMA $(p,1,q)$를 사용하는 것과 동등합니다.

$\mathrm{ARMA}(p,q)$를 사용하여 $Y$ 대신 로그 수익률 $\Delta Y$를 모델링하는 것이 유리할까요? 가격(또는 로그 가격)을 $\mathrm{ARMA}(p,q)$로 모델링할 때 얻은 것보다 시차 (lags) $p$와 $q$를 더 줄일 수 있다면 유리할 것입니다. 안타깝게도 저는 그것이 사실인 경우를 발견한 적이 없습니다. 예를 들어, $\mathrm{ARIMA}(p,1,q)$를 사용하여 AUD.USD 시계열의 로그를 모델링하면 $p=1$, $q=9$가 됩니다.

로그 가격에 대한 $\mathrm{ARIMA}(p,\ 1,\ q)$ 모델이 로그 수익률에 대한 $\mathrm{ARIMA}(p,\ 0,\ q)$ 모델과 동등하다는 사실을, 로그 가격에 대한 $\mathrm{ARMA}(p,\ q)=\mathrm{ARIMA}(p,\ 0,\ q)$ 모델이 로그 수익률에 대한 어떤 $\mathrm{ARMA}(p^{\prime},q^{\prime})$ 모델과 동등하다는 진술과 혼동해서는 안 됩니다. 후자의 진술은 거짓입니다. $\Delta Y_{\mathrm{S}}^{\prime}$의 ARMA 모델은 항상 $Y_{\mathrm{S}}^{\prime}$의 ARMA 모델로 변환될 수 있습니다. 그러나 $Y$에 대한 ARMA 모델이 항상 $\Delta Y$에 대한 ARMA 모델로 변환될 수 있는 것은 아닙니다. 이는 $\Delta Y$에 대한 ARMA 모델은 독립변수로 $\Delta Y$만 가질 수 있는 반면, Y에 대한 ARMA 모델은 $\Delta Y$($Y_{\mathrm{S}}$ 두 개의 차이일 뿐입니다)와

Y를 모두 독립변수로 가질 수 있기 때문입니다. 따라서 Y에 대한 모델이 더 유연하며 더 나은 결과를 제공합니다. $\Delta Y$와 Ys를 모두 독립변수로 갖는 $\Delta Y$에 대한 모델을 원한다면 ${\mathrm{VEC}}(p)$ 모델을 사용해야 하며, 이는 다음 절의 $\mathrm{VAR}(p)$에 대한 논의 끝부분에서 다룰 것입니다.

### VAR(p)

식 3.2의 단순 자기회귀 모델 $\operatorname{AR}(p)$는 m개의 다변량 시계열로 쉽게 일반화될 수 있습니다. 이 일반화된 모델을 벡터 자기회귀 모델 (vector autoregressive model), 또는 VAR(p)라고 합니다. 우리가 해야 할 일은 자기회귀 계수 $\boldsymbol{\Phi}$를 m × m 행렬로 해석하고, m-벡터인 잡음 $\boldsymbol{\varepsilon}$이 0이 아닌 횡단면 상관관계 (cross-sectional correlations)를 갖되 0의 시계열 상관관계 (serial correlations)를 갖도록 허용하는 것뿐입니다. 이는 임의의 $t \neq s,$에 대해 $\varepsilon_i(t)$가 $\varepsilon_j(s)$와 상관되지 않지만, $\varepsilon_i(t)$는 $\varepsilon_j(t)$와 상관될 수 있음을 의미합니다. 자기회귀 계수 행렬은 모든 시계열의 현재 가격을 모든 시계열의 지연 가격과 연결하므로, VAR 모델은 동일 산업군 내 주식 포트폴리오와 같이 상관된 수익률을 갖는 금융상품을 모델링하는 데 특히 적합합니다. 우리는 2007년 1월 3일 기준 S&P 500 지수 내 컴퓨터 하드웨어 그룹에 초점을 맞출 것이며, 이 그룹은 티커 AAPL, EMC, HPQ, NTAP, SNDK로 구성됩니다. 매수-매도 호가 반동 (bid-ask bounce)으로 인한 허위 평균회귀 효과를 제거하기 위해, 2007년 1월 3일부터 2013년 12월 31일까지 증권가격연구센터(Center for Research of Security Prices, CRSP)가 제공한 시장 마감 시점의 중간가격 (midprices)을 사용합니다.

$\operatorname{AR}(p)$ 절에서와 마찬가지로, 먼저 최적 지연 p를 결정해야 합니다. 이 결정을 위해 처음 6년의 데이터를 학습 세트로 사용할 것입니다. 필요한 코드에는 사소한 차이만 있습니다:4

```matlab
for p=1:length(P)
model=vgxset('n', size(mid, 2), 'nAR', p, 'Constant', true);
% with additive offset
[model,EstStdErrors,logL,W] = vgxvarx(model,mid(trainset, :));
[NumParam,∼] = vgxcount(model);
LOGL(p) = logL;
P(p) = NumParam;
```

$P = 1$이 BIC를 최소화한다는 사실은 만족스럽습니다(더 단순한 모델이 보통 더 좋습니다). 또한 이는 대부분의 산업군에서 전형적인 결과입니다. 이것이 결정되면, 모델의 다른 매개변수들은 ARIMA 모델의 estimate 함수에 해당하는 vgxvarx 함수로 결정할 수 있습니다. 동일한 학습 세트를 사용하여, 상수 오프셋, 자기회귀 계수, 그리고 잡음항의 공분산을 표 3.3에 제시했습니다. (이 표에서는 표 3.1 또는 3.2와 달리, 아래첨자가 시간 지연의 수가 아니라 주식을 가리킵니다.)

2013년의 표본외 (out-of-sample) 데이터에 대해 이 모델을 사용하여 예측하려면, ARIMA의 forecast 함수와 유사한 vgxpred 함수를 사용합니다.

```matlab
pMin=1;
yF=NaN(size(mid));
for t=testset(1):size(mid, 1)
FY = vgxpred(model,1, [], mid(t-pMin+1:t, :));
yF(t, :)=FY;
```

VAR 모델의 선형성과 일관되게, 우리도 선형 트레이딩 모델을 구성할 수 있습니다. 나아가 이를 섹터 중립적으로 만들도록 선택할 수도 있습니다. 매일 산업군 내 모든 주식의 평균 예측 수익률 $\left. r \right.$을 계산하고, 한 주식의 목표 달러 배분이 그 주식의 예측 수익률과 산업군 평균 간의 차이에 비례하도록 설정합니다.

표 3.3 컴퓨터 하드웨어 주식에 적용한 VAR(1) 모델의 상수 오프셋, 자기회귀 계수, 공분산
<table><tr><td colspan="2">상수 오프셋</td><td colspan="2">값</td><td colspan="2">표준오차</td></tr><tr><td colspan="2"> $\mu_{1}$ </td><td colspan="2">3.88363</td><td></td><td>1.15299</td></tr><tr><td colspan="2"> $\mu_{2}$ </td><td colspan="2">0.669367</td><td></td><td>0.0970334</td></tr><tr><td colspan="2"> $\mu_{3}$ </td><td colspan="2">1.75474</td><td></td><td>0.227636</td></tr><tr><td colspan="2"> $\mu_{4}$ </td><td colspan="2">1.701</td><td></td><td>0.249767</td></tr><tr><td colspan="2">$\mu_{5}$</td><td colspan="2">1.8752</td><td></td><td>0.282581</td></tr><tr><td> $\Phi_{\mathrm{i,j}}$ </td><td>AAPL</td><td>EMC</td><td>HPQ</td><td>NTAP</td><td>SNDK</td></tr><tr><td>AAPL</td><td>0.991815</td><td>0.0735881</td><td>-0.105676</td><td>0.0359698</td><td>-0.00619303</td></tr><tr><td>EMC</td><td>-7.15594e-05</td><td>0.970934</td><td>-0.0103416</td><td>0.00524778</td><td>0.00354032</td></tr><tr><td>HPQ</td><td>-0.00158962</td><td>-0.024093</td><td>0.965626</td><td>0.00898799</td><td>0.00190162</td></tr><tr><td>NTAP</td><td>-0.000771673</td><td>-0.0409408</td><td>-0.0284176</td><td>1.00662</td><td>0.00308001</td></tr><tr><td>SNDK</td><td>-0.000526824</td><td>-0.0579403</td><td>-0.0309631</td><td>0.01704</td><td>0.998657</td></tr><tr><td> $\langle \varepsilon_{i} \varepsilon_{j} \rangle$ </td><td>AAPL</td><td>EMC</td><td>HPQ</td><td>NTAP</td><td>SNDK</td></tr><tr><td>AAPL</td><td>36.2559</td><td></td><td></td><td></td><td></td></tr><tr><td>EMC</td><td>1.67571</td><td>0.256786</td><td></td><td></td><td></td></tr><tr><td>HPQ</td><td>3.37592</td><td>0.449846</td><td>1.41323</td><td></td><td></td></tr><tr><td>NTAP</td><td>3.78265</td><td>0.513747</td><td>1.20474</td><td>1.70138</td><td></td></tr><tr><td>SNDK</td><td>4.39542</td><td>0.522437</td><td>1.26443</td><td>1.41357</td><td>2.17779</td></tr></table>

$$
\mathrm{\Sigma}_{w_{i}} = (r_{i} - \langle r \rangle) / \sum_{j} | \mathrm{r}_{j} - \langle r \rangle | .\tag{3.4}
$$

우리는 포트폴리오의 초기 총 시장가치가 항상 \$1이 되도록 했습니다. 이 공식이 Chan (2013)의 식 4.1과 유사해 보일 수 있지만, 서로 다릅니다. 이전 저서의 공식에서는 사용된 수익률이 전일 수익률이며, 더 중요하게는 평균회귀를 가정했기 때문에 비례상수를 −1로 설정했습니다. 각 주식의 포지션(동등하게는 달러 배분)을 계산하기 위한 MATLAB 코드 조각5는 다음과 같습니다.

```matlab
retF=(yF-mid)./mid;
sectorRetF=mean(retF, 2);
pos=zeros(size(retF));
pos=(retF-repmat(sectorRetF, [1 size(retF, 2)]))./repmat
(smartsum(abs(retF-repmat(sectorRetF, [1 size(retF, 2)])), 2),
[1, size(retF, 2)]);
```

이 트레이딩 모델은 연율 수익률 48퍼센트와 샤프 비율 0.9를 산출합니다. 그 자산곡선은 그림 3.3을 참조하십시오.

우리는 종종 가격 Y 자체가 아니라 가격의 변화 $\Delta Y$를 예측하고자 합니다. 따라서 VAR 모델을 사용하는 것은 다소 어색하며, 그 결과로 얻어지는 AR 계수도 직관적으로 큰 의미를 갖지는 않습니다. 다행히 $\operatorname{VAR}(p)$는 $\Delta Y$를 종속변수로 하고, 다양한 지연된 $\Delta Y$들과 $Y$들을 독립변수로 갖는 모델로 변환될 수 있습니다. 이를 ${\mathrm{VEC}}(q)$ (벡터 오차수정, vector error correction) 모델이라고 하며, 다음과 같이 씁니다.

$$
\Delta Y(t) = M + C Y(t - 1) + A_{1} \Delta Y(t - 1) + \cdot \cdot \cdot + A_{k} \Delta Y(t - k) + \mathfrak{e}(t).\tag{3.5}
$$

식 3.5의 $m \times m$ 행렬 C를 오차수정 행렬이라고 합니다. $\operatorname{VAR}(p)$의 계수를 ${\mathrm{VEC}}(q)$로 변환하려면, 먼저 $q = p - 1$임에 유의하고 함수 vartovec를 사용할 수 있습니다. 이를 위에서 컴퓨터 하드웨어 주식에 대해 구축한 VAR 모델에 적용하면 다음과 같습니다.

[model\_vec, C]=vartovec(model);

컴퓨터 하드웨어 SPX 주식에 대한 VAR(1) 모델  
![](images/1e7bf8ef43f24da9f35f46f64bbc9a42561cbc4c0a28e622c0b0a61dfbf4a4e9.jpg)  
그림 3.3 컴퓨터 하드웨어 주식에 적용한 VAR(1) 매매 전략

C의 값을 표시한 표 3.4를 얻습니다.

표 3.4 컴퓨터 하드웨어 주식에 적용한 VEC(0) 모델의 오차수정 행렬
<table><tr><td> $c_{i,j}$ </td><td>AAPL</td><td>EMC</td><td>HPQ</td><td>NTAP</td><td>SNDK</td></tr><tr><td>AAPL</td><td>-0.0082</td><td>0.0736</td><td>-0.1057</td><td>0.0360</td><td>-0.0062</td></tr><tr><td>EMC</td><td>-0.0001</td><td>-0.0291</td><td>-0.0103</td><td>0.0052</td><td>0.0035</td></tr><tr><td>HPQ</td><td>-0.0016</td><td>-0.0241</td><td>-0.0344</td><td>0.0090</td><td>0.0019</td></tr><tr><td>NTAP</td><td>-0.0008</td><td>-0.0409</td><td>-0.0284</td><td>0.0066</td><td>0.0031</td></tr><tr><td>SNDK</td><td>-0.0005</td><td>-0.0579</td><td>-0.0310</td><td>0.0170</td><td>-0.0013</td></tr></table>

C의 값은 서로 다른 주식들의 움직임 사이 관계에 대해 더 직관적인 이해를 제공합니다. NTAP를 제외하면 모든 대각 원소가 음수 값을 갖는다는 점을 알 수 있습니다. 이는 NTAP를 제외한 모든 주식이 자기상관적으로 평균회귀적이며, 다만 일부는 그 정도가 매우 약하다는 것을 의미합니다.

식 3.5는 Chan (2013)의 식 2.7과 동일하며, 그곳에서는 공적분에 대한 Johansen 검정과 관련하여 논의되었습니다. 실제로 컴퓨터 하드웨어 주식 포트폴리오가 공적분한다면, C는 Johansen 검정에서 유의하게 음수인 고윳값을 발생시킬 것입니다. 그러나 예측에 VEC(q)를 사용하기 위해 공적분 포트폴리오가 필요한 것은 아닙니다. 표 3.4에서 보았듯이, 일부 주식은 추세를 보일 수 있고 다른 주식은 평균회귀적일 수 있습니다.

그런데 컴퓨터 하드웨어 주식만이 아니라 전체 SPX 유니버스에 VAR 모델을 시도해 보고 싶다면, 컴퓨터에 이례적으로 큰 메모리가 있는지 반드시 확인해야 합니다! 또한 앞서 언급했듯이, 이러한 모델은 가격 대신 로그 가격을 사용하면 더 잘 작동할 수 있습니다. (어쨌든 로그 가격 표현은 VAR 및 VEC의 연속시간 버전과 더 잘 연결될 수 있게 해 줍니다. Cartea, Jaimungal, and Penalva, 2015, p. 285를 참조하십시오.)

### 상태공간 모델 (State Space Models)


지금까지 살펴본 AR, ARMA, VAR, VEC 모델은 모두 관측 가능한 변수(여러 지연 시점의 가격)를 사용하여 그 미래 값을 예측합니다. 그러나 계량경제학자들은 상태(states)라고 불리는 은닉 변수 (hidden variables)를 포함하는 모델의 한 부류도 고안해 왔으며, 이러한 변수는 관측 잡음 (observation noise)의 영향을 받기는 하지만 관측 변수의 값을 결정할 수 있습니다. 이러한 모델을 상태공간 모델 (SSM)이라고 하며, 그 선형적 예가 칼만 필터 (Kalman filter)입니다. 이는 Chan (2013)의 3장에서 논의되며, 이 책의 5장에서 사용됩니다. 비선형 상태공간 모델도 있을 수 있지만, 이 절에서는 선형 버전만 논의합니다.

상태공간 모델은 보통 x로 표시되는 은닉 상태 변수의 시간적 진화를 명시하는 선형 관계에서 출발합니다.

$$
x(t)=A(t)*x(t-1)+B(t)*u(t)\tag{3.6}
$$

여기서 x는 m차원 벡터이고, A(t)와 B(t)는 시간 의존적일 수 있지만 관측 가능한 행렬입니다(A는 m × m이고, B는 m × k입니다). 또한 u(t)는 평균이 0이고, 분산이 1이며, 자기상관과 교차상관이 0인 k차원 가우스 백색잡음입니다. 식 3.6은 흔히 상태 전이 방정식 (state transition equation)이라고 불립니다. 관측 가능한 변수(측정값이라고도 함)는 또 다른 선형 방정식에 의해 은닉 변수와 관련됩니다.

$$
y(t)=C(t)*x(t)+D(t)*\varepsilon(t)\tag{3.7}
$$

여기서 y는 n-벡터이고, C(t)와 D(t)는 시간 의존적일 수 있지만 관측 가능한 행렬입니다(C는 $n \times m$이고, D는 $n \times h$입니다). 또한 ε(t)는 h차원 가우스 백색잡음으로, 역시 평균이 0이고, 분산이 1이며, 자기상관과 교차상관이 0입니다. 식 3.7은 흔히 측정 방정식 (measurement equation)이라고 불립니다.

이러한 은닉 변수란 무엇이며, 왜 우리는 그 존재를 가정하고자 할까요? 은닉 변수의 한 예는 익숙한 이동평균입니다. 우리는 보통 고정된 수의 지연 가격을 사용하여 가격의 이동평균을 계산하므로, 그것이 겉보기에는 관측 가능한 변수처럼 보입니다. 그러나 이러한 고정된 지연 수는 인위적인 구성물이라고 주장할 수 있습니다. 또한 이동평균 대신 지수이동평균을 사용하지 않을 이유는 무엇일까요? 표준적이고 유일한 이동평균 변수에 대해 누구도 합의할 수 없다는 사실은 그것이 은닉 변수로 취급될 수 있음을 시사합니다. 우리는 이 은닉 변수 x가 특히 단순한 방식으로 진화하도록 요구함으로써 그것에 어느 정도 구조를 부여할 수 있습니다.

$$
x(t)=x(t-1)+B*u(t)\tag{3.8}
$$

우리는 A(t)가 항등행렬이라고 가정했으며, 이는 물론 시간에 대해 불변입니다. 또한 B는 알려지지 않았지만 역시 시간 불변인 행렬로, 이동평균 x에 대한 추정 오차의 공분산을 결정합니다. (u 자체는 항등행렬인 공분산 행렬을 가진다는 점을 기억하십시오.) 앞서 B가 관측 가능해야 한다고 말했지만, 이는 학습 데이터에 최대우도추정 (maximum likelihood estimation)을 적용하여 추정해야 하는 미지의 모수로 취급될 수 있습니다. (다시 말해, B는 칼만 필터 업데이트 동안 각 시간 단계에서 그 값이 갱신되지 않는다는 정도에서만 “관측 가능”합니다.)

시계열의 이동평균 (moving average)(시계열이 다변량인 경우 복수의 이동평균)이 주어지면, 트레이더는 가격이 추세를 보인다고 가정할 수 있으며, 따라서 시간 $t$에서 관측된 가격에 대한 최선의 추정치는 시간 $t$에서 추정된 이동평균 그 자체입니다.

$$
y(t) = x(t) + D * \varepsilon(t)\tag{3.9}
$$

여기서 $D$는 MLE로 추정해야 하는 또 다른 미지의 시간 불변 행렬입니다.

이제 VAR(p)에 관한 절에서 살펴본 동일한 컴퓨터 하드웨어 주식 가격 계열에 적용하여, 식 3.8과 3.9의 이 “이동평균” 모델이 실제로 어떻게 작동하는지 살펴보겠습니다. 우리는 컴퓨터 하드웨어 산업군의 주식 수와 동일한 수의 은닉 상태 변수 (hidden state variables)(총 다섯 개)가 있다고 가정할 것입니다. 이는 전형적인 이동평균 모델도 가정하는 바입니다. 즉, 각 가격 계열은 자체적으로 독립적인 이동평균을 갖습니다. 더 나아가, 우리는 한 이동평균의 상태 잡음 (state noise)이 다른 어떤 이동평균의 상태 잡음과도 상관되지 않지만, 각각은 서로 다른 분산을 가질 수 있다고도 가정합니다. 따라서 $B$는 미지의 매개변수를 가진 $5 \times 5$ 대각행렬입니다. (미지의 매개변수는 MATLAB estimate 함수의 입력에서 NaN으로 표시됩니다.) 마찬가지로, 우리는 한 주식 가격의 측정 잡음 (measurement noise)이 다른 주식의 측정 잡음과 상관되지 않지만, 각각 또한 서로 다른 분산을 가질 수 있다고 가정할 것입니다. 따라서 $D$ 역시 미지의 매개변수를 가진 $5 \times 5$ 대각행렬입니다. 상태 잡음과 측정 잡음에 대한 이 무상관 제약을 완화할 수도 있었지만, 이는 추정해야 할 변수가 훨씬 많아짐을 의미하며, 최적화에 걸리는 시간을 크게 늘리고 과적합 (overfitting)의 위험을 크게 증가시킵니다.

상태 잡음과 측정 잡음의 미지의 분산($B$와 $D$의 매개변수)에 대한 추정치를 생성하기 위해 estimate 함수6을 사용하는 코드 조각은 다음과 같습니다.

```matlab
A=eye(size(y, 2)); % State transition matrix
B=diag(NaN(size(y, 2), 1))
C=eye(size(y, 2)); % Time-invariant measurement matrix
D=diag(NaN(size(y, 2), 1))
model=ssm(A, B, C, D);
param0=randn(2*size(B, 1)̂2, 1); % 50 unknown parameters per bar.
model=estimate(model, y(trainset, :), param0);
```

이는 표 3.5에 제시된 값을 생성합니다.

이 경우, $u(t)$와 $\varepsilon(t)$ 잡음이 교차상관 없이 영평균을 중심으로 대칭적으로 분포하므로, $B$와 $D$ 행렬의 대각 원소들의 부호는 중요하지 않습니다. 또한 가우스 잡음 가정이 더 타당해지도록, 대신 로그 가격에 SSM을 적용하는 것도 고려할 수 있습니다.

표 3.5 B 및 D 행렬의 추정값(비대각 원소는 0)
<table><tr><td>Bi,j</td><td>u1</td><td>u2</td><td>u3</td><td>u4</td><td></td><td>u5</td></tr><tr><td>X1</td><td>-3.74</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>X2</td><td></td><td>0.34</td><td></td><td></td><td></td><td></td></tr><tr><td>X3</td><td></td><td></td><td>-0.73</td><td></td><td></td><td></td></tr><tr><td>X4</td><td></td><td></td><td></td><td></td><td>-0.67</td><td></td></tr><tr><td>X5</td><td></td><td></td><td></td><td></td><td></td><td>-1.00</td></tr><tr><td>i,</td><td>x1</td><td></td><td>X2</td><td>x3</td><td>X4</td><td>X5</td></tr><tr><td>AAPL</td><td>-0.0000454</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>EMC</td><td></td><td></td><td>-0.08</td><td></td><td></td><td></td></tr><tr><td>HPQ</td><td></td><td></td><td></td><td>0.22</td><td></td><td></td></tr><tr><td>NTAP</td><td></td><td></td><td></td><td></td><td>0.19</td><td></td></tr><tr><td>SNDK</td><td></td><td></td><td></td><td></td><td></td><td>-0.15</td></tr></table>

상태 전이 방정식과 측정 방정식이 고정되면, filter 함수를 사용하여 상태값과 관측값 모두의 예측을 생성할 수 있습니다.

```javascript
[x, logL, output]=filter(model, y);
```

filter 함수의 출력에서 x(t) 변수는 시간 t까지의 관측 가격이 주어졌을 때 시간 t의 필터링된 가격(이동평균)입니다. 이 모델은 관측 가격과 매우 유사한 필터링된 가격을 생성하며, 대개 차이는 0.1퍼센트 미만입니다. 방정식 (3.8)과 (3.9)를 고려하면, 이는 다음 날 가격에 대한 우리의 예측도 오늘의 가격과 매우 유사할 것임을 의미합니다. 관측 가격이 주어졌을 때 시간 t의 이러한 예측 가격은 output(t).ForecastedObs에서 추출할 수 있습니다.

```matlab
for t=1:length(output)
yF(t, :)=output(t).ForecastedObs';
```

여기서 우리는 앞서 사용했던 것과 동일한 관례를 사용하여 시간 t의 예측 가격을 yF(t − 1)에 할당합니다. 이러한 예측 가격으로부터 예측 수익률을 계산할 수 있습니다.

retF=(yF-y)./y;

retF(t)는 시간 t − 1의 관측 가격 y가 주어졌을 때 t − 1에서 t까지의 예측 수익률이라는 점에 유의하십시오. 이러한 예측 수익률은 VAR 모델에서 했던 것과 동일한 방식으로 섹터 중립 거래 전략을 만드는 데 사용할 수 있습니다. 그림 3.4에는 학습 세트에서의 모델 누적 수익률을 제시하고, 그림 3.5에는 테스트 세트에서의 누적 수익률을 제시합니다. 단지 학습 데이터를 사용하여 상태 잡음과 측정 잡음의 분산을 추정했을 뿐이라는 점을 고려하면, 과적합의 정도는 놀랍습니다.

이동평균을 찾는 것이 칼만 필터 (Kalman filter)를 가격 예측에 사용할 수 있는 유일한 방법은 아닙니다. 추세적 움직임을 가정한다면, 최근 가격 추세의 기울기를 찾는 데에도 이를 사용할 수 있으며, 이는 그 기울기가 지속된다고 가정한 다음 가격의 예측으로 이어집니다. 이는 독자의 연습문제로 남겨 둡니다.

칼만 필터를 사용하여 관측값을 예측하는 것이 이를 거래에 적용하는 유일한 방법은 아닙니다. 은닉 상태 (hidden state) 자체의 추정값도 유용할 수 있습니다. 결국 그것은 이동평균이어야 하기 때문입니다. 잡음이 있는 상황에서 은닉 변수의 추정값을 찾는 것은 필터링의 원래 의미이며, 신호 처리에서 잘 알려진 개념입니다. 칼만 필터 외에도 금융 및 경제학에서 잘 알려진 다른 필터로는 Hodrick-Prescott 필터와 웨이블릿 필터가 있습니다.

학습셋: 컴퓨터 하드웨어 SPX 주식에 대한 칼만 필터 모델  
![](images/4fc3230ed9388e4c08f9acc9be9f96d0f25ce1dbe63be92dfa21cfd89b82341e.jpg)  
그림 3.4 컴퓨터 하드웨어 주식에 적용한 칼만 필터 거래 전략(표본 내)

![](images/2be4ec13438ab8b951098615aa8ec952f188bfb41c679e1efb60b1d54570b6e8.jpg)  
그림 3.5 컴퓨터 하드웨어 주식에 적용한 칼만 필터 거래 전략(표본 외)

칼만 필터링의 또 다른 적용은 Chan (2013)에서 논의된 바 있는데, 여기서는 두 공적분된 (cointegrated) 가격 시계열 사이의 헤지 비율 (hedge ratio)에 대한 최적 추정값을 찾는 데 사용되었습니다. 그곳에서 제시된 예는 ETF EWA($T \times 1$ 벡터)와 EWC(역시 $T \times 1$ 벡터)의 가격 시계열이며, 이들은 다음과 같은 관계가 있다고 가정됩니다.

$$
[\mathrm{EWC}] = [\mathrm{EWA}, 1] * \left[ \begin{array}{c} {\mathrm{hedge\ ratio}} \\ {\mathrm{offset}} \end{array} \right] + \mathrm{noise}
$$

그러나 두 가격 시계열을 측정값으로 취급하는 대신, EWC를 측정값 $y$로 취급하고, 1들이 추가된 EWA를 식 3.7의 $J{:}$ 시간가변 행렬 $C(t)$로 취급합니다. (1들은 EWA와 EWC 사이의 선형 회귀 관계에서 상수 절편 (offset)을 허용하기 위해 필요합니다.) 우리는 헤지 비율과 둘 사이의 상수 절편을 은닉 상태 $x$로 취급합니다. 따라서 다음을 얻습니다.

$$
x(t) = x(t - 1) + B * u(t)\tag{3.10}
$$

$$
y(t) = C(t) * x(t) + D * \varepsilon(t)\tag{3.11}
$$

여기서 $x$는 $2 \times 1$ 시간가변 벡터 $[\mathrm{hedge\ ratio}, \mathrm{offset}]'$이고, $y$는 스칼라 $[\mathrm{EWC}(t)]$이며, $C(t)$는 $1 \times 2$ 시간가변 행렬 $[\mathrm{EWA}(t), 1]$입니다. 이러한 명세에 대한 MATLAB 코드 조각은 다음과 같습니다.

```prolog
load('inputData_ETF', 'tday', 'syms', 'cl');
idxA=find(strcmp('EWA', syms));
idxC=find(strcmp('EWC', syms));
y=cl(:, idxC);
C=[cl(:, idxA) ones(size(cl, 1), 1)];
A=eye(2);
B=NaN(2);
C=mat2cell(C, ones(size(cl, 1), 1));
D=NaN;
```

여기서 NaN은 알려지지 않은 매개변수를 나타냅니다. 이전과 마찬가지로, 이러한 알려지지 않은 매개변수는 2006년 4월 26일부터 2012년 4월 9일까지의 학습셋에 estimate 함수7을 적용하여 추정됩니다.

trainset=1:1250;   
model=ssm(A, B, C(trainset, :), D);

그리고 $B$ 행렬은 표 3.6에 표시되어 있으며, 스칼라 $D$는 −0.08로 추정됩니다. 표 3.5와 달리, 우리는 상태 잡음이 0의 교차상관을 가진다는 제약을 부과하지 않습니다.

<table><tr><td>표 3.6</td><td>B에 대한 추정값</td></tr><tr><td> $\mathbf{B_{i,j}}$ </td><td> $\mathbf{}_{u_{1}}$   $\mathbf{} u_{2}$ </td></tr><tr><td> $x_{1}$ </td><td>-0.01 0.02</td></tr><tr><td> $\mathbf{} x_{2}$ </td><td>0.41 -0.32</td></tr></table>

이러한 잡음항은 Chan (2013)의 박스 3.1에서 우리가 가정했던 것들과 현저히 다르다는 점에 유의하십시오. 그곳에서는 헤지 비율에 대한 상태 혁신 잡음 (state innovation noise) $\omega_{1}(t)$와 절편에 대한 $\omega_{2}(t)$가 서로 상관되어 있지 않으며, 각각의 분산이 약 0.0001과 같다고 가정했습니다. 그러나 여기서는 $\omega_{1}(t) = -0.01 * u_{1}(t) + 0.02 * u_{2}(t)$이고 $\omega_{2}(t) = 0.41 * u_{1}(t) - 0.32 * u_{2}(t)$라고 추정했으며, $u_{1}(t)$와 $u_{2}(t)$가 서로 상관되어 있지 않다고 가정되므로, $\omega\mathrm{s}$는 다음 공분산 행렬을 가집니다.

$$
\left[ { \begin{array}{rr} {0.00055} & {-0.011} \\ {-0.011} & {0.27} \end{array} \right].
$$

마찬가지로, 측정 잡음 (measurement noise) $\varepsilon(t)$의 분산을 임의로 0.001로 설정하는 대신, 이제 그것이 $D^{2} = 0.0059$라고 추정했습니다. 이러한 추정값을 사용하고 데이터에 filter 함수를 적용하면, 처음에는 Chan (2013)의 그림 3.5 및 3.6과 상당히 달라 보이지만 결국 유사한 값으로 수렴하는 기울기(그림 3.6)와 절편(그림 3.7)의 추정값이 생성됩니다. 이제 이전 논의에서 설명한 것과 동일한 거래 전략을 적용할 수 있습니다. 즉, 관측된 $y$의 값이 관측값의 예측 표준편차보다 더 크게 예측값보다 작다고 판단되면 EWC($y$)를 매수하는 동시에 EWA를 공매도하고, 그 반대의 경우도 마찬가지입니다.

![](images/6cd10f613f8cc7b4faf5d9c34217a0c9819c1a18b09f9fe5f2781fdf874a819e.jpg)  
그림 3.6 EWC와 EWA 사이 기울기의 칼만 필터 추정치

![](images/c047abeb280cdfed5d71c104adf370fd5d41ddd7ce3a3e7b1fb0a3927b0035de.jpg)  
그림 3.7 EWC와 EWA 사이 오프셋의 칼만 필터 추정치

```matlab
yF=NaN(size(y));
ymse=NaN(size(y));
for t=1:length(output)
yF(t, :)=output(t).ForecastedObs';
ymse(t, :)=output(t).ForecastedObsCov';
e=y-yF; % forecast error
longsEntry=e < -sqrt(ymse); % a long position means
we should buy EWC
longsExit=e > -sqrt(ymse);
shortsEntry=e > sqrt(ymse);
shortsExit=e < sqrt(ymse);
```

EWC와 EWA의 실제 포지션 결정은 Chan (2013)에서와 동일하며, MATLAB 코드는 SSM\_beta\_EWA\_EWC.m으로 다운로드할 수 있습니다. 이 전략의 학습 집합과 테스트 집합에서의 누적 수익률은 각각 그림 3.8과 3.9에 제시되어 있습니다.

![](images/fdf1d8d3ac832eafafe0e6988d6ae08b0c15c0b8aed4dd26c401d6045f15b8f9.jpg)  
그림 3.8 EWC–EWA에 적용한 칼만 필터 거래 전략(표본 내)

![](images/27c4675f6cba50aae8ed415a300c88d576972b250b20ae1c2caedcc42d7a92fb.jpg)  
그림 3.9 EWC–EWA에 적용한 칼만 필터 거래 전략(표본 외)

자본 곡선이 학습 집합의 후반부에서조차 평탄해지기 시작했음을 알 수 있습니다. 이는 EWA와 EWC가 공적분 관계에서 벗어난 체제 변화의 결과였을 수도 있고, 더 가능성이 높게는 잡음 공분산 행렬 B를 과적합한 결과였을 수도 있습니다.

### ■ 요약


시계열 분석은 완전히 새로운 금융상품이나 시장에 직면했을 때, 그리고 아직 그에 대한 직관이 전혀 형성되지 않았을 때 가장 먼저 시도해야 할 기법입니다. 우리는 많은 퀀트 트레이더의 전략에 자리 잡은 가장 널리 쓰이는 시계열 선형 모델들 중 일부를 살펴보았습니다. 이러한 모델들은 선형적임에도 불구하고 추정해야 할 매개변수가 많은 경우가 흔하므로, 과적합은 항상 위험 요소입니다. 이는 상태공간 모델 (state space model)의 경우 특히 그러한데, 추정해야 하는 자체 동역학을 가진 추가적인 숨은 변수가 존재하기 때문입니다. 이러한 방법을 전략 구축에 성공적으로 적용하려면 미지의 매개변수 수를 줄이기 위해 신중한 제약을 부과해야 합니다. ARMA 또는 VAR 모델의 경우 널리 쓰이는 제약은 지연 수를 1로 제한하는 것이며, SSM의 경우에는 잡음 간 교차상관이 0이라고 가정하는 것입니다. 제약을 부과하는 것 외에, 대량의 데이터로 모델을 학습시키는 것이 궁극적인 해결책이며, 이는 이러한 모델들이 장중 거래에서 유망함을 시사합니다.

### 연습문제


3.1. 식 3.1의 $\mathrm{AR}(1)$ 과정에서 $Y(t)$가 약정상 (weakly stationary)이라면 $|\Phi| < 1$임을 보이십시오. 힌트: $Y(t)$의 분산을 고려하십시오.

3.2. $\operatorname{AR}(p)$에 관한 절에서, 중간가격을 사용하여 158퍼센트의 연평균 성장률 (CAGR)을 달성한 $\operatorname{AR}(1)$을 이용한 AUD.USD 백테스트를 설명했습니다. 동일한 .mat 데이터셋에는 매수 및 매도 호가도 별도로 포함되어 있습니다. 시장가 주문만 사용한다고 가정하고 동일한 전략을 백테스트하십시오. 그 결과 연평균 성장률은 얼마입니까?

3.3. MATLAB의 arima 및 estimate 함수를 사용하여, AUD.USD의 로그 수익률을 모델링하는 데 $\mathrm{ARIMA}(p,0,q)$를 사용하면 로그 가격을 모델링하는 데 $\mathrm{ARIMA}(p,1,q)$를 사용할 때와 동일한 자기회귀 계수가 얻어짐을 확인하십시오. 또한 $P$와 $q$에 대한 최적 추정값이 각각 1과 9임을 보이십시오.

3.4. EWA와 EWC에 VAR 모델을 적용하고, 예측된 일일 수익률이 양수/음수일 때 일일 매수/매도 거래 신호를 생성하십시오. 각 ETF당 항상 \$1를 거래한다고 가정할 때, 연평균 성장률과 샤프 지수 (Sharpe ratio)는 얼마입니까? 두 ETF의 거래 신호가 같은 부호를 갖는 시점이 있습니까?

3.5. 식 (3.8)과 (3.9)로 생성된 이동평균을 N일 지수이동평균(예: en.wikipedia.org/wiki/Moving\_average 참조)과 비교할 때, 추정된 상태 변수에 가장 잘 맞는 N은 무엇입니까? 더 큰 N을 강제하기 위해 식 3.8과 3.9의 B 또는 D 행렬에 어떤 제약을 적용해야 합니까?

3.6. 식 3.9에서 B가 대각행렬이라고 가정한다면, 2006년 4월 26일부터 2012년 4월 9일까지의 데이터를 사용하여 연평균 성장률 26.2퍼센트와 샤프 지수 2.4를 갖는 EWC 대 EWA 칼만 필터 거래 전략을 백테스트할 수 있습니까? (이는 우리가 Chan, 2013에서 얻은 결과입니다.)

3.7. 가격 대신 로그 가격을 사용하여, VAR(p)에 관한 절에서 보인 것처럼 컴퓨터 하드웨어 주식에 VAR과 VEC를 적용하십시오. 표본외 수익률과 샤프 지수가 개선됩니까?

3.8. 가격의 이동평균을 찾는 데 칼만 필터를 사용하는 대신, 최근 가격 추세의 기울기를 찾는 데 사용하십시오. 이 기울기가 미래에도 지속된다고 가정하고, 예를 들어 컴퓨터 하드웨어 주식에 대해 추세추종 전략을 백테스트하십시오.

### ■ 미주


1. 시계열은 시간 이동에 의해 그 행동의 모든 측면이 변하지 않으면 엄격 정상적(strictly stationary)입니다(Ruppert and Matteson, 2015). 약한 정상성을 갖는 시계열은 평균과 분산이 변하지 않을 것만 요구합니다. 이것이 다변량 시계열이라면 공분산도 변하지 않아야 합니다.

2. 이 전체 프로그램은 buildAR1.m으로 다운로드할 수 있습니다.

3. 전체 코드는 buildARp\_AUDUSD.m으로 다운로드할 수 있습니다.

4. 전체 코드는 buildVAR\_findP\_stocks.m으로 다운로드할 수 있습니다.

5. 전체 코드는 buildVAR1\_sectorNeutral\_ computerHardware.m으로 다운로드할 수 있습니다.

6. 전체 코드는 SSM\_MA\_computer Hardware\_diag.m으로 다운로드할 수 있습니다.

7. 전체 코드는 SSM\_beta\_EWA\_EWC.m으로 다운로드할 수 있습니다.
