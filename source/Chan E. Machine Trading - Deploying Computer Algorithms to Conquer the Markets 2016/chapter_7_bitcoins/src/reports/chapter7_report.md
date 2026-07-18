# Machine Trading 2016 Chapter 7 — Bitcoin

## 1. 결론 먼저

공식 ZIP 12개 member와 각 SHA-256을 고정했다. Bollinger 세 지표와 Bitstamp order-flow의 31개 일별 행·세 aggregate 지표는 원본과 정확히 일치한다. AR(16)은 MATLAB MLE 대신 conditional OLS로 근사해 계수와 성과가 출판값에 가깝지만, lag 선택과 ADF 구현 차이를 정확 재현으로 부르지 않는다. 높은 gross 수익은 10bp 비용에서 거의 전부 사라지므로 이 장의 핵심은 예측모형보다 시점·비용·거래소 위험이다.

## 2. provenance와 데이터

- 공식 페이지: https://epchan.com/book3
- archive SHA-256: `03267f6abd2fa35325d915e43677265b8b90da93fea0a02b16df4c7247b18dba`
- minute BBO: 479,535행, daily close: 345행, Bitstamp trade: 119,914행
- MATLAB source 9개, research data 3개

`HHMM`은 13,438행뿐이라 price의 479,535행과 맞지 않고, bid가 ask보다 큰 행도 517개다. 원본 전략이 midpoint만 쓰기 때문에 계산은 가능하지만, 보조 배열과 호가 품질의 이상을 숨기지 않는다.

## 3. 재현 분류와 coverage

| topic | status | evidence |
|---|---|---|
| BTC daily tail risk | independent source-data replay | 345 daily closes |
| AR(16), ADF, and forecast strategy | approximate replay + look-ahead audit | 479,535 minute mids and printed coefficients |
| AR/ARMA BIC selection | diagnostic + output-only source order | Yule-Walker AR(15) versus MATLAB MLE AR(16), ARMA(3,7) |
| Bollinger mean reversion | exact source replay | three published outputs |
| bagged regression trees | deterministic source-semantic adaptation | chronological half split and execution-cost stress |
| SVM and neural networks | code/output-only | nonportable MATLAB training defaults |
| Bitstamp order flow | exact source replay | 31 daily rows and three aggregate outputs |
| cross-exchange arbitrage | historical arithmetic illustration | book example; credit and transfer risk unquantified |

exact, approximate, source-semantic, output-only를 구분한다. MATLAB 숫자를 단순 복사한 항목은 Python 실험 결과로 가장하지 않는다.

## 4. BTC 위험

2014-01-20부터 2015-01-14까지 연율 변동성은 67.4%, 최악 일간수익률은 -24.5% (20150114), 최고는 19.7% (20140303), 최대 낙폭은 -79.2%다. fat tail과 79% 낙폭은 전략 수익률의 큰 숫자보다 먼저 봐야 할 위험이다.

## 5. AR(16), ADF, 그리고 look-ahead

원본은 훈련구간으로 lag 16을 고른 뒤 `estimate(model, mid)`로 전체 표본을 다시 fit한다. 따라서 test price가 계수 추정에 들어간다. 이를 source-faithful 경로로 보존하는 동시에 train-only correction을 별도 제공한다. Python Yule-Walker BIC는 lag 15를, MATLAB Gaussian MLE는 16을 고르므로 estimator가 다르면 model selection도 달라진다. ADF statistic -3.000998는 원본 5% 임계값 -2.871보다 작지만, 단일 표본과 가격수준 회귀만으로 안정적 시장 alpha를 보장하지 않는다.

## 6. 원본 MATLAB 비교

| topic | metric | Python | source | class |
|---|---|---|---|---|
| buildARp_BTCUSD.m | constant | 0.00670585459 | 0.00670585 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{1} | 0.685261199 | 0.685261 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{2} | 0.257701928 | 0.257702 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{3} | 0.0580413727 | 0.0580414 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{4} | 0.00443158709 | 0.00443159 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{5} | -0.00339982381 | -0.00339982 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{6} | -0.00496623755 | -0.00496624 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{7} | -0.0106181771 | -0.0106182 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{8} | -0.00189899279 | -0.00189899 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{9} | 0.00326403422 | 0.00326403 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{10} | 0.00457750093 | 0.0045775 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{11} | -0.00946253324 | -0.00946253 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{12} | 0.000956514594 | 0.000956515 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{13} | 0.00157189986 | 0.0015719 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{14} | -0.00177550115 | -0.0017755 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{15} | 0.00752874187 | 0.00752874 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | AR{16} | 0.00876892871 | 0.00876893 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | ADF statistic | -3.00099836 | -3.001533 | approximate Python OLS versus JPL/MATLAB ADF |
| buildARp_BTCUSD.m | test cumulative return | 201.957524 | 201.908136 | approximate conditional OLS versus MATLAB Gaussian MLE |
| buildARp_BTCUSD.m | test annual return | 41190.7565 | 41170.7116 | approximate conditional OLS versus MATLAB Gaussian MLE |
| bollinger.m | test cumulative return | 0.201024205 | 0.201024205 | exact source replay |
| bollinger.m | test annual return | 0.44245914 | 0.44245914 | exact source replay |
| bollinger.m | 10 bps test cumulative return | -0.994913612 | -0.994913612 | exact source replay |
| orderFlow2.m | total P&L USD | 32.54 | 32.54 | exact source replay |
| orderFlow2.m | P&L per trade USD | 0.0968452381 | 0.096845 | exact source replay |
| orderFlow2.m | number of trades | 336 | 336 | exact source replay |

AR 계수의 허용오차 5e-7은 출력 자릿수에 맞춘 근사 기준이다. ADF와 수익률은 JPL/MATLAB 및 MLE 차이를 반영한 별도 허용오차를 쓴다. Bollinger와 order flow는 훨씬 엄격한 1e-12/표시 반올림 기준이다.

## 7. Bollinger exact replay

60분 rolling mean, 표본 표준편차 `ddof=1`, 진입 z=2, 평균선 청산을 그대로 적용했다. test gross 누적수익은 0.201024204779, 연율수익은 0.442459140464로 원본과 일치한다. 그러나 position change마다 10bp를 차감하면 누적수익이 -99.491%가 된다. midpoint 체결, latency 0, market impact 0도 낙관적이므로 비용 포함 결과가 더 중요한 하한이다.

## 8. bagged tree 현대적 adaptation

1·5·10·30·60분 수익률로 다음 1분 수익률을 예측하고 시간순 절반 split, tree 5개, minimum leaf 100, seed 1을 고정했다. test gross 누적수익은 2.87e+07로 비현실적으로 크지만, 10bp turnover cost의 cumulative log return은 -198.5다. TreeBagger와 sklearn은 알고리즘이 달라 수치 비교하지 않으며, 높은 Sharpe를 경제적 타당성으로 해석하지 않는다.

## 9. selection bias와 데이터 누수

원본 AR full fit의 look-ahead는 명시적이다. tree in-sample prediction은 bootstrap tree가 본 관측치를 포함할 수 있어 독립 성능이 아니다. 여러 모델·lag·kernel·network 중 잘 보이는 결과를 선택하면 selection bias가 생긴다. 실전 검증에는 과거 window에서만 모델을 선택하고 다음 window에서 고정하는 walk-forward가 필요하다.

## 10. Bitstamp order flow exact replay

timestamp를 `datetime64[ns]`로 명시 변환하고, 현재 tick보다 60초 이전인 마지막 cumulative flow를 뺀다. Pandas 저장 단위를 암묵적으로 가정하면 60초가 잘못 스케일될 수 있다. 진입 threshold 90 BTC, exit 0, 포지션은 자정을 넘어 유지한다. 총손익 32.54달러, 336 legs, trade당 0.096845달러가 원본과 일치한다.

## 11. 거래비용과 원본 설명의 모순

책 본문은 spread 약 0.12달러와 one-way cost 0.06달러를 논하지만 `orderFlow2.m` 마지막 주석은 spread 약 0.5달러라고 적는다. 0.06달러를 각 leg에 적용하면 12.38달러, 0.25달러면 -51.46달러다. 어느 설명을 임의로 정답 처리하지 않고 둘 다 sensitivity로 제시한다.

## 12. cross-exchange arbitrage

역사적 예제에서 Bitfinex bid 239.19와 btc-e ask 233.546의 gross spread는 5.644달러다. 명시된 commission 1달러와 withdrawal 2달러를 빼면 2.644달러지만, transfer delay 동안의 가격위험, 거래소 credit risk, 출금 중단, 양쪽 inventory funding은 정량화되지 않았다. 현재 거래소나 가격에 대한 추천이 아니다.

## 13. output-only 참조

| topic | compared | reason |
|---|---|---|
| AR and ARMA lag selection | false | MATLAB Gaussian ARIMA maximum-likelihood BIC is not numerically equivalent to the lightweight Yule-Walker diagnostic; Python selects AR(15), so the source selections are preserved without claiming exact estimator equivalence |
| MATLAB TreeBagger results | false | scikit-learn RandomForestRegressor is a deterministic source-semantic adaptation, not MATLAB TreeBagger; split, lags, five trees and minimum leaf size are retained while numeric parity is not asserted |
| cross-validated SVM | false | the script selects one fold-trained model rather than refitting on all training data, and MATLAB fitcsvm defaults are not pinned sufficiently for a defensible cross-library replay |
| 100-network feed-forward ensemble | false | the MATLAB neural-network training algorithm and initialization contract are not portable to the pinned Python environment; the published output remains output-only |
| MXN, SPY, and HYG risk comparison | false | the official Chapter 7 archive contains BTC daily data but the script points to external local MXN and ETF files that are absent |

SVM은 각 fold model 중 하나를 골라 전체 훈련자료에 refit하지 않는다는 점도 주의한다. NN 100회와 ARMA 90개 MLE 탐색은 library 기본값과 실행 계약이 다르므로 출판값만 보존한다.

## 14. 위험지표와 연율화

crypto는 24/7인데 AR·Bollinger는 252일, tree/SVM/NN은 365일을 사용한다. 같은 minute return도 연율화 convention에 따라 CAGR과 Sharpe가 크게 바뀐다. 따라서 원본 비교용 convention은 보존하되, 경제적 비교에서는 cumulative return, drawdown, turnover와 비용을 함께 본다.

## 15. 자동 검증과 결론

검증 25/25 통과. checksum, member 수, 원본 출력, 31일 order-flow, datetime 단위 fixture, position lag, 비용 단조성, look-ahead 공개, 누락 입력의 output-only 처리를 검사한다. 이 노트북은 역사적 코드를 재현하고 함정을 진단하는 백테스트 실험이지 미래 수익을 증명하지 않는다. 결론은 단순하다. 암호자산에서는 model score보다 데이터 contract, 시간 단위, 체결비용, 거래소 생존과 출금 가능성이 먼저다.
