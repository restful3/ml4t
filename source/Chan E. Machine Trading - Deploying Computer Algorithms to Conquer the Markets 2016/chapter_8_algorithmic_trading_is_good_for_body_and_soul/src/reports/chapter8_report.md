# Machine Trading 2016 Chapter 8 — Algorithmic Trading Is Good for Your Body and Soul

## 1. 결론 먼저

Chapter 8에는 독립적인 공식 code/data ZIP이 없다. 따라서 숫자를 새로 발명하지 않고, 본문이 재검토하는 GLD-USO 전략을 공식 Chapter 5 ETF panel에서 재현했다. 이 panel의 2006-04-26~2012-04-09 GLD·USO 값 1,500행은 로컬 Chan 2013 CSV와 전부 동일하다. 원본 분석의 active-day APR 23.86%는 정확히 재현되지만, flat day를 포함한 calendar APR은 17.76%, 10bp 비용 후 14.43%다. 전략은 2012-2015에도 플러스였으나 더 낮아졌고, 이것은 2015 이후 생존을 의미하지 않는다.

## 2. provenance와 archive 부재

- 공식 Book 3 page: https://epchan.com/book3
- Chapter 8 전용 archive: 없음 (`official_archive_available=false`)
- 재사용 data: Chapter 5 official archive의 `data/raw/book3/chapter_5/inputDataOHLCDaily_ETF_20151125.mat`
- data SHA-256: `939245babc96c8dad661b6676ba3f07ea2b9d7b6ec627e702b1448f3e7e7e1b6`
- Chan 2013 source SHA-256: `99b8a6e8d40389130f4bcead274b1b45280b16491149ea21bf8c6953748b27a6`

전용 bundle이 없다는 사실은 누락이 아니라 장의 성격이다. lifestyle, service, business structure와 전략 lifecycle이 중심이며, 시장 데이터 계산은 GLD-GDX와 GLD-USO 예시뿐이다.

## 3. coverage와 재현 분류

| topic | status | evidence |
|---|---|---|
| GLD-USO 20-day Bollinger | cross-book empirical replay | official Book 3 Ch5 panel; values identical to Chan 2013 CSV over 1,500 days |
| active-day annualization | source-metric audit | 23.86% source clock versus 17.76% calendar clock |
| post-2012 strategy shelf life | temporal extension | 915 later observations through 2015-11-25 |
| source long-exit typo | code-semantic sensitivity | published Python uses -1 on the long exit while prose says exit z=0 |
| strategy births/deaths and diversification | deterministic conceptual simulation | 10 strategies, two alpha breaks, seed pinned |
| managed-account integer allocation | deterministic capacity illustration | four account sizes and ten margin requirements |
| GLD-GDX negative test | output-only | required 2006-2013 GDX panel absent |
| health, service, and regulation | narrative/output-only | not a market-data calculation |

GLD-USO는 cross-book empirical replay, 2012-2015는 temporal extension, 전략 pool과 account granularity는 conceptual simulation이다. GLD-GDX와 규제·건강 주장은 output-only다.

## 4. 데이터 진단

사용구간은 2,415일(20060426~20151125), asset 26개 중 GLD와 USO를 사용한다. 사용가격은 finite·positive이고 날짜는 엄격히 증가한다. GDX는 panel에 없으므로 Chapter 8 Figure 8.1의 2006-2013 negative test를 재현했다고 주장하지 않는다. GLD-USO 60일 상관도 시간에 따라 바뀌어, 경제적 story만으로 spread의 안정성을 가정할 수 없다.

## 5. GLD-USO 수식 → 코드

최근 20일 OLS로 $USO_t=\alpha_t+\beta_t GLD_t+\epsilon_t$를 적합하고 $S_t=USO_t-\beta_t GLD_t$를 구성한다. rolling sample mean/std의 z-score가 -1 아래면 long spread, +1 위면 short spread, 0에서 청산한다. signal은 오늘 종가로 계산하고 position은 다음 날 수익에 적용한다.

## 6. cross-book exact 비교

| metric | Python | source | error |
|---|---|---|---|
| annual_return | 0.238609066658 | 0.238609066658 | 0.0e+00 |
| sharpe | 1.106654196870 | 1.106654196870 | 0.0e+00 |
| maximum_drawdown | -0.218317700657 | -0.218317700657 | 0.0e+00 |

여기서 exact는 Machine Trading Ch8 자체 output이 아니라, Chan 2013 data와 수정된 chapter-analysis implementation의 active-day 결과에 대한 정확 일치다. 분류 범위를 좁게 쓴 이유다.

## 7. active-day clock 편향

원본 분석은 포지션이 없는 날의 return을 NaN으로 두고 삭제한 뒤 252일 연율화한다. active 1146일만 쓰면 APR 23.86%, Sharpe 1.107다. 전체 1500 calendar rows를 넣으면 APR 17.76%, Sharpe 0.967로 낮아진다. 투자자 자본의 시간은 flat day에도 흐르므로 calendar clock을 경제적 기본값으로 삼는다.

## 8. source long-exit typo

Chan 2013 `bollinger.py`는 `exitZscore=0`을 선언하지만 long exit에 `zScore > -entryZscore`를 사용한다. 수정된 chapter analyzer는 long/short 모두 0에서 청산한다. typo 경로의 calendar cumulative return 113.43%와 수정 경로 164.62%가 달라, 변수 이름이 아니라 실제 분기식을 감사해야 한다.

## 9. 거래비용과 표본 외

unit change마다 10bp를 부과하면 2006-2012 calendar APR은 14.43%, max drawdown은 -22.91%다. spread, slippage, borrow fee와 market impact는 별도이므로 여전히 낙관적이다. position lag는 지켰지만 hedge ratio·lookback·threshold가 같은 역사에서 선택되었다면 selection bias도 남는다.

## 10. shelf-life temporal extension

새로 시작한 2012-04-10~2015-11-25 915일 window에서 gross calendar APR 6.92%, 10bp 후 4.10%로 플러스다. 원래 기간보다 약해졌고 2015 이후 데이터는 없다. “계속 수익성”이라는 책의 당시 서술을 2026년 현재도 유효하다고 확대하지 않는다.

## 11. 전략 births/deaths 개념 실험

seed 20260718로 10개 synthetic strategy를 만들고 strategy 0은 day 500, strategy 1은 day 700 이후 alpha를 악화시켰다. 한 죽어가는 전략 집중의 max drawdown은 -61.9%, live pool equal-weight는 -6.0%다. 이는 diversification 원리를 설명하는 simulation이지 시장 backtest가 아니다.

## 12. drawdown budget와 look-ahead

각 strategy의 전일까지 wealth와 peak만으로 12% drawdown budget을 계산하고 오늘 weight를 정한다. 미래 return은 weight에 들어가지 않는다. adaptive pool의 성과가 좋은 것은 특정 seed의 교육용 결과이며 CPPI가 미래 성과를 보장한다는 증거가 아니다. 핵심은 죽은 전략의 레버리지를 자동으로 줄이는 state transition을 명시하는 것이다.

## 13. managed-account integer granularity

| account | active strategies | utilization | L1 error |
|---|---|---|---|
| $25,000 | 0 | 0.0% | 1.000 |
| $100,000 | 4 | 30.5% | 0.695 |
| $500,000 | 10 | 88.8% | 0.112 |
| $2,000,000 | 10 | 98.2% | 0.018 |

margin 요구액은 현재 quote가 아닌 illustrative fixture다. 작은 별도계좌는 목표 비중이 한 계약 margin보다 작아 0계약이 되고, 10전략 분산을 구현하지 못한다. 펀드와 managed account의 비교에는 transparency·trade-secret·principal-agent·규제가 더 있지만 이 표는 integer allocation 한 가지 메커니즘만 보여준다.

## 14. output-only와 법률 주의

| topic | compared | reason |
|---|---|---|
| GLD-GDX fixed-spread strategy death | false | Chapter 8 publishes a figure and directional statement but no Chapter 8 archive or the May 2006-May 2013 GDX panel; the available official Book 3 ETF panel does not contain GDX |
| health, autonomy, and productivity claims | false | these are narrative and cited lifestyle claims, not trading calculations that can be reproduced from the bundled market data |
| managed-account registration thresholds and service providers | false | the chapter describes a 2016 legal and business environment; this notebook provides no current legal, regulatory, or provider-availability advice |

Chapter 8의 투자자문 등록 threshold, capital-introduction service, provider 이름은 2016년 역사적 문맥이다. 현재 법률·서비스 상태를 확인하지 않았고 법률 또는 투자 조언으로 제공하지 않는다.

## 15. 자동 검증과 결론

검증 24/24 통과. checksum, asset/date/price, 1,500·915 rows, source active-day 숫자, calendar 보정, position lag, 비용 단조성, exit typo 차이, lifecycle seed/lag, account integer allocation과 output-only 사유를 검사한다. Chapter 8의 실험적 결론은 “한 전략이 영원히 산다”가 아니다. strategy가 태어나고 죽는다는 가정을 운영 규칙에 넣고, 성과 clock과 비용을 정직하게 정의하며, 충분한 independent pool과 실행 가능한 account size를 확보해야 한다.
