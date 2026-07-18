# Machine Trading 2016 Chapter 5 — Options Strategies

## 1. 결론 먼저

공식 저자 ZIP의 40개 멤버를 SHA-256으로 고정했다. 그중 현재 번들만으로 수치 계보를 끝까지 추적할 수 있는 `compareVolWithVXX.m`과 `XIV_SPY_rollreturn_lagged.m`은 정확 재현했다. 7개 비교가 모두 원본의 6자리 출력 반올림 허용오차 `5e-7` 안에 있다. 나머지는 데이터 라이선스나 버전 불일치가 있으므로 output-only 참조와 독립 Python 실험을 명시적으로 분리했다.

이 결과는 투자 조언이 아니며 실거래 백테스트가 아니다. 특히 합성 감마 스캘핑은 경로 의존성과 거래비용을 설명하는 교육 실험이다.

## 2. 데이터 구조와 provenance

- 출처: https://epchan.com/book3
- 공식 ZIP: https://epchan.com/img/book3/Chap5%20Options.zip
- archive SHA-256: `3ffe46a1c8ed37e5611e114b02ab1a714efb676c9ecca2548083aa7363da590f`
- 멤버: 40개 = MATLAB 30 + MAT 데이터/출력 10
- ETF 패널: 2500행, 26종목, 20051221~20151125, 결측 3,401개
- VX 패널: 2877행, 141열, 20040326~20150828
- environment versions와 `uv.lock` SHA는 metrics.json에 기록했다.

## 3. 학습 질문

1. 실현변동성 증가와 VXX 가격 상승을 같은 사건으로 취급해도 되는가?
2. 옵션 델타 레버리지는 행사가와 만기에 따라 얼마나 급해지는가?
3. 시작가와 종가가 같아도 감마 스캘핑 손익이 달라지는 이유는 무엇인가?
4. 데이터가 없을 때 원본 출력값과 재현값을 어떻게 구별해야 하는가?

## 4. 구현 범위와 재현 상태 coverage

| topic | status | evidence |
|---|---|---|
| realized-volatility versus VXX direction | exact source replay | bundled 20151125 ETF panel |
| XIV-SPY Kalman roll-return hedge | exact source replay | bundled VX and hedge-ratio MAT files |
| Black-Scholes, Greeks, leverage, IV inversion | independent Python experiment | parity and finite-difference checks |
| gamma-scalping path and costs | conceptual synthetic experiment | same endpoints, different paths |
| SPY GARCH | overlap diagnostic + output-only reference | original input version absent |
| VX-SPY Kelly comparison | output-only reference | input version absent and source metric mixing |
| event-driven CL/LO options | code/output-only | licensed tick data unavailable |
| cross-sectional IV and dispersion | output-only reference | incomplete historical option panels |

## 5. 수식 → 코드

Black–Scholes 콜 가격과 put-call parity는 다음과 같다.

$$
C=S\Phi(d_1)-K e^{-rT}\Phi(d_2), \qquad C-P=S-Ke^{-rT}.
$$

`black_scholes()`가 가격·델타·감마·베가·세타를 한 번에 계산한다. `implied_volatility()`는 관측 콜 가격에서 변동성을 역산한다. 중앙 유한차분으로 델타와 감마를 독립 확인했다.

- parity error: 1.421e-14
- delta finite-difference error: -1.710e-10
- gamma finite-difference error: -3.148e-09
- true/recovered volatility: 0.250000 / 0.250000

## 6. 원본 MATLAB 정확 비교

| source | metric | Python | MATLAB | abs error |
|---|---|---|---|---|
| compareVolWithVXX.m | all | 0.350680544 | 0.350681 | 4.56e-07 |
| compareVolWithVXX.m | positive_spy | 0.288419519 | 0.288420 | 4.81e-07 |
| compareVolWithVXX.m | negative_spy | 0.426273458 | 0.426273 | 4.58e-07 |
| XIV_SPY_rollreturn_lagged.m | annual_return | 0.128077502 | 0.128078 | 4.98e-07 |
| XIV_SPY_rollreturn_lagged.m | sharpe | 1.101081442 | 1.101081 | 4.42e-07 |
| XIV_SPY_rollreturn_lagged.m | max_drawdown | -0.131693278 | -0.131693 | 2.78e-07 |
| XIV_SPY_rollreturn_lagged.m | calmar | 0.972543961 | 0.972544 | 3.90e-08 |

XIV–SPY 전략은 20101130~20150819에 CAGR 12.8078%, Sharpe 1.1011, 최대낙폭 -13.1693%, Calmar 0.9725다. 신호에는 전일 VX/VIX를 쓰고 포지션은 다시 하루 늦게 손익에 반영한다. look-ahead를 막는 이 시차는 자동 검증한다.

## 7. VXX와 실현변동성은 같은 것이 아니다

전체 일자의 같은 방향 비율은 35.0681%뿐이다. VXX는 선물곡선, 롤, 만기 구조의 영향을 받으므로 다음 날 실현분산 변화와 동의어가 아니다. 높은 방향 정확도를 곧바로 거래 가능성으로 해석하면 selection bias가 생긴다.

## 8. 감마 스캘핑 경로·거래비용 실험

두 합성 경로는 시작과 끝이 모두 100이지만 중간 진동 횟수가 다르다. 델타 헤지는 직전 델타로 수행하고, 헤지 변경 명목에 0/1/5 bps transaction cost를 부과한다. 비용을 높였을 때 어느 경로에서도 손익이 개선되지 않는 것을 검증했다. 이 합성 실험은 CL/LO 백테스트가 아니며, 슬리피지·bid-ask·유동성·점프 위험을 실거래 수준으로 모델링하지 않는다.

## 9. output-only 원본 참조

| topic | compared | reason |
|---|---|---|
| Kelly-scaled long SPY versus short VX | false | The used ETF 20150828 panel is absent; the bundled 20151125 panel starts later. The source also overwrites both Kelly multipliers with 1 while retaining Kelly labels and combines unlevered CAGR with purportedly levered drawdown |
| SPY GARCH volatility forecast | false | variance_SPY.mat was produced from the missing 20150828 ETF panel; the later bundled panel cannot establish the same return history |
| event-driven crude-oil straddles and strangles | false | The book and source say the licensed Nanex tick files are not downloadable |
| crude-oil gamma-scalping strangle | false | The official archive contains code but not the licensed CL/LO tick data |
| cross-sectional implied-volatility mean reversion | false | The output MAT preserves state arrays but not the complete daily Ivy option files needed to reconstruct P&L |
| dispersion trading | false | The output MAT is a saved workspace state, not the complete historical option quote panel |

`VX_SPY_returns.m`은 Kelly 계수를 계산한 뒤 둘 다 1로 덮어쓰면서 Kelly 범례를 유지하고, 무레버리지 CAGR과 레버리지로 보이는 낙폭을 한 표에 섞는다. 공식 번들에는 그 코드가 실제로 읽은 20150828 ETF 패널도 없다. 따라서 이를 숫자 재현이라고 부르지 않는다.

## 10. 표본 외와 편향 경고

SPY GARCH 소스는 첫 1,500일을 훈련, 이후를 out-of-sample로 정의하지만, 저장된 `variance_SPY.mat`의 입력 버전이 빠져 있다. 제공된 20151125 ETF 패널과의 겹치는 구간은 diagnostic으로만 계산하며 `compared:false`다. 횡단면 전략은 survivorship bias, option selection bias, stale quote, corporate action, bid-ask, delisting을 모두 재검토해야 한다.

## 11. 위험지표와 한계

Sharpe, drawdown, Calmar는 경로를 요약할 뿐 꼬리 위험을 없애지 않는다. 옵션 매도는 짧은 표본에서 좋아 보이기 쉽고, 원유 이벤트 전략의 1년 결과는 일반화 근거가 약하다. 라이선스 제한 틱 데이터가 없으므로 이벤트·감마·dispersion·횡단면 IV의 책 수치는 `null/compared:false + reason + book_output` 계약으로만 보존한다.

## 12. 자동 검증

검증 22/22 통과. 엄격 비교, 시차, parity, 유한차분 Greeks, IV 역산, 거래비용 단조성, provenance와 누락 데이터 공개를 검사한다.

## 13. 결론

이 장의 핵심은 “옵션 공식”보다 실행 계약이다. 내재·실현변동성은 다르고, 레버리지는 정의를 섞으면 안 되며, 감마 스캘핑은 경로와 비용에 의존한다. 재현 불가능한 책 출력은 정직하게 output-only로 남기는 것이 임의의 근사값을 정확 재현처럼 포장하는 것보다 중요하다.
