# Machine Trading 2016 Chapter 6 — Intraday Trading and Market Microstructure

## 1. 결론 먼저

공식 ZIP 7개 멤버를 SHA-256으로 고정했다. `tickRule.m`의 bid-ask 경로는 원본 4개 출력과 정확히 일치한다. 그러나 단 하루의 ES 데이터이므로 전략 수익성이나 out-of-sample Sharpe를 주장하지 않는다. 이 장의 실험 목적은 체결가격, 분류 규칙, volume clock, 주문장 상태가 결과를 어떻게 바꾸는지 검증하는 것이다.

## 2. provenance와 데이터 구조

- 공식 URL: https://epchan.com/img/book3/Chap6%20Intraday%20Trading.zip
- archive SHA-256: `e8bceb329afcc9fe2c61bcd06f553a5fa5dbe569716b22949354b5991b6c9e3e`
- 7개 = MATLAB 4 + 연구 데이터 3
- ES TAQ: 77,390행, 2012-10-01 하루
- ES 500계약 volume bars: 3,639행
- Coinsetter execution reports: 246,060행
- `uv.lock` SHA와 environment versions는 metrics.json에 기록했다.

## 3. 학습 질문

1. 같은 신호라도 midpoint와 bid-ask 체결이 왜 정반대 손익을 만드는가?
2. tick rule의 동가 거래 방향 전파는 order flow에 어떤 영향을 주는가?
3. 시간 bar 대신 500계약 volume bar를 쓰면 관측 clock이 어떻게 달라지는가?
4. 주문 추가·취소·부분체결로 best bid/ask를 어떻게 재구성하는가?

## 4. coverage와 재현 상태

| topic | status | evidence |
|---|---|---|
| tick-rule trade classification | exact source replay | 77,390-row ES TAQ MAT |
| bid-ask order-flow strategy | exact source-output comparison | four published outputs |
| midpoint execution | current-code replay + output-only comment | published trade counts contradict branch invariance |
| 500-contract volume bars/BVC | source-semantic replay | 3,639-row MAT; source entry/exit thresholds 0.95/0.5 |
| Coinsetter order-book reconstruction | independent Python port | 246,060 execution events |
| Algoseek aggressor tags | code/output-only | aggressor input file absent |
| multi-day strategy performance | unavailable | single ES trading day only |

## 5. 수식 → 코드: tick rule

가격이 직전 거래보다 오르면 매수, 내리면 매도, 같으면 직전 분류를 이어받는다. 60초 signed volume은 다음과 같다.

$$
OF_t=\sum_{t-60s < i \leq t} q_i s_i, \qquad s_i \in \{-1,+1\}.
$$

`tick_rule_order_flow()`와 `rolling_flow()`가 이 계약을 구현한다. 현재 이벤트까지 신호에 포함하고 현재 quote로 체결하므로 실거래 지연을 모델링한 예측 백테스트가 아니라 원본의 동일 시점 실행 예제다.

## 6. 원본 MATLAB 정확 비교

| metric | Python | MATLAB | error |
|---|---|---|---|
| gross_pl_usd | -6687.500000000 | -6687.5 | 0.00e+00 |
| pl_per_trade_usd | -5.619747899 | -5.619748 | 1.01e-07 |
| num_trades | 1190.000000000 | 1190 | 0.00e+00 |
| total_fee_usd | 113.050000000 | 113.05 | 0.00e+00 |

bid-ask gross P&L은 -6687.50달러, 거래 1190건, 최소 수수료 차감 후 -6800.55달러다. 반면 현재 코드 의미의 rounded midpoint gross P&L은 887.50달러다. 체결 가정 하나가 신호 효과보다 크다.

## 7. 원본 midpoint 주석 모순

`useMidPrice` 분기는 가격만 바꾸며 포지션 결정에는 관여하지 않는다. 따라서 midpoint와 bid-ask의 거래 횟수는 같아야 한다. 실제 current-code replay는 둘 다 1190건이다. 원본 주석은 1,168건과 1,190건을 보고하므로 동시에 성립할 수 없어 midpoint 주석을 `compared:false`로 둔다.

## 8. volume bar와 BVC

`volumeBar.m`은 500계약마다 만든 bar에서 최근 100개 가격변화의 표준편차로 현재 변화의 정규 CDF를 계산한다. 이는 buy volume fraction의 근사다. 원본의 진입 기준은 매수 fraction 0.95 초과 또는 0.05 미만이고, 청산 기준은 0.5다. Python은 `buy_fraction - 0.5`로 중심화하여 원본의 보유 구간까지 유지한다. 결과는 gross P&L -3975.00달러, 거래 688건이지만 원본에 숫자 출력 주석이 없어 source-semantic replay로만 분류하고 책 비교 숫자를 만들지 않는다.

## 9. 주문장 재구성

Coinsetter의 246,060개 이벤트를 가격별 수량 dictionary로 처리한다. 추가는 수량을 더하고 취소·체결·부분체결은 줄인다. 매초 마지막 상태에서 최고 bid와 최저 ask를 기록한다. 결과는 233,072개 1초 bar, 양면 호가 232,940개, median spread -250.830000달러다. 음수 median은 정상 시장 spread가 아니라, 원본 스크립트도 assertion을 주석 처리한 crossed order-book state가 많다는 데이터 품질 경고다. 누락·과대 감소와 crossed state를 제거해 결과를 미화하지 않고 진단 수로 남긴다.

## 10. 거래비용과 슬리피지

책의 최소 fee 0.095달러, 더 일반적인 0.16달러, IB 예시 2.47달러를 비교한다. bid-ask 경로는 spread를 이미 체결가격에 포함하며 fee는 별도 차감한다. market impact, queue position, latency는 없다. midpoint는 지정가가 항상 체결된다는 낙관적 상한이지 실행 가능한 보장이 아니다.

## 11. 표본 외와 편향 경고

단 하루에 threshold 66을 선택하고 같은 날 평가하므로 selection bias가 크다. 이 결과에는 walk-forward도 out-of-sample도 없다. 여러 임계값과 lookback을 같은 날 탐색하면 data snooping이 더 심해진다. 장중 전략은 거래일·변동성 regime·뉴스일을 넓게 포함해야 한다.

## 12. output-only 참조

| topic | compared | reason |
|---|---|---|
| tick-rule midpoint execution comment | false | The execution-price branch cannot change signals or trade count, yet the midpoint and bid-ask comments report 1168 versus 1190 trades; at least one comment is stale |
| Algoseek aggressor-tag strategy | false | The downloadable TAQ MAT has prices and volume but no Algoseek aggressor flag or the CSV read by this script |
| multi-day profitability | false | The bundled ES examples contain one trading day, so no out-of-sample Sharpe or annual return is supportable |

Algoseek aggressor flag 입력은 공식 번들에 없다. TAQ MAT의 tick rule과 실제 aggressor tag를 같은 것으로 간주하지 않는다.

## 13. 위험지표 해석

하루의 달러 P&L과 trade count만 보고한다. CAGR, Sharpe, drawdown, Calmar는 정의하지 않는다. 일중 realized P&L path는 미청산 포지션의 mark-to-market을 포함하지 않으므로 완전한 equity curve도 아니다. 이 제한이 다일 백테스트처럼 보이는 것을 막는다.

## 14. 자동 검증

검증 21/21 통과. checksum, source output, timestamp, quote, 비용 단조성, BVC 범위, 주문장 synthetic partial fill, 누락 입력 공개를 검사한다.

## 15. 결론

미시구조 전략에서 alpha보다 먼저 검증해야 할 것은 clock, trade sign, quote side, fee와 order-state transition이다. midpoint 수익이 bid-ask에서 사라지는 사례는 실행 가정이 연구 결론을 뒤집는다는 가장 직접적인 경고다.
