# Chapter 4 — Artificial Intelligence Techniques 재현 보고서

## 1. 개요와 문제 정의

이 장의 질문은 “어떤 AI 모형이 가장 높은 백테스트를 만드는가”가 아니다. 같은 네 개의
중첩 수익률 피처로 복잡도를 높였을 때 훈련 성능과 표본 외 성능이 어떻게 갈라지는지,
그리고 검증·시차·비용 계약이 그 차이를 얼마나 정직하게 드러내는지가 핵심이다.

공식 SPY 일봉 2,357개를 2004-12-22부터
2014-06-02까지 사용한다. 달력상 앞 1,178개는 훈련,
뒤 1,179개는 테스트다. 다만 원본 회귀는 마지막 훈련일의
다음 날 target으로 첫 테스트 수익을 한 행 사용한다. exact replay는 이 경계 누수까지
보존하고, Python 적응은 마지막 predictor 행을 빼 테스트를 완전히 격리한다.

## 2. 핵심 수식과 수식-코드 지도

단순수익률과 다음 날 목표는 다음처럼 정의한다.

$$ r_t(k) = (P_t - P_{t-k}) / P_{t-k} $$

$$ y_t = r_{t+1}(1) $$

| 수식/계약 | 구현 함수 | 검증 |
|---|---|---|
| lagged simple return | `simple_returns` | MATLAB `calculateReturns.m`과 동일 |
| OLS와 coefficient p-value | `ols_fit` | 원본 계수와 수치 비교 |
| stepwise complete-case selection | `forward_stepwise_fit` | ret2 선택과 계수 비교 |
| signal(t) → position(t+1) | `backtest_signals` | 첫 포지션 0 assert |
| HMM one-step observation forecast | `hmm_signals` | 행렬 확률합과 원본 CAGR 비교 |
| ordered cross validation | `chronological_cv_*` | 모든 train 마지막 < validation 첫 행 |

## 3. 구현 범위와 재현 상태

| 주제 | 재현 상태 | 근거 |
|---|---|---|
| AI와 자동화된 패턴 탐색 | 개념 설명 | 예측보다 검증 설계가 우선 |
| 선형회귀와 과적합 | 정확 수치 재현 | lr.m, SPY 공식 MAT |
| stepwise regression | 정확 수치 재현 | 완전관측 행 고정과 ret2 선택 |
| regression tree | 정확 규칙 재현 + Python 적응 | 극단 leaf 규칙과 sklearn 전체 tree |
| cross validation | 실행된 방법론 수정 | random K-fold 대신 TimeSeriesSplit |
| bagging/random subspace | Python 적응 + output-only 원본 참조 | RandomForestRegressor, seed=1; 직접 비교 금지 |
| boosting | Python 적응 + 원본 한계 진단 | 훈련 CV로 iteration 선택 |
| classification tree | Python 적응 + provenance-uncertain 원본 참조 | 같은 주석이 5개 MATLAB 파일에 중복 |
| support vector machine | Python 적응 + output-only 원본 참조 | train-only kernel 선택 |
| hidden Markov model | published-parameter 정확 replay | 온라인 filtering, 행렬 재추정 아님 |
| neural network | 결정적 다중 seed 적응 + output-only 원본 참조 | MLP 10개와 seed 분산 |
| aggregation과 normalization | 실행 진단 | StandardScaler와 ensemble 평균 |
| technical stock selection | output-only | fundamentalData가 공식 ZIP에 없음 |
| fundamental stock selection | output-only | 저자 로컬 Dropbox 경로만 존재 |
| look-ahead/survivorship/selection bias | 개념 + 자동 검증 | 시계열 split과 테스트 격리 |
| 거래비용과 위험 | 실행 백테스트 | 2bps 편도 민감도, MDD와 duration |
| 연습문제·요약·endnotes | 커버리지 매핑 | 모형 확장과 검증 질문 |

원본 충실 재현은 같은 공식 입력, 같은 식, 같은 분할과 원본의 경계 누수까지 보존해
수치를 다시 계산했다는 뜻이다. 누수를 올바른 설계로 승인한다는 뜻은 아니다.
Python 적응은 라이브러리와 검증 설계를 현대화한 별도 실험이며 MATLAB 숫자와 직접
일치한다고 주장하지 않는다. output-only는 책/소스 주석값만 보존하고 전략 지표는
`null`로 둔다.

## 4. 출처·데이터·환경

- 저자 페이지: https://epchan.com/book3
- 직접 ZIP: https://epchan.com/img/book3/Chap4%20AI.zip
- ZIP SHA-256: `a556978ebe871446b3f32511a910b8fec74669bc02dddcced0c8537fd82a0358`
- 입력 MAT SHA-256: `b94aa1b85d10e4e41c021882ee90a7d945341baf8d4690ca7561561936437fc4`
- `uv.lock` SHA-256: `34057457773a614f058648f3ba9ba2f02f02354f64214ec14891fbb212c5ebaf`
- 공식 멤버: 24개, close 결측 0개,
  비양수 0개

`SOURCE_MANIFEST.json`은 MATLAB 22개와 데이터 2개의 개별 checksum을 고정한다.
offline 실행은 네트워크 없이 전 멤버를 재검증한다. `SPY.xls`는 원본 보존용이며 계산은
구조가 명확한 `inputData_SPY.mat`을 사용한다.

## 5. 선형회귀: 복잡한 모델이 자동으로 낫지 않다

| 실험 | 구간 | Python CAGR | 책/원본 CAGR | Python Sharpe | 책/원본 Sharpe | 분류 |
|---|---|---:|---:|---:|---:|---|
| Full linear | train | 0.343339 | 0.343339 | 1.365304 | 1.365304 | 원본 충실 재현(경계 누수 포함) |
| Full linear | test | 0.003582 | 0.003582 | 0.103952 | 0.103952 | 원본 충실 재현(경계 누수 포함) |
| Stepwise | train | 0.436383 | 0.436383 | 1.649068 | 1.649068 | 원본 충실 재현(경계 누수 포함) |
| Stepwise | test | 0.105685 | 0.105685 | 0.695228 | 0.695228 | 원본 충실 재현(경계 누수 포함) |
| Extreme tree rule | train | 0.287519 | 0.287519 | 1.529458 | 1.529458 | 원본 threshold 재현 |
| Extreme tree rule | test | 0.038665 | 0.038665 | 0.505706 | 0.505706 | 원본 threshold 재현 |
| HMM | train | 0.086644 | 0.086644 | 0.467205 | 0.467205 | 계수 replay |
| HMM | test | -0.010173 | -0.010173 | 0.019724 | 0.019724 | 계수 replay |

네 피처 전체 회귀는 훈련 CAGR 34.3%에서
테스트 0.4%로 거의 소멸한다. 반면 stepwise는
`ret2` 하나만 남겨 테스트 10.6%를 기록한다.
이는 stepwise가 항상 우월하다는 증거가 아니라 이 한 표본에서 분산 감소가 유리했다는
결과다. 후보와 진입 규칙을 테스트를 본 뒤 바꾸면 selection bias가 된다.

### 5.1 원본의 1행 경계 누수와 수정본

`retFut1=fwdshift(1, ret1)` 뒤 `trainset=1:floor(N/2)`를 그대로 쓰면 훈련 마지막
predictor 날짜 2009-09-15의 target이 첫 테스트일
2009-09-16 수익이다. 따라서 위 표의 exact test는 엄밀한 완전 격리
OOS가 아니라 **source-defined test replay**다. 수정본은 predictor를
2009-09-14까지만 적합해 마지막 target이 훈련 종료일에 실현되게 한다.

| 모델 | source-defined test CAGR | boundary-corrected test CAGR | source fit rows | corrected fit rows |
|---|---:|---:|---:|---:|
| Full linear | 0.003582 | 0.011817 | 1158 | 1157 |
| Stepwise | 0.105685 | 0.107317 | 1158 | 1157 |

모든 sklearn 적응, ordered CV, scaling, MLP target normalization은 1,157개 corrected row만
사용한다. `source_boundary_target_leakage_detected`와
`python_training_targets_precede_test`가 두 경로를 각각 검증한다.

## 6. Regression tree와 extreme-leaf 규칙

원본 tree 전체 예측은 훈련 수익이 매우 높지만 테스트에서 음수가 된다. 책이 별도로
제시한 극단 leaf 두 개만 쓰는 규칙은 flat 상태를 허용해 테스트 CAGR
3.9%, Sharpe
0.51를 정확 재현한다. leaf threshold가 학습자료에서
선택됐다는 사실은 유지되며, 테스트에 맞춘 재최적화는 하지 않았다.

## 7. Cross validation 교정

MATLAB 예제는 무작위 5-fold를 만든 뒤 loss가 가장 낮은 한 fold의 fitted model을 꺼내 쓴다.
금융 시계열에서는 미래 행이 과거 행의 모델 선택에 섞일 수 있고, 최종 훈련 전체를 다시
적합하지 않는 문제도 있다. Python 적응은 `TimeSeriesSplit`으로 항상 과거→미래 순서를
지키고, 첫 테스트 target을 제외한 훈련 행 전체에 재적합한다. 따라서 아래 수치는 원본 수치 재현이 아니라
검증 설계를 바꾼 방법론적 적응이다.

## 8. Bagging, random subspace, boosting, classification, SVM, NN

| Python 적응 | gross CAGR | net CAGR (2bps) | gross Sharpe | max drawdown |
|---|---:|---:|---:|---:|
| tree | -0.1084 | -0.1449 | -0.618 | -0.455 |
| classification_tree | 0.0397 | 0.0005 | 0.319 | -0.260 |
| random_forest | 0.0870 | 0.0478 | 0.591 | -0.256 |
| boosting | -0.0220 | -0.0612 | -0.054 | -0.348 |
| svm | 0.1120 | 0.1116 | 0.730 | -0.264 |
| mlp_ensemble | -0.0304 | -0.0597 | -0.106 | -0.317 |

Random forest는 MATLAB `TreeBagger(K=5)`와 같은 tree 수와 seed를 쓰지만 split 규칙과
random-subspace 기본값이 달라 직접 수치 비교하지 않는다. classification tree 원본값은
같은 주석이 서로 다른 MATLAB 5개에 복제돼 provenance도 확정할 수 없다. 아래 값은
정확성 판정에 쓰지 않는 output-only reference다.

| 원본 주제 | 원본 CAGR / Sharpe | Python 적응 CAGR / Sharpe | 비교 상태 |
|---|---:|---:|---|
| bagging/random subspace | 0.071967 / 0.505925 | 0.087037 / 0.590903 | output-only reference; 직접 비교 안 함 |
| classification tree | 0.047740 / 0.366381 | 0.039669 / 0.319308 | output-only reference; 직접 비교 안 함 |
| support vector machine | 0.133530 / 0.847489 | 0.112014 / 0.730285 | output-only reference; 직접 비교 안 함 |
| neural-network average | 0.078454 / 0.542809 | -0.030418 / -0.106211 | output-only reference; 직접 비교 안 함 |

Boosting iteration과 SVM kernel은 훈련 구간의 ordered CV만으로 선택한다. MLP는 내부
무작위 validation을 쓰지 않고 고정 500 iteration으로 seed 1~10을 모두 실행한 뒤 평균
예측을 사용한다. seed별 성능 분산은 한 번의 좋은 초기화를 일반화 성능으로 오인하면
안 된다는 진단이다.

## 9. HMM replay와 상태 해석

HMM은 `hmm_train.m`이 출력한 prior, transition, emission을 고정해 온라인 filtering을
replay한다. 훈련 CAGR 8.7%와 테스트
-1.0%가 원본과 일치하지만, 이는 Python에서 EM
최적화를 독립 재현했다는 뜻이 아니다. `hmm.m`에는 `INCOMPLETE!`, `hmm_train2.m`에는
작동하지 않는다는 assert가 있어 published-parameter replay로 분류했다.

## 10. 누락된 cross-sectional 데이터

`rTree_SPX.m`과 `stepwiseLR_SPX.m`은 공식 ZIP에 없는 저자 로컬
`C:/Users/Ernest/Dropbox/AI_WS/fundamentalData`를 읽는다. 따라서 기술 피처 주식선택
2.3%/0.9와 fundamental 주식선택 4.0%/1.1은 output-only다. 다른 최신 S&P 데이터로
바꾸면 survivorship, point-in-time fundamentals, universe가 달라지므로 정확 재현으로
표시할 수 없다. `metrics.json`의 두 `test` 값은 0이 아니라 `null`이며 이유가 함께 있다.

## 11. 거래비용·위험·적용 범위

모든 전략은 신호를 하루 뒤 포지션에 적용한다. 편도 2bps × 포지션 변화량을 차감한
민감도와 gross를 함께 저장한다. 이는 spread, slippage, borrow fee, market impact를 모두
포함한 실거래 견적이 아니며 일봉 회전율의 최소 마찰 진단이다. CAGR과 Sharpe 외에
maximum drawdown, drawdown duration, annual turnover를 기록한다.

이 결과는 SPY 한 종목, 한 번의 고정 분할에 대한 백테스트다. 모델 우열의 보편적 증거나
현재 시장 추천이 아니다. test 결과를 보고 seed, leaf, kernel, 비용을 다시 선택하면 그
test도 훈련자료가 되므로 새 표본 외 기간이 필요하다.

## 12. 편향과 원본 코드 감사

- look-ahead: 원본의 첫 테스트 target 1행 사용을 exact replay에 공개하고 Python 적응에서는 제외했으며, random K-fold도 ordered split으로 교체했다.
- survivorship: SPY 자체에는 구성종목 생존 편향이 없지만 cross-sectional 결과는 원자료가 없어 검증 불가다.
- selection bias: model family, boosting iteration, SVM kernel은 train CV 안에서만 선택했다.
- data snooping: 책에 인쇄된 좋은 모델을 이미 알고 있다는 사후 지식을 분류에 명시했다.
- source completeness: hmm.m, hmm_train2.m, nn_retrain.m, stepwiseLR_SPX_fillFwd.m의 중단 표식을 보존했다.

## 13. 자동 검증과 결론

검증 23/23개가 통과했다. 독립/경험 비교
13개와 계산 계약 invariant
10개를 구분했다.

Chapter 4의 가장 재사용 가능한 결론은 “더 복잡한 AI”가 아니라 “더 엄격한 정보 경계”다.
공식 OLS·stepwise·tree rule·HMM 수치와 경계 누수를 함께 보존하면서도 Python 실험은
첫 테스트 target을 제외한 시간순 CV, corrected-train 재적합, seed ensemble, 비용 전후를
별도 계약으로 둔다. 데이터가 없는 주식선택 예제는
숫자를 꾸며내지 않고 output-only로 남긴다. 다른 책에서도 동일하게 원본 수치 replay와
현대적 방법론 적응을 한 표의 다른 행으로 분리해야 한다.
