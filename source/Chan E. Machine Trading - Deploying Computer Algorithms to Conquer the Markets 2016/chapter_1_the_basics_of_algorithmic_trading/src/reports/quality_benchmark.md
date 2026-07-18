# Chapter 1 품질 벤치마크

## 평가 목적과 재생성 계약

현재 Ch1을 다른 Chan 책의 실행 노트북과 같은 정적 rubric 및 수동 수치 검토로
비교한다. 이 파일은 수동으로 고치지 않는다. `run_chapter1_analysis.py`가 실행된
노트북, `metrics.json`, 아래 감사 스크립트에서 매번 재생성한다.

- 감사 스크립트: `.codex/skills/create-chan-chapter-analysis/scripts/audit_chapter_artifacts.py`
- 감사 스크립트 SHA-256: `b0583cd05b5f3d14b58661cab019fe2d2bb53456d4eabb823ffe657fa64a22a9`
- 실행 셀·그림·문자 수: 현재 `chapter1_full_report.ipynb`에서 계산
- 수치 검증 수: 현재 `metrics.json`에서 계산

## 비교군 정적 감사

| 책 / Chapter | 실행 | 재현성 | 엄밀성 | 교육 | 트레이딩 맥락 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| Algorithmic Trading 2013 Ch2 | 20 | 5 | 5 | 17 | 15 | 62 |
| Algorithmic Trading 2013 Ch3 | 20 | 5 | 5 | 17 | 12 | 59 |
| Algorithmic Trading 2013 Ch5 | 20 | 5 | 9 | 17 | 9 | 60 |
| Quantitative Trading 2021 Ch3 | 20 | 5 | 5 | 17 | 15 | 62 |
| Quantitative Trading 2021 Ch7 | 20 | 5 | 2 | 15 | 12 | 54 |
| **Machine Trading 2016 Ch1 — 현재** | **20** | **20** | **25** | **20** | **15** | **100** |

정적 점수는 품질의 하한이다. 글자 수나 헤딩을 늘리는 것으로 수치·결론 모순을
가릴 수 없으므로, 공식 MATLAB 결과·독립 솔버·표본운·편향도 별도로 검토한다.

## 현재 실행 산출물 계측

- 총 28셀: Markdown 18, code 10.
- code 10/10 실행, 오류 출력 0.
- 실행 중 인라인 생성 PNG 5개; 사전 생성 PNG 읽기 없음.
- Markdown 7,415자, 한국어 문자 3,266자, 헤딩 20개.
- 데이터 2,500행, 결측 0, 비양수 0.
- 검증 15개 통과: 독립/경험 검증 8개, 계약 invariant 7개.
- normalized `C^-1 M` gross exposure `8.4224`.
- 전·후반 롱온리 tangency 비중 L1 거리 `2.0000`.

## 반복 개선 근거

초기 구현도 공식 해시·잠금 환경·책 비중 비교와 15개 검증을 갖췄지만, 장 전체
성과지표·운영 맥락과 검증 강도 분류가 부족했다. 현재 버전은 CAGR·scalar Kelly,
200시드 분포, global minimum variance, 제외 자산 결측 근거, 검증 8+7 분류,
공통 재현 계층과 pytest 교차검증을 추가했다.

## 판정

현재 Ch1은 실행·재현성·엄밀성·교육·트레이딩 맥락에서 100/100이다.
진짜 성과 백테스트가 없는 것은 1장 공식 예제의 범위이며, 노트북은 정적
포트폴리오 결과를 표본 외 성과로 과장하지 않는다. 이 문서의 계측값은 파이프라인
재실행 때 함께 갱신되므로 노트북과 한 리비전 어긋나지 않는다.
