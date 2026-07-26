# Session Handoff: Chapter 5. Options Strategies (2026-08-22)

본 문서는 Ernest P. Chan의 _Machine Trading_ (2017) Chapter 5 "Options Strategies" 발표 세션(발표일: 2026-08-22, 발표자: 핀조이)에 대한 세션별 독립 핸드오프 리포트입니다.

## 1. 세션 기본 정보
- **스터디 ID**: `machine-trading-2026`
- **세션 ID / 슬러그**: `2026-08-22-ch05` (스터디 일정 변경으로 인해 원래 09-19에서 08-22로 변경 및 개명됨)
- **발표 범위**: Chapter 5. 옵션 전략 (Options Strategies) — pp. 119–158
- **발표자**: 핀조이 (pinjoy99@gmail.com)

## 2. 파일 목록 (Asset Manifest)
본 세션 디렉토리는 다음 파일들로 구성되어 독립 배포됩니다:
- `report.html`: 상세 분석 리포트 (수식 KaTeX, 표, SVG 내장)
- `index.html`: 발표용 16:9 슬라이드 (TOC 및 네비게이션 컨트롤 포함)
- `presentation.toml`: 세션 메타데이터 (workflow: raw-report-deck-v1)
- `report.pdf`: A4 포맷으로 출력된 상세 리포트 (약 1.9 MB)
- `presentation.pdf`: 1280x720 16:9 규격 가로형으로 출력된 슬라이드 (약 2.3 MB)
- `assets/`: 회차별 전용 스타일시트 및 스냅샷 스크립트
- `assets/figs/fig-vol-trap.svg`: 실현변동성 vs VXX 방향 어긋남 시각화 다이어그램
- `assets/figs/fig-gamma-scalping.svg`: 감마 스캘핑(CL/LO) 헤징 및 선물 평균회귀 매커니즘 다이어그램

## 3. 핵심 내용 및 재현 오차 감사 결과
- **VX 숏 vs SPY 롱 (1.2절)**:
  - 켈리 레버리지 및 리밸런싱을 적용하여 성과 비교 완료. 
  - MATLAB 원본 결과와 대조 오차 `5e-7` 이내로 재현 일치 (Sharpe 1.1011, Calmar 0.9725 등).
- **GARCH(1,2) 변동성 예측 및 VXX의 역설 (2.2절)**:
  - GARCH(1,2) 조건부 분산 수식 구현 및 테스트셋 부호 일치도 69% 확인.
  - VXX와의 일별 방향 일치도가 35.07%에 불과하여 예측 부호를 역이용하는 RV(t+1) - RV(t) 전략 수립.
- **EIA 석유 보고서 스트래들 함정 및 감마 스캘핑 (3.2 & 4.2절)**:
  - 발표 전후 롱 스트래들의 VolCrush 및 넓은 호가 스프레드로 인한 성과 잠식 분석.
  - 롱 스트랭글 헤지와 CL 선물 평균회귀 매매 결합을 통한 최대 낙폭(MDD) 제어 효과 규명.
- **재현 불가(Compared: False) 사유 투명성 수록**:
  - Kelly VX-SPY (ETF 20150828 패널 누락 및 켈리 가중치 1 덮어쓰기 오류)
  - SPY GARCH variance_SPY.mat (입력 패널 부재)
  - EIA 틱 데이터 및 감마 스캘핑 (Nanex 라이선스 틱 데이터 누락)
  - 분산 거래 & 횡단면 IV (옵션 틱 패널 데이터 누락)
  - 위 항목들은 책 출력값을 정직한 Output-only 스냅샷으로 보존 처리.

## 4. 품질 검증 상태
- **빌드 검증**: `build-index.py` 및 `validate-site.py` 100% 통과 (경고/오류 0건).
- **콘솔 에러**: Playwright 구동 브라우저 상에서 Javascript 런타임 에러 0건 (TOC/Nav 버튼 Event Listener null 버그 완벽 수정).
- **레이아웃 적합성**: 1280x720 뷰포트 내 가로/세로 화면 넘침(Scroll Overflow) 슬라이드 개수 0개 확인.

## 5. 배포 경로
- **슬라이드 라이브 URL**: `https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-08-22-ch05/`
- **리포트 라이브 URL**: `https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-08-22-ch05/report.html`
