# 세션 핸드오프 — ML4T 발표자료 시스템 + Chapter 1·2·5 배포 및 QuantConnect 리서치 완성
_최종 갱신: 2026-08-25 KST_

## 🎯 목표
ds4th_study의 HTML 리포트+발표덱 생성 시스템을 ml4t로 이식하고, 그 시스템으로 각 스터디 회차(상세 리포트 + 발표 덱)를 만들어 GitHub Pages에 배포·누적하며, QuantConnect Cloud Research 환경을 통한 실증 재현 및 2015–2026 OOS 분석을 병행한다.

## ✅ 완료
- **시스템 이식** (`9f562ef`): ds4th_study의 study-presentation 시스템 전체 이식 — `agent-support/`(study-report-v1·study-deck-v1 템플릿, new-presentation/build-index/validate-site, 절차·테스트, studies.toml, legacy-pages.toml), `.claude`·`.agents` 스킬, `docs/` 게시 셸, AGENTS/QUICKSTART/CLAUDE.md. **자체 호스팅 KaTeX 0.16.22**(woff2, code/pre 제외)를 두 템플릿에 추가 — 퀀트 수식용.
- **Pages 안전화** (`10bfff1`, 피어 Fable5): `.gitignore` `!docs/**/`+`!docs/**`(회차 자산 드롭 방지), `.github/workflows/pages.yml`(sparse CI: 테스트+build-index+validate, main push 시 인덱스 자동 재생성 guard 포함).
- **Chapter 1 산출물** (`6e6fe2c`, 이 세션/Opus): `2026-07-25-ch01` — report.html(7섹션+부록, 9표, 6그림[새 SVG 4 + 재현차트 2], KaTeX 수식), index.html(23슬라이드, 전 슬라이드 data-report-refs 추적, 7섹션·4필수그림 커버). 근거는 재현 실험(공식 epchan.com/book3, SHA-256). build-index --check·validate-site --check-materials·headless Chrome(데스크톱/모바일390/A4, 덱 16:9) 전부 통과.
- **Chapter 2 산출물** (`b5c74d9`, 2026-08-01/Opus): `2026-08-01-ch02` — report.html(8섹션+부록 2, 9표, 11그림[새 SVG 6 + 재현차트 5]), index.html(29슬라이드, 8개 본문 절·9개 필수그림 전부 커버).
- **Chapter 5 입문자/전문가 리포트 이원화 및 배포** (`aabbc56`, 2026-08-21/pinjoy):
  - `docs/studies/machine-trading/presentations/2026-08-22-ch05/`:
    - `report.html`: 상단 탭 스위처(Beginner vs Expert)를 탑재하여 한 페이지 내에서 입문자용과 전문가용을 동적으로 전환하는 마스터 리포트.
    - `report_beginner.html` & `report_beginner.pdf`: 비전공 입문자를 위한 금융 기초 지식(VIX, Contango, 켈리 기준, 칼만 필터, 그리스 등) 설명이 보강된 입문자용 리포트 및 A4 PDF.
    - `report_expert.html` & `report_expert.pdf`: 퀀트 관점의 비판(XIV Volmageddon 붕괴, GARCH 정상성 결함, 호가 스프레드 확대 유동성 블랙홀 등)을 수록한 전문가용 리포트 및 A4 PDF.
    - `index.html` & `presentation.pdf`: 16:9 가로형 슬라이드 덱 및 22장 슬라이드 PDF.
- **Chapter 5 QuantConnect Cloud Research 전용 분석 계층 구축** (`aabbc56`):
  - `source/.../chapter_5_options_strategies/src/`:
    - `qc_cloud_research_ch5_in_sample_replication.ipynb`: 2004–2015 표본 내 엄밀 재현 주피터 노트북 (지표 스키마 통일 및 MultiIndex 보정).
    - `qc_cloud_research_ch5_out_of_sample_testing.ipynb`: 2015–2026 표본 외 11개년 스트레스 테스트 주피터 노트북 (2018 Volmageddon, 2020 마이너스 유가, 2022 금리인상, 2024–2026 0DTE 시장 분석).
    - `QC_IN_SAMPLE_REPLICATION_REPORT.md`: 표본 내 정량 대조 및 저자 원본 MATLAB 코드 지표 혼용(SPY CAGR 7.2% vs 11.76%) 원인 규명 보고서.
    - `QC_OUT_OF_SAMPLE_TESTING_REPORT.md`: 11개년 4대 거시 레짐별 스트레스 테스트 결과 및 퀀트 통찰 보고서.
    - `QC_RESEARCH_PLAN.md` & `QC_RESEARCH_GUIDE.md`: 연구 계획서 및 클라우드 실행 가이드.
- **배포·라이브**: Pages(main/docs) 빌드 및 검증 통과 (11 HTML 파일, 0 에러/경고). 원격 저장소(`origin/main`) 푸시 완료.
  - 사이트: https://restful3.github.io/ml4t/
  - ch01: https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-07-25-ch01/
  - ch02: https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-08-01-ch02/
  - ch05: https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-08-22-ch05/

## 🔄 진행 중
- 없음. Chapter 1, 2, 5 산출물 및 QuantConnect 리서치 파이프라인 배포 완료.

## ⏭️ 다음 단계
1. **Chapter 3 준비** (시계열 분석 / 주간 스터디 일정에 따른 스캐폴딩).
2. (선택) collaborator 초대 및 기타 챕터 리포트의 QuantConnect 확장 적용.

## 🧠 대화에만 있던 핵심 컨텍스트
- **결정(리포트 탭 스위처)**: `report.html`에 동적 탭 스위처를 도입하여 `validate-site.py`의 `data-report-refs` 추적 계약을 완벽히 보존하면서 입문자용/전문가용 뷰를 유연하게 제공.
- **발견(Chan 코드 버그 규명)**: 교재 표 5.1의 SPY CAGR(7.2%)은 MATLAB 코드 내부에서 2.15배 레버리지 시계열을 생성한 후 1.0배 무레버리지 시계열로 CAGR을 잘못 계산한 Metric Mixing 버그에 기인함을 밝힘 (실제 2.15배 복리는 11.76%).
- **발견(OOS 팻 테일)**: 2018 Volmageddon 당시 Short Vol 포트폴리오 왜도 -18.02, 첨도 637.7 폭발 실증. 롱 감마 스캘핑만이 유일하게 +$20.50으로 위기 방어 성능을 입증함.

## ⚠️ 클리어 전 주의
- 커밋 안 됨: `chapter1_full_report.ipynb` kernelspec 1줄(로컬 venv 환경). 자동 커밋 금지.
- 정리 완료: 불필요한 빌더 스크립트 및 node_modules 정리 완료.
- 백그라운드: 없음.
- 미완료 todo: 없음.

## 📂 관련 파일
- `docs/studies/machine-trading/presentations/2026-08-22-ch05/` — ch05 report.html, report_beginner.html, report_expert.html, index.html, PDFs 4종.
- `source/.../chapter_5_options_strategies/src/` — QC 리서치 노트북 2종 및 종합 분석 마크다운 보고서 4종.
- `agent-support/scripts/{build-index,validate-site}.py` — 인덱스 및 검증 스크립트.
