# 세션 핸드오프 — ML4T 발표자료 시스템 + Chapter 1·2 배포
_최종 갱신: 2026-08-01 KST_

## 🎯 목표
ds4th_study의 HTML 리포트+발표덱 생성 시스템을 ml4t로 이식하고, 그 시스템으로 각 스터디 회차(상세 리포트 + 발표 덱)를 만들어 GitHub Pages에 배포·누적한다.

## ✅ 완료
- **시스템 이식** (`9f562ef`): ds4th_study의 study-presentation 시스템 전체 이식 — `agent-support/`(study-report-v1·study-deck-v1 템플릿, new-presentation/build-index/validate-site, 절차·테스트, studies.toml, legacy-pages.toml), `.claude`·`.agents` 스킬, `docs/` 게시 셸, AGENTS/QUICKSTART/CLAUDE.md. **자체 호스팅 KaTeX 0.16.22**(woff2, code/pre 제외)를 두 템플릿에 추가 — 퀀트 수식용(원래 ml4t 고유, 2026-07-22 ds4th에도 역이식되어 이제 두 repo 동일).
- **Pages 안전화** (`10bfff1`, 피어 Fable5): `.gitignore` `!docs/**/`+`!docs/**`(회차 자산 드롭 방지), `.github/workflows/pages.yml`(sparse CI: 테스트+build-index+validate, main push 시 인덱스 자동 재생성 guard 포함).
- **Chapter 1 산출물** (`6e6fe2c`, 이 세션/Opus): `2026-07-25-ch01` — report.html(7섹션+부록, 9표, 6그림[새 SVG 4 + 재현차트 2], KaTeX 수식), index.html(23슬라이드, 전 슬라이드 data-report-refs 추적, 7섹션·4필수그림 커버). 근거는 재현 실험(공식 epchan.com/book3, SHA-256). build-index --check·validate-site --check-materials·headless Chrome(데스크톱/모바일390/A4, 덱 16:9) 전부 통과.
- **favicon 정비** (`28850e0`, 2026-07-25): study-report 템플릿에 `<link rel="icon">`이 없어 생성되는 모든 report.html이 favicon 없이 나가던 버그 수정(덱 템플릿엔 원래 있었음) — 템플릿 + 게시된 ch01·ch05 report.html에 추가. `docs/assets/favicon.svg`는 이식 때 딸려온 ds4th의 "D" 마크였어서 같은 파랑(#3157d5)에 "M"으로 다시 그림. 사이트 6개 HTML 전부 favicon 링크 보유·경로 해석 확인.
- **리포트 푸터 정리** (`d7bc74a`, 2026-07-25): 부록 끝의 `.report-jump-links`(발표자료·회차 목록·문서 처음) 제거 — 설정 패널의 `Index / Report / Slides`와 중복. 템플릿 + ch01·ch05 report.html, 그리고 죽은 `.report-jump-links` CSS 3벌(`@media print`는 `.skip-link`만 남김)까지. report.css 3벌은 계속 byte-identical 유지.
- **참고 자료 GitHub 링크**: 리포트에서 저장소에 실제 있는 파일(재현 스크립트·노트북·재현 리포트·원본 MATLAB·해설판)을 GitHub로 링크하는 규칙. 참고 자료 부록뿐 아니라 본문의 그림 캡션·소스코드 표 셀까지 전부 링크(파서로 미링크 0건 확인, 임베드된 14개 링크 curl 200). 규칙은 `procedures/study-presentation.md`·`study-report/DESIGN.md`·`STUDY_SESSION_BLUEPRINT.md`·양쪽 `SKILL.md`·리포트 템플릿 주석에 기록.
- **링크는 커밋 SHA 고정**: `blob/main/source/...`은 학습자료가 `archive/`로 옮겨지는 순간 404가 되므로 `blob/9198e9e11eb425eccaefbc1095181e3b01657efc/`로 고정했다. 고정 커밋은 불변이라 아카이브 때 손댈 것이 없다(그래서 `archive-study.md` 7번은 '갱신'이 아니라 '손대지 않는다'로 바뀜). 반대급부: 원본 파일이 수정돼도 링크는 옛 버전을 가리키므로 크게 바뀌면 SHA를 갱신한다. `validate-site.py`는 SHA 고정 링크는 shape만(ref·인코딩) 검사하고 워킹트리 경로 검사를 건너뛴다 — 아카이브 후 오탐을 막기 위해서다. `main` 링크는 기존대로 경로 존재까지 검사하되 CI sparse checkout(`docs/`+`agent-support/`)에 없는 top-level은 건너뛴다. 테스트 7개.
- **Chapter 2 산출물** (`b5c74d9`, 2026-08-01/Opus): `2026-08-01-ch02` — report.html(8섹션+부록 2, 9표, 11그림[새 SVG 6 + 재현차트 5]), index.html(29슬라이드, 8개 본문 절·9개 필수그림 전부 커버). **책 보고값과 재현값을 끝까지 분리**했고, 라이선스 데이터(OptionMetrics·Compustat) 부재로 재현 못 한 절은 표 6 「재현 경계」에 명시했다. 링크 SHA는 `2f7e380` 고정, 5종 curl 200. 검증 3종 + headless Chrome(데스크톱/모바일390/A4, 덱 16:9) 통과.
  - **작업 경위**: 07-31 심야에 스캐폴딩과 그림 11장까지만 만들고 중단된 것을 08-01 아침에 이어받아 리포트 본문·덱을 완성했다. 어젯밤 산출물은 전부 미커밋 상태였다.
  - **렌더에서 잡은 결함**: 이미지+표를 함께 둔 슬라이드 4장(12·15·19·23)이 1280×720에서 lead 문장이 차트에 가려지고 takeaway가 footer와 겹쳤다. 이미지 `max-height`를 vh에서 px로 고정해 해결했고, 19번은 도해와 two-col 패널이 내용 중복이라 패널을 걷어냈다. **덱에 이미지와 표/패널을 함께 넣을 때는 반드시 16:9 렌더를 눈으로 확인할 것.**
  - **남은 gap**: 회차 폴더의 재현 차트 PNG 5장(`fig-bug-equity`·`fig-fundamental-equity`·`fig-pca-equity`·`fig-loadings-spread`·`fig-cost-sensitivity`)을 그린 **플로팅 스크립트가 저장소에 없다.** 수치는 전부 `src/reports/metrics.json`에서 추적되지만 차트 자체는 재생성 경로가 끊겨 있다. ch01은 `run_chapter1_analysis.py`가 그림까지 만들어 링크가 걸렸는데 ch02는 그렇지 않다 — 다음에 그 스크립트를 `src/` 아래로 복원하면 캡션에 GitHub 링크를 걸 수 있다.
  - **책 본문 vs 공식 코드 불일치**: 예제 2.1 표본 내 성과를 책 본문은 CAGR 242%·Sharpe 3.7로 적었지만, 재현의 대조 기준 책 수치는 103.6%·2.46이고 Python 재현은 105.0%·2.47이다. 리포트는 대조 기준 수치를 쓰고 이 불일치를 본문에 명시했다. **원인 미규명** — 발표 때 질문이 나올 수 있다.
- **배포·라이브**: Pages(main/docs) 빌드 완료. ch05(`2026-08-22-ch05`)는 pinjoy99가, ch01 폴더의 `index_정훈.html`은 Junghoon Park(`dc1ae11`, 웹 업로드)이 추가. **참여자 3명이 main에 직접 push하므로 작업 전 `git pull --rebase` 필수.**
- **Chapter 5 PDF 인쇄 추가**: 2026-08-22-ch05 하위에 리포트(report.pdf, 1.9MB)와 발표자료(presentation.pdf, 2.3MB)를 Playwright를 통해 16:9 및 A4 규격으로 출력하여 빌드 및 커밋 완료.
- **ch05 회차 폴더 개명**: 매주 전환으로 Ch5가 09-19 → 08-22가 되어 디렉터리·`session_id`를 `2026-09-19-ch05` → `2026-08-22-ch05`로 바꿨다. **사용자가 pinjoy99 허가를 받아 지시한 것.** 기존 공개 URL `.../2026-09-19-ch05/`는 이제 404다 — 외부에 그 링크를 공유했다면 새 URL로 안내가 필요하다.
  - 사이트 https://restful3.github.io/ml4t/
  - ch01 덱 https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-07-25-ch01/
  - ch01 리포트 위 경로 + `report.html`

## 🔄 진행 중
- **ch02는 로컬 커밋까지 완료, push 안 됨** (`b5c74d9`). 사용자 확인 후 push해야 라이브에 뜬다.
- **ch02 발표자는 종훈** (`presentation.toml`). 태영이 자료를 만든 것이라 **종훈에게 전달·검토가 필요**하다. 종훈이 자기 버전을 따로 만들었다면 ch01의 `index_정훈.html`처럼 병존시킬지 정해야 한다.

## ⏭️ 다음 단계
1. **ch02 push 여부 확인** → push 후 Pages 반영 확인(`https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-08-01-ch02/`).
2. **ch02 PDF 인쇄물 추가**(선택) — ch05처럼 `report.pdf`·`presentation.pdf`를 만들려면 별도 작업. 현재 ch02에는 없다.
3. **ch02 재현 차트 스크립트 복원**(위 gap 참조) — 복원하면 그림 캡션에 GitHub 링크를 걸 수 있다.
4. (선택) collaborator 초대 — 사용자가 직접.
5. (선택) 두 repo CI 파일명 통일(ml4t `pages.yml` ↔ ds4th `validate-study-site.yml`).

## 🧠 대화에만 있던 핵심 컨텍스트
- **결정(KaTeX)**: 수식은 자체 호스팅 KaTeX(외부 CDN 금지 유지). 원래 ml4t 고유였으나 2026-07-22 ds4th에도 역이식(3bd49288)되어 두 repo 동일.
- **결정(기준 예시 각색)**: 이식 문서들이 참조하던 kg-llm ch01 경로를 ml4t엔 없어 중립 문구로 교체. 첫 완성 회차(=현재 ch01)가 이제 구체적 기준 예시.
- **발견(멀티에이전트 충돌)**: ml4t↔ds4th가 쌍둥이 구조 + 같은 머신 공유라 aiml 세션 Fable5가 대상 repo를 헷갈려 ds4th용 수정을 ml4t에 잘못 커밋한 사건 발생 → 조율로 해결. **역할 고정: ml4t=이 세션 / ds4th=aiml**. ds4th엔 Codex(aiml:2.1)·Tori(aiml:4.1)도 있음. 형제 repo git 작업 전 대상 확인·`git add -A` 금지·push 전 `git pull --rebase`.
- **발견(gotcha)**: GitHub Pages 소스를 API(PUT)로 바꿔도 자동 재빌드 안 됨 → `POST /pages/builds` 필요.
- 상세 결정은 영속 메모리 `~/.claude/projects/-home-restful3/memory/reference_ml4t_ds4th_presentation_system.md`에도 기록됨.

## ⚠️ 클리어 전 주의
- 커밋 안 됨: `chapter1_full_report.ipynb` kernelspec 1줄(`display_name`이 로컬 venv 이름으로 바뀜). 머신 종속 잡음이라 의도적으로 미커밋 — 자동 커밋·되돌림 금지.
- 정리 완료(2026-07-25): `.claude/.headroom_wrap_marker.json` 삭제(죽은 pid 마커), `.claude/settings.local.json`은 repo `.gitignore`에 추가(`b1b1af5`) — 이전엔 전역 ignore에만 있어 다른 참여자에겐 안 걸렸음. `HANDOFF.md`는 이제 추적 대상(커밋됨).
- 백그라운드: 없음(띄웠던 http.server 8137/8138 모두 종료 확인).
- 미완료 todo: 없음.

## 📂 관련 파일
- `docs/studies/machine-trading/presentations/2026-07-25-ch01/` — ch01 report.html·index.html·assets(figs 6, katex). 라이브.
- `agent-support/templates/study-{report,deck}/` — 템플릿(+KaTeX). 새 회차 품질 기준.
- `agent-support/scripts/{new-presentation,build-index,validate-site}.py` — 회차 생성·인덱스·검증.
- `agent-support/studies.toml` — machine-trading(active, 2026-07-25 – 2026-09-12). 2026-07-25 격주→매주 전환(`189e30c`)으로 종료일이 10-31에서 당겨졌다.
- `.claude/skills/study-presentation/SKILL.md` — 회차 작업 스킬(이제 하네스에 등록됨).
- `source/Chan E. Machine Trading .../chapter_N_*/` — 원자료(한국어 해설판 + 재현 실험 `src/reports/`).
