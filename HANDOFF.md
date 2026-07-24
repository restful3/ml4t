# 세션 핸드오프 — ML4T 발표자료 시스템 + Chapter 1 배포
_최종 갱신: 2026-07-22 KST_

## 🎯 목표
ds4th_study의 HTML 리포트+발표덱 생성 시스템을 ml4t로 이식하고, 그 시스템으로 각 스터디 회차(상세 리포트 + 발표 덱)를 만들어 GitHub Pages에 배포·누적한다.

## ✅ 완료
- **시스템 이식** (`9f562ef`): ds4th_study의 study-presentation 시스템 전체 이식 — `agent-support/`(study-report-v1·study-deck-v1 템플릿, new-presentation/build-index/validate-site, 절차·테스트, studies.toml, legacy-pages.toml), `.claude`·`.agents` 스킬, `docs/` 게시 셸, AGENTS/QUICKSTART/CLAUDE.md. **자체 호스팅 KaTeX 0.16.22**(woff2, code/pre 제외)를 두 템플릿에 추가 — 퀀트 수식용(원래 ml4t 고유, 2026-07-22 ds4th에도 역이식되어 이제 두 repo 동일).
- **Pages 안전화** (`10bfff1`, 피어 Fable5): `.gitignore` `!docs/**/`+`!docs/**`(회차 자산 드롭 방지), `.github/workflows/pages.yml`(sparse CI: 테스트+build-index+validate, main push 시 인덱스 자동 재생성 guard 포함).
- **Chapter 1 산출물** (`6e6fe2c`, 이 세션/Opus): `2026-07-25-ch01` — report.html(7섹션+부록, 9표, 6그림[새 SVG 4 + 재현차트 2], KaTeX 수식), index.html(23슬라이드, 전 슬라이드 data-report-refs 추적, 7섹션·4필수그림 커버). 근거는 재현 실험(공식 epchan.com/book3, SHA-256). build-index --check·validate-site --check-materials·headless Chrome(데스크톱/모바일390/A4, 덱 16:9) 전부 통과.
- **배포·라이브**: main=origin/main=`6e6fe2c` 동기화, Pages(main/docs) 빌드 완료.
  - 사이트 https://restful3.github.io/ml4t/
  - ch01 덱 https://restful3.github.io/ml4t/studies/machine-trading/presentations/2026-07-25-ch01/
  - ch01 리포트 위 경로 + `report.html`

## 🔄 진행 중
- 없음. Chapter 1은 완성·배포·검증 완료.

## ⏭️ 다음 단계
1. **Chapter 2 준비**(2026-08-08, 발표자 미정) — 원자료 `source/Chan E. Machine Trading .../chapter_2_factor_models/`. `new-presentation.py --study machine-trading-2026 --session 2026-08-08-ch02 …` 스캐폴딩 → 리포트 먼저 완성·검증(리포트 게이트) → 덱 파생. study-presentation 스킬/절차 사용.
2. (선택) collaborator 초대 — 사용자가 직접.
3. (선택) 두 repo CI 파일명 통일(ml4t `pages.yml` ↔ ds4th `validate-study-site.yml`).

## 🧠 대화에만 있던 핵심 컨텍스트
- **결정(KaTeX)**: 수식은 자체 호스팅 KaTeX(외부 CDN 금지 유지). 원래 ml4t 고유였으나 2026-07-22 ds4th에도 역이식(3bd49288)되어 두 repo 동일.
- **결정(기준 예시 각색)**: 이식 문서들이 참조하던 kg-llm ch01 경로를 ml4t엔 없어 중립 문구로 교체. 첫 완성 회차(=현재 ch01)가 이제 구체적 기준 예시.
- **발견(멀티에이전트 충돌)**: ml4t↔ds4th가 쌍둥이 구조 + 같은 머신 공유라 aiml 세션 Fable5가 대상 repo를 헷갈려 ds4th용 수정을 ml4t에 잘못 커밋한 사건 발생 → 조율로 해결. **역할 고정: ml4t=이 세션 / ds4th=aiml**. ds4th엔 Codex(aiml:2.1)·Tori(aiml:4.1)도 있음. 형제 repo git 작업 전 대상 확인·`git add -A` 금지·push 전 `git pull --rebase`.
- **발견(gotcha)**: GitHub Pages 소스를 API(PUT)로 바꿔도 자동 재빌드 안 됨 → `POST /pages/builds` 필요.
- 상세 결정은 영속 메모리 `~/.claude/projects/-home-restful3/memory/reference_ml4t_ds4th_presentation_system.md`에도 기록됨.

## ⚠️ 클리어 전 주의
- 커밋 안 됨(전부 사용자 소유 — 자동 커밋·되돌림·삭제 금지): `chapter1_full_report.ipynb` kernelspec 1줄 수정(M, 소유권 미확인), `.claude/.headroom_wrap_marker.json`(?? 사용자 소유), `HANDOFF.md`(?? — untracked 유지가 관례). **이 세션 작업은 전부 커밋·push 완료(6e6fe2c 동기화)라 클리어에 안전.**
- 백그라운드: 없음(띄웠던 http.server 8137/8138 모두 종료 확인).
- 미완료 todo: 없음.

## 📂 관련 파일
- `docs/studies/machine-trading/presentations/2026-07-25-ch01/` — ch01 report.html·index.html·assets(figs 6, katex). 라이브.
- `agent-support/templates/study-{report,deck}/` — 템플릿(+KaTeX). 새 회차 품질 기준.
- `agent-support/scripts/{new-presentation,build-index,validate-site}.py` — 회차 생성·인덱스·검증.
- `agent-support/studies.toml` — machine-trading(active, 2026-07-25 – 2026-10-31).
- `.claude/skills/study-presentation/SKILL.md` — 회차 작업 스킬(이제 하네스에 등록됨).
- `source/Chan E. Machine Trading .../chapter_N_*/` — 원자료(한국어 해설판 + 재현 실험 `src/reports/`).
