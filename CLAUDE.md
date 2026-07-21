@AGENTS.md

## Claude Code

- 반복 가능한 회차 리포트·발표자료 작업은 `.claude/skills/study-presentation/`의 프로젝트 스킬을 사용한다.
- 해당 스킬이 안내하는 `agent-support/templates/STUDY_SESSION_BLUEPRINT.md`와 두 템플릿의 완성 조건을 리포트·슬라이드·자동 목차의 공통 품질 기준으로 사용한다.
- 재현 가능한 챕터 실험은 별개의 `/create-chapter-analysis` 명령·`create-chan-chapter-analysis` 스킬로 만들며, 그 산출물은 발표용 리포트의 원자료다.
- `.claude/settings.local.json`은 개인 설정이므로 공유하거나 공용 설정으로 복사하지 않는다.
