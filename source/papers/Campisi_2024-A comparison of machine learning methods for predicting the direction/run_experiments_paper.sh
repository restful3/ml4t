#!/bin/bash
# =============================================================================
# Campisi 2024 논문 재현 - 30일 Paper Mode 전용 실행 스크립트
# =============================================================================
# 사용법:
#   ./run_experiments_paper.sh        # 기본 30일 실행 (논문 설정)
#   NO_FEAT_SEL=1 ./run_experiments_paper.sh   # Feature Selection 끄기
#   WALKFWD=1 ./run_experiments_paper.sh       # Debug용 walk-forward 모드
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "가상환경 활성화 완료: $(which python)"
else
    echo "ERROR: .venv 디렉토리가 없습니다."
    exit 1
fi

FORECAST_DAYS=30
NO_FEAT_SEL="${NO_FEAT_SEL:-0}"
WALKFWD="${WALKFWD:-0}"

if [ "$WALKFWD" -eq 1 ]; then
    EVAL_MODE="walk-forward"
else
    EVAL_MODE="paper"
fi

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/experiment_paper_${TIMESTAMP}.log"

COMMAND="python campisi_2024_replication_paper.py --forecast-days $FORECAST_DAYS --evaluation-mode $EVAL_MODE"

if [ "$NO_FEAT_SEL" -eq 1 ]; then
    COMMAND="$COMMAND --no-feature-selection"
fi

if [ "$WALKFWD" -ne 1 ]; then
    # default: 논문 설정 → 비중첩, class_weight, threshold 최적화 유지
    :
else
    COMMAND="$COMMAND --allow-overlap-labels --disable-balanced-class-weights --disable-optimize-threshold"
fi

echo "실행 명령: $COMMAND"
echo "로그 파일: $MAIN_LOG"

echo "======================================================"
echo "Campisi 2024 Paper Mode 실행"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

if $COMMAND 2>&1 | tee "$MAIN_LOG"; then
    echo "실행 완료"
else
    echo "실행 실패"
fi
