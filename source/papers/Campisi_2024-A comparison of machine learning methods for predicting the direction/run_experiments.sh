#!/bin/bash
# =============================================================================
# Campisi 2024 논문 실증 재현 - 다중 예측 기간 실험
# =============================================================================
# 사용법:
#   ./run_experiments.sh              # 기본 실행 (7, 15, 30, 60일)
#   ./run_experiments.sh 7 14 21      # 특정 기간만 실행
# =============================================================================

set -e  # 에러 발생시 중단

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "가상환경 활성화 완료: $(which python)"
else
    echo "ERROR: .venv 디렉토리가 없습니다. 가상환경을 먼저 생성하세요."
    exit 1
fi

# 예측 기간 설정 (인자가 없으면 기본값 사용)
if [ $# -eq 0 ]; then
    FORECAST_DAYS=(1 7 15 30 60)
else
    FORECAST_DAYS=("$@")
fi

# 로그 디렉토리 생성
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/experiment_${TIMESTAMP}.log"

# 헤더 출력
echo "======================================================"
echo "Campisi 2024 실험 시작"
echo "======================================================"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "예측 기간: ${FORECAST_DAYS[*]} 일"
echo "로그 파일: $MAIN_LOG"
echo "======================================================"
echo ""

# 로그 파일에도 헤더 기록
{
    echo "======================================================"
    echo "Campisi 2024 실험 로그"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "예측 기간: ${FORECAST_DAYS[*]} 일"
    echo "======================================================"
    echo ""
} >> "$MAIN_LOG"

# 전체 시작 시간
TOTAL_START=$(date +%s)

# 성공/실패 카운터
SUCCESS_COUNT=0
FAIL_COUNT=0

# 각 예측 기간에 대해 순차 실행
for days in "${FORECAST_DAYS[@]}"; do
    echo "------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] ${days}일 예측 시작..."
    echo "------------------------------------------------------"

    START_TIME=$(date +%s)

    # 실행 및 결과 캡처
    if python campisi_2024_replication.py --forecast-days "$days" 2>&1 | tee -a "$MAIN_LOG"; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        MINUTES=$((ELAPSED / 60))
        SECONDS=$((ELAPSED % 60))

        echo ""
        echo "[$(date +%H:%M:%S)] ${days}일 예측 완료 (소요시간: ${MINUTES}분 ${SECONDS}초)"
        echo ""
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "[$(date +%H:%M:%S)] ${days}일 예측 실패!"
        echo ""
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# 전체 종료 시간
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
TOTAL_MINUTES=$((TOTAL_ELAPSED / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

# 요약 출력
echo "======================================================"
echo "모든 실험 완료!"
echo "======================================================"
echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "총 소요시간: ${TOTAL_MINUTES}분 ${TOTAL_SECONDS}초"
echo "성공: ${SUCCESS_COUNT}개 / 실패: ${FAIL_COUNT}개"
echo ""
echo "생성된 리포트:"
for days in "${FORECAST_DAYS[@]}"; do
    REPORT=$(ls -t campisi_2024_results_${days}d_*.md 2>/dev/null | head -1)
    if [ -n "$REPORT" ]; then
        echo "  - $REPORT"
    fi
done
echo ""
echo "로그 파일: $MAIN_LOG"
echo "======================================================"

# 로그 파일에도 요약 기록
{
    echo ""
    echo "======================================================"
    echo "실험 완료 요약"
    echo "종료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "총 소요시간: ${TOTAL_MINUTES}분 ${TOTAL_SECONDS}초"
    echo "성공: ${SUCCESS_COUNT}개 / 실패: ${FAIL_COUNT}개"
    echo "======================================================"
} >> "$MAIN_LOG"
