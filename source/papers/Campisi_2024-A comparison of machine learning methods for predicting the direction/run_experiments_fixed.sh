#!/bin/bash
# =============================================================================
# Campisi 2024 논문 실증 재현 - 다중 예측 기간 실험 (FIXED VERSION)
# =============================================================================
# 사용법:
#   ./run_experiments_fixed.sh                           # 기본 실행 (7, 15, 30, 60일)
#   ./run_experiments_fixed.sh 7 14 21                   # 특정 기간만 실행
#   MAX_ITER=10 ./run_experiments_fixed.sh               # 빠른 테스트 (10 iterations)
#   REFIT_FREQ=30 ./run_experiments_fixed.sh             # 30일마다 재학습
#   NO_FEAT_SEL=1 ./run_experiments_fixed.sh             # Feature selection 없이
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

# 최대 이터레이션 수 설정 (환경변수로 제어 가능)
# 예: MAX_ITER=10 ./run_experiments_fixed.sh
MAX_ITER="${MAX_ITER:-}"  # 기본값: 비어있음 (전체 실행)

# Refit Frequency 설정 (환경변수로 제어 가능)
# 예: REFIT_FREQ=30 ./run_experiments_fixed.sh
REFIT_FREQ="${REFIT_FREQ:-1}"  # 기본값: 1 (매일 재학습)

# Feature Selection 설정 (환경변수로 제어 가능)
# 예: NO_FEAT_SEL=1 ./run_experiments_fixed.sh
NO_FEAT_SEL="${NO_FEAT_SEL:-0}"  # 기본값: 0 (feature selection 사용)

# 로그 디렉토리 생성
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/experiment_fixed_${TIMESTAMP}.log"

# 헤더 출력
echo "======================================================"
echo "Campisi 2024 실험 시작 (FIXED VERSION)"
echo "======================================================"
echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo "예측 기간: ${FORECAST_DAYS[*]} 일"
if [ -n "$MAX_ITER" ]; then
    echo "최대 이터레이션: $MAX_ITER 개 (빠른 테스트 모드)"
else
    echo "최대 이터레이션: 전체 (~755개)"
fi
echo "Refit Frequency: 매 $REFIT_FREQ 일"
if [ "$NO_FEAT_SEL" -eq 1 ]; then
    echo "Feature Selection: 비활성화"
else
    echo "Feature Selection: 활성화"
fi
echo "로그 파일: $MAIN_LOG"
echo "======================================================"
echo ""
echo "주요 개선사항:"
echo "  ✅ Data Leakage 문제 해결 (Pipeline 사용)"
echo "  ✅ Diebold-Mariano 통계 검정 추가"
echo "  ✅ Performance 최적화 (Refit Frequency)"
echo ""

# 로그 파일에도 헤더 기록
{
    echo "======================================================"
    echo "Campisi 2024 실험 로그 (FIXED VERSION)"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "예측 기간: ${FORECAST_DAYS[*]} 일"
    if [ -n "$MAX_ITER" ]; then
        echo "최대 이터레이션: $MAX_ITER 개 (빠른 테스트 모드)"
    else
        echo "최대 이터레이션: 전체 (~755개)"
    fi
    echo "Refit Frequency: 매 $REFIT_FREQ 일"
    if [ "$NO_FEAT_SEL" -eq 1 ]; then
        echo "Feature Selection: 비활성화"
    else
        echo "Feature Selection: 활성화"
    fi
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

    # 명령어 구성
    CMD="python campisi_2024_replication_fixed.py --forecast-days $days"

    # 옵션 추가
    if [ -n "$MAX_ITER" ]; then
        CMD="$CMD --max-iterations $MAX_ITER"
    fi

    if [ "$REFIT_FREQ" -ne 1 ]; then
        CMD="$CMD --refit-frequency $REFIT_FREQ"
    fi

    if [ "$NO_FEAT_SEL" -eq 1 ]; then
        CMD="$CMD --no-feature-selection"
    fi

    echo "실행 명령: $CMD"
    echo ""

    # 실행 및 결과 캡처
    if $CMD 2>&1 | tee -a "$MAIN_LOG"; then
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
echo "생성된 리포트 (FIXED VERSION):"
for days in "${FORECAST_DAYS[@]}"; do
    REPORT=$(ls -t campisi_2024_results_fixed_${days}d_*.md 2>/dev/null | head -1)
    if [ -n "$REPORT" ]; then
        echo "  - $REPORT"
    fi
done
echo ""
echo "생성된 시각화 (FIXED VERSION):"
if [ -d "figures" ]; then
    ls -t figures/*_fixed.png 2>/dev/null | head -10 | while read -r fig; do
        echo "  - $fig"
    done
fi
echo ""
echo "로그 파일: $MAIN_LOG"
echo ""
echo "참고: FIXES_APPLIED.md를 읽어보세요 (수정 사항 상세 설명)"
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

# 결과 비교 스크립트 제안
if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "💡 TIP: 원본 코드와 수정 코드의 성능을 비교하려면:"
    echo "   1. 원본: ./run_experiments.sh"
    echo "   2. 수정: ./run_experiments_fixed.sh"
    echo "   3. 두 결과를 비교하여 data leakage의 영향 확인"
    echo ""
fi
