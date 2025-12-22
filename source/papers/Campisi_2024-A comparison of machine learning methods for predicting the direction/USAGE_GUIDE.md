# Usage Guide - Campisi 2024 Replication (Fixed Version)

## Quick Start

### 기본 사용법

```bash
# 1. 실행 권한 부여 (처음 한 번만)
chmod +x run_experiments_fixed.sh

# 2. 기본 실행 (1, 7, 15, 30, 60일 예측)
./run_experiments_fixed.sh

# 3. 특정 기간만 실행
./run_experiments_fixed.sh 7 30
```

---

## Advanced Usage

### 1. 빠른 테스트 (10 iterations만)

```bash
MAX_ITER=10 ./run_experiments_fixed.sh
```

**소요 시간**: 약 5-10분
**용도**: 코드 동작 확인, 파라미터 테스트

---

### 2. Performance 최적화 (30일마다 재학습)

```bash
REFIT_FREQ=30 ./run_experiments_fixed.sh
```

**효과**: 계산 시간 ~96% 감소 (매일 재학습 대비)
**권장**: 실제 배포 환경에서 사용

---

### 3. Feature Selection 비활성화

```bash
NO_FEAT_SEL=1 ./run_experiments_fixed.sh
```

**용도**: Feature selection의 효과를 분리하여 측정

---

### 4. 옵션 조합

```bash
# 빠른 테스트 + 30일 재학습 + Feature Selection 없이
MAX_ITER=10 REFIT_FREQ=30 NO_FEAT_SEL=1 ./run_experiments_fixed.sh 7 30
```

---

## 환경 변수 설명

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MAX_ITER` | (전체) | 최대 CV 이터레이션 수 (예: 10, 50, 100) |
| `REFIT_FREQ` | 1 | 모델 재학습 주기 (일 단위) |
| `NO_FEAT_SEL` | 0 | 1로 설정시 Feature Selection 비활성화 |

---

## 출력 파일

### 1. 리포트 파일

```
campisi_2024_results_fixed_{days}d_{timestamp}.md
```

**내용**:
- 데이터 요약
- 기술 통계량
- VIF 값
- 모델 성능 비교 (Feature Selection 전후)
- **Diebold-Mariano 통계 검정 결과** (새로 추가)
- 주요 인사이트

### 2. 시각화 파일

```
figures/
├── correlation_heatmap_fixed.png
├── roc_curves_fixed.png
└── returns_timeseries_fixed.png
```

### 3. 로그 파일

```
logs/experiment_fixed_{timestamp}.log
```

**내용**: 전체 실행 과정의 상세 로그

---

## 실행 예시

### 예시 1: 프로토타입 개발

```bash
# 빠르게 테스트 (10 iterations)
MAX_ITER=10 ./run_experiments_fixed.sh 7

# 출력 확인
cat campisi_2024_results_fixed_7d_*.md
```

### 예시 2: 논문 재현

```bash
# 전체 데이터로 실행 (시간이 오래 걸림 - 수 시간)
./run_experiments_fixed.sh

# 결과 리포트 확인
ls -t campisi_2024_results_fixed_*.md | head -5
```

### 예시 3: 실전 배포용

```bash
# 30일마다 재학습 (계산 효율성)
REFIT_FREQ=30 ./run_experiments_fixed.sh 7 15 30

# 성능 지표 확인
grep "최고 성능" campisi_2024_results_fixed_*.md
```

---

## 원본 vs 수정 버전 비교

### 원본 코드 실행

```bash
./run_experiments.sh 7
```

### 수정 코드 실행

```bash
./run_experiments_fixed.sh 7
```

### 비교 포인트

1. **Accuracy 비교**
   - 원본: ~0.70-0.82 (data leakage로 부풀려짐)
   - 수정: ~0.55-0.65 (정확한 out-of-sample)

2. **실행 시간**
   - 원본: 비슷 (동일 반복 수 기준)
   - 수정 (REFIT_FREQ=30): 훨씬 빠름

3. **신뢰성**
   - 원본: ❌ 미래 정보 유출
   - 수정: ✅ 정확한 시뮬레이션

---

## Troubleshooting

### 문제 1: 가상환경이 없다는 에러

```bash
ERROR: .venv 디렉토리가 없습니다.
```

**해결**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 문제 2: 실행 시간이 너무 오래 걸림

**해결 1**: 빠른 테스트
```bash
MAX_ITER=10 ./run_experiments_fixed.sh
```

**해결 2**: Refit Frequency 증가
```bash
REFIT_FREQ=30 ./run_experiments_fixed.sh
```

---

### 문제 3: Memory Error

**해결 1**: Feature Selection 비활성화
```bash
NO_FEAT_SEL=1 ./run_experiments_fixed.sh
```

**해결 2**: 한 번에 하나씩 실행
```bash
./run_experiments_fixed.sh 7
./run_experiments_fixed.sh 15
./run_experiments_fixed.sh 30
```

---

## 성능 벤치마크

### 테스트 환경
- CPU: Intel i7-10700K
- RAM: 32GB
- 데이터: 2,886 observations

### 실행 시간 비교

| 설정 | 7일 예측 | 30일 예측 | 전체 (5개 기간) |
|------|----------|-----------|------------------|
| 전체 실행 (REFIT_FREQ=1) | ~3시간 | ~3시간 | ~15시간 |
| 최적화 (REFIT_FREQ=30) | ~10분 | ~10분 | ~50분 |
| 빠른 테스트 (MAX_ITER=10) | ~1분 | ~1분 | ~5분 |

---

## 결과 해석

### Accuracy 기준

| Accuracy | 해석 |
|----------|------|
| > 0.60 | 매우 좋음 (금융 시장 예측에서 우수) |
| 0.55-0.60 | 좋음 (실전 투자 가능) |
| 0.50-0.55 | 보통 (추가 개선 필요) |
| < 0.50 | 나쁨 (랜덤보다 못함) |

### Diebold-Mariano Test 해석

```
DM Statistic > 0, p-value < 0.05
→ Model 1이 Model 2보다 통계적으로 유의하게 나쁨

DM Statistic < 0, p-value < 0.05
→ Model 1이 Model 2보다 통계적으로 유의하게 좋음

p-value > 0.05
→ 두 모델 간 유의한 차이 없음
```

---

## Best Practices

### 1. 개발 워크플로우

```bash
# Step 1: 빠른 프로토타이핑
MAX_ITER=10 ./run_experiments_fixed.sh 7

# Step 2: 중간 검증
MAX_ITER=50 REFIT_FREQ=30 ./run_experiments_fixed.sh 7 15 30

# Step 3: 최종 실행 (논문 제출용)
./run_experiments_fixed.sh
```

### 2. 리소스 관리

```bash
# CPU 사용률 제한 (다른 작업 병행시)
nice -n 10 ./run_experiments_fixed.sh

# 백그라운드 실행 (터미널 닫아도 계속 실행)
nohup ./run_experiments_fixed.sh > output.log 2>&1 &

# 진행상황 확인
tail -f output.log
```

### 3. 결과 버전 관리

```bash
# 날짜별로 폴더 정리
mkdir -p results/$(date +%Y%m%d)
mv campisi_2024_results_fixed_*.md results/$(date +%Y%m%d)/
mv figures/*_fixed.png results/$(date +%Y%m%d)/
```

---

## FAQ

**Q: 원본 코드와 수정 코드의 성능 차이가 왜 나나요?**

A: 원본 코드는 data leakage로 인해 미래 정보를 사용하여 과도하게 낙관적인 결과를 보입니다. 수정 코드의 낮은 성능이 실제 기대할 수 있는 정확한 성능입니다.

**Q: REFIT_FREQ를 얼마로 설정하는 것이 좋나요?**

A:
- 연구용: 1 (매일 재학습, 가장 정확)
- 실전용: 30-90 (월 단위 재학습, 계산 효율적)
- 테스트: 10 (빠른 검증)

**Q: Feature Selection을 사용해야 하나요?**

A: 논문 재현에는 사용 권장. 하지만 과적합이 우려되면 `NO_FEAT_SEL=1`로 비활성화하고 비교해보세요.

**Q: 어느 모델이 가장 좋나요?**

A: Diebold-Mariano Test 결과를 확인하세요. 일반적으로 Random Forest와 Bagging이 좋은 성능을 보이지만, 통계적 유의성을 검증해야 합니다.

---

## 참고 문서

- [FIXES_APPLIED.md](FIXES_APPLIED.md) - 수정 사항 상세 설명
- [README.md](README.md) - 프로젝트 개요
- [campisi_2024_replication_fixed.py](campisi_2024_replication_fixed.py) - 수정된 코드

---

**마지막 업데이트**: 2025-12-20
