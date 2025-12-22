# Campisi et al. (2024) Fixed Replication – Final Report

## 1. 배경 및 목적
- `campisi_2024_replication.py` 기반 초기 실험은 Gemini 코드 리뷰(`Gemini_Code_Review.pdf`)에서 **데이터 누수와 통계 검정 미활용** 문제가 지적됐다. 이를 반영해 `campisi_2024_replication_fixed.py`가 작성되었고, 실행은 `run_experiments_fixed.sh`로 자동화됐다.
- 본 보고서는 기존 결론 문서(`Campisi_2024_Final_Report_Conclusion.md`)에서 요약된 결과와 **수정 코드 실행 결과**를 대비해, 수정 사항과 성능 변화, 그리고 원 논문(Campisi et al., 2024)과의 차이를 정리한다.

## 2. 적용된 핵심 수정 요약
| 구분 | 수정 내용 | 근거 |
|------|-----------|------|
| 스케일링 | 표준화를 Walk-Forward CV 루프 외부에서 수행하던 것을 `Pipeline`으로 이동시켜 각 폴드의 훈련 데이터로만 `StandardScaler`를 맞춤 | `FIXES_APPLIED.md:17-55` |
| Feature Selection | 전체 데이터로 Lasso를 실행하던 방식을 `SelectFromModel(Lasso)` 스텝으로 교체해 폴드별 훈련 샘플로만 변수 선택 | `FIXES_APPLIED.md:58-93` |
| 성능 비교 검정 | 정의만 되어 있던 Diebold-Mariano 함수 호출 로직을 추가하여 모델 간 유의한 성능 차이 판단 | `FIXES_APPLIED.md:96-138` |
| 실행 옵션 | Refit 빈도, 최대 이터레이션, Feature Selection on/off 등을 CLI 인자로 제어하도록 Walk-Forward CV와 main 파이프라인을 리팩터링 | `FIXES_APPLIED.md:142-160` |

**효과**: 원 코드에 존재하던 데이터 누수 제거로 인해 전반적인 정확도가 현실적인 수준으로 낮아졌으며, DM 검정으로 모델 우열에 대한 통계적 근거를 확보했다.

## 3. 기존 리포트 대비 성능 변화
| 예측 기간 | 기존 리포트 Accuracy | 수정 후 Accuracy | 변화폭 (Δ) | 수정 후 최고 모델 | 출처 |
|-----------|----------------------|------------------|-----------|------------------|------|
| 7일 | 67.6% (`Campisi_2024_Final_Report_Conclusion.md:34-41`) | 62.38% | -5.22%p | Lasso Regression (`campisi_2024_results_fixed_7d_20251220_133545.md:83-100`) | Feature Selection 유지 시 규제 회귀만이 0.62 수준 확보 |
| 15일 | 80.2% | 69.80% | -10.40%p | Linear/Ridge/Lasso Regression (`campisi_2024_results_fixed_15d_20251220_145134.md:93-101`) | 선형 모델이 0.698 동율로 최상, 앙상블과 DM 차이 4~6σ |
| 30일 | 85.4% | 72.45% | -12.95%p | Lasso Regression (`campisi_2024_results_fixed_30d_20251220_160519.md:83-100`) | 분류 Logistic/LDA는 0.667 수준에 머무름 |
| 60일 | 85.6% | 66.75% | -18.85%p | Random Forest (Classification) (`campisi_2024_results_fixed_60d_20251220_171822.md:83-90`) | 장기 구간에서도 0.67 이하로 하락 |

### 해석
1. **데이터 누수 제거 영향**: 스케일링과 Feature Selection을 CV 내부로 옮기면서, 과거 결과에서 보고된 80~85% 정확도가 62~72% 범위로 하락했다. 이는 이전 보고서의 고성능이 미래 정보를 활용한 결과였음을 시사한다.
2. **모델 포지션 변화**: 기존 결론에서는 Logistic Regression이 30·60일 구간에서 85% 이상의 정확도를 냈다고 보고했지만(`Campisi_2024_Final_Report_Conclusion.md:20-45`), 수정 후에는 Lasso/선형 회귀가 가장 안정적이며 분류 모델과의 차이를 DM 검정으로 확인했다 (`campisi_2024_results_fixed_30d_20251220_160519.md:104-137`).
3. **Feature Selection 재평가**: 30·60일 구간에서 Feature Selection이 오히려 평균 정확도를 하락시켰다는 기존 인사이트가 유지되지만, 정확도 절대 수준이 낮아져 **전체 변수 사용 권장**의 근거가 더 강해졌다 (`campisi_2024_results_fixed_30d_20251220_160-167`, `campisi_2024_results_fixed_60d_20251220_151-159`).

## 4. 원 논문과의 재비교 (30일 예측)
| 분류/회귀 | Campisi et al. 보고값 | Fixed 재현 | 차이 | 출처 |
|-----------|----------------------|------------|------|------|
| 분류 최고 Accuracy | Bagging 0.8275 | Logistic/LDA 0.6675 | -0.1600 | 논문: `Campisi_2024-A comparison...md:407-415`, 재현: `campisi_2024_results_fixed_30d_20251220_160519.md:83-89` |
| 분류 최고 AUC | Random Forest 0.8495 | Random Forest 0.6603 | -0.1892 | 동일 |
| 회귀 최고 Accuracy | Random Forest 0.8003 | Lasso 0.7245 | -0.0758 | 논문: `Campisi_2024-A comparison...md:421-428`, 재현: `campisi_2024_results_fixed_30d_20251220_160519.md:93-100` |
| 회귀 최고 AUC | Random Forest 0.8412 | Bagging 0.6598 | -0.1814 | 동일 |

**차이 원인 요약**
- 원 논문은 교차 검증 내 전처리 세부를 명시하지 않으며, 실행 코드에서도 Data Leakage가 존재했던 것으로 판단된다. Fixed 버전은 Pipeline 기반 처리를 통해 이를 제거했다.
- 논문은 앙상블 모델 우위를 주장하지만, DM 검정 결과 선형 모델이 통계적으로 우수하다 (`campisi_2024_results_fixed_30d_20251220_160519.md:104-137`). 따라서 “Ensemble supremacy” 결론은 재현 시 지지되지 않는다.

## 5. Horizon별 핵심 인사이트 (Fixed 결과)
- **7일**: 모든 모델이 0.62 이하의 Accuracy, DM 검정에서 Lasso가 기타 모델 대비 2~5σ 수준으로 우월 (`campisi_2024_results_fixed_7d_20251220_133545.md:104-135`). 단기 예측은 규제 회귀 + 완전 Feature Selection이 적절.
- **15일**: 0.70 근처 정확도 달성, 선형 모델 간 차이는 없고 앙상블보다 5σ 이상 유리 (`campisi_2024_results_fixed_15d_20251220_145134.md:108-134`). 중기 전략은 단순 모델로 충분.
- **30일**: 회귀 모델 평균 정확도가 분류 모델보다 높고 Feature Selection이 평균 정확도를 -1.36%p 낮춤 (`campisi_2024_results_fixed_30d_20251220_160-167`). 장기 전략에서는 전체 변수를 사용한 선형 회귀 채택 권장.
- **60일**: Random Forest 분류기가 Accuracy 0.6675로 최상이나 AUC 0.5476으로 확률 추정 신뢰도가 낮음 (`campisi_2024_results_fixed_60d_20251220_171822.md:83-100`). Horizon이 길수록 클래스 불균형과 확률 보정 필요성이 확대.

## 6. 향후 권장 사항
1. **확률 캘리브레이션 도입**: Platt scaling 또는 isotonic regression을 파이프라인에 추가해 AUC 및 포지션 크기 결정의 신뢰도를 높일 것.
2. **하이브리드 전략 재설계**: 장기 구간에서는 회귀 출력(연속값)을 직접 사용하거나, threshold를 동적으로 조정하여 낮아진 정확도를 보완할 필요가 있다.
3. **Rerun 1일/다른 Feature Set**: 기존 리포트는 1일 구간도 포함했으므로, 동일한 수정된 파이프라인으로 1일 결과를 재추정하여 전체 서사를 일관되게 맞출 것을 권장한다.

---
*작성일: 2025-12-20*
