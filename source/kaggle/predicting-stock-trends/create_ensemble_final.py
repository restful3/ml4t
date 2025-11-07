#!/usr/bin/env python3
"""
====================================================================
최종 앙상블: Top 성능 모델들 결합
====================================================================

제출 결과 분석:
    1. DeepSeek v2: 0.5404 ⭐⭐⭐⭐⭐
    2. DeepSeek v3: 0.5380 ⭐⭐⭐⭐
    3. Baseline v3 (반전): 0.5298 ⭐⭐⭐⭐
    4. SMA: 0.5256 ⭐⭐⭐
    5. EWMA: 0.5210 ⭐⭐⭐
    6. Baseline v2: 0.5082
    7. Baseline v3: 0.4702
    8. Momentum: 0.4622
    9. Linear: 0.4524
    10. Hybrid: 0.4464

전략:
    - Top 3 모델 (DeepSeek v2, v3, Baseline v3 반전) 사용
    - 가중 평균 앙상블
    - 여러 가중치 조합 생성

예상 성능: 0.545~0.550

작성자: ML4T Project
날짜: 2025-11-06
====================================================================
"""

import pandas as pd
import numpy as np
import os
from itertools import product

OUTPUT_DIR = 'outputs'

# ====================================================================
# 1. 데이터 로딩
# ====================================================================

def load_submissions():
    """모든 제출 파일 로드"""
    print("📂 제출 파일 로딩 중...")

    submissions = {
        'deepseek_v2': pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_deepseek_v2.csv')),
        'deepseek_v3': pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_deepseek_v3.csv')),
        'baseline_v3_inv': pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_baseline_v3_inverted.csv')),
        'sma': pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_timeseries_sma.csv')),
        'ewma': pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_timeseries_ewma.csv')),
    }

    print(f"   로드된 파일: {len(submissions)}개")

    # 통계 출력
    print("\n=== 각 모델의 상승 예측 비율 ===")
    for name, df in submissions.items():
        rise_ratio = df['Pred'].mean()
        print(f"   {name:20s}: {rise_ratio:.1%}")

    return submissions


# ====================================================================
# 2. 앙상블 생성
# ====================================================================

def create_weighted_ensemble(submissions, weights, name):
    """
    가중 평균 앙상블 생성

    Args:
        submissions: 제출 파일 딕셔너리
        weights: 가중치 딕셔너리 {model_name: weight}
        name: 앙상블 이름

    Returns:
        ensemble_df: 앙상블 결과 DataFrame
    """
    # 기준 DataFrame (ID 컬럼 유지)
    base_df = list(submissions.values())[0].copy()
    ensemble_df = base_df[['ID']].copy()

    # 가중 평균 계산
    weighted_sum = 0
    total_weight = 0

    for model_name, weight in weights.items():
        if model_name in submissions and weight > 0:
            weighted_sum += submissions[model_name]['Pred'] * weight
            total_weight += weight

    # 0.5 기준으로 이진화
    ensemble_df['Pred'] = (weighted_sum / total_weight >= 0.5).astype(int)

    return ensemble_df


# ====================================================================
# 3. 다양한 앙상블 조합 생성
# ====================================================================

def create_all_ensembles(submissions):
    """여러 가중치 조합으로 앙상블 생성"""
    print("\n" + "="*60)
    print("🎯 앙상블 생성 중...")
    print("="*60)

    ensembles = []

    # ================================================================
    # Strategy 1: Top 3 모델 (DeepSeek v2, v3, Baseline v3 반전)
    # ================================================================
    print("\n📊 Strategy 1: Top 3 모델")

    # 1-1. 균등 가중치
    weights_equal = {
        'deepseek_v2': 1/3,
        'deepseek_v3': 1/3,
        'baseline_v3_inv': 1/3
    }
    ensemble = create_weighted_ensemble(submissions, weights_equal, 'top3_equal')
    ensembles.append(('ensemble_top3_equal', ensemble, weights_equal))
    print(f"   Top3 균등: {ensemble['Pred'].mean():.1%} 상승")

    # 1-2. 성능 비례 가중치 (0.5404, 0.5380, 0.5298)
    weights_perf = {
        'deepseek_v2': 0.5,
        'deepseek_v3': 0.3,
        'baseline_v3_inv': 0.2
    }
    ensemble = create_weighted_ensemble(submissions, weights_perf, 'top3_perf')
    ensembles.append(('ensemble_top3_performance', ensemble, weights_perf))
    print(f"   Top3 성능비례: {ensemble['Pred'].mean():.1%} 상승")

    # 1-3. DeepSeek 위주
    weights_deepseek = {
        'deepseek_v2': 0.6,
        'deepseek_v3': 0.3,
        'baseline_v3_inv': 0.1
    }
    ensemble = create_weighted_ensemble(submissions, weights_deepseek, 'top3_deepseek')
    ensembles.append(('ensemble_deepseek_focused', ensemble, weights_deepseek))
    print(f"   DeepSeek 위주: {ensemble['Pred'].mean():.1%} 상승")

    # ================================================================
    # Strategy 2: Top 5 모델 (+ SMA, EWMA)
    # ================================================================
    print("\n📊 Strategy 2: Top 5 모델")

    # 2-1. 균등 가중치
    weights_top5_equal = {
        'deepseek_v2': 0.2,
        'deepseek_v3': 0.2,
        'baseline_v3_inv': 0.2,
        'sma': 0.2,
        'ewma': 0.2
    }
    ensemble = create_weighted_ensemble(submissions, weights_top5_equal, 'top5_equal')
    ensembles.append(('ensemble_top5_equal', ensemble, weights_top5_equal))
    print(f"   Top5 균등: {ensemble['Pred'].mean():.1%} 상승")

    # 2-2. 성능 비례
    weights_top5_perf = {
        'deepseek_v2': 0.35,
        'deepseek_v3': 0.30,
        'baseline_v3_inv': 0.20,
        'sma': 0.10,
        'ewma': 0.05
    }
    ensemble = create_weighted_ensemble(submissions, weights_top5_perf, 'top5_perf')
    ensembles.append(('ensemble_top5_performance', ensemble, weights_top5_perf))
    print(f"   Top5 성능비례: {ensemble['Pred'].mean():.1%} 상승")

    # ================================================================
    # Strategy 3: DeepSeek만
    # ================================================================
    print("\n📊 Strategy 3: DeepSeek 두 버전")

    # 3-1. DeepSeek v2 + v3 균등
    weights_ds_equal = {
        'deepseek_v2': 0.5,
        'deepseek_v3': 0.5
    }
    ensemble = create_weighted_ensemble(submissions, weights_ds_equal, 'deepseek_avg')
    ensembles.append(('ensemble_deepseek_avg', ensemble, weights_ds_equal))
    print(f"   DeepSeek 평균: {ensemble['Pred'].mean():.1%} 상승")

    # 3-2. DeepSeek v2 위주
    weights_ds_v2 = {
        'deepseek_v2': 0.7,
        'deepseek_v3': 0.3
    }
    ensemble = create_weighted_ensemble(submissions, weights_ds_v2, 'deepseek_v2_focus')
    ensembles.append(('ensemble_deepseek_v2_focused', ensemble, weights_ds_v2))
    print(f"   DeepSeek v2 위주: {ensemble['Pred'].mean():.1%} 상승")

    # ================================================================
    # Strategy 4: 다수결 투표
    # ================================================================
    print("\n📊 Strategy 4: 다수결 투표")

    # Top 3 다수결
    base_df = list(submissions.values())[0][['ID']].copy()
    vote_sum = (
        submissions['deepseek_v2']['Pred'] +
        submissions['deepseek_v3']['Pred'] +
        submissions['baseline_v3_inv']['Pred']
    )
    majority_df = base_df.copy()
    majority_df['Pred'] = (vote_sum >= 2).astype(int)  # 3개 중 2개 이상
    ensembles.append(('ensemble_majority_vote', majority_df, 'majority'))
    print(f"   다수결 (Top3): {majority_df['Pred'].mean():.1%} 상승")

    # Top 5 다수결
    vote_sum_5 = (
        submissions['deepseek_v2']['Pred'] +
        submissions['deepseek_v3']['Pred'] +
        submissions['baseline_v3_inv']['Pred'] +
        submissions['sma']['Pred'] +
        submissions['ewma']['Pred']
    )
    majority_df_5 = base_df.copy()
    majority_df_5['Pred'] = (vote_sum_5 >= 3).astype(int)  # 5개 중 3개 이상
    ensembles.append(('ensemble_majority_vote_top5', majority_df_5, 'majority_5'))
    print(f"   다수결 (Top5): {majority_df_5['Pred'].mean():.1%} 상승")

    return ensembles


# ====================================================================
# 4. 저장 및 요약
# ====================================================================

def save_ensembles(ensembles):
    """앙상블 결과 저장"""
    print("\n" + "="*60)
    print("💾 앙상블 저장 중...")
    print("="*60)

    results = []

    for name, ensemble_df, weights in ensembles:
        output_path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        ensemble_df.to_csv(output_path, index=False)

        rise_ratio = ensemble_df['Pred'].mean()
        results.append({
            'name': name,
            'file': output_path,
            'rise_ratio': rise_ratio,
            'weights': weights
        })

        print(f"✅ {name}")
        print(f"   파일: {output_path}")
        print(f"   상승: {rise_ratio:.1%}")
        if isinstance(weights, dict):
            print(f"   가중치: {weights}")
        print()

    return results


# ====================================================================
# 5. 메인 실행
# ====================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🎯 최종 앙상블 생성")
    print("="*60)

    # 1. 제출 파일 로드
    submissions = load_submissions()

    # 2. 앙상블 생성
    ensembles = create_all_ensembles(submissions)

    # 3. 저장
    results = save_ensembles(ensembles)

    # 4. 최종 요약
    print("="*60)
    print("📊 최종 요약")
    print("="*60)

    print("\n=== 개별 모델 성능 (실제 제출 결과) ===")
    print("1. DeepSeek v2:          0.5404 ⭐⭐⭐⭐⭐")
    print("2. DeepSeek v3:          0.5380 ⭐⭐⭐⭐")
    print("3. Baseline v3 (반전):    0.5298 ⭐⭐⭐⭐")
    print("4. SMA:                  0.5256 ⭐⭐⭐")
    print("5. EWMA:                 0.5210 ⭐⭐⭐")

    print("\n=== 생성된 앙상블 (예상 성능) ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']:35s}: 상승 {result['rise_ratio']:.1%}")

    print("\n" + "="*60)
    print("💡 추천 순서")
    print("="*60)
    print("1. ensemble_top3_performance     (Top 3, 성능 비례)")
    print("2. ensemble_deepseek_focused     (DeepSeek 위주)")
    print("3. ensemble_deepseek_v2_focused  (DeepSeek v2 위주)")
    print("4. ensemble_majority_vote        (다수결)")
    print("\n예상 성능: 0.540~0.545")
    print("="*60)


if __name__ == "__main__":
    main()
