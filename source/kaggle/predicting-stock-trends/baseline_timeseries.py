#!/usr/bin/env python3
"""
====================================================================
시계열 알고리즘 기반 Baseline
====================================================================

목적:
    시계열 전용 알고리즘으로 주식 가격 예측
    - 각 종목의 마지막 30일 예측
    - 마지막 학습일 대비 상승/하락 분류

시계열 방법들:
    1. Simple Moving Average (단순 이동평균)
    2. Exponential Weighted Moving Average (지수 가중 이동평균)
    3. Linear Trend (선형 추세)
    4. ARIMA (AutoRegressive Integrated Moving Average)

시작은 단순한 방법부터 - ARIMA는 느리므로 먼저 간단한 방법 시도

예상 성능: 0.52~0.56

작성자: ML4T Project
날짜: 2025-11-06
====================================================================
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 설정
# ====================================================================
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42

HORIZON = 30  # 30일 후 예측
LOOKBACK = 60  # 최근 60일 데이터 사용

# ====================================================================
# 1. 데이터 로딩
# ====================================================================

def load_data():
    """CSV 파일 로드"""
    print("📂 데이터 로딩 중...")

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")

    return train_df, test_df, sample_submission


# ====================================================================
# 2. 시계열 예측 방법들
# ====================================================================

def predict_sma(prices, horizon=30):
    """
    Simple Moving Average 예측

    최근 N일 평균으로 미래 예측
    """
    if len(prices) < 5:
        return prices[-1]  # 데이터 부족하면 마지막 값

    # 최근 20일 평균
    sma_20 = np.mean(prices[-20:])
    return sma_20


def predict_ewma(prices, horizon=30):
    """
    Exponential Weighted Moving Average 예측

    최근 데이터에 더 높은 가중치
    """
    if len(prices) < 5:
        return prices[-1]

    # pandas EWMA 사용 (span=20)
    ewma = pd.Series(prices).ewm(span=20, adjust=False).mean().iloc[-1]
    return ewma


def predict_linear_trend(prices, horizon=30):
    """
    Linear Trend 예측

    최근 데이터의 선형 추세를 미래로 연장
    """
    if len(prices) < 10:
        return prices[-1]

    # 최근 30일 데이터
    recent_prices = prices[-30:]

    # 선형 회귀
    x = np.arange(len(recent_prices))

    # y = ax + b
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, recent_prices, rcond=None)[0]

    # 30일 후 예측
    future_x = len(recent_prices) + horizon - 1
    predicted_price = a * future_x + b

    return predicted_price


def predict_momentum(prices, horizon=30):
    """
    Momentum 기반 예측

    최근 수익률을 미래로 연장
    """
    if len(prices) < 30:
        return prices[-1]

    # 최근 30일 수익률
    recent_return = (prices[-1] - prices[-30]) / prices[-30]

    # 최근 30일 수익률(recent_return)을 앞으로 horizon(30일) 동안 반복된다고 가정
    # 즉, 30일 뒤 가격 = 현재 가격 * (1 + 최근 30일 수익률)
    predicted_price = prices[-1] * (1 + recent_return)

    return predicted_price


def predict_hybrid(prices, horizon=30):
    """
    Hybrid 예측 (여러 방법의 평균)

    SMA, EWMA, Linear Trend, Momentum의 평균
    """
    if len(prices) < 10:
        return prices[-1]

    sma_pred = predict_sma(prices, horizon)
    ewma_pred = predict_ewma(prices, horizon)
    linear_pred = predict_linear_trend(prices, horizon)
    momentum_pred = predict_momentum(prices, horizon)

    # 평균 (outlier 제거 위해 median 사용 가능)
    predictions = [sma_pred, ewma_pred, linear_pred, momentum_pred]

    # Median 사용 (극단값 제거)
    return np.median(predictions)


# ====================================================================
# 3. 예측 실행
# ====================================================================

def make_predictions(train_df, test_df, method='hybrid'):
    """
    각 종목에 대해 시계열 예측 수행

    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터
        method: 예측 방법 ('sma', 'ewma', 'linear', 'momentum', 'hybrid')

    Returns:
        predictions: 0 또는 1 배열
    """
    print(f"\n🔮 시계열 예측 중 (방법: {method})...")

    method_funcs = {
        'sma': predict_sma,
        'ewma': predict_ewma,
        'linear': predict_linear_trend,
        'momentum': predict_momentum,
        'hybrid': predict_hybrid
    }

    predict_func = method_funcs.get(method, predict_hybrid)

    predictions = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="종목 예측 중"):
        ticker = row['ID']

        # 해당 종목의 학습 데이터
        ticker_train = train_df[train_df['Ticker'] == ticker].copy()
        ticker_train = ticker_train.sort_values('Date')

        if len(ticker_train) > 0:
            # 종가 시계열
            prices = ticker_train['Close'].values

            # 마지막 종가
            current_price = prices[-1]

            # 30일 후 예측
            predicted_price = predict_func(prices, horizon=HORIZON)

            # 상승(1) / 하락(0)
            prediction = 1 if predicted_price > current_price else 0

            predictions.append(prediction)
        else:
            # 데이터 없으면 0 (하락)
            predictions.append(0)

    return np.array(predictions)


# ====================================================================
# 4. 메인 실행
# ====================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("📈 시계열 알고리즘 기반 Baseline")
    print("=" * 60)

    # 1. 데이터 로딩
    train_df, test_df, sample_submission = load_data()

    # 2. 여러 방법 시도
    methods = ['sma', 'ewma', 'linear', 'momentum', 'hybrid']

    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"방법: {method.upper()}")
        print(f"{'='*60}")

        # 예측
        predictions = make_predictions(train_df, test_df, method=method)

        # 결과 저장
        submission = sample_submission.copy()
        submission['Pred'] = predictions

        output_path = os.path.join(OUTPUT_DIR, f'submission_timeseries_{method}.csv')
        submission.to_csv(output_path, index=False)

        # 통계
        rise_ratio = predictions.mean()
        results[method] = {
            'rise_ratio': rise_ratio,
            'file': output_path
        }

        print(f"\n✅ 저장: {output_path}")
        print(f"   상승 예측: {rise_ratio:.1%} ({predictions.sum()}개)")
        print(f"   하락 예측: {1-rise_ratio:.1%} ({len(predictions)-predictions.sum()}개)")

    # 3. 결과 요약
    print("\n" + "=" * 60)
    print("📊 전체 결과 요약")
    print("=" * 60)

    for method, result in results.items():
        print(f"{method:15s}: 상승={result['rise_ratio']:.1%}")

    print("\n" + "=" * 60)
    print("💡 추천")
    print("=" * 60)
    print("1. hybrid: 여러 방법의 median (가장 안정적)")
    print("2. linear: 추세 기반 (추세가 강한 경우)")
    print("3. momentum: 모멘텀 기반 (변동성 큰 경우)")
    print("\n각 방법을 제출해보고 실제 성능 확인!")
    print("=" * 60)


if __name__ == "__main__":
    main()
