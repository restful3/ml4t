#!/usr/bin/env python3
"""
====================================================================
Kaggle 주식 트렌드 예측 - 올바른 Baseline 모델
====================================================================

목적:
    주식의 마지막 학습일 종가 대비 30 거래일 후 가격이
    상승(1) 또는 하락(0)할지 예측

대회 요구사항:
    - test.csv의 각 종목에 대해
    - 마지막 학습일(2024-09-23) 종가 대비
    - 30일 후(2024-11-04) 종가가 높을지(1) 낮을지(0) 예측

올바른 접근:
    - 과거 데이터에서 각 시점의 "30일 후 실제 가격"으로 타겟 생성
    - 기술적 지표로 패턴 학습
    - Random Forest로 분류

예상 성능: 0.58~0.62 (기존 0.50 대비 큰 개선)

작성자: ML4T Project
날짜: 2025-11-06
====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# ====================================================================
# 설정
# ====================================================================
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42

# 학습 설정
HORIZON = 30  # 30 거래일 후 예측
LOOKBACK = 252  # 최근 1년(252 거래일) 데이터 사용

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
# 2. 특징 생성
# ====================================================================

def create_features(df):
    """
    OHLCV 데이터로부터 기술적 특징 생성

    특징 (16개):
        - 가격 기반 (6개): returns, high_low_ratio, close_open_ratio, daily_range
        - 거래량 (2개): volume, volume_log
        - 이동평균 (3개): ma_5, ma_10, ma_20
        - 거래량 MA (2개): volume_ma_5, volume_ma_10
        - 가격/MA 비율 (2개): price_to_ma5, price_to_ma20
        - 기업활동 (2개): has_dividend, has_split
    """
    features = pd.DataFrame()

    # 가격 기반 특징
    features['returns'] = (df['Close'] - df['Open']) / df['Open']
    features['high_low_ratio'] = df['High'] / df['Low']
    features['close_open_ratio'] = df['Close'] / df['Open']
    features['daily_range'] = (df['High'] - df['Low']) / df['Open']

    # 거래량 특징
    features['volume'] = df['Volume']
    features['volume_log'] = np.log1p(df['Volume'])

    # 이동평균
    features['ma_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    features['ma_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    features['ma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()

    # 거래량 이동평균
    features['volume_ma_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    features['volume_ma_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()

    # 가격/MA 비율
    features['price_to_ma5'] = df['Close'] / features['ma_5']
    features['price_to_ma20'] = df['Close'] / features['ma_20']

    # 기업 활동
    features['has_dividend'] = (df['Dividends'] > 0).astype(int)
    features['has_split'] = (df['Stock Splits'] > 0).astype(int)

    # 무한대/결측값 처리
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)

    return features


# ====================================================================
# 3. 학습 데이터 준비
# ====================================================================

def prepare_training_data(train_df, horizon=HORIZON, lookback=LOOKBACK):
    """
    올바른 방법으로 학습 데이터 준비

    각 종목의 과거 데이터에서:
        - t 시점 특징 → t+30일 실제 가격 상승/하락을 타겟으로 학습

    Args:
        train_df: 학습 데이터
        horizon: 예측 기간 (30일)
        lookback: 사용할 과거 데이터 기간 (252일 = 1년)

    Returns:
        X: 특징 DataFrame
        y: 타겟 배열 (0 또는 1)
    """
    print("\n📊 학습 데이터 준비 중...")
    print(f"   - 예측 기간: {horizon} 거래일")
    print(f"   - 사용 기간: 최근 {lookback} 거래일")

    all_features = []
    all_targets = []

    tickers = train_df['Ticker'].unique()

    for ticker in tqdm(tickers, desc="종목 처리 중"):
        # 해당 종목 데이터
        ticker_data = train_df[train_df['Ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date').reset_index(drop=True)

        # 최소 데이터 체크
        if len(ticker_data) < lookback + horizon:
            continue

        # 최근 lookback + horizon일 데이터만 사용
        recent_data = ticker_data.iloc[-(lookback + horizon):].reset_index(drop=True)

        # 특징 생성 (전체 데이터로)
        features = create_features(recent_data)

        # 각 시점마다 샘플 생성
        for i in range(len(recent_data) - horizon):
            # 현재 시점 종가
            current_close = recent_data.iloc[i]['Close']

            # 30일 후 실제 종가
            future_close = recent_data.iloc[i + horizon]['Close']

            # 타겟: 30일 후 > 현재?
            target = 1 if future_close > current_close else 0

            # 현재 시점 특징
            current_features = features.iloc[i]

            all_features.append(current_features)
            all_targets.append(target)

    # DataFrame으로 변환
    X = pd.DataFrame(all_features).reset_index(drop=True)
    y = np.array(all_targets)

    print(f"\n✅ 학습 데이터 생성 완료")
    print(f"   - 총 샘플: {len(X):,}개")
    print(f"   - 상승(1): {y.sum():,}개 ({y.mean():.1%})")
    print(f"   - 하락(0): {len(y) - y.sum():,}개 ({1-y.mean():.1%})")

    return X, y


# ====================================================================
# 4. 테스트 데이터 준비
# ====================================================================

def prepare_test_features(test_df, train_df):
    """
    테스트 데이터 특징 준비

    각 종목의 마지막 학습일(2024-09-23) 데이터로 특징 생성

    Args:
        test_df: 테스트 데이터 (ID, Date)
        train_df: 학습 데이터

    Returns:
        test_features: 테스트 특징 DataFrame
        test_tickers: 종목 리스트
    """
    print("\n📋 테스트 데이터 준비 중...")

    test_features = []
    test_tickers = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="테스트 샘플 처리 중"):
        ticker = row['ID']

        # 해당 종목의 학습 데이터
        ticker_train = train_df[train_df['Ticker'] == ticker].copy()
        ticker_train = ticker_train.sort_values('Date')

        if len(ticker_train) > 0:
            # 최근 데이터로 특징 생성 (이동평균 계산 위해 충분한 데이터 필요)
            recent_data = ticker_train.iloc[-30:] if len(ticker_train) >= 30 else ticker_train

            # 특징 생성
            features = create_features(recent_data)

            # 마지막 행 (가장 최근 데이터)
            last_features = features.iloc[-1]

            test_features.append(last_features)
            test_tickers.append(ticker)
        else:
            # 데이터 없으면 0으로 채움
            test_features.append(pd.Series(0, index=range(16)))
            test_tickers.append(ticker)

    test_X = pd.DataFrame(test_features).reset_index(drop=True)

    print(f"✅ 테스트 데이터 준비 완료: {len(test_X)}개 샘플")

    return test_X, test_tickers


# ====================================================================
# 5. 모델 학습
# ====================================================================

def train_model(X_train, y_train):
    """
    Random Forest 모델 학습

    하이퍼파라미터:
        - n_estimators: 200 (더 많은 트리)
        - max_depth: 15 (더 깊게)
        - min_samples_split: 10 (더 세밀하게)
        - min_samples_leaf: 5
        - random_state: 42
    """
    print("\n🌲 Random Forest 모델 학습 중...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    print(f"✅ 모델 학습 완료")

    return model


# ====================================================================
# 6. 메인 실행
# ====================================================================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("🎯 주식 트렌드 예측 - 올바른 Baseline 모델")
    print("=" * 60)

    # 1. 데이터 로딩
    train_df, test_df, sample_submission = load_data()

    # 2. 학습 데이터 준비
    X, y = prepare_training_data(train_df, horizon=HORIZON, lookback=LOOKBACK)

    # 3. Train/Validation 분할
    print("\n📊 데이터 분할 중...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"   - Train: {len(X_train):,}개")
    print(f"   - Validation: {len(X_val):,}개")

    # 4. 스케일링
    print("\n📏 데이터 정규화 중...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. 모델 학습
    model = train_model(X_train_scaled, y_train)

    # 6. Validation 평가
    print("\n📈 Validation 성능 평가...")
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"\n{'='*60}")
    print(f"✨ Validation Accuracy: {val_accuracy:.4f}")
    print(f"{'='*60}")

    print("\n분류 리포트:")
    print(classification_report(y_val, y_val_pred, target_names=['하락(0)', '상승(1)']))

    # 7. 테스트 데이터 준비
    test_X, test_tickers = prepare_test_features(test_df, train_df)

    # 8. 스케일링
    test_X_scaled = scaler.transform(test_X)

    # 9. 예측
    print("\n🔮 테스트 데이터 예측 중...")
    test_predictions = model.predict(test_X_scaled)

    # 10. 제출 파일 생성
    submission = sample_submission.copy()
    submission['Pred'] = test_predictions

    output_path = os.path.join(OUTPUT_DIR, 'submission_baseline_v3.csv')
    submission.to_csv(output_path, index=False)

    print(f"\n✅ 제출 파일 저장: {output_path}")
    print(f"\n예측 분포:")
    print(f"   - 상승(1): {test_predictions.sum()}개 ({test_predictions.mean():.1%})")
    print(f"   - 하락(0): {len(test_predictions) - test_predictions.sum()}개 ({1-test_predictions.mean():.1%})")

    # 11. 특징 중요도
    print("\n📊 Top 10 중요 특징:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:20s}: {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print("🎉 완료!")
    print("=" * 60)
    print(f"\n💡 예상 성능: 0.58~0.62 (기존 0.50 대비 큰 개선)")
    print(f"   - Validation: {val_accuracy:.4f}")
    print(f"   - 제출 파일: {output_path}")
    print()


if __name__ == "__main__":
    main()
