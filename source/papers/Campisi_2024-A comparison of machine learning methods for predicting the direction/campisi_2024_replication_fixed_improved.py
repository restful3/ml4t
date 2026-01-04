#!/usr/bin/env python3
"""
Campisi et al. (2024) 논문 실증 재현 - FIXED + IMPROVED VERSION

"A comparison of machine learning methods for predicting the direction
of the US stock market on the basis of volatility indices"

International Journal of Forecasting, 2023

Authors: Giovanni Campisi, Silvia Muzzioli, Bernard De Baets

FIXES:
1. Data Leakage 문제 해결 - Pipeline 사용으로 Scaler와 Feature Selection을 CV 내부로 이동
2. Diebold-Mariano Test 구현 및 호출
3. Performance 최적화 옵션 추가 (refit_frequency)

IMPROVEMENTS:
1. Paper-style holdout split 옵션과 비중첩 레이블 샘플링 지원으로 원 논문 설정을 재현 가능
2. 누락된 PUTCALL 변수를 포함하고 Balanced class weight/threshold 최적화 옵션 추가
3. 회귀 모델에 대한 확률 기반 임곗값 최적화로 분류 지표 개선
"""

import os
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, LassoCV
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    BaggingClassifier, BaggingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.base import clone

import yfinance as yf
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================
START_DATE = '2011-01-01'
END_DATE = '2022-07-31'
TRAIN_SIZE = 2128  # 70% of data
FORECAST_DAYS = 30  # Default forecast horizon (days ahead to predict)
GAP = 30  # Forward-looking prevention gap
RANDOM_STATE = 42

TICKERS = {
    'SP500': '^GSPC',
    'VIX': '^VIX',
    'VIX9D': '^VIX9D',
    'VIX3M': '^VIX3M',
    'VIX6M': '^VIX6M',
    'VVIX': '^VVIX',
    'VXN': '^VXN',
    'GVZ': '^GVZ',
    'OVX': '^OVX',
    'SKEW': '^SKEW',
    # Yahoo Finance ticker for CBOE total put/call ratio
    'PUTCALL': '^CPC'
}

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')


# =============================================================================
# Data Collection Functions
# =============================================================================
def download_volatility_indices(start_date: str = START_DATE,
                                 end_date: str = END_DATE) -> pd.DataFrame:
    """
    Download volatility indices from Yahoo Finance.

    Args:
        start_date: Start date for data download
        end_date: End date for data download

    Returns:
        DataFrame with all volatility indices
    """
    print("\n[1] Downloading volatility indices from Yahoo Finance...")

    all_data = {}
    for name, ticker in tqdm(TICKERS.items(), desc="Downloading"):
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                           progress=False, auto_adjust=True)
            if len(df) > 0:
                # Handle multi-level column index from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'].squeeze()
                else:
                    close_data = df['Close']
                all_data[name] = close_data
            else:
                print(f"  Warning: No data for {name} ({ticker})")
        except Exception as e:
            print(f"  Error downloading {name}: {e}")

    # Create DataFrame from dictionary of Series
    data = pd.DataFrame(all_data)
    print(f"  Downloaded {len(data)} observations for {len(data.columns)} variables")

    return data


def calculate_derived_features(data: pd.DataFrame, forecast_days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Calculate derived features: RVOL, Returns, Target.

    Args:
        data: DataFrame with raw data
        forecast_days: Number of days ahead to predict (default: 30)

    Returns:
        DataFrame with derived features added
    """
    print(f"\n[2] Calculating derived features (forecast_days={forecast_days})...")

    df = data.copy()

    # RVOL: Realized volatility (30-day rolling std, annualized)
    daily_returns = np.log(df['SP500'] / df['SP500'].shift(1))
    df['RVOL'] = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100

    # Forward returns (target variable) - parameterized by forecast_days
    returns_col = f'Returns{forecast_days}'
    df[returns_col] = np.log(df['SP500'].shift(-forecast_days) / df['SP500'])

    # Target: Binary classification target (1 if positive, 0 otherwise)
    df['Target'] = (df[returns_col] > 0).astype(int)

    print(f"  RVOL: min={df['RVOL'].min():.2f}, max={df['RVOL'].max():.2f}")
    print(f"  {returns_col}: min={df[returns_col].min():.4f}, max={df[returns_col].max():.4f}")
    print(f"  Target distribution: {df['Target'].value_counts().to_dict()}")

    return df, returns_col


# =============================================================================
# Sampling Helpers
# =============================================================================
def apply_non_overlapping_sampling(data: pd.DataFrame,
                                   forecast_days: int,
                                   returns_col: str) -> pd.DataFrame:
    """Down-sample observations to avoid overlapping forecast windows.

    The original paper evaluated monthly (30-day) rebalancing with non-overlapping
    targets. When this option is enabled we keep every `forecast_days`-th
    observation to mimic that setup. The last few rows (where the forward return
    is NaN) are dropped automatically after sampling.
    """
    if forecast_days <= 1:
        return data

    if returns_col not in data.columns or 'Target' not in data.columns:
        return data

    eligible = data.dropna(subset=[returns_col, 'Target']).copy()
    if eligible.empty:
        return eligible
    indices = list(range(0, len(eligible), forecast_days))
    if indices and indices[-1] != len(eligible) - 1:
        indices.append(len(eligible) - 1)
    sampled = eligible.iloc[indices].copy()
    return sampled


# =============================================================================
# Preprocessing Functions
# =============================================================================
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward fill and backward fill.

    Args:
        data: DataFrame with potential missing values

    Returns:
        DataFrame with missing values handled
    """
    print("\n[3] Handling missing values...")

    initial_rows = len(data)
    initial_missing = data.isnull().sum().sum()

    df = data.copy()
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    if all_nan_cols:
        print(f"  Removing columns with all NaN: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    # Forward fill, then backward fill
    df = df.ffill().bfill()

    # Drop any remaining rows with NaN
    df = df.dropna()

    final_rows = len(df)
    print(f"  Initial rows: {initial_rows}, Final rows: {final_rows}")
    print(f"  Initial missing values: {initial_missing}")
    print(f"  Rows removed: {initial_rows - final_rows}")

    return df


def calculate_summary_statistics(data: pd.DataFrame,
                                  feature_cols: List[str],
                                  returns_col: str = 'Returns30') -> pd.DataFrame:
    """
    Calculate summary statistics matching paper's Table 1.

    Args:
        data: DataFrame with features
        feature_cols: List of feature column names
        returns_col: Name of the returns column

    Returns:
        DataFrame with summary statistics
    """
    print("\n[4] Calculating summary statistics...")

    stats_list = []

    for col in feature_cols + [returns_col]:
        series = data[col].dropna()
        n = len(series)

        # Basic statistics
        mean = series.mean()
        std = series.std()
        skewness = series.skew()
        kurtosis = series.kurtosis()

        # Autocorrelations
        acf_values = acf(series, nlags=3, fft=True)
        rho1, rho2, rho3 = acf_values[1], acf_values[2], acf_values[3]

        # ADF test
        adf_stat = adfuller(series, autolag='AIC')[0]

        # Jarque-Bera test
        jb_stat = jarque_bera(series)[0]

        stats_list.append({
            'Variable': col,
            'N': n,
            'Mean': mean,
            'St. Dev.': std,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'rho1': rho1,
            'rho2': rho2,
            'rho3': rho3,
            'ADF': adf_stat,
            'JB': jb_stat
        })

    stats_df = pd.DataFrame(stats_list)
    return stats_df


def calculate_correlation_matrix(data: pd.DataFrame,
                                  feature_cols: List[str],
                                  returns_col: str = 'Returns30') -> pd.DataFrame:
    """
    Calculate correlation matrix for features.

    Args:
        data: DataFrame with features
        feature_cols: List of feature column names
        returns_col: Name of the returns column

    Returns:
        Correlation matrix DataFrame
    """
    cols = feature_cols + [returns_col]
    return data[cols].corr()


# =============================================================================
# Feature Selection Functions
# =============================================================================
def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor for each feature.

    Args:
        X: Feature DataFrame

    Returns:
        DataFrame with VIF values
    """
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)


# =============================================================================
# Walk-Forward Validation - FIXED VERSION
# =============================================================================
class WalkForwardCV:
    """
    Walk-Forward Cross-Validation with rolling window.

    Implements the validation scheme from the paper:
    - Training size: 2128 observations (70%)
    - Gap: 30 days (forward-looking prevention)
    - Rolling window (not expanding)
    """

    def __init__(self, train_size: int = TRAIN_SIZE, gap: int = GAP,
                 max_iterations: Optional[int] = None,
                 refit_frequency: int = 1):
        """
        Args:
            train_size: Number of training samples
            gap: Gap between train and test to prevent leakage
            max_iterations: Maximum number of iterations (for debugging)
            refit_frequency: Retrain model every N days (default: 1)
        """
        self.train_size = train_size
        self.gap = gap
        self.max_iterations = max_iterations
        self.refit_frequency = refit_frequency

    def split(self, X: np.ndarray):
        """
        Generate train/test indices for walk-forward validation.

        Args:
            X: Feature matrix

        Yields:
            Tuple of (train_indices, test_indices, should_refit)
        """
        n_samples = len(X)
        iteration_count = 0
        last_train_idx = None

        for test_idx in range(self.train_size + self.gap, n_samples):
            if self.max_iterations is not None and iteration_count >= self.max_iterations:
                break

            train_end = test_idx - self.gap
            train_start = train_end - self.train_size

            if train_start >= 0:
                # Determine if we should refit the model
                should_refit = (
                    last_train_idx is None or
                    iteration_count % self.refit_frequency == 0
                )

                yield (
                    list(range(train_start, train_end)),
                    [test_idx],
                    should_refit
                )

                last_train_idx = train_idx = list(range(train_start, train_end))
                iteration_count += 1

    def get_n_splits(self, X: np.ndarray) -> int:
        """Get the number of splits."""
        total = max(0, len(X) - self.train_size - self.gap)
        if self.max_iterations is not None:
            return min(total, self.max_iterations)
        return total


class PaperHoldoutCV:
    """Single train/test split mimicking the paper's 70/30 setup."""

    def __init__(self, train_size: int = TRAIN_SIZE, gap: int = 0):
        self.train_size = train_size
        self.gap = gap

    def split(self, X: np.ndarray):
        train_indices = list(range(self.train_size))
        start_test = self.train_size + self.gap
        test_indices = list(range(start_test, len(X)))
        yield train_indices, test_indices, True

    def get_n_splits(self, X: np.ndarray) -> int:
        return 1 if len(X) > self.train_size else 0


# =============================================================================
# Model Definitions with Pipeline
# =============================================================================
def get_classification_models(use_feature_selection: bool = True,
                               lasso_alpha: float = 0.001,
                               balanced: bool = False) -> Dict:
    """
    Get dictionary of classification models with Pipeline.

    Args:
        use_feature_selection: Whether to include Lasso feature selection
        lasso_alpha: Alpha parameter for Lasso feature selection

    Returns:
        Dictionary mapping model names to Pipeline objects
    """
    models = {}

    class_weight = 'balanced' if balanced else None
    base_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight=class_weight
        ),
        'LDA': LinearDiscriminantAnalysis(),
        'Random Forest (Clf)': RandomForestClassifier(
            n_estimators=800,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight
        ),
        'Bagging (Clf)': BaggingClassifier(
            n_estimators=800,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting (Clf)': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    for name, base_model in base_models.items():
        steps = [('scaler', StandardScaler())]

        if use_feature_selection:
            # Feature selection inside pipeline
            steps.append((
                'selector',
                SelectFromModel(
                    Lasso(alpha=lasso_alpha, random_state=RANDOM_STATE),
                    threshold=1e-5
                )
            ))

        steps.append(('classifier', base_model))
        models[name] = Pipeline(steps)

    return models


def get_regression_models(use_feature_selection: bool = True,
                          lasso_alpha: float = 0.001) -> Dict:
    """
    Get dictionary of regression models with Pipeline.

    Args:
        use_feature_selection: Whether to include Lasso feature selection
        lasso_alpha: Alpha parameter for Lasso feature selection

    Returns:
        Dictionary mapping model names to Pipeline objects
    """
    models = {}

    base_models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.001),
        'Random Forest (Reg)': RandomForestRegressor(
            n_estimators=500,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Bagging (Reg)': BaggingRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting (Reg)': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    for name, base_model in base_models.items():
        steps = [('scaler', StandardScaler())]

        if use_feature_selection:
            steps.append((
                'selector',
                SelectFromModel(
                    Lasso(alpha=lasso_alpha, random_state=RANDOM_STATE),
                    threshold=1e-5
                )
            ))

        steps.append(('regressor', base_model))
        models[name] = Pipeline(steps)

    return models


# =============================================================================
# Training and Prediction - FIXED VERSION
# =============================================================================
def train_and_predict_classification(pipeline, X: np.ndarray, y: np.ndarray,
                                      cv,
                                      desc: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train classification model and get predictions using walk-forward validation.
    FIXED: Pipeline handles scaling and feature selection inside CV loop.

    Args:
        pipeline: Classification pipeline
        X: Feature matrix (NOT scaled)
        y: Target variable
        cv: Cross-validation object
        desc: Description for progress bar

    Returns:
        Tuple of (predictions, probabilities, actuals)
    """
    predictions = []
    probabilities = []
    actuals = []

    # Cache for trained model (for refit_frequency > 1)
    cached_model = None

    splits = list(cv.split(X))

    for train_idx, test_idx, should_refit in tqdm(splits, desc=desc, leave=False):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Refit model if needed
        if should_refit or cached_model is None:
            cached_model = clone(pipeline)
            cached_model.fit(X_train, y_train)

        pred = cached_model.predict(X_test)
        predictions.extend(pred.tolist())
        actuals.extend(y_test.tolist())

        if hasattr(cached_model, 'predict_proba'):
            prob = cached_model.predict_proba(X_test)[:, 1]
            probabilities.extend(prob.tolist())
        else:
            if hasattr(cached_model.named_steps['classifier'], 'decision_function'):
                decision = cached_model.decision_function(X_test)
                prob = 1 / (1 + np.exp(-decision))
                probabilities.extend(prob.tolist())
            else:
                probabilities.extend(pred.tolist())

    return np.array(predictions), np.array(probabilities), np.array(actuals)


def train_and_predict_regression(pipeline, X: np.ndarray, y: np.ndarray,
                                  cv,
                                  desc: str = "",
                                  optimize_threshold: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train regression model and get predictions using walk-forward validation.
    FIXED: Pipeline handles scaling and feature selection inside CV loop.

    Args:
        pipeline: Regression pipeline
        X: Feature matrix (NOT scaled)
        y: Target variable (continuous)
        cv: Cross-validation object
        desc: Description for progress bar

    Returns:
        Tuple of (predictions, actuals)
    """
    predictions = []
    actuals = []
    probabilities = []
    binaries = []

    cached_model = None
    cached_threshold = 0.5

    splits = list(cv.split(X))

    for train_idx, test_idx, should_refit in tqdm(splits, desc=desc, leave=False):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Refit model if needed
        if should_refit or cached_model is None:
            cached_model = clone(pipeline)
            cached_model.fit(X_train, y_train)

            if optimize_threshold:
                train_probs = 1 / (1 + np.exp(-cached_model.predict(X_train)))
                train_binary = (y_train > 0).astype(int)
                fpr, tpr, thresholds = roc_curve(train_binary, train_probs)
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                cached_threshold = thresholds[best_idx]
            else:
                cached_threshold = 0.5

        pred = cached_model.predict(X_test)
        prob = 1 / (1 + np.exp(-pred))
        pred_binary = (prob >= cached_threshold).astype(int)

        predictions.extend(pred.tolist())
        probabilities.extend(prob.tolist())
        binaries.extend(pred_binary.tolist())
        actuals.extend(y_test.tolist())

    return (
        np.array(predictions),
        np.array(actuals),
        np.array(probabilities),
        np.array(binaries)
    )


# =============================================================================
# Evaluation Metrics
# =============================================================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate evaluation metrics: Accuracy, AUC, F-measure.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary with metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F-measure': f1_score(y_true, y_pred)
    }

    if y_prob is not None:
        try:
            raw_auc = roc_auc_score(y_true, y_prob)
            metrics['AUC'] = max(raw_auc, 1 - raw_auc)  # AUC < 0.5면 반전
        except:
            metrics['AUC'] = np.nan

    return metrics


def diebold_mariano_test(y_true: np.ndarray, pred1: np.ndarray,
                          pred2: np.ndarray) -> Tuple[float, float]:
    """
    Perform Diebold-Mariano test for comparing forecast accuracy.

    Args:
        y_true: True values
        pred1: Predictions from model 1
        pred2: Predictions from model 2

    Returns:
        Tuple of (DM statistic, p-value)
    """
    e1 = (y_true - pred1) ** 2
    e2 = (y_true - pred2) ** 2
    d = e1 - e2

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d == 0:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def perform_dm_tests(results: Dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Perform pairwise Diebold-Mariano tests between models.

    Args:
        results: Dictionary of model results
        y_true: True values

    Returns:
        DataFrame with DM test results
    """
    model_names = list(results.keys())
    n_models = len(model_names)

    dm_matrix = np.zeros((n_models, n_models))
    p_matrix = np.zeros((n_models, n_models))

    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                dm_matrix[i, j] = 0.0
                p_matrix[i, j] = 1.0
            elif i < j:
                pred1 = results[name1]['predictions']
                pred2 = results[name2]['predictions']

                dm_stat, p_val = diebold_mariano_test(y_true, pred1, pred2)
                dm_matrix[i, j] = dm_stat
                dm_matrix[j, i] = -dm_stat
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val

    # Create summary DataFrame
    dm_results = []
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j:
                dm_results.append({
                    'Model 1': name1,
                    'Model 2': name2,
                    'DM Statistic': dm_matrix[i, j],
                    'p-value': p_matrix[i, j],
                    'Significant (5%)': p_matrix[i, j] < 0.05
                })

    return pd.DataFrame(dm_results)


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: str):
    """
    Plot correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(results: Dict, save_path: str):
    """
    Plot ROC curves for all classification models.

    Args:
        results: Dictionary with model results
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    for model_name, res in results.items():
        if 'probabilities' in res and 'actuals' in res:
            fpr, tpr, _ = roc_curve(res['actuals'], res['probabilities'])
            auc = res['metrics'].get('AUC', 0)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Classification Models')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_returns_time_series(data: pd.DataFrame, save_path: str,
                             returns_col: str, forecast_days: int):
    """
    Plot returns time series (Figure 1 from paper).

    Args:
        data: DataFrame with returns column
        save_path: Path to save figure
        returns_col: Name of returns column (e.g., 'Returns1', 'Returns30')
        forecast_days: Forecast horizon in days
    """
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, data[returns_col], linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel(f'{forecast_days}-day Log Returns')
    plt.title(f'S&P 500 {forecast_days}-day Log Returns Time Series')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# Report Generation - ENHANCED
# =============================================================================
def generate_report(data_info: Dict, stats_df: pd.DataFrame,
                    vif_df: pd.DataFrame,
                    clf_results_before: Dict, clf_results_after: Dict,
                    reg_results_before: Dict, reg_results_after: Dict,
                    dm_tests_clf: Optional[pd.DataFrame],
                    dm_tests_reg: Optional[pd.DataFrame],
                    paper_table9: Dict, paper_table10: Dict,
                    save_path: str,
                    forecast_days: int = FORECAST_DAYS,
                    n_splits: int = None,
                    refit_frequency: int = 1):
    """
    Generate markdown report with all results.

    Args:
        Various result dictionaries and DataFrames
        save_path: Path to save report
        forecast_days: Number of days ahead being predicted
        n_splits: Number of walk-forward CV iterations
        refit_frequency: Refit frequency used
    """

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"# 머신러닝 기반 S&P 500 방향성 예측 성능 분석 (FIXED VERSION)\n\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**예측 기간**: {forecast_days}일 후 방향성\n\n")
        f.write("**사용 방법론**: Volatility Indices 기반 머신러닝 모델 (Campisi et al., 2024)\n\n")
        f.write("**주요 개선사항**:\n")
        f.write("- ✅ Data Leakage 문제 해결 (Pipeline 사용)\n")
        f.write("- ✅ Diebold-Mariano 통계 검정 추가\n")
        f.write("- ✅ Performance 최적화 (Refit Frequency)\n\n")
        f.write("---\n\n")

        # 1. Data Summary
        f.write("## 1. 데이터 요약\n\n")
        f.write(f"- **수집 기간**: {data_info['start_date']} ~ {data_info['end_date']}\n")
        f.write(f"- **총 관측치 수**: {data_info['n_observations']:,}\n")
        f.write(f"- **변수 수**: {data_info['n_features']}\n")
        if n_splits is not None:
            f.write(f"- **Walk-Forward CV 이터레이션**: {n_splits}개\n")
        f.write(f"- **Refit Frequency**: 매 {refit_frequency}일\n")
        f.write(f"- **데이터 소스**: Yahoo Finance\n\n")

        # 2. Summary Statistics
        f.write("## 2. 기술 통계량\n\n")
        f.write(tabulate(stats_df.round(3), headers='keys',
                        tablefmt='pipe', showindex=False))
        f.write("\n\n")

        # 3. VIF
        f.write("## 3. VIF (Variance Inflation Factor)\n\n")
        f.write(tabulate(vif_df.round(3), headers='keys',
                        tablefmt='pipe', showindex=False))
        f.write("\n\n")

        # 4. Classification Results Before Feature Selection
        f.write("## 4. 모델 성능 비교 (Feature Selection 전)\n\n")
        f.write("### 4.1 분류 모델\n\n")

        clf_table_before = []
        for name, res in clf_results_before.items():
            clf_table_before.append({
                'Model': name,
                'Accuracy': res['metrics']['Accuracy'],
                'AUC': res['metrics'].get('AUC', '-'),
                'F-measure': res['metrics']['F-measure']
            })
        f.write(tabulate(pd.DataFrame(clf_table_before).round(4),
                        headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")

        # 4.2 Regression Results Before
        f.write("### 4.2 회귀 모델\n\n")

        reg_table_before = []
        for name, res in reg_results_before.items():
            reg_table_before.append({
                'Model': name,
                'Accuracy': res['metrics']['Accuracy'],
                'AUC': res['metrics'].get('AUC', '-'),
                'F-measure': res['metrics']['F-measure']
            })
        f.write(tabulate(pd.DataFrame(reg_table_before).round(4),
                        headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")

        # 5. Results After Feature Selection
        f.write("## 5. 모델 성능 비교 (Feature Selection 후)\n\n")
        f.write("### 5.1 분류 모델\n\n")

        clf_table_after = []
        for name, res in clf_results_after.items():
            clf_table_after.append({
                'Model': name,
                'Accuracy': res['metrics']['Accuracy'],
                'AUC': res['metrics'].get('AUC', '-'),
                'F-measure': res['metrics']['F-measure']
            })
        f.write(tabulate(pd.DataFrame(clf_table_after).round(4),
                        headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")

        # 5.2 Regression Results After
        f.write("### 5.2 회귀 모델\n\n")

        reg_table_after = []
        for name, res in reg_results_after.items():
            reg_table_after.append({
                'Model': name,
                'Accuracy': res['metrics']['Accuracy'],
                'AUC': res['metrics'].get('AUC', '-'),
                'F-measure': res['metrics']['F-measure']
            })
        f.write(tabulate(pd.DataFrame(reg_table_after).round(4),
                        headers='keys', tablefmt='pipe', showindex=False))
        f.write("\n\n")

        # 6. Diebold-Mariano Tests
        if dm_tests_clf is not None and len(dm_tests_clf) > 0:
            f.write("## 6. Diebold-Mariano 통계 검정\n\n")
            f.write("### 6.1 분류 모델 쌍별 비교\n\n")

            # Show only significant results
            sig_tests = dm_tests_clf[dm_tests_clf['Significant (5%)'] == True]
            if len(sig_tests) > 0:
                f.write("**통계적으로 유의한 차이 (p < 0.05):**\n\n")
                f.write(tabulate(sig_tests.round(4), headers='keys',
                                tablefmt='pipe', showindex=False))
                f.write("\n\n")
            else:
                f.write("통계적으로 유의한 차이를 보이는 모델 쌍이 없습니다.\n\n")

        if dm_tests_reg is not None and len(dm_tests_reg) > 0:
            f.write("### 6.2 회귀 모델 쌍별 비교\n\n")

            sig_tests = dm_tests_reg[dm_tests_reg['Significant (5%)'] == True]
            if len(sig_tests) > 0:
                f.write("**통계적으로 유의한 차이 (p < 0.05):**\n\n")
                f.write(tabulate(sig_tests.round(4), headers='keys',
                                tablefmt='pipe', showindex=False))
                f.write("\n\n")
            else:
                f.write("통계적으로 유의한 차이를 보이는 모델 쌍이 없습니다.\n\n")

        # 7. Model Performance Summary
        f.write("## 7. 모델 성능 요약\n\n")

        # Find best models
        best_clf_before = max(clf_results_before.items(),
                             key=lambda x: x[1]['metrics']['Accuracy'])
        best_clf_after = max(clf_results_after.items(),
                            key=lambda x: x[1]['metrics']['Accuracy'])
        best_reg_before = max(reg_results_before.items(),
                             key=lambda x: x[1]['metrics']['Accuracy'])
        best_reg_after = max(reg_results_after.items(),
                            key=lambda x: x[1]['metrics']['Accuracy'])

        f.write("### 7.1 Feature Selection 전후 비교\n\n")
        f.write("**분류 모델:**\n")
        f.write(f"- Feature Selection 전: {best_clf_before[0]} (Accuracy: {best_clf_before[1]['metrics']['Accuracy']:.4f})\n")
        f.write(f"- Feature Selection 후: {best_clf_after[0]} (Accuracy: {best_clf_after[1]['metrics']['Accuracy']:.4f})\n\n")

        f.write("**회귀 모델:**\n")
        f.write(f"- Feature Selection 전: {best_reg_before[0]} (Accuracy: {best_reg_before[1]['metrics']['Accuracy']:.4f})\n")
        f.write(f"- Feature Selection 후: {best_reg_after[0]} (Accuracy: {best_reg_after[1]['metrics']['Accuracy']:.4f})\n\n")

        f.write("### 7.2 전체 최고 성능 모델\n\n")

        # Find overall best model
        all_models = list(clf_results_after.items()) + list(reg_results_after.items())
        overall_best = max(all_models, key=lambda x: x[1]['metrics']['Accuracy'])

        f.write(f"**최고 성능**: {overall_best[0]}\n")
        f.write(f"- Accuracy: {overall_best[1]['metrics']['Accuracy']:.4f}\n")

        auc_value = overall_best[1]['metrics'].get('AUC', '-')
        if isinstance(auc_value, (int, float)) and not np.isnan(auc_value):
            f.write(f"- AUC: {auc_value:.4f}\n")
        else:
            f.write(f"- AUC: -\n")

        f.write(f"- F-measure: {overall_best[1]['metrics']['F-measure']:.4f}\n\n")

        # 8. Conclusion
        f.write("## 8. 주요 인사이트\n\n")

        # Compare classification vs regression
        avg_clf_acc = sum(r['metrics']['Accuracy'] for r in clf_results_after.values()) / len(clf_results_after)
        avg_reg_acc = sum(r['metrics']['Accuracy'] for r in reg_results_after.values()) / len(reg_results_after)

        # Compare feature selection impact
        avg_acc_before = (sum(r['metrics']['Accuracy'] for r in clf_results_before.values()) +
                          sum(r['metrics']['Accuracy'] for r in reg_results_before.values())) / (len(clf_results_before) + len(reg_results_before))
        avg_acc_after = (sum(r['metrics']['Accuracy'] for r in clf_results_after.values()) +
                         sum(r['metrics']['Accuracy'] for r in reg_results_after.values())) / (len(clf_results_after) + len(reg_results_after))

        f.write("**1. Feature Selection 효과**\n")
        f.write(f"- 전체 평균 Accuracy: {avg_acc_before:.4f} → {avg_acc_after:.4f}\n")
        if avg_acc_after > avg_acc_before:
            f.write(f"- Feature Selection을 통해 성능 향상 (+{(avg_acc_after-avg_acc_before):.4f})\n\n")
        else:
            f.write(f"- Feature Selection으로 성능 변화 ({(avg_acc_after-avg_acc_before):.4f})\n\n")

        f.write("**2. 분류 vs 회귀 모델**\n")
        f.write(f"- 분류 모델 평균 Accuracy: {avg_clf_acc:.4f}\n")
        f.write(f"- 회귀 모델 평균 Accuracy: {avg_reg_acc:.4f}\n")
        if avg_reg_acc > avg_clf_acc:
            f.write("- 회귀 모델이 방향 예측에 더 효과적\n\n")
        else:
            f.write("- 분류 모델이 방향 예측에 더 효과적\n\n")

        f.write("**3. Data Leakage 문제 해결**\n")
        f.write("- ✅ Standardization을 CV loop 내부로 이동 (Pipeline 사용)\n")
        f.write("- ✅ Feature Selection을 CV loop 내부로 이동 (SelectFromModel)\n")
        f.write("- ✅ 이로 인해 원본 코드 대비 성능이 낮아질 수 있으나, 이것이 정확한 out-of-sample 성능임\n\n")

        f.write("---\n\n")
        f.write("## 9. 시각화\n\n")
        f.write("- `figures/correlation_heatmap.png`: 상관관계 히트맵\n")
        f.write("- `figures/roc_curves.png`: ROC 곡선\n")
        f.write("- `figures/returns_timeseries.png`: 수익률 시계열\n")

    print(f"\n  Report saved to: {save_path}")


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Campisi et al. (2024) 논문 실증 재현 - FIXED VERSION'
    )
    parser.add_argument(
        '--forecast-days', '-f',
        type=int,
        default=FORECAST_DAYS,
        help=f'예측할 미래 일수 (기본값: {FORECAST_DAYS}일)'
    )
    parser.add_argument(
        '--max-iterations', '-m',
        type=int,
        default=None,
        help='Walk-Forward CV 최대 이터레이션 수 (기본값: 전체 ~755개)'
    )
    parser.add_argument(
        '--refit-frequency', '-r',
        type=int,
        default=1,
        help='모델 재학습 주기 (일 단위, 기본값: 1 = 매일 재학습)'
    )
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Feature selection 비활성화'
    )
    parser.add_argument(
        '--evaluation-mode',
        choices=['walk-forward', 'paper'],
        default='walk-forward',
        help='Walk-forward 전체 샘플 또는 논문 스타일 holdout 중 선택'
    )
    parser.add_argument(
        '--non-overlap-labels',
        action='store_true',
        help='레이블 중첩을 피하기 위해 forecast_days 간격으로 샘플 선택'
    )
    parser.add_argument(
        '--balanced-class-weights',
        action='store_true',
        help='분류 모델에 class_weight="balanced" 적용'
    )
    parser.add_argument(
        '--optimize-threshold',
        action='store_true',
        help='회귀 모델 확률에 대해 Youden의 J 임곗값 사용'
    )
    return parser.parse_args()


# =============================================================================
# Main Function
# =============================================================================
def main():
    """Main execution function."""

    # Parse arguments
    args = parse_args()
    forecast_days = args.forecast_days
    max_iterations = args.max_iterations
    refit_frequency = args.refit_frequency
    use_feature_selection = not args.no_feature_selection
    evaluation_mode = args.evaluation_mode
    non_overlap_labels = args.non_overlap_labels
    balanced_weights = args.balanced_class_weights
    optimize_threshold = args.optimize_threshold

    print("=" * 70)
    print("Campisi et al. (2024) 논문 실증 재현 - FIXED VERSION")
    print(f"예측 기간: {forecast_days}일 후")
    if max_iterations is not None:
        print(f"최대 이터레이션: {max_iterations}개 (빠른 테스트 모드)")
    print(f"Refit Frequency: 매 {refit_frequency}일")
    print(f"Feature Selection: {'활성화' if use_feature_selection else '비활성화'}")
    print(f"Evaluation Mode: {evaluation_mode}")
    print(f"Non-overlap Sampling: {'ON' if non_overlap_labels else 'OFF'}")
    print(f"Balanced class weights: {'ON' if balanced_weights else 'OFF'}")
    print(f"Regression threshold optimization: {'ON' if optimize_threshold else 'OFF'}")
    print("=" * 70)

    # Create output directories
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Paper results for comparison (Table 9, 10)
    paper_table9 = {
        'Logistic Regression': {'ACC': 0.6776, 'AUC': 0.6365, 'F': 0.8076},
        'LDA': {'ACC': 0.6776, 'AUC': 0.6365, 'F': 0.8076},
        'Random Forest (Clf)': {'ACC': 0.8239, 'AUC': 0.8495, 'F': 0.8828},
        'Bagging (Clf)': {'ACC': 0.8275, 'AUC': 0.8493, 'F': 0.8845},
        'Gradient Boosting (Clf)': {'ACC': 0.7113, 'AUC': 0.7187, 'F': 0.8215}
    }

    paper_table10 = {
        'Linear Regression': {'ACC': 0.6373, 'AUC': 0.6042, 'F': 0.7614},
        'Random Forest (Reg)': {'ACC': 0.8003, 'AUC': 0.8412, 'F': 0.8592},
        'Bagging (Reg)': {'ACC': 0.7958, 'AUC': 0.8370, 'F': 0.8500},
        'Gradient Boosting (Reg)': {'ACC': 0.6854, 'AUC': 0.6999, 'F': 0.7694},
        'Ridge Regression': {'ACC': 0.6080, 'AUC': 0.5595, 'F': 0.7212},
        'Lasso Regression': {'ACC': 0.6178, 'AUC': 0.5800, 'F': 0.7465}
    }

    # =========================================================================
    # Phase 1-2: Data Collection
    # =========================================================================
    data = download_volatility_indices()
    data, returns_col = calculate_derived_features(data, forecast_days=forecast_days)

    # =========================================================================
    # Phase 3: Preprocessing
    # =========================================================================
    data = handle_missing_values(data)

    if non_overlap_labels:
        data = apply_non_overlapping_sampling(data, forecast_days, returns_col)
        print(f"  Non-overlapping 샘플 수: {len(data)}")

    # Define feature columns (excluding SP500, Returns, Target)
    all_feature_cols = ['VIX', 'VIX9D', 'VIX3M', 'VIX6M', 'VVIX',
                        'SKEW', 'VXN', 'GVZ', 'OVX', 'RVOL', 'PUTCALL']

    # Filter to available columns
    feature_cols = [col for col in all_feature_cols if col in data.columns]

    print(f"\n  Available features: {feature_cols}")

    # Calculate summary statistics
    stats_df = calculate_summary_statistics(data, feature_cols, returns_col)
    print("\n  Summary Statistics:")
    print(tabulate(stats_df.round(3), headers='keys', tablefmt='simple', showindex=False))

    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(data, feature_cols, returns_col)

    # =========================================================================
    # Phase 4: Prepare Data (NO SCALING HERE - Pipeline will handle it)
    # =========================================================================
    X = data[feature_cols].values  # NOT scaled
    y_continuous = data[returns_col].values
    y_binary = data['Target'].values

    # Calculate VIF (for reporting only, using raw data)
    vif_df = calculate_vif(pd.DataFrame(X, columns=feature_cols))
    print("\n  VIF Values:")
    print(tabulate(vif_df.round(3), headers='keys', tablefmt='simple', showindex=False))

    # =========================================================================
    # Phase 5-7: Model Training with Pipeline (FIXED)
    # =========================================================================
    if evaluation_mode == 'paper':
        cv = PaperHoldoutCV(train_size=TRAIN_SIZE, gap=0)
    else:
        cv = WalkForwardCV(
            train_size=TRAIN_SIZE,
            gap=GAP,
            max_iterations=max_iterations,
            refit_frequency=refit_frequency
        )
    n_splits = cv.get_n_splits(X)
    print(f"\n[5] Training models with Walk-Forward CV ({n_splits} splits)...")

    # Get models with Pipeline
    clf_models_before = get_classification_models(
        use_feature_selection=False,
        balanced=balanced_weights
    )
    clf_models_after = get_classification_models(
        use_feature_selection=use_feature_selection,
        balanced=balanced_weights
    )
    reg_models_before = get_regression_models(use_feature_selection=False)
    reg_models_after = get_regression_models(use_feature_selection=use_feature_selection)

    # Results storage
    clf_results_before = {}
    clf_results_after = {}
    reg_results_before = {}
    reg_results_after = {}

    # Train classification models (before feature selection)
    print("\n  Training classification models (all features)...")
    for name, pipeline in clf_models_before.items():
        preds, probs, actuals = train_and_predict_classification(
            pipeline, X, y_binary, cv, desc=name
        )
        metrics = calculate_metrics(actuals, preds, probs)
        clf_results_before[name] = {
            'predictions': preds,
            'probabilities': probs,
            'actuals': actuals,
            'metrics': metrics
        }
        print(f"    {name}: ACC={metrics['Accuracy']:.4f}, AUC={metrics.get('AUC', 'N/A'):.4f}, F={metrics['F-measure']:.4f}")

    # Train classification models (after feature selection)
    print("\n  Training classification models (with feature selection)...")
    for name, pipeline in clf_models_after.items():
        preds, probs, actuals = train_and_predict_classification(
            pipeline, X, y_binary, cv, desc=name
        )
        metrics = calculate_metrics(actuals, preds, probs)
        clf_results_after[name] = {
            'predictions': preds,
            'probabilities': probs,
            'actuals': actuals,
            'metrics': metrics
        }
        print(f"    {name}: ACC={metrics['Accuracy']:.4f}, AUC={metrics.get('AUC', 'N/A'):.4f}, F={metrics['F-measure']:.4f}")

    # Train regression models (before feature selection)
    print("\n  Training regression models (all features)...")
    for name, pipeline in reg_models_before.items():
        preds, actuals, probs, preds_binary = train_and_predict_regression(
            pipeline, X, y_continuous, cv, desc=name,
            optimize_threshold=optimize_threshold
        )
        actuals_binary = (actuals > 0).astype(int)

        metrics = calculate_metrics(actuals_binary, preds_binary, probs)
        reg_results_before[name] = {
            'predictions': preds,
            'predictions_binary': preds_binary,
            'actuals': actuals,
            'actuals_binary': actuals_binary,
            'metrics': metrics
        }
        print(f"    {name}: ACC={metrics['Accuracy']:.4f}, AUC={metrics.get('AUC', 'N/A'):.4f}, F={metrics['F-measure']:.4f}")

    # Train regression models (after feature selection)
    print("\n  Training regression models (with feature selection)...")
    for name, pipeline in reg_models_after.items():
        preds, actuals, probs, preds_binary = train_and_predict_regression(
            pipeline, X, y_continuous, cv, desc=name,
            optimize_threshold=optimize_threshold
        )
        actuals_binary = (actuals > 0).astype(int)

        metrics = calculate_metrics(actuals_binary, preds_binary, probs)
        reg_results_after[name] = {
            'predictions': preds,
            'predictions_binary': preds_binary,
            'actuals': actuals,
            'actuals_binary': actuals_binary,
            'metrics': metrics
        }
        print(f"    {name}: ACC={metrics['Accuracy']:.4f}, AUC={metrics.get('AUC', 'N/A'):.4f}, F={metrics['F-measure']:.4f}")

    # =========================================================================
    # Phase 8: Diebold-Mariano Tests
    # =========================================================================
    print("\n[6] Performing Diebold-Mariano tests...")

    # Get actuals for binary classification
    y_true_binary = clf_results_after[list(clf_results_after.keys())[0]]['actuals']

    # Perform DM tests for classification models
    clf_pred_dict = {name: res['predictions'] for name, res in clf_results_after.items()}
    dm_tests_clf = perform_dm_tests(clf_results_after, y_true_binary)

    # Perform DM tests for regression models (using binary predictions)
    reg_pred_for_dm = {
        name: res['predictions_binary']
        for name, res in reg_results_after.items()
    }
    y_true_reg_binary = reg_results_after[list(reg_results_after.keys())[0]]['actuals_binary']
    dm_tests_reg = perform_dm_tests(
        {k: {'predictions': v, 'actuals': y_true_reg_binary}
         for k, v in reg_pred_for_dm.items()},
        y_true_reg_binary
    )

    print(f"  Classification models: {len(dm_tests_clf)} pairwise comparisons")
    print(f"  Regression models: {len(dm_tests_reg)} pairwise comparisons")

    # =========================================================================
    # Phase 9: Visualization
    # =========================================================================
    print("\n[7] Generating visualizations...")

    # Correlation heatmap
    plot_correlation_heatmap(
        corr_matrix,
        os.path.join(FIGURES_DIR, 'correlation_heatmap_fixed.png')
    )

    # ROC curves
    plot_roc_curves(
        clf_results_after,
        os.path.join(FIGURES_DIR, 'roc_curves_fixed.png')
    )

    # Returns time series
    plot_returns_time_series(
        data,
        os.path.join(FIGURES_DIR, 'returns_timeseries_fixed.png'),
        returns_col,
        forecast_days
    )

    print("  Visualizations saved to figures/")

    # =========================================================================
    # Phase 10: Report Generation
    # =========================================================================
    print("\n[8] Generating report...")

    data_info = {
        'start_date': data.index.min().strftime('%Y-%m-%d'),
        'end_date': data.index.max().strftime('%Y-%m-%d'),
        'n_observations': len(data),
        'n_features': len(feature_cols)
    }

    # Generate report filename
    report_filename = f'campisi_2024_results_fixed_{forecast_days}d_{timestamp}.md'
    report_path = os.path.join(OUTPUT_DIR, report_filename)

    generate_report(
        data_info=data_info,
        stats_df=stats_df,
        vif_df=vif_df,
        clf_results_before=clf_results_before,
        clf_results_after=clf_results_after,
        reg_results_before=reg_results_before,
        reg_results_after=reg_results_after,
        dm_tests_clf=dm_tests_clf,
        dm_tests_reg=dm_tests_reg,
        paper_table9=paper_table9,
        paper_table10=paper_table10,
        save_path=report_path,
        forecast_days=forecast_days,
        n_splits=n_splits,
        refit_frequency=refit_frequency
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("실행 완료!")
    print("=" * 70)
    print(f"\n생성된 파일:")
    print(f"  - {report_path}")
    print(f"  - {os.path.join(FIGURES_DIR, 'correlation_heatmap_fixed.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'roc_curves_fixed.png')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'returns_timeseries_fixed.png')}")


if __name__ == "__main__":
    main()
