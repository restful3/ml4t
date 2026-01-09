import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor
import warnings
import os
import datetime

warnings.filterwarnings('ignore')

class ReportGenerator:
    def __init__(self):
        self.content = []
        self.add_header("Rank IC Optimization Model Report")
        self.add_text(f"**생성일시:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_section("1. 코드 개요", "이 스크립트는 **Rank IC (순위 상관계수)** 최적화를 목표로 개선된 모델입니다. \n주요 특징으로는 **순위 변수(Rank Features)** 도입, **변동성(Volatility)** 변수 추가, 그리고 **Rank IC 기반 검증**이 있습니다.")

    def add_text(self, text):
        self.content.append(text)

    def add_header(self, text, level=1):
        self.content.append(f"{'#' * level} {text}")

    def add_section(self, title, text):
        self.add_header(title, level=2)
        self.content.append(text)

    def add_list(self, items):
        for item in items:
            self.content.append(f"- {item}")

    def add_code_block(self, code, language=""):
        self.content.append(f"```{language}\n{code}\n```")

    def save(self, filename="baseline_rank_ic.md"):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n\n".join(self.content))
        print(f"Report saved to {filename}")

report = ReportGenerator()

print("Loading data...")
with open('data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

report.add_section("2. 데이터 준비", "")
report.add_header("2.1 데이터 로드", level=3)
report.add_list([
    f"Train data shape: {train_data.shape}",
    f"Test data shape: {test_data.shape}"
])

# Data preprocessing
print("\nPreprocessing data...")

# Convert f_3 to numeric first (to handle dirty data), then to category
train_data['f_3'] = pd.to_numeric(train_data['f_3'], errors='coerce')
test_data['f_3'] = pd.to_numeric(test_data['f_3'], errors='coerce')

# Fill NaN values for categorical
train_data['f_3'].fillna(-1, inplace=True)
test_data['f_3'].fillna(-1, inplace=True)

# Convert to category type for LGBM
train_data['f_3'] = train_data['f_3'].astype('category')
test_data['f_3'] = test_data['f_3'].astype('category')

# Outlier clipping for f_5
print("Clipping outliers in f_5...")
lower_bound = train_data['f_5'].quantile(0.01)
upper_bound = train_data['f_5'].quantile(0.99)
train_data['f_5'] = train_data['f_5'].clip(lower_bound, upper_bound)
test_data['f_5'] = test_data['f_5'].clip(lower_bound, upper_bound)

report.add_header("2.2 데이터 전처리", level=3)
report.add_list([
    "f_3: 결측치 -1 처리 및 Category 타입 변환 (LGBM 전용)",
    f"f_5: 이상치 Clipping 적용 (1% ~ 99% 구간, {lower_bound:.4f} ~ {upper_bound:.4f})"
])

# Feature engineering
def add_features(df):
    """Add engineered features optimizing for Rank IC"""
    # Sort by code and date for lag features
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 1. Price ratios & Volume
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_normalized'] = np.log1p(df['f_4'])
    
    # 2. Date cyclic feature
    if 'date' in df.columns:
        df['date_mod_5'] = df['date'] % 5

    # 3. Moving averages (Trends)
    for window in [5, 10, 20]:
        df[f'f_0_ma_{window}'] = df.groupby('code')['f_0'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Volatility Feature: Standard deviation of returns (price_ratio)
        df[f'volatility_{window}'] = df.groupby('code')['price_ratio'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # 4. Stock-level statistics
    df['stock_mean_f_0'] = df.groupby('code')['f_0'].transform('mean')
    df['stock_std_f_0'] = df.groupby('code')['f_0'].transform('std')

    # 5. Cross-Sectional Rank Features (Crucial for Rank IC)
    # Rank features across all stocks on the same date
    print("  Adding cross-sectional rank features...")
    rank_cols = ['f_0', 'f_1', 'f_2', 'f_4', 'f_5', 'price_ratio', 'volume_normalized']
    
    # Note: This operation can be slow on large data, but essential for Rank IC
    # Using transform to keep the index aligned
    for col in rank_cols:
        df[f'rank_{col}'] = df.groupby('date')[col].transform(lambda x: x.rank(pct=True))

    return df

print("Adding features to train data...")
train_data = add_features(train_data)

print("Adding features to test data...")
test_data = add_features(test_data)

# Fill NaN values created by lag/rolling features
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
if 'y' in numeric_cols:
    numeric_cols.remove('y') # Exclude target from features to fill

train_data[numeric_cols] = train_data[numeric_cols].fillna(0)
test_data[numeric_cols] = test_data[numeric_cols].fillna(0)

# Define features
# Base features + New Engineered features
base_features = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6']
new_features = ['price_ratio', 'volume_normalized', 'date_mod_5',
                'f_0_ma_5', 'f_0_ma_10', 'f_0_ma_20',
                'volatility_5', 'volatility_10', 'volatility_20',
                'stock_mean_f_0', 'stock_std_f_0',
                # 'sector_mean_f_0' - dropped as grouping by f_3 might be redundant if f_3 is categorical
                ]
# Add rank columns dynamically
rank_features = [c for c in train_data.columns if c.startswith('rank_')]
feature_cols = base_features + new_features + rank_features

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

report.add_header("2.3 Feature Engineering (변수 생성)", level=3)
report.add_text(f"총 {len(feature_cols)}개의 Feature를 사용하였습니다. 주요 파생 변수는 다음과 같습니다.")
report.add_list([
    "**Cross-Sectional Rank Features (핵심):** 날짜별 각 종목의 상대적 순위 (rank_f_0, rank_f_4 등)",
    "**Volatility Features:** 주가 변동성 (volatility_5, volatility_10 등)",
    "**Trends:** 이동평균 (f_0_ma_5, f_0_ma_20 등)",
    "**Stock Stats:** 종목별 평균 및 표준편차",
    "**Date Cyclic:** 날짜 주기성 (date_mod_5)"
])

# Prepare data
X = train_data[feature_cols]
y = train_data['y']

# Time-series cross-validation
print("\nTraining model with time-series cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)

report.add_section("3. 모델 학습", "")
report.add_header("3.1 모델 설정", level=3)
report.add_text("Rank IC 향상을 위해 모델 복잡도를 높였습니다.")
report.add_code_block("""
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
""", language="python")

report.add_header("3.2 교차 검증 (Cross-Validation) 결과", level=3)
report.add_text("TimeSeriesSplit (n_splits=5)을 사용하여 검증하였습니다.")
table_header = "| Fold | RMSE | Rank IC (Spearman) |\n|---|---|---|"
report.add_text(table_header)

cv_rmse_scores = []
cv_ic_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"\nFold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train model
    # Note: Rank IC is roughly maintained by RMSE minimization, but we could check correlation
    model = LGBMRegressor(
        n_estimators=1000, # Increased estimators
        learning_rate=0.05,
        max_depth=10,      # Slightly deeper
        num_leaves=127,    # More leaves
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            # We can perform early stopping manually if needed, but keeping simple for now
        ]
    )

    # Predict
    y_pred = model.predict(X_val)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # Calculate Rank IC (Spearman Correlation)
    # Using dataframe for correlation to handle potential index issues safely
    score_df = pd.DataFrame({'pred': y_pred, 'actual': y_val})
    ic, _ = spearmanr(score_df['pred'], score_df['actual'])

    print(f"  RMSE: {rmse:.4f}")
    print(f"  Rank IC: {ic:.4f}")

    cv_rmse_scores.append(rmse)
    cv_ic_scores.append(ic)
    models.append(model)
    report.add_text(f"| {fold} | {rmse:.4f} | **{ic:.4f}** |")

print(f"\nMean CV RMSE: {np.mean(cv_rmse_scores):.4f} (+/- {np.std(cv_rmse_scores):.4f})")
print(f"Mean CV Rank IC: {np.mean(cv_ic_scores):.4f} (+/- {np.std(cv_ic_scores):.4f})")

report.add_text(f"\n**Mean CV Rank IC: {np.mean(cv_ic_scores):.4f}** (+/- {np.std(cv_ic_scores):.4f})")

# Train final model on all data
print("\nTraining final model on all data...")
final_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

final_model.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15))

report.add_header("3.3 변수 중요도 (Top 15)", level=3)
report.add_text("Rank 변수와 변동성 변수가 상위권을 차지하는지 확인합니다.")
top15_features = feature_importance.head(15)
fi_table = "| Rank | Feature | Importance |\n|---|---|---|\n"
for i, (idx, row) in enumerate(top15_features.iterrows(), 1):
    fi_table += f"| {i} | {row['feature']} | {row['importance']:.0f} |\n"
report.add_text(fi_table)

# Make predictions on test data
print("\nMaking predictions on test data...")
X_test = test_data[feature_cols]
predictions = final_model.predict(X_test)

# Create submission file
print("\nCreating submission file...")
# Ensure directories exist
os.makedirs('submissions', exist_ok=True)

sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission['y_pred'] = predictions

submission_file = 'submissions/submission_rank_ic.csv'
sample_submission.to_csv(submission_file, index=False)
print(f"Submission file saved as '{submission_file}'")

# Prediction statistics
print("\nPrediction Statistics:")
print(f"Mean: {predictions.mean():.4f}")
print(f"Std: {predictions.std():.4f}")
print(f"Min: {predictions.min():.4f}")
print(f"Max: {predictions.max():.4f}")

print("\nRank IC Optimized model complete!")

report.add_section("4. 최종 제출", "")
report.add_header("4.1 예측 결과 통계", level=3)
report.add_list([
    f"Mean: {predictions.mean():.4f}",
    f"Std: {predictions.std():.4f}",
    f"Min: {predictions.min():.4f}",
    f"Max: {predictions.max():.4f}"
])
report.add_header("4.2 저장 파일", level=3)
report.add_text(f"제출 파일이 '{submission_file}'에 저장되었습니다.")

report.save("baseline_rank_ic.md")
