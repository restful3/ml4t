import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import warnings
import datetime

warnings.filterwarnings('ignore')

class ReportGenerator:
    def __init__(self):
        self.content = []
        self.add_header("Baseline Model Report")
        self.add_text(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.add_section("1. 코드 설명", "이 코드는 주가 수익률 예측을 위한 Baseline 모델 학습 및 추론 스크립트입니다. \nLightGBM Regressor를 사용하며, 시계열 교차 검증(Time-Series Cross Validation)을 수행합니다.")

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

    def save(self, filename="baseline.md"):
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

report.add_section("2. 작업 내용", "")
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

# Fill NaN values (using mode or -1 is better for categorical than median usually, but let's stick to median for consistency or use -1)
# Using -1 for missing categorical
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
    "f_3 변수: Numeric 변환 및 결측치 -1로 대체 후 Category 타입으로 변환",
    f"f_5 변수: 이상치 Clipping (1% ~ 99% quantile 적용, range: {lower_bound:.4f} ~ {upper_bound:.4f})"
])

# Feature engineering
def add_features(df):
    """Add engineered features"""
    # Sort by code and date for lag features
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # Price ratios
    # Price ratios
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_normalized'] = np.log1p(df['f_4'])
    
    # Date cyclic feature (assuming daily data)
    if 'date' in df.columns:
        df['date_mod_5'] = df['date'] % 5

    # Moving averages (using f_0 as proxy for price)
    for window in [5, 10]:
        df[f'f_0_ma_{window}'] = df.groupby('code')['f_0'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'f_5_ma_{window}'] = df.groupby('code')['f_5'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # Lag features (for train data with y)
    if 'y' in df.columns:
        for lag in [1, 3, 5]:
            df[f'y_lag_{lag}'] = df.groupby('code')['y'].shift(lag)

        # Rolling statistics
        for window in [5, 10]:
            df[f'y_rolling_mean_{window}'] = df.groupby('code')['y'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'y_rolling_std_{window}'] = df.groupby('code')['y'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )

    # Stock-level statistics
    df['stock_mean_f_0'] = df.groupby('code')['f_0'].transform('mean')
    df['stock_std_f_0'] = df.groupby('code')['f_0'].transform('std')

    # Sector-level statistics (using f_3)
    df['sector_mean_f_0'] = df.groupby(['date', 'f_3'])['f_0'].transform('mean')

    return df

print("Adding features to train data...")
train_data = add_features(train_data)

print("Adding features to test data...")
test_data = add_features(test_data)

# Fill NaN values created by lag features
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Define features (only those available in test data)
# Exclude lag features since they require target variable
base_features = ['f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6']
engineered_features = ['price_ratio', 'volume_normalized', 'date_mod_5',
                       'f_0_ma_5', 'f_5_ma_5', 'f_0_ma_10', 'f_5_ma_10',
                       'stock_mean_f_0', 'stock_std_f_0', 'sector_mean_f_0']
feature_cols = base_features + engineered_features

print(f"Features: {feature_cols}")

report.add_header("2.3 Feature Engineering", level=3)
report.add_text("기본 변수 외에 다음과 같은 파생 변수를 생성하였습니다.")
report.add_list(engineered_features)
report.add_text(f"\n총 Feature 개수: {len(feature_cols)}개")

# Prepare data
X = train_data[feature_cols]
y = train_data['y']

# Time-series cross-validation
print("\nTraining model with time-series cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)

report.add_section("3. 모델 설명", "")
report.add_text("LightGBM Regressor를 사용하였습니다. 주요 파라미터는 다음과 같습니다.")
report.add_code_block("""
n_estimators=500
learning_rate=0.05
max_depth=8
num_leaves=63
colsample_bytree=0.8
subsample=0.8
random_state=42
""", language="python")

report.add_section("4. 학습 결과", "")
report.add_header("4.1 교차 검증 (Cross-Validation) 결과", level=3)
report.add_text("TimeSeriesSplit (n_splits=5)을 사용하여 검증하였습니다.")
table_header = "| Fold | RMSE | MAE |\n|---|---|---|"
report.add_text(table_header)

cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"\nFold {fold}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train model
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    # Predict
    y_pred = model.predict(X_val)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    print(f"  mae: {mae:.4f}")

    cv_scores.append(rmse)
    models.append(model)
    report.add_text(f"| {fold} | {rmse:.4f} | {mae:.4f} |")

print(f"\nMean CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
report.add_text(f"\n**Mean CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})**")

# Train final model on all data
print("\nTraining final model on all data...")
final_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

final_model.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

report.add_header("4.2 변수 중요도 (Top 10)", level=3)
report.add_text("전체 데이터로 학습한 모델 기준 상위 10개 중요 변수입니다.")
top10_features = feature_importance.head(10)
fi_table = "| Rank | Feature | Importance |\n|---|---|---|\n"
for i, (idx, row) in enumerate(top10_features.iterrows(), 1):
    fi_table += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"
report.add_text(fi_table)

# Make predictions on test data
print("\nMaking predictions on test data...")
X_test = test_data[feature_cols]
predictions = final_model.predict(X_test)

# Create submission file
print("\nCreating submission file...")
sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission['y_pred'] = predictions

submission_file = 'submissions/submission_baseline.csv'
sample_submission.to_csv(submission_file, index=False)
print(f"Submission file saved as '{submission_file}'")

# Prediction statistics
print("\nPrediction Statistics:")
print(f"Mean: {predictions.mean():.4f}")
print(f"Std: {predictions.std():.4f}")
print(f"Min: {predictions.min():.4f}")
print(f"Max: {predictions.max():.4f}")

# Compare with train distribution
print("\nTrain Target Distribution:")
print(f"Mean: {y.mean():.4f}")
print(f"Std: {y.std():.4f}")
print(f"Min: {y.min():.4f}")
print(f"Max: {y.max():.4f}")

print("\nBaseline model complete!")

report.add_section("5. Test 결과 및 제출", "")
report.add_header("5.1 Prediction Statistics", level=3)
report.add_list([
    f"Mean: {predictions.mean():.4f}",
    f"Std: {predictions.std():.4f}",
    f"Min: {predictions.min():.4f}",
    f"Max: {predictions.max():.4f}"
])

report.add_header("5.2 제출 파일", level=3)
report.add_text(f"제출 파일이 '{submission_file}'에 저장되었습니다.")

report.save("baseline.md")
