import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
with open('data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# Data preprocessing
print("\nPreprocessing data...")

# Convert f_3 to numeric in train (it's object type)
train_data['f_3'] = pd.to_numeric(train_data['f_3'], errors='coerce')

# Fill any NaN values that might have been created
train_data['f_3'].fillna(train_data['f_3'].median(), inplace=True)

# Feature engineering
def add_features(df):
    """Add engineered features"""
    # Sort by code and date for lag features
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # Price ratios
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_normalized'] = np.log1p(df['f_4'])

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
engineered_features = ['price_ratio', 'volume_normalized',
                       'f_0_ma_5', 'f_5_ma_5', 'f_0_ma_10', 'f_5_ma_10',
                       'stock_mean_f_0', 'stock_std_f_0', 'sector_mean_f_0']
feature_cols = base_features + engineered_features

print(f"\nNumber of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# Prepare data
X = train_data[feature_cols]
y = train_data['y']

# Time-series cross-validation
print("\nTraining model with time-series cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)

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

    cv_scores.append(rmse)
    models.append(model)

print(f"\nMean CV RMSE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

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
