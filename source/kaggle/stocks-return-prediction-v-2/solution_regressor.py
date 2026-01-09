import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
import os

warnings.filterwarnings('ignore')

def add_features(df):
    """Add engineered features for Rank IC Optimization"""
    # Sort by code and date
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 1. Base transformations
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_log'] = np.log1p(df['f_4'])
    
    # 2. Sector Neutralization (f_0 - SectorMean)
    if 'f_3' in df.columns:
        if df['f_3'].dtype == 'object':
             df['f_3'] = pd.to_numeric(df['f_3'], errors='coerce').fillna(-1)
    
    print("  Adding Sector Neutral features...")
    sector_means = df.groupby(['date', 'f_3'])['f_0'].transform('mean')
    df['f_0_sector_neutral'] = df['f_0'] - sector_means
    
    # 3. Time Series Trends (Rolling)
    for window in [5, 20]:
        df[f'f_0_ma_{window}'] = df.groupby('code')['f_0'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Volatility
        df[f'vol_{window}'] = df.groupby('code')['price_ratio'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # 4. Cross-Sectional Rank Features (Global Rank per Date)
    rank_cols = ['f_0', 'f_4', 'f_5', 'price_ratio', 'f_0_sector_neutral']
    print("  Adding Cross-Sectional Rank features...")
    for col in rank_cols:
         df[f'rank_{col}'] = df.groupby('date')[col].transform(lambda x: x.rank(pct=True))
         
    return df

def train_and_evaluate_regressor():
    print("Loading data for Regressor Solution...")
    with open('data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
        
    train_data['f_3'].fillna(-1, inplace=True)
    test_data['f_3'].fillna(-1, inplace=True)
    
    print("Adding features...")
    train_data = add_features(train_data)
    test_data = add_features(test_data)
    
    features = [c for c in train_data.columns if c not in ['code', 'date', 'y', 'f_3']]
    print(f"Features: {features}")
    
    train_data = train_data.sort_values('date').reset_index(drop=True)
    
    X = train_data[features]
    y = train_data['y']
    
    n_splits = 5
    dates = train_data['date'].unique()
    fold_size = len(dates) // (n_splits + 1)
    gap = 20 # Purged Gap to prevent leakage
    
    cv_scores = []
    
    print("\nStarting Cross-Validation (Regressor)...")
    for i in range(n_splits):
        train_end_idx = (i + 1) * fold_size
        val_start_idx = train_end_idx + gap # Add gap
        val_end_idx = val_start_idx + fold_size
        
        # Check bounds
        if val_start_idx >= len(dates):
             break
        
        date_values = sorted(dates)
        train_dates = date_values[:train_end_idx]
        val_dates = date_values[val_start_idx:val_end_idx] # Might be shorter if near end
        
        if len(val_dates) == 0:
            break
            
        train_mask = train_data['date'].isin(train_dates)
        val_mask = train_data['date'].isin(val_dates)
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            colsample_bytree=0.8,
            subsample=0.8
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50), 
                lgb.log_evaluation(0) # Silent to reduce noise
            ]
        )
        
        pred = model.predict(X_val)
        
        # Calculate Rank IC
        val_res = pd.DataFrame({'date': train_data.loc[val_mask, 'date'], 'y': y_val, 'pred': pred})
        daily_ics = []
        for d in val_res['date'].unique():
            tmp = val_res[val_res['date'] == d]
            if len(tmp) > 1:
                ic, _ = spearmanr(tmp['y'], tmp['pred'])
                daily_ics.append(ic)
        
        mean_ic = np.mean(daily_ics)
        print(f"Fold {i+1} Rank IC: {mean_ic:.4f}")
        cv_scores.append(mean_ic)
        
    print(f"Mean CV Rank IC: {np.mean(cv_scores):.4f}")
    
    # Logic check
    if np.mean(cv_scores) < 0:
        print("WARNING: Negative correlation detected! Inverting predictions approach required?? No, fix model.")
    
    # Final Model
    print("Training final model...")
    final_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        colsample_bytree=0.8,
        subsample=0.8
    )
    final_model.fit(train_data[features], train_data['y'])
    
    pred_test = final_model.predict(test_data[features])
    
    sub = pd.read_csv('data/sample_submission.csv')
    sub['y_pred'] = pred_test
    sub.to_csv('submissions/submission_regressor.csv', index=False)
    print("Saved submissions/submission_regressor.csv")

if __name__ == "__main__":
    train_and_evaluate_regressor()
