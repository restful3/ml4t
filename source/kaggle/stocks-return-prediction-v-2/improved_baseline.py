import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
import os
import datetime

warnings.filterwarnings('ignore')

def add_features(df):
    """Add engineered features for Rank IC Optimization"""
    # Sort by code and date
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 1. Base transformations
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_log'] = np.log1p(df['f_4'])
    
    # 2. Sector Neutralization (f_0 - SectorMean)
    # Ensure f_3 is numeric or handled
    # We will compute mean of f_0 per date per sector
    if df['f_3'].dtype == 'object':
        df['f_3_cat'] = df['f_3'].astype('category').cat.codes
    else:
        df['f_3_cat'] = df['f_3']
        
    print("  Adding Sector Neutral features...")
    # Group by [date, f_3] -> mean(f_0)
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
    # This is crucial for Rank IC
    rank_cols = ['f_0', 'f_4', 'f_5', 'price_ratio', 'f_0_sector_neutral']
    print("  Adding Cross-Sectional Rank features...")
    for col in rank_cols:
         df[f'rank_{col}'] = df.groupby('date')[col].transform(lambda x: x.rank(pct=True))
         
    return df

def get_groups(df):
    """Calculate group sizes for LambdaRank (rows per date)"""
    # Assumes df is sorted by date!
    return df.groupby('date', sort=False).size().to_numpy()

def train_and_evaluate():
    print("Loading data...")
    with open('data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
        
    # Preprocessing
    train_data['f_5'] = train_data['f_5'].clip(train_data['f_5'].quantile(0.01), train_data['f_5'].quantile(0.99))
    test_data['f_5'] = test_data['f_5'].clip(test_data['f_5'].quantile(0.01), test_data['f_5'].quantile(0.99))
    
    # Fill NA common
    train_data['f_3'].fillna(-1, inplace=True)
    test_data['f_3'].fillna(-1, inplace=True)
    
    print("Adding features...")
    train_data = add_features(train_data)
    test_data = add_features(test_data)
    
    features = [c for c in train_data.columns if c not in ['code', 'date', 'y', 'f_3']]
    # Include rank features
    print(f"Features ({len(features)}): {features}")
    
    # Sort by date for splitting
    train_data = train_data.sort_values('date').reset_index(drop=True)
    
    X = train_data[features]
    y = train_data['y']
    dates = train_data['date'].unique()
    
    # Custom TimeSeriesSplit respecting Dates
    n_splits = 5
    fold_size = len(dates) // (n_splits + 1)
    
    print("\nStarting Cross-Validation with LGBMRanker...")
    
    cv_scores = []
    
    for i in range(n_splits):
        # Split dates
        train_end_date_idx = (i + 1) * fold_size
        val_start_date_idx = train_end_date_idx
        val_end_date_idx = val_start_date_idx + fold_size
        
        # Get actual date values
        date_values = sorted(train_data['date'].unique()) # Should be 0..1701
        
        train_dates = date_values[:train_end_date_idx]
        val_dates = date_values[val_start_date_idx:val_end_date_idx]
        
        # Create masks
        train_mask = train_data['date'].isin(train_dates)
        val_mask = train_data['date'].isin(val_dates)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        # Calculate groups for Ranker
        q_train = X_train.groupby(train_data.loc[train_mask, 'date'], sort=False).size().to_numpy()
        q_val = X_val.groupby(train_data.loc[val_mask, 'date'], sort=False).size().to_numpy()
        
        # Train LGBMRanker
        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            metric='ndcg', # internal metric
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            n_jobs=-1
        )
        
        ranker.fit(
            X_train, y_train,
            group=q_train,
            eval_set=[(X_val, y_val)],
            eval_group=[q_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
        )
        
        # Predict
        pred = ranker.predict(X_val)
        
        # Calculate Spearman Rank IC
        # Group by date and mean
        val_df = pd.DataFrame({'date': train_data.loc[val_mask, 'date'], 'y': y_val, 'pred': pred})
        
        # Calculate IC per date
        daily_ics = []
        for d in val_df['date'].unique():
            tmp = val_df[val_df['date'] == d]
            ic, _ = spearmanr(tmp['y'], tmp['pred'])
            daily_ics.append(ic)
            
        mean_ic = np.mean(daily_ics)
        print(f"Fold {i+1} Rank IC: {mean_ic:.4f}")
        cv_scores.append(mean_ic)
        
    print(f"\nMean CV Rank IC: {np.mean(cv_scores):.4f}")
    
    # Train Full Model
    print("\nTraining final model...")
    # Re-sort full data just in case
    train_data = train_data.sort_values('date')
    q_full = train_data.groupby('date', sort=False).size().to_numpy()
    
    final_model = lgb.LGBMRanker(
        objective='lambdarank',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(train_data[features], train_data['y'], group=q_full)
    
    # Predict Test
    print("Predicting test...")
    test_pred = final_model.predict(test_data[features])
    
    # Save submission
    sub = pd.read_csv('data/sample_submission.csv')
    sub['y_pred'] = test_pred
    sub.to_csv('submissions/submission_improved.csv', index=False)
    print("Saved submission_improved.csv")
    
    # Generate Report
    with open('improved_baseline.md', 'w') as f:
        f.write(f"# Improved Rank IC Model Report\n\n")
        f.write(f"Mean CV Rank IC: {np.mean(cv_scores):.4f}\n\n")
        f.write(f"## Features used:\n{features}\n")

if __name__ == "__main__":
    train_and_evaluate()
