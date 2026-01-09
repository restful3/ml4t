import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
import os
import gc

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

def run_phase2():
    print("Loading data for Phase 2...")
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
    print(f"Features ({len(features)}): {features}")
    
    train_data = train_data.sort_values('date').reset_index(drop=True)
    
    # Target Transformation: Rank(y)
    print("Transforming Target to Rank(pct=True)...")
    train_data['y_rank'] = train_data.groupby('date')['y'].transform(lambda x: x.rank(pct=True))
    
    X = train_data[features]
    y = train_data['y_rank'] # Train on Rank
    y_raw = train_data['y']  # Evaluate on Raw (Rank IC)
    
    dates = train_data['date'].unique()
    n_splits = 5
    fold_size = len(dates) // (n_splits + 1)
    gap = 20
    
    seeds = [42, 2024, 777, 123, 999]
    ensemble_preds = np.zeros(len(test_data))
    cv_mean_scores = []
    
    print(f"\nStarting 5-Seed Ensemble Training (Seeds: {seeds})...")
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        
        seed_cv_scores = []
        
        # Cross-Validation Loop
        for i in range(n_splits):
            train_end_idx = (i + 1) * fold_size
            val_start_idx = train_end_idx + gap
            val_end_idx = val_start_idx + fold_size
            
            if val_start_idx >= len(dates): break
            
            date_values = sorted(dates)
            train_dates = date_values[:train_end_idx]
            val_dates = date_values[val_start_idx:val_end_idx]
            
            if len(val_dates) == 0: break
            
            train_mask = train_data['date'].isin(train_dates)
            val_mask = train_data['date'].isin(val_dates)
            
            X_tr, y_tr = X[train_mask], y[train_mask] # Train on Rank
            X_val, y_val_raw = X[val_mask], y_raw[val_mask] # Val on Raw for IC
            # But we also need y_val (rank) if we want to early stop on 'rmse' of rank?
            # Actually, standard RMSE on rank is fine.
            y_val = y[val_mask] 
            
            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=-1,
                colsample_bytree=0.8,
                subsample=0.8
            )
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            pred = model.predict(X_val)
            
            # Calculate Rank IC (Prediction vs Raw Y)
            # Both are monotonic, so correlation should be high
            val_res = pd.DataFrame({'date': train_data.loc[val_mask, 'date'], 'y': y_val_raw, 'pred': pred})
            daily_ics = []
            for d in val_res['date'].unique():
                tmp = val_res[val_res['date'] == d]
                if len(tmp) > 1:
                    ic, _ = spearmanr(tmp['y'], tmp['pred'])
                    daily_ics.append(ic)
            
            mean_ic = np.mean(daily_ics)
            seed_cv_scores.append(mean_ic)
            print(f"  Fold {i+1} Rank IC: {mean_ic:.4f}")
            
        seed_mean = np.mean(seed_cv_scores)
        print(f"  > Seed {seed} Mean CV: {seed_mean:.4f}")
        cv_mean_scores.append(seed_mean)
        
        # Full Train on Seed
        print(f"  Training Full Model for Seed {seed}...")
        final_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            colsample_bytree=0.8,
            subsample=0.8
        )
        final_model.fit(X, y) # Train on Full Rank Y
        
        print("  Predicting Test...")
        # Accumulate predictions
        # We can sum them up, then rank/average later.
        # Since we trained on Ranks (0..1), average is fine.
        seed_pred = final_model.predict(test_data[features])
        ensemble_preds += seed_pred
        
        # Cleanup
        del model, final_model, X_tr, X_val
        gc.collect()

    # Final Average
    ensemble_preds /= len(seeds)
    print(f"\nGlobal Mean CV Rank IC across seeds: {np.mean(cv_mean_scores):.4f}")
    
    # Create Submission
    sub = pd.read_csv('data/sample_submission.csv')
    sub['y_pred'] = ensemble_preds
    sub.to_csv('submissions/submission_phase2.csv', index=False)
    print("Saved submissions/submission_phase2.csv")

if __name__ == "__main__":
    run_phase2()
