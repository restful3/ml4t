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
    """Add engineered features including Interactions for Phase 3"""
    # Sort by code and date
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 1. Base transformations (Phase 1/2)
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_log'] = np.log1p(df['f_4'])
    
    # 2. Sector Neutralization
    if 'f_3' in df.columns:
        if df['f_3'].dtype == 'object':
             df['f_3'] = pd.to_numeric(df['f_3'], errors='coerce').fillna(-1)
    
    print("  Adding Sector Neutral features...")
    sector_means = df.groupby(['date', 'f_3'])['f_0'].transform('mean')
    df['f_0_sector_neutral'] = df['f_0'] - sector_means
    
    # 3. Time Series Trends & Volatility
    for window in [5, 20]:
        df[f'f_0_ma_{window}'] = df.groupby('code')['f_0'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'vol_{window}'] = df.groupby('code')['price_ratio'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # 4. Phase 3: Interaction Features
    print("  Adding Phase 3 Interaction features...")
    # Price * Tech (f_0 * f_5)
    df['inter_price_tech'] = df['f_0'] * df['f_5']
    # Price / Volume (f_0 / f_4) - Liquidity proxy?
    df['inter_price_vol'] = df['f_0'] / (df['f_4'] + 1e-8)
    # Volatility Adjusted Price (Sharpe-like?)
    df['inter_price_adj_vol'] = df['f_0'] / (df['vol_20'] + 1e-8)
    # Tech Diff (f_5 - f_6)
    df['inter_tech_diff'] = df['f_5'] - df['f_6']

    # 5. Cross-Sectional Rank Features (Global Rank per Date)
    # Include new interactions in ranking
    rank_cols = ['f_0', 'f_4', 'f_5', 'price_ratio', 'f_0_sector_neutral', 
                 'inter_price_tech', 'inter_price_vol', 'inter_price_adj_vol']
    
    print("  Adding Cross-Sectional Rank features...")
    for col in rank_cols:
         df[f'rank_{col}'] = df.groupby('date')[col].transform(lambda x: x.rank(pct=True))
         
    return df

def run_phase3():
    print("Loading data for Phase 3...")
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
    y = train_data['y_rank']
    y_raw = train_data['y']
    
    dates = train_data['date'].unique()
    n_splits = 5
    fold_size = len(dates) // (n_splits + 1)
    gap = 20
    
    # Phase 3: Extended Ensemble (10 Seeds)
    seeds = [42, 2024, 777, 123, 999, 1111, 2222, 3333, 4444, 5555]
    ensemble_preds = np.zeros(len(test_data))
    cv_mean_scores = []
    
    print(f"\nStarting 10-Seed Ensemble Training (Seeds: {len(seeds)})...")
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        seed_cv_scores = []
        
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
            
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val_raw = X[val_mask], y_raw[val_mask] # Raw y for IC calc
            y_val = y[val_mask] # Rank y for early stopping
            
            # Phase 3: Stronger Bagging
            model = lgb.LGBMRegressor(
                n_estimators=1200,      # Slightly more trees
                learning_rate=0.04,     # Slower learning
                num_leaves=31,
                random_state=seed,
                n_jobs=-1,
                colsample_bytree=0.7,   # More random feature selection
                subsample=0.7,          # Bagging (70% data)
                subsample_freq=1
            )
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            pred = model.predict(X_val)
            
            val_res = pd.DataFrame({'date': train_data.loc[val_mask, 'date'], 'y': y_val_raw, 'pred': pred})
            daily_ics = []
            for d in val_res['date'].unique():
                tmp = val_res[val_res['date'] == d]
                if len(tmp) > 1:
                    ic, _ = spearmanr(tmp['y'], tmp['pred'])
                    daily_ics.append(ic)
            
            mean_ic = np.mean(daily_ics)
            seed_cv_scores.append(mean_ic)
            # print(f"  Fold {i+1} Rank IC: {mean_ic:.4f}") # Silent for speed
            
        seed_mean = np.mean(seed_cv_scores)
        print(f"  > Seed {seed} Mean CV: {seed_mean:.4f}")
        cv_mean_scores.append(seed_mean)
        
        # Full Train
        final_model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            colsample_bytree=0.7,
            subsample=0.7,
            subsample_freq=1
        )
        final_model.fit(X, y)
        
        seed_pred = final_model.predict(test_data[features])
        ensemble_preds += seed_pred
        
        del model, final_model, X_tr, X_val
        gc.collect()

    ensemble_preds /= len(seeds)
    print(f"\nGlobal Mean CV Rank IC across seeds: {np.mean(cv_mean_scores):.4f}")
    
    sub = pd.read_csv('data/sample_submission.csv')
    sub['y_pred'] = ensemble_preds
    sub.to_csv('submissions/submission_phase3.csv', index=False)
    print("Saved submissions/submission_phase3.csv")

if __name__ == "__main__":
    run_phase3()
