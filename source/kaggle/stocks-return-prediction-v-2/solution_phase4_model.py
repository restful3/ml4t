"""
Phase 4 Step B: Heterogeneous Stacking Model
Combines LightGBM, XGBoost, and CatBoost predictions via weighted blending.

Loads data with genetic features (from solution_phase4_gen.py) and trains ensemble.
"""
import pickle
import pandas as pd
import numpy as np
import warnings
import os
import gc
from datetime import datetime
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Import ML libraries
import lightgbm as lgb

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SUBMISSIONS_DIR = os.path.join(BASE_DIR, 'submissions')

# Ensure submissions directory exists
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def calculate_daily_rank_ic(y_true, y_pred, dates):
    """Calculate mean daily Rank IC (Spearman correlation)"""
    df = pd.DataFrame({'date': dates, 'y': y_true, 'pred': y_pred})
    daily_ics = []
    for d in df['date'].unique():
        tmp = df[df['date'] == d]
        if len(tmp) > 1:
            ic, _ = spearmanr(tmp['y'], tmp['pred'])
            if not np.isnan(ic):
                daily_ics.append(ic)
    return np.mean(daily_ics) if daily_ics else 0.0


def train_lightgbm(X_train, y_train, X_val, y_val, seed=42):
    """Train LightGBM model"""
    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=8,
        colsample_bytree=0.7,
        subsample=0.7,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost model"""
    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=8,
        colsample_bytree=0.7,
        subsample=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist'  # Fast histogram-based
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def train_catboost(X_train, y_train, X_val, y_val, seed=42):
    """Train CatBoost model"""
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=50,
        task_type='CPU'  # Use 'GPU' if available
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    return model


def run_phase4():
    print("="*70)
    print("PHASE 4: HETEROGENEOUS STACKING MODEL")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    print(f"LightGBM: Available")
    print(f"XGBoost: {'Available' if XGB_AVAILABLE else 'Not Available'}")
    print(f"CatBoost: {'Available' if CAT_AVAILABLE else 'Not Available'}")
    
    # Try loading genetic feature data first
    gen_train_path = os.path.join(DATA_DIR, 'train_data_gen.pkl')
    gen_test_path = os.path.join(DATA_DIR, 'test_data_gen.pkl')
    
    if os.path.exists(gen_train_path) and os.path.exists(gen_test_path):
        print("\nLoading data with Genetic Features...")
        with open(gen_train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(gen_test_path, 'rb') as f:
            test_data = pickle.load(f)
        print("Loaded pre-generated genetic features data.")
    else:
        print("\n⚠️  Genetic feature data not found!")
        print("Please run solution_phase4_gen.py first, or loading raw data...")
        
        # Fall back to raw data and generate base features inline
        with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
            train_data = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        
        # Add base features (minimal version)
        print("Adding base features...")
        for df in [train_data, test_data]:
            df.sort_values(['code', 'date'], inplace=True)
            if df['f_3'].dtype == 'object':
                df['f_3'] = pd.to_numeric(df['f_3'], errors='coerce')
            df['f_3'].fillna(-1, inplace=True)
            
            df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
            df['volume_log'] = np.log1p(df['f_4'])
            
            sector_means = df.groupby(['date', 'f_3'])['f_0'].transform('mean')
            df['f_0_sector_neutral'] = df['f_0'] - sector_means
            
            for window in [5, 20]:
                df[f'f_0_ma_{window}'] = df.groupby('code')['f_0'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            
            df['inter_price_tech'] = df['f_0'] * df['f_5']
            df['inter_tech_diff'] = df['f_5'] - df['f_6']
            
            rank_cols = ['f_0', 'f_4', 'f_5', 'price_ratio', 'f_0_sector_neutral']
            for col in rank_cols:
                df[f'rank_{col}'] = df.groupby('date')[col].transform(lambda x: x.rank(pct=True))
            
            df.fillna(0, inplace=True)
    
    print(f"\nTrain shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    # Define features
    exclude_cols = ['code', 'date', 'y', 'f_3']
    features = [c for c in train_data.columns if c not in exclude_cols]
    features = [c for c in features if train_data[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    print(f"Features: {len(features)}")
    gp_features = [c for c in features if c.startswith('gp_')]
    print(f"  - GP features: {len(gp_features)}")
    
    # Sort by date
    train_data = train_data.sort_values('date').reset_index(drop=True)
    
    # Target transformation: Rank(y) per date
    print("\nTransforming target to daily rank...")
    train_data['y_rank'] = train_data.groupby('date')['y'].transform(lambda x: x.rank(pct=True))
    
    X = train_data[features]
    y = train_data['y_rank']
    y_raw = train_data['y']
    dates = train_data['date']
    
    # Time Series Split setup
    unique_dates = sorted(train_data['date'].unique())
    n_splits = 5
    fold_size = len(unique_dates) // (n_splits + 1)
    gap = 20
    
    # Ensemble configuration
    seeds = [42, 2024, 777]  # 3 seeds per model type
    
    # Model weights for blending
    if XGB_AVAILABLE and CAT_AVAILABLE:
        weights = {'lgb': 0.4, 'xgb': 0.3, 'cat': 0.3}
    elif XGB_AVAILABLE:
        weights = {'lgb': 0.6, 'xgb': 0.4}
    elif CAT_AVAILABLE:
        weights = {'lgb': 0.6, 'cat': 0.4}
    else:
        weights = {'lgb': 1.0}
    
    print(f"\nModel weights: {weights}")
    print(f"Seeds per model: {seeds}")
    
    # Results storage
    cv_results = {'lgb': [], 'xgb': [], 'cat': [], 'ensemble': []}
    test_preds = {'lgb': np.zeros(len(test_data)),
                  'xgb': np.zeros(len(test_data)),
                  'cat': np.zeros(len(test_data))}
    
    # Cross-validation and training
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION")
    print(f"{'='*70}")
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(seeds)}) ---")
        
        seed_cv = {'lgb': [], 'xgb': [], 'cat': [], 'ensemble': []}
        seed_test_preds = {'lgb': None, 'xgb': None, 'cat': None}
        
        for fold_i in range(n_splits):
            train_end_idx = (fold_i + 1) * fold_size
            val_start_idx = train_end_idx + gap
            val_end_idx = val_start_idx + fold_size
            
            if val_start_idx >= len(unique_dates):
                break
            
            train_dates = unique_dates[:train_end_idx]
            val_dates = unique_dates[val_start_idx:min(val_end_idx, len(unique_dates))]
            
            if len(val_dates) == 0:
                break
            
            train_mask = train_data['date'].isin(train_dates)
            val_mask = train_data['date'].isin(val_dates)
            
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            y_val_raw = y_raw[val_mask]
            val_dates_arr = dates[val_mask]
            
            fold_preds = {}
            
            # Train LightGBM
            lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, seed)
            fold_preds['lgb'] = lgb_model.predict(X_val)
            
            # Train XGBoost
            if XGB_AVAILABLE:
                xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, seed)
                fold_preds['xgb'] = xgb_model.predict(X_val)
            
            # Train CatBoost
            if CAT_AVAILABLE:
                cat_model = train_catboost(X_tr, y_tr, X_val, y_val, seed)
                fold_preds['cat'] = cat_model.predict(X_val)
            
            # Ensemble prediction
            ensemble_pred = np.zeros(len(X_val))
            for key, w in weights.items():
                if key in fold_preds:
                    ensemble_pred += w * fold_preds[key]
            
            # Calculate Rank IC
            for key in fold_preds:
                ic = calculate_daily_rank_ic(y_val_raw.values, fold_preds[key], val_dates_arr.values)
                seed_cv[key].append(ic)
            
            ensemble_ic = calculate_daily_rank_ic(y_val_raw.values, ensemble_pred, val_dates_arr.values)
            seed_cv['ensemble'].append(ensemble_ic)
            
            del lgb_model
            if XGB_AVAILABLE:
                del xgb_model
            if CAT_AVAILABLE:
                del cat_model
            gc.collect()
        
        # Report seed CV results
        print(f"  LightGBM CV: {np.mean(seed_cv['lgb']):.4f}")
        if XGB_AVAILABLE:
            print(f"  XGBoost CV:  {np.mean(seed_cv['xgb']):.4f}")
        if CAT_AVAILABLE:
            print(f"  CatBoost CV: {np.mean(seed_cv['cat']):.4f}")
        print(f"  ENSEMBLE CV: {np.mean(seed_cv['ensemble']):.4f}")
        
        for key in seed_cv:
            cv_results[key].extend(seed_cv[key])
        
        # Train on full data for this seed
        print(f"  Training full models for seed {seed}...")
        
        # LightGBM full
        lgb_full = lgb.LGBMRegressor(
            n_estimators=1500, learning_rate=0.03, num_leaves=31, max_depth=8,
            colsample_bytree=0.7, subsample=0.7, subsample_freq=1,
            reg_alpha=0.1, reg_lambda=0.1, random_state=seed, n_jobs=-1, verbose=-1
        )
        lgb_full.fit(X, y)
        test_preds['lgb'] += lgb_full.predict(test_data[features])
        
        # XGBoost full
        if XGB_AVAILABLE:
            xgb_full = xgb.XGBRegressor(
                n_estimators=1500, learning_rate=0.03, max_depth=8,
                colsample_bytree=0.7, subsample=0.7,
                reg_alpha=0.1, reg_lambda=0.1, random_state=seed, n_jobs=-1, verbosity=0, tree_method='hist'
            )
            xgb_full.fit(X, y)
            test_preds['xgb'] += xgb_full.predict(test_data[features])
            del xgb_full
        
        # CatBoost full
        if CAT_AVAILABLE:
            cat_full = CatBoostRegressor(
                iterations=1500, learning_rate=0.03, depth=8, l2_leaf_reg=3,
                random_seed=seed, verbose=False, task_type='CPU'
            )
            cat_full.fit(X, y)
            test_preds['cat'] += cat_full.predict(test_data[features])
            del cat_full
        
        del lgb_full
        gc.collect()
    
    # Average predictions across seeds
    for key in test_preds:
        if np.any(test_preds[key] != 0):
            test_preds[key] /= len(seeds)
    
    # Final ensemble prediction
    final_pred = np.zeros(len(test_data))
    for key, w in weights.items():
        if np.any(test_preds[key] != 0):
            final_pred += w * test_preds[key]
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"LightGBM Mean CV: {np.mean(cv_results['lgb']):.4f} ± {np.std(cv_results['lgb']):.4f}")
    if XGB_AVAILABLE:
        print(f"XGBoost Mean CV:  {np.mean(cv_results['xgb']):.4f} ± {np.std(cv_results['xgb']):.4f}")
    if CAT_AVAILABLE:
        print(f"CatBoost Mean CV: {np.mean(cv_results['cat']):.4f} ± {np.std(cv_results['cat']):.4f}")
    print(f"ENSEMBLE Mean CV: {np.mean(cv_results['ensemble']):.4f} ± {np.std(cv_results['ensemble']):.4f}")
    
    # Save submission
    print("Generating submission file with safe merge...")
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    
    # Create a dense dataframe of predictions with keys
    # Note: final_pred aligns with test_data because we used test_data[features] to predict
    pred_df = test_data[['code', 'date']].copy()
    pred_df['y_pred'] = final_pred
    
    # Merge predictions into sample submission
    # Use left join to ensure we keep exactly the rows from sample_submission
    sub = sub.drop('y_pred', axis=1, errors='ignore')
    sub = sub.merge(pred_df, on=['code', 'date'], how='left')
    
    # Handle any missing values (shouldn't happen if alignment is correct)
    null_preds = sub['y_pred'].isnull().sum()
    if null_preds > 0:
        print(f"⚠️ Warning: {null_preds} missing predictions after merge! Filling with 0.5")
        sub['y_pred'].fillna(0.5, inplace=True)
        
    submission_file = os.path.join(SUBMISSIONS_DIR, 'submission_phase4_fixed.csv')
    sub.to_csv(submission_file, index=False)
    print(f"\nSaved: {submission_file}")
    
    # Generate report
    report_file = os.path.join(BASE_DIR, 'phase4_report.md')
    with open(report_file, 'w') as f:
        f.write("# Phase 4 Results Report\n\n")
        f.write(f"**Generated:** {datetime.now()}\n\n")
        f.write("## Model Configuration\n\n")
        f.write(f"- **Weights:** {weights}\n")
        f.write(f"- **Seeds:** {seeds}\n")
        f.write(f"- **Features:** {len(features)} (GP: {len(gp_features)})\n\n")
        f.write("## Cross-Validation Results\n\n")
        f.write("| Model | Mean Rank IC | Std |\n")
        f.write("|-------|--------------|-----|\n")
        f.write(f"| LightGBM | {np.mean(cv_results['lgb']):.4f} | {np.std(cv_results['lgb']):.4f} |\n")
        if XGB_AVAILABLE:
            f.write(f"| XGBoost | {np.mean(cv_results['xgb']):.4f} | {np.std(cv_results['xgb']):.4f} |\n")
        if CAT_AVAILABLE:
            f.write(f"| CatBoost | {np.mean(cv_results['cat']):.4f} | {np.std(cv_results['cat']):.4f} |\n")
        f.write(f"| **Ensemble** | **{np.mean(cv_results['ensemble']):.4f}** | {np.std(cv_results['ensemble']):.4f} |\n\n")
        f.write("## Submission\n\n")
        f.write(f"- File: `{submission_file}`\n")
        f.write(f"- Prediction range: [{final_pred.min():.4f}, {final_pred.max():.4f}]\n")
    
    print(f"Saved: {report_file}")
    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    run_phase4()
