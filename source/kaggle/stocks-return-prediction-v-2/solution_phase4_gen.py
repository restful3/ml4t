"""
Phase 4 Step A: Optimized Feature Generation
Generates features efficiently without slow groupby().transform() operations.
"""
import pickle
import pandas as pd
import numpy as np
import warnings
import os
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def fast_rank_by_group(df, group_col, value_col):
    """Faster rank computation using sort instead of groupby transform"""
    df = df.sort_values([group_col, value_col])
    df['_rank_temp'] = 1
    df['_count'] = df.groupby(group_col)['_rank_temp'].cumsum()
    df['_total'] = df.groupby(group_col)['_rank_temp'].transform('sum')
    rank_pct = df['_count'] / df['_total']
    df.drop(['_rank_temp', '_count', '_total'], axis=1, inplace=True)
    return rank_pct.values


def add_base_features_fast(df):
    """Add base features with optimized operations"""
    print("  Sorting data...")
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # Convert f_3 to numeric
    if 'f_3' in df.columns:
        if df['f_3'].dtype == 'object':
            df['f_3'] = pd.to_numeric(df['f_3'], errors='coerce').fillna(-1).astype(int)
        else:
            df['f_3'] = df['f_3'].fillna(-1).astype(int)
    
    # 1. Simple transformations (fast)
    print("  Adding simple features...")
    df['price_ratio'] = df['f_0'] / (df['f_1'] + 1e-8)
    df['volume_log'] = np.log1p(df['f_4'])
    
    # 2. Sector mean - use merge instead of transform
    print("  Adding sector neutral features...")
    sector_means = df.groupby(['date', 'f_3'])['f_0'].mean().reset_index()
    sector_means.columns = ['date', 'f_3', 'sector_mean_f_0']
    df = df.merge(sector_means, on=['date', 'f_3'], how='left')
    df['f_0_sector_neutral'] = df['f_0'] - df['sector_mean_f_0']
    df.drop('sector_mean_f_0', axis=1, inplace=True)
    
    # 3. Simplified volatility (avoid groupby rolling)
    print("  Adding volatility proxies...")
    df['vol_proxy'] = np.abs(df['f_0'] - 1)  # Simple proxy
    df['vol_proxy_2'] = np.abs(df['price_ratio'] - 1)
    
    # 4. Interaction features
    print("  Adding interaction features...")
    df['inter_price_tech'] = df['f_0'] * df['f_5']
    df['inter_price_vol'] = df['f_0'] / (df['f_4'] + 1e-8)
    df['inter_tech_diff'] = df['f_5'] - df['f_6']
    df['inter_f0_f2'] = df['f_0'] * df['f_2']
    df['inter_f1_f5'] = df['f_1'] * df['f_5']
    
    # 5. Cross-sectional ranks using efficient method
    print("  Adding cross-sectional rank features (optimized)...")
    rank_cols = ['f_0', 'f_4', 'f_5', 'price_ratio', 'f_0_sector_neutral']
    
    for col in rank_cols:
        print(f"    Ranking {col}...")
        # Use sort-based ranking which is faster
        df = df.sort_values(['date', col]).reset_index(drop=True)
        df[f'rank_{col}'] = df.groupby('date').cumcount() + 1
        date_counts = df.groupby('date').size()
        df['_date_count'] = df['date'].map(date_counts)
        df[f'rank_{col}'] = df[f'rank_{col}'] / df['_date_count']
        df.drop('_date_count', axis=1, inplace=True)
    
    df.fillna(0, inplace=True)
    return df


def add_advanced_features(df):
    """Add advanced polynomial and interaction features"""
    print("\n  Adding advanced features...")
    
    # Polynomial
    df['sq_f_0'] = df['f_0'] ** 2
    df['sq_f_5'] = df['f_5'] ** 2
    df['sq_price_ratio'] = df['price_ratio'] ** 2
    
    # Ratios
    df['ratio_f5_f6'] = df['f_5'] / (df['f_6'] + 1e-8)
    df['ratio_f0_f2'] = df['f_0'] / (df['f_2'] + 1e-8)
    
    # Differences
    df['diff_f0_f1'] = df['f_0'] - df['f_1']
    df['diff_f0_f2'] = df['f_0'] - df['f_2']
    
    # Alpha-style features
    df['alpha_momentum'] = (df['f_0'] - df['f_1']) / (df['f_0'] + df['f_1'] + 1e-8)
    df['alpha_tech_div'] = (df['f_5'] - df['f_6']) * df.get('rank_f_0', 0.5)
    df['alpha_vol_price'] = df['volume_log'] * df.get('rank_f_0', 0.5)
    
    # Risk-adjusted
    df['alpha_risk_adj'] = df['f_0_sector_neutral'] / (df['vol_proxy'] + 1e-8)
    df['alpha_risk_adj'] = df['alpha_risk_adj'].clip(-100, 100)
    
    # Rank combinations
    if 'rank_f_5' in df.columns and 'rank_f_4' in df.columns:
        df['alpha_rank_combo'] = df['rank_f_5'] * 0.6 + df['rank_f_4'] * 0.4
        df['alpha_rank_min'] = np.minimum(df['rank_f_0'], df['rank_f_5'])
        df['alpha_rank_max'] = np.maximum(df['rank_f_0'], df['rank_f_5'])
    
    # Complex interactions
    df['alpha_price_vol_int'] = df['f_0'] * df['volume_log']
    df['alpha_sharpe_proxy'] = df['price_ratio'] / (df['vol_proxy_2'] + 1e-8)
    df['alpha_sharpe_proxy'] = df['alpha_sharpe_proxy'].clip(-100, 100)
    
    # Additional ranks for new features
    print("  Ranking key alpha features...")
    for col in ['alpha_momentum', 'alpha_tech_div', 'diff_f0_f1']:
        if col in df.columns:
            df = df.sort_values(['date', col]).reset_index(drop=True)
            df[f'rank_{col}'] = df.groupby('date').cumcount() + 1
            date_counts = df.groupby('date').size()
            df['_date_count'] = df['date'].map(date_counts)
            df[f'rank_{col}'] = df[f'rank_{col}'] / df['_date_count']
            df.drop('_date_count', axis=1, inplace=True)
    
    df.fillna(0, inplace=True)
    return df


def main():
    print("="*60)
    print("PHASE 4: OPTIMIZED FEATURE GENERATION")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # Load data
    print("\n[1/4] Loading data...")
    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Process train data
    print("\n[2/4] Processing TRAIN data...")
    train_data['_orig_idx'] = np.arange(len(train_data))
    train_data = add_base_features_fast(train_data)
    train_data = add_advanced_features(train_data)
    
    # Restore order
    print("  Restoring original order...")
    train_data = train_data.sort_values('_orig_idx').reset_index(drop=True)
    train_data.drop('_orig_idx', axis=1, inplace=True)
    
    gc.collect()
    
    # Process test data
    print("\n[3/4] Processing TEST data...")
    test_data['_orig_idx'] = np.arange(len(test_data))
    test_data = add_base_features_fast(test_data)
    test_data = add_advanced_features(test_data)
    
    # Restore order
    print("  Restoring original order...")
    test_data = test_data.sort_values('_orig_idx').reset_index(drop=True)
    test_data.drop('_orig_idx', axis=1, inplace=True)
    
    gc.collect()
    
    # Save
    print("\n[4/4] Saving augmented data...")
    train_out = os.path.join(DATA_DIR, 'train_data_gen.pkl')
    test_out = os.path.join(DATA_DIR, 'test_data_gen.pkl')
    
    with open(train_out, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"Saved: {train_out}")
    
    with open(test_out, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"Saved: {test_out}")
    
    # Summary
    new_cols = [c for c in train_data.columns if c not in ['code', 'date', 'y', 'f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6']]
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")
    print(f"New features: {len(new_cols)}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
