import pickle
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def analyze_leak():
    print("Loading train data...")
    with open('data/train_data.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Sort by code and date
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    features = [c for c in df.columns if c.startswith('f_')]
    target = 'y'
    
    print(f"Data shape: {df.shape}")
    print(f"Features: {features}")
    
    # 1. Direct Correlation
    print("\n--- Direct Correlation with y ---")
    for f in features:
        corr = df[[f, target]].corr().iloc[0, 1]
        rank_ic = df[[f, target]].corr(method='spearman').iloc[0, 1]
        print(f"{f}: Pearson={corr:.4f}, Spearman={rank_ic:.4f}")
        
    # 2. Next Day Feature Correlation (Lookahead)
    # Check if target is related to FUTURE features
    print("\n--- Correlation with Next Day Features (Shift -1) ---")
    # Group by code to shift properly
    df_shifted = df.groupby('code')[features].shift(-1)
    df_shifted.columns = [f"{c}_next" for c in features]
    df_combined = pd.concat([df[[target]], df_shifted], axis=1)
    
    for f in features:
        f_next = f"{f}_next"
        # Drop last rows where next is NaN
        temp = df_combined[[target, f_next]].dropna()
        if len(temp) > 0:
            corr = temp.corr().iloc[0, 1]
            rank_ic = temp.corr(method='spearman').iloc[0, 1]
            print(f"{f}_next: Pearson={corr:.4f}, Spearman={rank_ic:.4f}")
            
    # 3. Return-like relationships
    # Check if y ~ (Feature_next / Feature_current) - 1
    print("\n--- Correlation with Calculated Returns ---")
    for f in features:
        if df[f].dtype == 'object' or df[f].dtype.name == 'category':
            continue
            
        # Calculate return: (Next / Current) - 1
        # Need original unshifted values
        current_vals = df[f]
        next_vals = df_shifted[f"{f}_next"]
        
        # Avoid division by zero
        calc_ret = (next_vals / (current_vals + 1e-8)) - 1
        
        # Check correlation with y
        temp = pd.DataFrame({'y': df[target], 'calc_ret': calc_ret}).dropna()
        if len(temp) > 0:
             corr = temp.corr().iloc[0, 1]
             rank_ic = temp.corr(method='spearman').iloc[0, 1]
             print(f"Return({f}): Pearson={corr:.4f}, Spearman={rank_ic:.4f}")

    # 4. Check for 'Id' leakage or similar? 
    # Not applicable here as anonymous features usually imply time-series leak
    
    # 5. Check if f_0..f_2 sum up to something?
    
if __name__ == "__main__":
    analyze_leak()
