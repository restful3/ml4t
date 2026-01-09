import pickle
import pandas as pd
import numpy as np

def check_features():
    print("Loading train data...")
    with open('data/train_data.pkl', 'rb') as f:
        df = pickle.load(f)
        
    print(f"Data shape: {df.shape}")
    
    # Check Dates
    dates = df['date'].unique()
    print(f"Num dates: {len(dates)}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    
    # Check Stocks per Date
    stocks_per_date = df.groupby('date').size()
    print("\nStocks per date stats:")
    print(stocks_per_date.describe())
    
    # Check f_3 (Sector)
    print("\nf_3 (Sector) Analysis:")
    print(f"Unique values: {df['f_3'].nunique()}")
    print("Top 10 sectors by count:")
    print(df['f_3'].value_counts().head(10))
    
    # Check if f_3 changes for a stock
    # Consitency of f_3 for each code
    f3_counts = df.groupby('code')['f_3'].nunique()
    print("\nSector changes per stock:")
    print(f3_counts.value_counts())
    
    # Prepare for LambdaRank
    # LambdaRank needs group info (counts per query/date)
    # We need to ensure data is sorted by date
    df_sorted = df.sort_values('date')
    group_counts = df_sorted.groupby('date').size().to_list()
    print(f"\nGroup counts for LambdaRank (first 5): {group_counts[:5]}")
    
if __name__ == "__main__":
    check_features()
