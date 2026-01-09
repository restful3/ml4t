import pandas as pd
import numpy as np

print("Checking files...")
try:
    sub = pd.read_csv('submissions/submission_phase4_fixed.csv')
    sample = pd.read_csv('data/sample_submission.csv')

    print(f"Sub columns: {sub.columns.tolist()}")
    print(f"Sample columns: {sample.columns.tolist()}")
    print(f"Sub shape: {sub.shape}")
    print(f"Sample shape: {sample.shape}")

    if len(sub) != len(sample):
        print("LENGTH MISMATCH!")

    cols_match = list(sub.columns) == list(sample.columns)
    print(f"Columns match: {cols_match}")

    # Check specific columns if present
    if 'id' in sub.columns:
        id_match = (sub['id'] == sample['id']).all()
        print(f"ID match: {id_match}")
    else:
        print("ID column missing in submission")

    if 'y_pred' in sub.columns:
        nan_count = sub['y_pred'].isnull().sum()
        print(f"NaNs in y_pred: {nan_count}")
        inf_count = np.isinf(sub['y_pred']).sum()
        print(f"Infs in y_pred: {inf_count}")
        
    # Check if any extra columns
    extra = set(sub.columns) - set(sample.columns)
    print(f"Extra cols: {extra}")
    missing = set(sample.columns) - set(sub.columns)
    print(f"Missing cols: {missing}")

except Exception as e:
    print(f"Error: {e}")
