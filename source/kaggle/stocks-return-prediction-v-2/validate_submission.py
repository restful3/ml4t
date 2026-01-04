import pandas as pd
import numpy as np

print("Validating submission file...")

# Load files
sample = pd.read_csv('data/sample_submission.csv')
submission = pd.read_csv('submissions/submission_baseline.csv')

print(f"\nSample submission shape: {sample.shape}")
print(f"Our submission shape: {submission.shape}")

# Check if shapes match
if sample.shape == submission.shape:
    print("✓ Shape matches!")
else:
    print("✗ Shape mismatch!")

# Check columns
print(f"\nSample columns: {sample.columns.tolist()}")
print(f"Our columns: {submission.columns.tolist()}")

if list(sample.columns) == list(submission.columns):
    print("✓ Columns match!")
else:
    print("✗ Columns mismatch!")

# Check for missing values
print(f"\nMissing values in submission:")
print(submission.isnull().sum())

if submission.isnull().sum().sum() == 0:
    print("✓ No missing values!")
else:
    print("✗ Missing values found!")

# Check prediction statistics
print(f"\nPrediction statistics:")
print(submission['y_pred'].describe())

# Check for infinities
if np.isinf(submission['y_pred']).sum() > 0:
    print(f"✗ Found {np.isinf(submission['y_pred']).sum()} infinite values!")
else:
    print("✓ No infinite values!")

# Verify ID, code, date match
if (submission['id'] == sample['id']).all():
    print("✓ ID column matches!")
else:
    print("✗ ID column mismatch!")

if (submission['code'] == sample['code']).all():
    print("✓ Code column matches!")
else:
    print("✗ Code column mismatch!")

if (submission['date'] == sample['date']).all():
    print("✓ Date column matches!")
else:
    print("✗ Date column mismatch!")

print("\n✓ Submission file is valid and ready to upload!")
