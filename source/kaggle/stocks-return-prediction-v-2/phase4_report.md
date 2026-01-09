# Phase 4 Results Report

**Generated:** 2026-01-06 09:25:11.636346

## Model Configuration

- **Weights:** {'lgb': 0.4, 'xgb': 0.3, 'cat': 0.3}
- **Seeds:** [42, 2024, 777]
- **Features:** 40 (GP: 0)

## Cross-Validation Results

| Model | Mean Rank IC | Std |
|-------|--------------|-----|
| LightGBM | 0.1072 | 0.0215 |
| XGBoost | 0.0921 | 0.0095 |
| CatBoost | 0.1069 | 0.0211 |
| **Ensemble** | **0.1045** | 0.0182 |

## Submission

- File: `/media/restful3/data/workspace/ml4t/source/kaggle/stocks-return-prediction-v-2/submissions/submission_phase4_fixed.csv`
- Prediction range: [0.0827, 0.9797]
