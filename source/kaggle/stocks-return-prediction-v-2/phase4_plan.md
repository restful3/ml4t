# Phase 4 Improvement Plan: Genetic Features & Heterogeneous Stacking

## 1. Goal Description
Break the 0.09 Rank IC barrier (current best: 0.08957) by introducing:
1.  **Automated Feature Discovery**: Using Genetic Programming (`gplearn`) to find complex, non-intuitive interactions between features.
2.  **Heterogeneous Ensemble**: Combining predictions from **LightGBM**, **XGBoost**, and **CatBoost** to exploit different algorithmic strengths.

## 2. Prerequisites
The following libraries need to be installed:
- `gplearn`: For genetic feature generation.
- `xgboost`: Gradient boosting library.
- `catboost`: Gradient boosting library handling categoricals well.

## 3. Proposed Changes

### A. Feature Generation Script (`solution_phase4_gen.py`)
This script will:
1.  Load data and existing features.
2.  Run `gplearn.SymbolicTransformer` to generate ~10-20 new interaction features.
    - **Function Set**: `add`, `sub`, `mul`, `div`, `sqrt`, `log`, `abs`, `neg`, `inv`, `max`, `min`.
    - **Generations**: 20
    - **Population**: 2000
3.  Save the new feature set data to `data/train_data_gen.pkl` and `data/test_data_gen.pkl` to optimize training time.

### B. Modeling Script (`solution_phase4_model.py`)
This script will:
1.  Load the data with Genetic Features.
2.  Train 3 different models on **Rank(y)** targets:
    - **LightGBM**: Fast, accurate, handles large data well.
    - **XGBoost**: Highly optimized, different tree growing strategy.
    - **CatBoost**: Excellent with categorical features (if any) and robust to overfitting.
3.  **Ensemble (Stacking/Blending)**:
    - Blend predictions using weighted average (e.g., 0.4*LGBM + 0.3*XGB + 0.3*CAT).
    - Or use a Meta-Learner (Linear Regression) to stack predictions.

## 4. Verification Plan

### Automated Tests
- Monitor CV Rank IC for each individual model and the final ensemble.
- **Success Criteria**: CV Rank IC > 0.1150.

### Manual Verification
- Execute feature generation -> model training.
- Submit `submissions/submission_phase4.csv`.
- **Target Leaderboard Score**: > 0.09.
