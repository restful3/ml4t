# Kaggle Stock Trends Prediction Project

## 대회 개요
**Predicting Stock Trends: Rise or Fall?**
- 목표: 30 거래일 후 주식 종가가 현재보다 상승(1) 또는 하락(0)할지 예측
- 데이터: 5,000개 주식의 OHLCV + 배당/분할 데이터
- 평가: Accuracy (정확도)

## 프로젝트 구조
```
predicting-stock-trends/
├── .venv/                 # Python 가상환경
├── data/                  # Kaggle 데이터
│   ├── train.csv         # 학습 데이터 (2.2GB)
│   ├── test.csv          # 테스트 데이터
│   └── sample_submission.csv
├── outputs/              # 모델 출력 파일
│   └── submission.csv    # 제출 파일
├── baseline.py           # 베이스라인 모델
├── requirements.txt      # 패키지 의존성
├── .gitignore           # Git 제외 설정
└── README.md            # 프로젝트 문서
```

## 환경 설정

### 1. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python3 -m venv .venv

# 활성화 (Linux/Mac)
source .venv/bin/activate

# 활성화 (Windows)
.venv\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Kaggle API 설정
```bash
# ~/.kaggle/kaggle.json 파일이 있어야 함
# Kaggle 계정 설정에서 API 토큰 다운로드 필요
chmod 600 ~/.kaggle/kaggle.json
```

### 4. 데이터 다운로드 (이미 완료됨)
```bash
kaggle competitions download -c predicting-stock-trends-rise-or-fall -p data/
cd data && unzip -q predicting-stock-trends-rise-or-fall.zip
```

## 베이스라인 모델 실행

```bash
# 베이스라인 모델 실행
python baseline.py
```

### 베이스라인 모델 특징
- **알고리즘**: Random Forest Classifier
- **Feature Engineering**:
  - 가격 기반: returns, high/low ratio, close/open ratio
  - 거래량: volume, log volume
  - 기술적 지표: 이동평균 (5, 10, 20일)
  - 기업 활동: 배당 및 주식 분할 여부
- **출력**: `outputs/submission.csv`

## 개선 아이디어

### 1. Feature Engineering
- RSI, MACD, Bollinger Bands 등 기술적 지표 추가
- 섹터별 특성 반영
- 시장 전체 트렌드 지표 추가
- 계절성 패턴 분석

### 2. 모델 개선
- XGBoost, LightGBM 앙상블
- LSTM 등 시계열 딥러닝 모델
- Feature selection 및 hyperparameter tuning
- Cross-validation 전략 개선

### 3. 데이터 처리
- 이상치 제거 및 정규화
- Missing value imputation 전략
- 시계열 특성을 고려한 train/validation split

## 주요 파일 설명

### baseline.py
- `load_data()`: 데이터 로딩
- `create_features()`: 기본 feature 생성
- `prepare_training_data()`: 학습 데이터 준비
- `train_model()`: Random Forest 모델 학습
- `prepare_test_features()`: 테스트 데이터 처리
- `main()`: 전체 파이프라인 실행

### requirements.txt
- 데이터 처리: pandas, numpy
- ML: scikit-learn, xgboost, lightgbm
- 기술적 분석: ta library
- 시각화: matplotlib, seaborn, plotly
- 유틸리티: tqdm, jupyter

## 제출 방법

1. 모델 실행 후 `outputs/submission.csv` 생성 확인
2. Kaggle 웹사이트에서 제출 또는:
```bash
kaggle competitions submit -c predicting-stock-trends-rise-or-fall -f outputs/submission.csv -m "Baseline submission"
```

## 참고사항
- 데이터 크기가 크므로 (2.2GB) 충분한 메모리 필요
- 베이스라인 모델은 간단한 시작점으로, 실제 성능 향상을 위해 개선 필요
- 시계열 특성을 고려한 validation 전략 중요

## License
Competition Rules에 따름