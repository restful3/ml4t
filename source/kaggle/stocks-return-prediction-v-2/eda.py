"""
Exploratory Data Analysis for Stocks Return Prediction V2
통합된 데이터 탐색 및 분석 스크립트
실행 시 eda.md 리포트를 자동으로 생성합니다.
"""
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

# Optional visualization
try:
    import matplotlib
    matplotlib.use('Agg') # Set non-interactive backend to prevent segfaults
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib or seaborn not found. Plots will be skipped.")

warnings.filterwarnings('ignore')

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
REPORT_FILE = os.path.join(BASE_DIR, 'eda.md')
IMG_DIR = os.path.join(BASE_DIR, 'images')

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# Markdown 리포트 저장용
md_lines = []

def add_md(text):
    """마크다운 리포트에 추가"""
    md_lines.append(text)

def add_section(title, level=2):
    """섹션 헤더 추가"""
    md_lines.append(f"\n{'#' * level} {title}\n")

def safe_to_markdown(df, index=True):
    """Pandas to_markdown wrapper with fallback"""
    try:
        # Try default pandas method (needs tabulate)
        return df.to_markdown(index=index)
    except ImportError:
        # Fallback implementation
        if index:
            df = df.reset_index()
        
        columns = df.columns.tolist()
        # Header
        md = f"| {' | '.join(map(str, columns))} |\n"
        # Separator
        md += f"| {' | '.join(['---'] * len(columns))} |\n"
        # Rows
        for _, row in df.iterrows():
            md += f"| {' | '.join(map(str, row.values))} |\n"
        return md

def add_table(df, index=True):
    """DataFrame을 마크다운 테이블로 변환"""
    # Simply use safe_to_markdown
    md_lines.append(safe_to_markdown(df, index=index))
    md_lines.append("")

def add_image(filename, caption):
    """이미지 추가 (상대 경로 사용)"""
    if VISUALIZATION_AVAILABLE:
        md_lines.append(f"\n![{caption}](images/{filename})")
        md_lines.append(f"*{caption}*\n")

def interpret_skewness(skew_val):
    if skew_val > 1: return "오른쪽으로 치우침 (양의 왜도)"
    if skew_val < -1: return "왼쪽으로 치우침 (음의 왜도)"
    return "대략적으로 대칭"

def interpret_kurtosis(kurt_val):
    if kurt_val > 3: return "뾰족한 분포 (두꺼운 꼬리, 이상치 가능성 높음)"
    if kurt_val < 3: return "완만한 분포 (얇은 꼬리)"
    return "정규분포와 유사"

def interpret_correlation_strength(corr_val):
    abs_val = abs(corr_val)
    if abs_val > 0.7: return "강한 상관관계"
    if abs_val > 0.3: return "중간 상관관계"
    if abs_val > 0.1: return "약한 상관관계"
    return "매우 약함/없음"

# ============================================================================
# Main Execution
# ============================================================================

print("=" * 80)
print("STOCKS RETURN PREDICTION V2 - EDA")
print("=" * 80)

# 리포트 헤더
add_md("# Stocks Return Prediction V2 - EDA 리포트")
add_md(f"\n**생성일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
add_md(f"\n**대회 정보:** [Stocks Return Prediction V2](https://www.kaggle.com/competitions/stocks-return-prediction-v-2)")
add_md("\n---\n")

# ============================================================================
# 1. 데이터 로딩
# ============================================================================
print("\n[1/7] 데이터 로딩 중...")
add_section("1. 데이터 로딩 (Data Loading)")

try:
    with open(os.path.join(DATA_DIR, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    
    print("✓ 데이터 로딩 완료")
    add_md("✓ 데이터 파일 로딩 성공:\n")
    add_md(f"- `train_data.pkl`: {len(train_data):,} 행")
    add_md(f"- `test_data.pkl`: {len(test_data):,} 행")
    add_md(f"- `sample_submission.csv`: {len(sample_submission):,} 행\n")
    
except FileNotFoundError as e:
    print(f"데이터 로딩 오류: {e}")
    add_md(f"❌ 데이터 로딩 오류: {e}")
    # Stop execution if critical data is missing
    exit(1)

# ============================================================================
# 2. 데이터 구조 확인
# ============================================================================
print("\n[2/7] 데이터 구조 확인")
add_section("2. 데이터 구조 (Data Structure)")

add_section("2.1 학습 데이터 (Train Data)", 3)
add_md(f"- **크기 (Shape):** {train_data.shape[0]:,} 행 × {train_data.shape[1]} 열")
add_md(f"- **메모리 사용량:** {train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
add_md(f"- **컬럼 목록:** {', '.join([f'`{col}`' for col in train_data.columns])}\n")

add_md("**샘플 데이터 (상위 5행):**\n")
add_table(train_data.head())

add_md("**데이터 타입:**\n")
dtypes_df = pd.DataFrame({
    '컬럼명': train_data.dtypes.index,
    '데이터타입': train_data.dtypes.values.astype(str)
})
add_table(dtypes_df, index=False)

add_section("2.2 테스트 데이터 (Test Data)", 3)
add_md(f"- **크기 (Shape):** {test_data.shape[0]:,} 행 × {test_data.shape[1]} 열")
add_md(f"- **컬럼 목록:** {', '.join([f'`{col}`' for col in test_data.columns])}\n")
add_table(test_data.head())

# ============================================================================
# 3. 기본 통계량
# ============================================================================
# ============================================================================
# 3. 기본 통계량
# ============================================================================
print("\n[3/7] 기본 통계량 분석 (Statistical Summary)")
add_section("3. 기본 통계량 (Statistical Summary)")

# Target variable analysis
add_section("3.1 타겟 변수 (Target Variable, y)", 3)
target_col = 'y'
if target_col in train_data.columns:
    target_stats = train_data[target_col].describe()
    skew = train_data[target_col].skew()
    kurt = train_data[target_col].kurtosis()
    
    skew_desc = interpret_skewness(skew)
    kurt_desc = interpret_kurtosis(kurt)
    
    add_md(f"**통계량:**\n")
    add_md(f"- 왜도 (Skewness): {skew:.4f} ({skew_desc})")
    add_md(f"- 첨도 (Kurtosis): {kurt:.4f} ({kurt_desc})\n")
    
    target_summary = pd.DataFrame({
        '통계량': target_stats.index,
        '값': target_stats.values 
    }).T
    add_table(target_summary, index=False)
    
    # Visualization: Target Distribution
    if VISUALIZATION_AVAILABLE:
        print("타겟 분포 시각화 생성 중...")
        try:
            plt.figure(figsize=(10, 6))
            # Downsample if data is too large to prevent memory issues/crashes
            plot_data = train_data[target_col]
            if len(plot_data) > 100000:
                plot_data = plot_data.sample(100000, random_state=42)
                print(f"  (데이터가 많아 100,000개 샘플링하여 시각화합니다)")
                
            sns.histplot(plot_data, kde=True, bins=100)
            plt.title('Distribution of Target Variable (y)')
            plt.xlabel('Target Value')
            plt.ylabel('Count')
            if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)
            plt.savefig(os.path.join(IMG_DIR, 'target_dist.png'))
            plt.close()
            add_image('target_dist.png', '타겟 변수 분포 (Distribution of Target Variable)')
        except Exception as e:
            print(f"시각화 실패: {e}")

# Numeric features analysis (Dynamic selection)
add_section("3.2 수치형 특성 (Numeric Features)", 3)
numeric_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
if 'y' in numeric_features: numeric_features.remove('y')

feat_stats = []
for feat in numeric_features:
    stats = {
        '특성 (Feature)': feat,
        '평균 (Mean)': f"{train_data[feat].mean():.6f}",
        '표준편차 (Std)': f"{train_data[feat].std():.6f}",
        '최소 (Min)': f"{train_data[feat].min():.6f}",
        '최대 (Max)': f"{train_data[feat].max():.6f}",
        '결측치 (Missing)': train_data[feat].isna().sum()
    }
    feat_stats.append(stats)

add_table(pd.DataFrame(feat_stats), index=False)

# Categorical Feature Analysis (f_3)
cols_obj = train_data.select_dtypes(include=['object']).columns.tolist()
for col in cols_obj:
    add_section(f"3.3 범주형 특성 (Categorical Feature: {col})", 3)
    add_md(f"- **Train 데이터 타입:** `{train_data[col].dtype}`")
    if col in test_data.columns:
        train_type = train_data[col].dtype
        test_type = test_data[col].dtype
        if train_type != test_type:
             add_md(f"- **Test 데이터 타입:** `{test_type}` ⚠️ **타입 불일치!**")
        else:
             add_md(f"- **Test 데이터 타입:** `{test_type}`")
             
    display_top = 10
    top_vals = train_data[col].value_counts().head(display_top).reset_index()
    top_vals.columns = ['값 (Value)', '빈도 (Count)']
    top_vals['비율 (Percentage)'] = (top_vals['빈도 (Count)'] / len(train_data) * 100).round(2).astype(str) + '%'
    
    add_md(f"**상위 {display_top}개 최빈값:**\n")
    add_table(top_vals, index=False)

# ----------------------------------------------------------------------------
# [NEW] 3.4 종목별 데이터 분석 (Records per Stock Code)
# ----------------------------------------------------------------------------
if 'code' in train_data.columns:
    add_section("3.4 종목별 레코드 수 (Records per Stock)", 3)
    stock_counts = train_data['code'].value_counts()
    
    add_md(f"- **고유 종목 수:** {len(stock_counts):,} 개")
    add_md(f"- **종목별 평균 레코드 수:** {stock_counts.mean():.2f}")
    add_md(f"- **종목별 최소 레코드 수:** {stock_counts.min()}")
    add_md(f"- **종목별 최대 레코드 수:** {stock_counts.max()}")
    
    if stock_counts.nunique() == 1:
        add_md("\n✓ **모든 종목이 동일한 레코드 수를 가지고 있습니다.**")
    else:
        add_md("\n⚠️ **종목별 레코드 수가 다릅니다. (Imbalanced per stock)**\n")
        
        # Top 5 and Bottom 5 DataFrame display
        top5 = stock_counts.head(5).reset_index()
        top5.columns = ['종목코드', '레코드 수']
        
        bottom5 = stock_counts.tail(5).reset_index()
        bottom5.columns = ['종목코드', '레코드 수']
        
        add_md("**상위 5개 종목 (많은 레코드):**\n")
        add_table(top5, index=False)
        
        add_md("**하위 5개 종목 (적은 레코드):**\n")
        add_table(bottom5, index=False)

# ----------------------------------------------------------------------------
# [NEW] 3.5 시계열 데이터 확인 (Time Series Check)
# ----------------------------------------------------------------------------
date_cols = [c for c in train_data.columns if 'date' in c.lower() or 'time' in c.lower()]
if date_cols:
    add_section("3.5 시계열 데이터 확인 (Time Series)", 3)
    for d_col in date_cols:
        add_md(f"**날짜 컬럼 발견: `{d_col}`**")
        try:
             # Try converting to datetime if object/int
            if train_data[d_col].dtype == 'object':
                 temp_dates = pd.to_datetime(train_data[d_col], errors='coerce')
            else:
                 temp_dates = train_data[d_col]
            
            min_date = temp_dates.min()
            max_date = temp_dates.max()
            duration = (max_date - min_date).days if hasattr(max_date, 'days') else "N/A"
            
            add_md(f"- **시작일:** {min_date}")
            add_md(f"- **종료일:** {max_date}")
            if duration != "N/A":
                add_md(f"- **기간:** {duration} 일")
            
            add_md("")
        except Exception as e:
            add_md(f"- 날짜 파싱 중 오류 발생: {e}\n")


# ============================================================================
# 4. 데이터 품질 분석
# ============================================================================
# ============================================================================
# 4. 데이터 품질 분석
# ============================================================================
print("\n[4/7] 데이터 품질 점검 (Data Quality Check)")
add_section("4. 데이터 품질 (Data Quality)")

# Missing values
add_section("4.1 결측치 (Missing Values)", 3)
missing_train = train_data.isnull().sum()
if missing_train.sum() == 0:
    add_md("✓ **학습 데이터에 결측치가 없습니다.**\n")
else:
    add_md("**결측치가 있는 컬럼:**\n")
    missing_df = missing_train[missing_train > 0].reset_index()
    missing_df.columns = ['컬럼명', '결측치 수']
    add_table(missing_df, index=False)

# Duplicates
add_section("4.2 중복 데이터 (Duplicates)", 3)
train_dups = train_data.duplicated().sum()
test_dups = test_data.duplicated().sum()
add_md(f"- **학습 데이터 중복 수:** {train_dups}")
add_md(f"- **테스트 데이터 중복 수:** {test_dups}\n")

# Infinite values
inf_found = False
inf_report = []
for col in numeric_features:
    inf_count = np.isinf(train_data[col]).sum()
    if inf_count > 0:
        inf_report.append({'컬럼명': col, '무한대값 수': inf_count})
        inf_found = True

if inf_found:
    add_md("⚠️ **무한대 값(Infinite values) 발견:**\n")
    add_table(pd.DataFrame(inf_report), index=False)
else:
    add_md("✓ **무한대 값이 없습니다.**\n")

# ----------------------------------------------------------------------------
# [NEW] 4.3 이상치 탐지 (Outliers Detection - IQR Method)
# ----------------------------------------------------------------------------
add_section("4.3 이상치 탐지 (Outliers - IQR Method)", 3)
outlier_summary = []

for col in numeric_features:
    Q1 = train_data[col].quantile(0.25)
    Q3 = train_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_data[(train_data[col] < lower_bound) | (train_data[col] > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        outlier_percent = (outlier_count / len(train_data)) * 100
        outlier_summary.append({
            '컬럼명': col,
            '이상치 수': outlier_count,
            '비율 (%)': f"{outlier_percent:.2f}%"
        })

if outlier_summary:
    add_md("**IQR 방식으로 탐지된 이상치 요약 (상위 10개):**\n")
    outlier_df = pd.DataFrame(outlier_summary).sort_values('이상치 수', ascending=False).head(10)
    add_table(outlier_df, index=False)
else:
    add_md("✓ **IQR 기준으로 탐지된 이상치가 없습니다.**")


# ============================================================================
# 5. 특성 간 관계 분석
# ============================================================================
print("\n[5/7] 특성 관계 분석 (Feature Relationships)")
add_section("5. 특성 관계 (Feature Relationships)")

# Correlation
add_section("5.1 타겟과의 상관관계 (Correlation with Target)", 3)
correlations = train_data[numeric_features].corrwith(train_data['y']).sort_values(ascending=False)
max_corr = abs(correlations).max()
corr_strength = interpret_correlation_strength(max_corr)
add_md(f"- **최대 절대 상관계수:** {max_corr:.4f} ({corr_strength})\n")

corr_df = pd.DataFrame({
    '특성 (Feature)': correlations.index,
    '상관계수 (Correlation)': correlations.values.round(6)
}).sort_values('상관계수 (Correlation)', ascending=False)
add_table(corr_df, index=False)

if VISUALIZATION_AVAILABLE:
    print("상관관계 히트맵 생성 중...")
    try:
        plt.figure(figsize=(10, 8))
        # Correlation matrix of all numeric features + target
        # Use simple correlation on full data (usually fast enough) or sample if needed
        # For heatmap, calculating corr on full data is usually fine, but let's be safe
        if len(train_data) > 500000:
             corr_mat = train_data[numeric_features + ['y']].sample(500000, random_state=42).corr()
             print(f"  (데이터가 많아 500,000개 샘플링하여 상관관계를 계산합니다)")
        else:
             corr_mat = train_data[numeric_features + ['y']].corr()
             
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, 'correlation_heatmap.png'))
        plt.close()
        add_image('correlation_heatmap.png', '특성 상관관계 히트맵 (Feature Correlation Heatmap)')
    except Exception as e:
        print(f"시각화 실패: {e}")

# ============================================================================
# 6. Train-Test 분석
# ============================================================================
# ============================================================================
# 6. Train-Test 분석
# ============================================================================
print("\n[6/7] 학습-테스트 데이터 비교 분석 (Train-Test Analysis)")
add_section("6. 학습-테스트 데이터 분석 (Train-Test Analysis)")

# Stock overlap
if 'code' in train_data.columns and 'code' in test_data.columns:
    train_stocks = set(train_data['code'].unique())
    test_stocks = set(test_data['code'].unique())
    overlap_stocks = train_stocks & test_stocks
    cold_start_count = len(test_stocks - train_stocks)
    
    add_section("6.1 종목 중복 여부 (Stock Overlap)", 3)
    if cold_start_count > 0:
        add_md(f"⚠️ **Cold Start 문제:** 테스트 셋에만 존재하는 종목이 {cold_start_count:,}개 있습니다.\n")
    
    overlap_stats = pd.DataFrame({
        '구분': ['Train에만 있는 종목', 'Test에만 있는 종목 (Cold Start)', '양쪽에 모두 있는 종목', '중복 비율'],
        '수 (Count)': [
            f"{len(train_stocks - test_stocks):,}",
            f"{cold_start_count:,}",
            f"{len(overlap_stocks):,}",
            f"{len(overlap_stocks) / len(test_stocks) * 100:.2f}%"
        ]
    })
    add_table(overlap_stats, index=False)

# ============================================================================
# Summary & Save
# ============================================================================
print("\n" + "=" * 80)
print("eda.md 리포트 저장 중...")

report_content = "\n".join(md_lines)
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✓ EDA 리포트 저장 완료: '{REPORT_FILE}'")
if VISUALIZATION_AVAILABLE:
    print(f"✓ 이미지 저장 완료: '{IMG_DIR}'")
print("=" * 80)
print("\n✓ 분석이 완료되었습니다!")
