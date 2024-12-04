# 필요한 라이브러리 임포트
import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import cufflinks as cf
from plotly.offline import iplot

# cufflinks의 오프라인 모드 설정
cf.go_offline()

# 데이터 관련 함수 정의
@st.cache
def get_sp500_components():
    # S&P 500 기업 목록을 Wikipedia에서 가져오기
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = df[0]
    tickers = df["Symbol"].to_list()  # 티커 목록 추출
    tickers_companies_dict = dict(
        zip(df["Symbol"], df["Security"])  # 티커와 회사 이름 매핑
    )
    return tickers, tickers_companies_dict

@st.cache
def load_data(symbol, start, end):
    # 주어진 티커의 주식 데이터 다운로드
    return yf.download(symbol, start, end)

@st.cache
def convert_df_to_csv(df):
    # DataFrame을 CSV 형식으로 변환
    return df.to_csv().encode("utf-8")

# 사이드바 설정
st.sidebar.header("주식 매개변수")

available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "티커", 
    available_tickers, 
    format_func=tickers_companies_dict.get
)
start_date = st.sidebar.date_input(
    "시작 날짜", 
    datetime.date(2019, 1, 1)
)
end_date = st.sidebar.date_input(
    "종료 날짜", 
    datetime.date.today()
)

if start_date > end_date:
    st.sidebar.error("종료 날짜는 시작 날짜 이후여야 합니다")

# 기술 분석을 위한 입력
st.sidebar.header("기술 분석 매개변수")

volume_flag = st.sidebar.checkbox(label="거래량 추가")

exp_sma = st.sidebar.expander("단순 이동 평균")
sma_flag = exp_sma.checkbox(label="단순 이동 평균 추가")
sma_periods= exp_sma.number_input(
    label="SMA 기간", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)

exp_bb = st.sidebar.expander("볼린저 밴드")
bb_flag = exp_bb.checkbox(label="볼린저 밴드 추가")
bb_periods= exp_bb.number_input(label="BB 기간", 
                                min_value=1, max_value=50, 
                                value=20, step=1)
bb_std= exp_bb.number_input(label="표준편차 수", 
                            min_value=1, max_value=4, 
                            value=2, step=1)

exp_rsi = st.sidebar.expander("상대 강도 지수")
rsi_flag = exp_rsi.checkbox(label="RSI 추가")
rsi_periods= exp_rsi.number_input(
    label="RSI 기간", 
    min_value=1, 
    max_value=50, 
    value=20, 
    step=1
)
rsi_upper= exp_rsi.number_input(label="RSI 상한", 
                                min_value=50, 
                                max_value=90, value=70, 
                                step=1)
rsi_lower= exp_rsi.number_input(label="RSI 하한", 
                                min_value=10, 
                                max_value=50, value=30, 
                                step=1)

# 메인 본문
st.title("기술 분석을 위한 간단한 웹 앱")
st.write("""
### 사용자 매뉴얼
* S&P 지수의 구성 요소인 회사 중 하나를 선택할 수 있습니다
* 관심 있는 기간을 선택할 수 있습니다
* 선택한 데이터를 CSV 파일로 다운로드할 수 있습니다
* 다음 기술 지표를 플롯에 추가할 수 있습니다: 단순 이동 평균, 볼린저 밴드, 상대 강도 지수
* 지표의 다양한 매개변수를 실험할 수 있습니다
""")

# 선택된 주식 데이터 로드
# load_data 함수를 사용하여 선택된 티커, 시작 날짜, 종료 날짜에 해당하는 주식 데이터를 가져옵니다.
df = load_data(ticker, start_date, end_date)

# 데이터 미리보기 섹션
# st.expander를 사용하여 접을 수 있는 섹션을 생성합니다.
data_exp = st.expander("데이터 미리보기")

# 데이터프레임의 모든 열 이름을 가져옵니다.
available_cols = df.columns.tolist()

# 사용자가 표시할 열을 선택할 수 있는 다중 선택 위젯을 생성합니다.
# 기본값으로 모든 열이 선택되어 있습니다.
columns_to_show = data_exp.multiselect(
    "열", 
    available_cols, 
    default=available_cols
)

# 선택된 열만 포함하는 데이터프레임을 표시합니다.
data_exp.dataframe(df[columns_to_show])

# CSV 다운로드 버튼
# 선택된 열의 데이터를 CSV 형식으로 변환합니다.
csv_file = convert_df_to_csv(df[columns_to_show])

# 다운로드 버튼을 생성합니다. 클릭하면 CSV 파일이 다운로드됩니다.
data_exp.download_button(
    label="선택한 데이터를 CSV로 다운로드",
    data=csv_file,
    file_name=f"{ticker}_주가.csv",
    mime="text/csv",
)

# 기술 분석 플롯 생성
# 차트의 제목을 설정합니다. 회사 이름과 "의 주가"를 조합합니다.
title_str = f"{tickers_companies_dict[ticker]}의 주가"

# cufflinks의 QuantFig 객체를 생성합니다. 이는 기술적 분석 차트를 그리는 데 사용됩니다.
qf = cf.QuantFig(df, title=title_str)

# 사용자 선택에 따라 지표를 추가합니다.
# 거래량 추가
if volume_flag:
    qf.add_volume()

# 단순 이동 평균(SMA) 추가
if sma_flag:
    qf.add_sma(periods=sma_periods)

# 볼린저 밴드 추가
if bb_flag:
    qf.add_bollinger_bands(periods=bb_periods,
                           boll_std=bb_std)

# 상대 강도 지수(RSI) 추가
if rsi_flag:
    qf.add_rsi(periods=rsi_periods,
               rsi_upper=rsi_upper,
               rsi_lower=rsi_lower,
               showbands=True)

# 플롯 생성 및 표시
# iplot 메서드를 사용하여 인터랙티브 차트를 생성합니다.
fig = qf.iplot(asFigure=True)

# Streamlit의 plotly_chart 함수를 사용하여 생성된 차트를 웹 앱에 표시합니다.
st.plotly_chart(fig)