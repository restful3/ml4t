# 필요한 라이브러리 임포트
import sqlite3
import exchange_calendars as xcals
import pandas as pd
from IPython.display import Markdown, display
from openbb import obb
import os
from sys import argv

# datasets 디렉토리가 없으면 생성
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")

# OpenBB 출력 타입을 데이터프레임으로 설정
obb.user.preferences.output_type = "dataframe"

def get_stock_data(symbol, start_date=None, end_date=None):
    """
    주어진 심볼과 날짜 범위에 대한 주식 데이터를 가져오는 함수
    
    Parameters:
        symbol (str): 주식 심볼 (예: 'AAPL', 'MSFT' 등)
        start_date (str, optional): 시작 날짜. 기본값은 None
        end_date (str, optional): 종료 날짜. 기본값은 None
        
    Returns:
        pandas.DataFrame: 주식 데이터가 포함된 데이터프레임
            - date: 날짜
            - open: 시가
            - high: 고가 
            - low: 저가
            - close: 종가
            - volume: 거래량
            - symbol: 주식 심볼
    """
    # OpenBB API를 통해 주식 데이터 가져오기
    data = obb.equity.price.historical(
        symbol,
        start_date=start_date,
        end_date=end_date,
        provider="yfinance",
    )
    # 인덱스 리셋 및 심볼 컬럼 추가
    data.reset_index(inplace=True)
    data["symbol"] = symbol
    return data

def save_data_range(symbol, conn, start_date=None, end_date=None):
    """
    주어진 심볼과 날짜 범위의 주식 데이터를 SQLite 데이터베이스에 저장하는 함수
    
    Parameters:
        symbol (str): 주식 심볼 (예: 'AAPL', 'MSFT' 등)
        conn: SQLite 데이터베이스 연결 객체
        start_date (str, optional): 시작 날짜. 기본값은 None
        end_date (str, optional): 종료 날짜. 기본값은 None
    """
    # 데이터 가져와서 SQLite DB에 저장
    data = get_stock_data(symbol, start_date, end_date)
    data.to_sql("stock_data", conn, if_exists="append", index=False)


def save_last_trading_session(symbol, conn, date=None):
    """
    마지막 거래일의 주식 데이터를 SQLite 데이터베이스에 저장하는 함수
    
    Parameters:
        symbol (str): 주식 심볼 (예: 'AAPL', 'MSFT' 등)
        conn: SQLite 데이터베이스 연결 객체
        date (datetime.date, optional): 저장할 날짜. 기본값은 None
    """
    # 날짜가 지정되지 않은 경우 오늘 날짜 사용
    if date is None:
        date = pd.Timestamp.today()
    # 해당 날짜의 데이터 가져와서 SQLite DB에 저장
    data = get_stock_data(symbol, date, date)
    data.to_sql("stock_data", conn, if_exists="append", index=False)    


if __name__ == "__main__":
    try:
        # datasets 디렉토리 확인 및 생성
        if not os.path.exists("./datasets"):
            os.makedirs("./datasets")
    
        # SQLite DB 연결
        conn = sqlite3.connect("./datasets/market_data.sqlite")    

        # bulk 모드: 지정된 기간의 데이터 저장
        if argv[1] == "bulk":
            symbol = argv[2]
            start_date = argv[3]
            end_date = argv[4]
            save_data_range(symbol, conn, start_date=start_date, end_date=end_date)
            print(f"{symbol} saved between {start_date} and {end_date}")
        # last 모드: 마지막 거래일 데이터 저장
        elif argv[1] == "last":
            symbol = argv[2]
            calendar = argv[3]
            cal = xcals.get_calendar(calendar)
            today = pd.Timestamp.today().date()
            if cal.is_session(today):
                save_last_trading_session(symbol, conn, today)
                print(f"{symbol} saved")
            else:
                print(f"{today} is not a trading day. Doing nothing.")
        else:
            print("Enter bulk or last")
    finally:
        # DB 연결 종료
        conn.close()