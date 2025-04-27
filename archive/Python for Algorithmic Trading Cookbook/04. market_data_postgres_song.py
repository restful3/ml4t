# 필요한 라이브러리 임포트
import exchange_calendars as xcals
import pandas as pd
from IPython.display import Markdown, display
from openbb import obb
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from sys import argv

def create_database_and_get_engine(db_name, base_engine):
    """
    데이터베이스를 생성하고 해당 데이터베이스에 대한 엔진을 반환하는 함수
    
    매개변수:
        db_name (str): 생성할 데이터베이스 이름
        base_engine: 기본 데이터베이스 연결을 위한 SQLAlchemy 엔진
        
    반환값:
        Engine: 새로 생성된 데이터베이스에 대한 SQLAlchemy 엔진
    """
    # 기본 데이터베이스에 연결
    conn = base_engine.connect()
    # AUTOCOMMIT 모드로 설정
    conn = conn.execution_options(isolation_level="AUTOCOMMIT")

    try:
        # 데이터베이스 생성 시도
        conn.execute(text(f"CREATE DATABASE {db_name};"))
    except ProgrammingError:
        # 이미 데이터베이스가 존재하는 경우 무시
        pass
    finally:
        # 연결 종료
        conn.close()

    # 새 데이터베이스에 대한 연결 문자열 생성
    conn_str = base_engine.url.set(database=db_name)

    # 새 데이터베이스에 대한 엔진 반환
    return create_engine(conn_str)

def get_stock_data(symbol, start_date=None, end_date=None):
    """
    주어진 심볼과 날짜 범위에 대한 주식 데이터를 가져오는 함수
    
    매개변수:
        symbol (str): 주식 심볼
        start_date (str, optional): 시작 날짜. 기본값은 None
        end_date (str, optional): 종료 날짜. 기본값은 None
        
    반환값:
        DataFrame: 주식 데이터가 포함된 데이터프레임
    """
    # yfinance를 통해 주식 데이터 가져오기
    data = obb.equity.price.historical(
        symbol,
        start_date=start_date,
        end_date=end_date,
        provider="yfinance",
    )
    # 인덱스를 컬럼으로 변환
    data.reset_index(inplace=True)
    # 심볼 정보 추가
    data["symbol"] = symbol
    return data

def save_data_range(symbol, engine, start_date=None, end_date=None):
    """
    주어진 심볼과 날짜 범위의 주식 데이터를 데이터베이스에 저장하는 함수
    
    매개변수:
        symbol (str): 주식 심볼
        engine: SQLAlchemy 엔진 객체
        start_date (str, optional): 시작 날짜. 기본값은 None
        end_date (str, optional): 종료 날짜. 기본값은 None
    """
    # 데이터 가져와서 DB에 저장
    data = get_stock_data(symbol, start_date, end_date)
    data.to_sql("stock_data", engine, if_exists="append", index=False)

def save_last_trading_session(symbol, engine):
    """
    마지막 거래 세션의 주식 데이터를 데이터베이스에 저장하는 함수
    
    매개변수:
        symbol (str): 주식 심볼
        engine: SQLAlchemy 엔진 객체
    """
    # 오늘 날짜의 데이터 가져오기
    today = pd.Timestamp.today()
    data = get_stock_data(symbol, today, today)
    data.to_sql("stock_data", engine, if_exists="append", index=False)

# 메인 실행 부분
if __name__ == "__main__":
    # OpenBB 출력 타입을 데이터프레임으로 설정
    obb.user.preferences.output_type = "dataframe"

    # 데이터베이스 접속 정보 설정
    username = "admin"
    password = "postgre"
    host = "127.0.0.1"
    port = "5432"
    database = "/market_data"

    # PostgreSQL 데이터베이스 연결 URL 생성
    DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/postgres"
    # SQLAlchemy 엔진 생성
    base_engine = create_engine(DATABASE_URL)
    
    # DB 연결 및 엔진 생성
    DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/postgres"
    base_engine = create_engine(DATABASE_URL)
    engine = create_database_and_get_engine("stock_data", base_engine)
    
    # 명령행 인자에 따른 처리
    if argv[1] == "bulk":  # bulk 모드: 기간 데이터 저장
        symbol = argv[2]
        start_date = argv[3]
        end_date = argv[4]
        save_data_range(symbol, engine, start_date=start_date, end_date=end_date)
        print(f"{symbol} saved between {start_date} and {end_date}")
    elif argv[1] == "last":  # last 모드: 마지막 거래일 데이터 저장
        symbol = argv[2]
        calendar = argv[3]
        cal = xcals.get_calendar(calendar)
        today = pd.Timestamp.today().date()
        if cal.is_session(today):
            save_last_trading_session(symbol, engine)
            print(f"{symbol} saved")
        else:
            print(f"{today} is not a trading day. Doing nothing.")        