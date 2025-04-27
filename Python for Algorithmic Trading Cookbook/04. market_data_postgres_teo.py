from sys import argv
import pandas as pd
from sqlalchemy import create_engine, text                 # PostgreSQL 같은 데이터베이스와 연결을 설정 / 원시 SQL 쿼리를 작성하고 실행하기 위해 사용
from sqlalchemy.exc import ProgrammingError                # 잘못된 SQL 문법이나 존재하지 않는 테이블 접근 등으로 인해 발생할 수 있는 에러를 포착하여 적절히 처리
import exchange_calendars as xcals
from openbb import obb
obb.user.preferences.output_type = "dataframe"
 
def create_database_and_get_engine(db_name, base_engine):           # db 이름, db와 연결된 기본 SQLAlchemy엔진
    conn = base_engine.connect()                                    # PostgreSQL 서버에 연결을 생성
    conn = conn.execution_options(isolation_level="AUTOCOMMIT")                               # AUTOCOMMIT 모드를 활성화
                                                                    # 데이터베이스 생성과 같은 작업은 트랜잭션 없이 수행되어야 하므로 필요
    try:
        conn.execute(text(f"CREATE DATABASE {db_name};"))
                                                                    # SQL 명령문 CREATE DATABASE {db_name};를 실행하여 데이터베이스를 생성                                        
    except ProgrammingError:                                        # 이미 데이터베이스가 존재하면 ProgrammingError가 발생
        pass                                                        # 무시하고 프로그램이 계속 실행되도록 처리
    finally:
        conn.close()                                                # 연결을 닫아 리소스를 반환
    conn_str = base_engine.url.set(database=db_name)                # 기존 연결 문자열에서 database 값을 새로 생성한 데이터베이스 이름(db_name)으로 변경          
    return create_engine(conn_str)                                  # 생성된 데이터베이스와 연결된 새로운 SQLAlchemy 엔진을 생성하여여 반환


def save_data_range(symbol, engine, start_date=None, end_date=None):
    data = get_stock_data(symbol, start_date,end_date)
    data.to_sql(
        "stock_data",
        engine,
        if_exists="append",
        index=False
    )

def get_stock_data(symbol, start_date=None, end_date=None):
    data = obb.equity.price.historical(
        symbol,
        start_date=start_date,
        end_date=end_date,
        provider="yfinance",
    )
    data.reset_index(inplace=True)                              # 날짜 인덱스인 경우가 많이 때문에 날짜를 컬럼으로 포함시키기 위해....
    data['symbol'] = symbol
    return data    
    
def save_last_trading_session(symbol, engine, today):
    data = get_stock_data(symbol, today, today)
    data.to_sql(
        "stock_data",
        engine,
        if_exists="append",
        index=False
    )
    

if __name__ == "__main__":
    username = "postgres"#"postgres"
    password = "12345678"
    host = "127.0.0.1"
    port = "5432"
    database = "/market_data"
      
    DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/postgres" # PostgreSQL 연결 문자열을 생성
    base_engine = create_engine(DATABASE_URL)                                   # SQLAlchemy를 사용하여 PostgreSQL 서버와 연결된 엔진을 생성
    engine = create_database_and_get_engine("dddd_data", base_engine)                                              # base_engine을 사용해 PostgreSQL 서버에 연결
                                                                                # "stock_data" 데이터베이스를 생성
                                                                                # "stock_data" 데이터베이스와 연결된 새 엔진(engine)을 반환
    if argv[1] == "bulk":
        symbol = argv[2]
        start_date = argv[3]
        end_date = argv[4]                                                      # 원본 오타 수정정
        save_data_range(symbol, engine, start_date=start_date, end_date=end_date)   # None 제거
        print(f"{symbol} saved between {start_date} and {end_date}@@@@")
    elif argv[1] == "last":
        symbol = argv[2]
        calendar = argv[3]
        cal = xcals.get_calendar(calendar)
        today = pd.Timestamp.today().date()
        if cal.is_session(today):
            save_last_trading_session(symbol, engine, today)
            print(f"{symbol} saved")
        else:
            print(f"{today} is not a trading day. Doing nothing.")