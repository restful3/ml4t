from sys import argv
import sqlite3
import pandas as pd
import exchange_calendars as xcals
from openbb import obb
obb.user.preferences.output_type = "dataframe"
 
def get_stock_data(symbol, start_date=None, end_date=None):
    data = obb.equity.price.historical(
        symbol,
        start_date=start_date,
        end_date=end_date,
        provider="yfinance",
    )
    data.reset_index(inplace=True)
    data['symbol'] = symbol
    return data

def save_data_range(symbol, conn, start_date,
   end_date):
    data = get_stock_data(symbol, start_date,
   end_date)
    data.to_sql(
        "stock_data",
        conn,
        if_exists="replace",
        index=False
    )

def save_last_trading_session(symbol, conn, today):
    data = get_stock_data(symbol, today, today)
    data.to_sql(
        "stock_data",
        conn,
        if_exists="append",
        index=False
    )
    
if __name__ == "__main__":
    conn = sqlite3.connect("market_data.sqlite")
    if argv[1] == "bulk":
        symbol = argv[2]
        start_date = argv[3]
        end_date = argv[4]
        # save_data_range(symbol, conn, start_date=None,
        #     end_date=None)
        save_data_range(symbol, conn, start_date=start_date,
            end_date=end_date)
        print(f"{symbol} saved between {
           start_date} and {end_date}")
        # cursor = conn.cursor()
        # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # tables = cursor.fetchall()
        # print("테이블 목록:", tables)
        df = pd.read_sql_query("SELECT * FROM stock_data LIMIT 10;", conn)
        print(df)
        
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