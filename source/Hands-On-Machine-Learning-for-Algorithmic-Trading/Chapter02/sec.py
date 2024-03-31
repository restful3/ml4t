from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time
import pandas as pd
from datetime import date

def download_financial_statement(filing, filename):
    
    download_directory = '/home/restful3/datasets_bigpond/ml4t/sec_fs'

    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory' : download_directory}
    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # 웹 페이지 열기
        driver.get("https://www.sec.gov/dera/data/financial-statement-and-notes-data-set")

        # 파일 다운로드 요소 찾기
        print(f'{filing} 받기 시작')

        download_link = driver.find_element(By.LINK_TEXT, filing)
        downloaded_file_path = os.path.join(download_directory, filename)
        if not os.path.exists(downloaded_file_path):
            # 다운로드 링크 클릭
            download_link.click()
            timeout = 60  # 최대 대기 시간 (초)
            start_time = time.time()
            while not os.path.exists(downloaded_file_path):
                time.sleep(1)
                if time.time() - start_time > timeout:
                    print("다운로드 시간 초과")
                    break       

            print(f'{filing} 받기 완료')
            time.sleep(3)
        else:
            print(f'{filing}는 존재하여 skip')

    except Exception as e:
        print(f"다운로드 중 오류 발생: {e}")

    finally:
        # 브라우저 닫기
        driver.quit()    


from datetime import datetime, timedelta

def generate_quarters(start_year, end_year):
    for year in range(start_year, end_year + 1):
        end_quarter = 5 if year < end_year else datetime.now().month // 4   # 현재 분기보다 더 큰 분기가 있으면 현재 분기를 설정
        for quarter in range(1, end_quarter):
            filing = f"{year} Q{quarter}"
            filename = f"{year}q{quarter}_notes.zip" 
            yield filing, filename

def generate_months(start_year, start_month, end_year, end_month, exclude_year, exclude_month):
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    while start_date <= end_date:
        if start_date.year != exclude_year or start_date.month != exclude_month:
            year = start_date.strftime("%Y")
            month = start_date.strftime("%m")
            
            filing = f"{year} {month}"
            filename = f"{year}_{month}_notes.zip"           
            yield filing, filename
        start_date += timedelta(days=32)  # 한 달 더하기
        start_date = start_date.replace(day=1)  # 다음 달의 1일로 설정

# 2009년부터 2020년 Q3까지 출력
for filing, filename in generate_quarters(2009, 2021):
    download_financial_statement(filing, filename)

# 2020년 10월부터 전 달까지 출력 (2024년 3월은 제외)
current_year = datetime.now().year
current_month = datetime.now().month
previous_month = current_month - 1 if current_month != 1 else 12
previous_year = current_year - 1 if current_month == 1 else current_year
for filing, filename in generate_months(2020, 10, current_year, current_month, 2024, 3):
    download_financial_statement(filing, filename)