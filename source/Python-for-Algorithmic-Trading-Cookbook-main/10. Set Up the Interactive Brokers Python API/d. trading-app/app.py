import threading
import time

from wrapper import IBWrapper
from client import IBClient
from contract import stock, future, option

import pandas as pd

class IBApp(IBWrapper, IBClient):
    """
    Interactive Brokers API 애플리케이션 클래스
    
    IBWrapper와 IBClient를 상속받아 IB API와의 연결 및 데이터 처리를 담당합니다.
    """
    def __init__(self, ip, port, client_id):
        """
        IBApp 초기화
        
        Args:
            ip: TWS/IB Gateway 접속 IP
            port: TWS/IB Gateway 접속 포트
            client_id: 클라이언트 식별자
        """
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)

        # TWS/IB Gateway에 연결
        self.connect(ip, port, client_id)

        # API 메시지 처리를 위한 데몬 스레드 시작
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        time.sleep(2)  # 연결 설정을 위한 대기 시간


if __name__ == "__main__":
    # 애플리케이션 인스턴스 생성
    app = IBApp("127.0.0.1", 7497, client_id=10)

    # 다양한 종류의 계약 객체 생성
    aapl = stock("AAPL", "SMART", "USD")  # 애플 주식
    gbl = future("GBL", "EUREX", "202403")  # 유로 국채 선물
    pltr = option("PLTR", "BOX", "20240315", 20, "C")  # Palantir 콜옵션

    # 애플 주식의 과거 데이터 조회
    data = app.get_historical_data(
        request_id=99, contract=aapl, duration="2 D", bar_size="30 secs"
    )

    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'wap', 'count'])
    
    # DataFrame을 CSV 파일로 저장
    df.to_csv('aapl_historical_data.csv', index=True)

    # 데이터 수신을 위한 대기
    time.sleep(30)
    # TWS/IB Gateway 연결 종료
    app.disconnect()
