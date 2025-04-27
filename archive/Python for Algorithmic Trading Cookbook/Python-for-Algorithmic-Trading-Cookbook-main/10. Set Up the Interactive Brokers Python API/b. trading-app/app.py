# 필요한 모듈 임포트
import threading
import time

from wrapper import IBWrapper  # IB API 래퍼 클래스
from client import IBClient   # IB API 클라이언트 클래스 
from contract import stock, future, option  # 계약 생성 함수들


class IBApp(IBWrapper, IBClient):
    def __init__(self, ip, port, client_id):
        # 부모 클래스 초기화
        IBWrapper.__init__(self)
        IBClient.__init__(self, wrapper=self)

        # IB TWS/Gateway에 연결
        self.connect(ip, port, client_id)

        # API 메시지 수신을 위한 데몬 스레드 시작
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        time.sleep(2)  # 연결 대기


if __name__ == "__main__":
    # IB 애플리케이션 인스턴스 생성 및 연결
    app = IBApp("127.0.0.1", 7497, client_id=10)

    # 다양한 계약 객체 생성
    aapl = stock("AAPL", "SMART", "USD")  # 애플 주식
    gbl = future("GBL", "EUREX", "202403")  # 유렉스 국채선물
    pltr = option("PLTR", "BOX", "20240315", 20, "C")  # Palantir 콜옵션

    # 30초 대기 후 연결 종료
    time.sleep(30)
    app.disconnect()
