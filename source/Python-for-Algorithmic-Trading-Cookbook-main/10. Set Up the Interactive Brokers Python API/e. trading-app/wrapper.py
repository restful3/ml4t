from ibapi.wrapper import EWrapper


class IBWrapper(EWrapper):
    """
    Interactive Brokers API 래퍼 클래스
    
    IB API로부터 수신한 데이터를 처리하고 저장하는 역할을 담당합니다.
    """
    def __init__(self):
        """
        IBWrapper 초기화
        
        과거 데이터와 실시간 시장 데이터를 저장할 딕셔너리를 초기화합니다.
        """
        EWrapper.__init__(self)
        self.historical_data = {}  # 과거 데이터 저장용 딕셔너리
        self.market_data = {}      # 실시간 시장 데이터 저장용 딕셔너리

    def historicalData(self, request_id, bar):
        """
        과거 데이터 수신 시 호출되는 콜백 함수
        
        Args:
            request_id: 요청 식별자
            bar: 수신된 봉 데이터
        """
        bar_data = (
            bar.date,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        if request_id not in self.historical_data:
            self.historical_data[request_id] = []
        self.historical_data[request_id].append(bar_data)

    def tickPrice(self, request_id, tick_type, price, attrib):
        """
        실시간 가격 데이터 수신 시 호출되는 콜백 함수
        
        Args:
            request_id: 요청 식별자
            tick_type: 틱 데이터 유형
            price: 수신된 가격
            attrib: 가격 속성
        """
        if request_id not in self.market_data:
            self.market_data[request_id] = {}
        self.market_data[request_id][tick_type] = float(price)
