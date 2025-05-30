import time
import pandas as pd
from dataclasses import dataclass, field

from ibapi.client import EClient

TRADE_BAR_PROPERTIES = ["time", "open", "high", "low", "close", "volume"]


@dataclass
class Tick:
    """
    시장 데이터 틱을 나타내는 데이터 클래스
    
    Attributes:
        time: Unix 타임스탬프 형식의 시간
        bid_price: 매수 호가
        ask_price: 매도 호가  
        bid_size: 매수 호가 수량
        ask_size: 매도 호가 수량
        timestamp_: pandas Timestamp 형식으로 변환된 시간 (자동 생성)
    """
    time: int
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp_: pd.Timestamp = field(init=False)

    def __post_init__(self):
        """
        데이터 클래스 초기화 후 호출되는 메서드
        Unix 타임스탬프를 pandas Timestamp로 변환하고 데이터 타입을 변환
        """
        self.timestamp_ = pd.to_datetime(self.time, unit="s")
        self.bid_price = float(self.bid_price)
        self.ask_price = float(self.ask_price)
        self.bid_size = int(self.bid_size)
        self.ask_size = int(self.ask_size)


class IBClient(EClient):
    """IB API 클라이언트 클래스"""
    
    def __init__(self, wrapper):
        """
        IB 클라이언트 초기화
        
        Args:
            wrapper: IB API 래퍼 객체
        """
        EClient.__init__(self, wrapper)

    def get_historical_data(self, request_id, contract, duration, bar_size):
        """
        특정 계약의 과거 데이터를 조회하는 함수
        
        Args:
            request_id: 요청 식별자
            contract: 데이터를 조회할 계약 객체
            duration: 데이터 조회 기간
            bar_size: 봉 크기 설정
            
        Returns:
            DataFrame: 과거 데이터를 담은 데이터프레임
        """
        self.reqHistoricalData(
            reqId=request_id,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="MIDPOINT",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )
        time.sleep(5)

        # 날짜 형식 설정
        bar_sizes = ["day", "D", "week", "W", "month"]
        if any(x in bar_size for x in bar_sizes):
            fmt = "%Y%m%d"
        else:
            fmt = "%Y%m%d  %H:%M:%S"  # 공백 2개로 수정하고 %Z 제거

        data = self.historical_data[request_id]

        # 데이터프레임 생성 및 가공
        df = pd.DataFrame(data, columns=TRADE_BAR_PROPERTIES)
        df.set_index(pd.to_datetime(df.time, format=fmt), inplace=True)
        df.drop("time", axis=1, inplace=True)
        df["symbol"] = contract.symbol
        df.request_id = request_id

        return df

    def get_historical_data_for_many(
        self, request_id, contracts, duration, bar_size, col_to_use="close"
    ):
        """
        여러 계약의 과거 데이터를 한번에 조회하는 함수
        
        Args:
            request_id: 시작 요청 식별자
            contracts: 데이터를 조회할 계약 객체 리스트
            duration: 데이터 조회 기간
            bar_size: 봉 크기 설정
            col_to_use: 피벗할 때 사용할 컬럼 (기본값: 'close')
            
        Returns:
            DataFrame: 여러 계약의 과거 데이터를 피벗한 데이터프레임
        """
        dfs = []
        for contract in contracts:
            data = self.get_historical_data(request_id, contract, duration, bar_size)
            dfs.append(data)
            request_id += 1
        return (
            pd.concat(dfs)
            .reset_index()
            .pivot(index="time", columns="symbol", values=col_to_use)
        )

    def get_market_data(self, request_id, contract, tick_type=4):
        """
        실시간 시장 데이터를 조회하는 함수
        
        Args:
            request_id: 요청 식별자
            contract: 데이터를 조회할 계약 객체
            tick_type: 틱 데이터 유형 (기본값: 4, 최종거래가)
            
        Returns:
            float: 요청한 시장 데이터 값
        """
        self.reqMktData(
            reqId=request_id,
            contract=contract,
            genericTickList="4",
            snapshot=True,
            regulatorySnapshot=False,
            mktDataOptions=[],
        )
        time.sleep(5)

        self.cancelMktData(reqId=request_id)

        return self.market_data[request_id][tick_type]

    def get_streaming_data(self, request_id, contract):
        """
        실시간 틱 데이터 스트리밍을 시작하는 함수
        
        Args:
            request_id: 요청 식별자
            contract: 데이터를 조회할 계약 객체
            
        Yields:
            Tick: 실시간으로 수신된 틱 데이터 객체
        """
        # 틱 단위 데이터 요청 설정
        self.reqTickByTickData(
            reqId=request_id,
            contract=contract,
            tickType="BidAsk",  # 매수/매도 호가 데이터 요청
            numberOfTicks=0,     # 0: 스트리밍 모드
            ignoreSize=True,     # 크기 업데이트 무시
        )
        time.sleep(10)  # 초기 데이터 수신 대기

        # 스트리밍 데이터 지속적 모니터링
        while True:
            if self.stream_event.is_set():
                yield Tick(*self.streaming_data[request_id])  # 새로운 틱 데이터 반환
                self.stream_event.clear()  # 이벤트 플래그 초기화

    def stop_streaming_data(self, request_id):
        """
        실시간 틱 데이터 스트리밍을 중지하는 함수
        
        Args:
            request_id: 중지할 스트리밍 요청의 식별자
        """
        self.cancelTickByTickData(reqId=request_id)
