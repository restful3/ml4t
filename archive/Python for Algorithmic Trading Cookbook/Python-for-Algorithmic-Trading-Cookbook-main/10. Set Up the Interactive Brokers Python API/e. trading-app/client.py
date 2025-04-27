import time
import pandas as pd

from ibapi.client import EClient

# 거래 데이터의 속성들을 정의
TRADE_BAR_PROPERTIES = ["time", "open", "high", "low", "close", "volume"]


class IBClient(EClient):
    """
    Interactive Brokers API 클라이언트 클래스
    """
    def __init__(self, wrapper):
        """
        IBClient 초기화
        Args:
            wrapper: IB API 응답을 처리하는 래퍼 객체
        """
        EClient.__init__(self, wrapper)

    # def get_historical_data(self, request_id, contract, duration, bar_size):
    #     """
    #     특정 계약의 과거 데이터를 조회하는 함수
        
    #     Args:
    #         request_id: 요청 식별자
    #         contract: 데이터를 조회할 계약 객체
    #         duration: 데이터 조회 기간
    #         bar_size: 봉 크기 설정
            
    #     Returns:
    #         DataFrame: 과거 데이터를 담은 데이터프레임
    #     """
    #     self.reqHistoricalData(
    #         reqId=request_id,
    #         contract=contract,
    #         endDateTime="",
    #         durationStr=duration,
    #         barSizeSetting=bar_size,
    #         whatToShow="MIDPOINT",
    #         useRTH=1,
    #         formatDate=1,
    #         keepUpToDate=False,
    #         chartOptions=[],
    #     )
    #     time.sleep(5)

    #     # 날짜 형식 설정
    #     bar_sizes = ["day", "D", "week", "W", "month"]
    #     if any(x in bar_size for x in bar_sizes):
    #         fmt = "%Y%m%d"
    #     else:
    #         fmt = "%Y%m%d %H:%M:%S %Z"

    #     data = self.historical_data[request_id]

    #     # 데이터프레임 생성 및 가공
    #     df = pd.DataFrame(data, columns=TRADE_BAR_PROPERTIES)
    #     df.set_index(pd.to_datetime(df.time, format=fmt), inplace=True)
    #     df.drop("time", axis=1, inplace=True)
    #     df["symbol"] = contract.symbol
    #     df.request_id = request_id

    #     return df

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
