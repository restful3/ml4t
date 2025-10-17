# 단기 반전 효과 (Short-Term Reversal Effect) 전략
# 참조: https://quantpedia.com/strategies/short-term-reversal-in-stocks/
#
# 전략 개요:
# - 투자 유니버스: 시가총액 상위 100개 기업
# - 롱 포지션: 지난 주(5일) 성과가 가장 낮은 10개 종목
# - 숏 포지션: 지난 월(21일) 성과가 가장 높은 10개 종목
# - 리밸런싱: 매주 실행
#
# QuantConnect 구현상 변경사항:

#region imports
# QuantConnect 기본 라이브러리 임포트
from AlgorithmImports import *
from pandas.core.frame import DataFrame
from typing import List, Dict
#endregion

class ShortTermReversalEffectinStocks(QCAlgorithm):
    """
    단기 반전 효과 전략을 구현하는 메인 알고리즘 클래스
    
    이 전략은 시장의 단기 반전 효과를 활용하여:
    1. 최근 성과가 나쁜 종목은 반등할 가능성이 높다고 가정 (롱 포지션), 매수 후 보유하는 전통적인 투자 방식, 주식을 사서 가격 상승을 기대하는 포지션
    2. 최근 성과가 좋은 종목은 하락할 가능성이 높다고 가정 (숏 포지션), 빌려서 먼저 팔고, 나중에 사서 갚는 투자 방식, 주식 가격이 하락할 것을 예상할 때 사용
    """

    def Initialize(self) -> None:
        """
        알고리즘 초기화 함수
        전략 파라미터, 데이터 구조, 스케줄링 등을 설정
        """
        # 백테스트 시작일과 초기 자본 설정
        self.SetStartDate(2000, 1, 1)  
        self.SetCash(100000)

        # SPY를 기준 시장으로 설정 (스케줄링 기준)
        # AddEquity: 거래할 주식 종목을 알고리즘에 추가
        market:Symbol = self.AddEquity('SPY', Resolution.Daily).Symbol
        
        # 전략 파라미터 설정
        self.fundamental_count:int = 100  # 유니버스 종목 수 (시가총액 상위 100개)
        self.fundamental_sorting_key = lambda x: x.MarketCap  # 시가총액 기준 정렬

        self.period:int = 21  # 월간 성과 계산 기간 (21영업일)
        self.week_period:int = 5  # 주간 성과 계산 기간 (5영업일)
        self.stock_selection:int = 10  # 롱/숏 각각 선택할 종목 수
        # 내 자본: $10,000
        # 1배 (레버리지 없음): $10,000로 $10,000 투자
        # 5배 레버리지: $10,000로 $50,000 투자 가능
        # 차입금: $40,000 (브로커가 대출)
        self.leverage:int = 5  # 레버리지 배수
        # $1 미만 주식 제외 → 극단적 펜니스톡 배제
        self.min_share_price:float = 1.  # 최소 주가 필터 (펜니스톡 제외)
        
        # 포트폴리오 구성 종목 리스트
        self.long:List[Symbol] = []  # 롱 포지션 종목들
        self.short:List[Symbol] = []  # 숏 포지션 종목들
        
        # 종목별 가격 데이터 저장소
        self.data:Dict[Symbol, SymbolData] = {}
        
        # 리밸런싱 스케줄 관리 변수
        self.day:int = 1  # 현재 날짜 카운터 (1~5) # 첫 거래일을 1일차로 시작
        self.selection_flag:bool = False  # 종목 선택 실행 플래그 # 리밸런싱 플래그
        
        # 유니버스 설정
        self.UniverseSettings.Resolution = Resolution.Daily # 얼마나 자주 데이터를 받을지 결정
        self.AddUniverse(self.FundamentalSelectionFunction) # QuantConnect에서 투자 유니버스를 동적으로 선택하는 핵심 함
        
        # 마진 설정 (0%로 설정하여 현금 부족 시에도 거래 가능)
        # 현금 보유 의무 없음
        # 포트폴리오 가치의 0%만 현금으로 보유하면 됨
        # 이론적으로 모든 자금을 투자에 사용 가능 
        
        # 마진 50% 설정 시 문제점:
        # 포트폴리오 가치: $100,000
        # 필요 현금: $50,000  
        # 투자 가능: $50,000 → 전략 실행 불가능

        # 마진 0% 설정으로 해결:
        # 포트폴리오 가치: $100,000  
        #  필요 현금: $0
        # 투자 가능: $100,000 → 5배 레버리지로 $500,000 투자 가능
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0.						
        
        # 매일 장 시작 후 Selection 함수 실행 스케줄링
        # QuantConnect의 스케줄링 시스템
        # 언제(When) + 무엇을(What) 실행할지 정의
        # (날짜, 시간, 실행할 함수)
        # 매일 장이 열린 직후에 Selection 함수를 자동으로 실행해라
        self.Schedule.On(self.DateRules.EveryDay(market), self.TimeRules.AfterMarketOpen(market), self.Selection)

        # 일일 정확한 종료 시간 비활성화 (성능 최적화)
        # 이미 일별 해상도 → 정확한 초/분 단위 불필요
        # 대략적인 장 마감 시간 사용  
        # 예: 2023-01-15 16:00:00 EST (초/밀리초 무시)
        # 복잡한 시간 계산 생략
        self.settings.daily_precise_end_time = False
        # 만역 True 라면,
        # 매일 정확한 장 마감 시간까지 계산
        # 예: 2023-01-15 16:00:00.000 EST (정확히 4시)
        # 모든 시간대, 공휴일, 조기 마감 등을 정밀하게 계산
        
        # True 설정:
        # 백테스트 시간: 45분
        # 메모리 사용: 높음
        # CPU 사용률: 높음

        # False 설정:  
        # 백테스트 시간: 30분 (33% 단축)
        # 메모리 사용: 보통
        # CPU 사용률: 보통

    def OnSecuritiesChanged(self, changes: SecurityChanges) -> None:
        """
        QuantConnect에서 유니버스가 변경될 때 자동으로 호출되는 콜백 함수 ... 그럼 이 함수도 자동으로 호출되게 하면 되지 않나?
        새로 추가된 증권에 대해 수수료 모델과 레버리지 설정
        changes.RemovedSecurities  # 제거된 종목들의 리스트
        # 1주차 금요일: [AAPL, MSFT, GOOGL, ..., TSLA] (20개)
        # 2주차 금요일: [AAPL, AMZN, META, ..., NVDA] (20개)
        #
        # 변경사항:
        # AddedSecurities: [AMZN, META, NVDA] (새로 추가)
        # RemovedSecurities: [MSFT, GOOGL, TSLA] (제거됨)
        """
        for security in changes.AddedSecurities: # 새로 추가된 종목들의 리스트
            # 커스텀 수수료 모델 적용 (0.005% 비례 수수료)
            security.SetFeeModel(CustomFeeModel())
            # 레버리지 설정 (5배)
            security.SetLeverage(self.leverage)
        
    def FundamentalSelectionFunction(self, fundamental: List[Fundamental]) -> List[Symbol]:
        """
        유니버스 선택 함수 - QuantConnect가 매일 자동으로 호출하여 투자 대상 종목을 선택
        
        단기 반전 효과 전략의 핵심 로직을 담당:
        1. 매일 모든 종목의 가격 데이터 업데이트
        2. 리밸런싱일에 새로운 포트폴리오 구성
        3. 주간 하락 종목(롱) + 월간 상승 종목(숏) 선택
        
        Args:
            fundamental: QuantConnect가 제공하는 모든 종목의 펀더멘털 데이터 리스트 
                        (약 3000개 미국 상장 주식의 실시간 데이터)
            
        Returns:
            선택된 종목 심볼 리스트 (롱 10개 + 숏 10개 = 총 20개)
            또는 Universe.Unchanged (리밸런싱일이 아닐 때)
        """
        
        # ==================== 1단계: 일일 데이터 업데이트 ====================
        # 매일 실행되는 핵심 작업 - 모든 종목의 가격을 롤링 윈도우에 저장
        # 이는 리밸런싱일이 아니어도 반드시 수행되어야 함 (데이터 누락 방지)
        for stock in fundamental:
            symbol:Symbol = stock.Symbol

            # 기존에 추적 중인 종목이면 오늘의 조정 주가를 롤링 윈도우에 추가
            # stock.AdjustedPrice: 배당, 분할 등을 반영한 조정 주가 (더 정확한 성과 계산)
            if symbol in self.data:
                self.data[symbol].update(stock.AdjustedPrice)

        # ==================== 2단계: 리밸런싱 여부 확인 ====================
        # self.selection_flag는 Selection() 함수에서 매주 금요일에만 True로 설정됨
        # 월~목요일에는 False이므로 기존 유니버스를 그대로 유지 (불필요한 계산 방지)
        if not self.selection_flag:
            return Universe.Unchanged  # QuantConnect 특수 상수: "유니버스 변경 없음"

        # ==================== 3단계: 기본 품질 필터링 ====================
        # 약 3000개 종목 중에서 거래 가능한 우량 종목만 1차 선별
        # 리스트 컴프리헨션으로 4가지 조건을 동시에 만족하는 종목만 추출
        selected:List[Fundamental] = [x for x in fundamental if 
            x.HasFundamentalData and        # 재무제표 등 펀더멘털 데이터가 존재하는 종목
            x.Market == 'usa' and           # 미국 주식 시장 (NYSE, NASDAQ 등)
            x.Price >= self.min_share_price and  # $1 이상 (펜니스톡 제외로 유동성 확보)
            x.MarketCap != 0]               # 시가총액 정보가 유효한 종목 (0 또는 None 제외)
        
        # ==================== 4단계: 시가총액 기준 상위 종목 선택 ====================
        # 필터링된 종목이 100개를 초과하면 시가총액 상위 100개만 선택
        # 대형주 위주로 구성하여 유동성과 안정성 확보
        if len(selected) > self.fundamental_count:  # fundamental_count = 100
            # lambda x: x.MarketCap으로 시가총액 기준 내림차순 정렬 후 상위 100개 추출
            selected = [x for x in sorted(selected, key=self.fundamental_sorting_key, reverse=True)[:self.fundamental_count]]

        # ==================== 5단계: 성과 계산 준비 ====================
        # 각 종목의 과거 성과를 저장할 딕셔너리 초기화
        month_performances:Dict[Symbol, float] = {}  # 21일(월간) 수익률 저장
        week_performances:Dict[Symbol, float] = {}   # 5일(주간) 수익률 저장

        # ==================== 6단계: 개별 종목 데이터 처리 및 성과 계산 ====================
        for stock in selected:
            symbol:Symbol = stock.Symbol

            # 신규 종목 처리: 처음 유니버스에 포함되는 종목에 대한 초기화
            if symbol not in self.data:
                # 21일 성과 = (현재가 / 21일전 가격) - 1
                # 21일 성과를 계산하려면:
                # 현재일 (0번째) : 오늘 종가
                # 1일전 (1번째) : 어제 종가  
                # 2일전 (2번째) : 그저께 종가
                # ...
                # 21일전 (21번째) : 21일 전 종가  ← 이 값이 필요!
                #
                # 필요한 날짜들 (영업일 기준):
                #  [0] 2024-01-22 (월) : $150.00  ← 현재가
                # [1] 2024-01-19 (금) : $149.50
                # [2] 2024-01-18 (목) : $148.80
                # [3] 2024-01-17 (수) : $147.90
                # ...
                # [20] 2023-12-26 (화) : $141.20
                # [21] 2023-12-22 (금) : $140.00  ← 21일 전 가격

                # 21일 성과 계산:
                # performance = ($150.00 / $140.00) - 1 = 0.0714 = 7.14%
                #
                # 22일(21+1)치 데이터를 저장할 수 있는 롤링 윈도우 생성
                # +1을 하는 이유: 21일 성과 계산 시 현재일 포함 22일 필요
                self.data[symbol] = SymbolData(self.period+1)  
                
                # QuantConnect History API로 과거 22일간의 일별 데이터 요청
                history:DataFrame = self.History(symbol, self.period+1, Resolution.Daily)
                
                # 데이터가 없으면 (신규 상장 등) 해당 종목 건너뛰기
                if history.empty:
                    self.Log(f"{symbol} 종목의 충분한 데이터가 없습니다")
                    continue
                
                # 과거 데이터 추출 및 롤링 윈도우 초기화
                closes:pd.Series = history.loc[symbol]  # MultiIndex에서 해당 종목 데이터만 추출
                
                # 과거 종가 데이터를 시간 순서대로 롤링 윈도우에 추가
                # 이렇게 해야 올바른 순서로 데이터가 쌓임 (가장 오래된 것부터)
                for time, row in closes.iterrows():
                    self.data[symbol].update(row['close'])
            
            # 성과 계산: 충분한 데이터가 쌓인 종목만 처리
            if self.data[symbol].is_ready():  # 22일치 데이터가 모두 채워졌는지 확인
                # 월간 성과: (현재가 / 21일전 가격) - 1
                month_performances[symbol] = self.data[symbol].performance(self.period)      # period = 21
                # 주간 성과: (현재가 / 5일전 가격) - 1  
                week_performances[symbol] = self.data[symbol].performance(self.week_period)  # week_period = 5

        # ==================== 7단계: 포트폴리오 구성 ====================
        # 충분한 종목이 있을 때만 포트폴리오 구성 (최소 20개 이상 필요)
        # stock_selection * 2 = 10 * 2 = 20개 (롱 10개 + 숏 10개)
        if len(month_performances) > self.stock_selection * 2:
            
            # 주간 성과 기준 오름차순 정렬 - 성과가 나쁜 순서대로 (최악 → 최고)
            # sorted()의 key=lambda로 딕셔너리의 값(성과)을 기준으로 정렬
            # x[0]로 종목 심볼만 추출 (x[1]은 성과 값)
            sorted_by_week_perf:List[Symbol] = [x[0] for x in sorted(week_performances.items(), key=lambda item: item[1])]
            
            # 월간 성과 기준 내림차순 정렬 - 성과가 좋은 순서대로 (최고 → 최악)
            # reverse=True로 높은 성과부터 정렬
            sorted_by_month_perf:List[Symbol] = [x[0] for x in sorted(month_performances.items(), key=lambda item: item[1], reverse=True)]
            
            # 단기 반전 효과 전략의 핵심 로직:
            
            # 롱 포지션: 주간 성과 최하위 10개 종목 선택, "최근 5일간 가장 많이 떨어진 종목들 매수"
            # 논리: 최근 일주일간 많이 떨어진 종목들이 과매도되어 반등할 가능성이 높음
            self.long = sorted_by_week_perf[:self.stock_selection]  # stock_selection = 10
            
            # 숏 포지션: 월간 성과 최상위 10개 종목 선택, "최근 21일간 가장 많이 오른 종목들 공매도"  
            # 논리: 최근 한 달간 많이 오른 종목들이 과매수되어 조정받을 가능성이 높음
            self.short = sorted_by_month_perf[:self.stock_selection]  # stock_selection = 10
        
        # ==================== 8단계: 결과 반환 ====================
        # 선택된 롱 + 숏 포지션 종목들을 하나의 리스트로 합쳐서 반환
        # QuantConnect는 이 리스트를 새로운 유니버스로 설정하고 OnData에서 해당 종목들의 데이터만 제공
        return self.long + self.short
    
    def OnData(self, data: Slice) -> None:
        """
        데이터 수신 시 호출되는 핵심 거래 실행 함수
        
        QuantConnect가 시장 데이터를 받을 때마다 자동으로 호출되지만,
        실제 거래는 매주 금요일(리밸런싱일)에만 실행됩니다.
        
        주요 작업:
        1. 리밸런싱 일인지 확인
        2. 기존 포지션 중 불필요한 것들 청산
        3. 새로운 롱/숏 포지션 진입
        4. 포트폴리오 가중치 균등 분배
        
        Args:
            data: QuantConnect가 제공하는 현재 시점의 시장 데이터 슬라이스
                 (현재 유니버스에 포함된 종목들의 가격, 거래량 등 포함)
        """
        
        # ==================== 1단계: 리밸런싱 일인지 확인 ====================
        # selection_flag는 Selection() 함수에서 매주 금요일에만 True로 설정됨
        # 월~목요일에는 False이므로 거래하지 않고 즉시 함수 종료
        if not self.selection_flag:
            return  # 리밸런싱일이 아니면 아무것도 하지 않음
        
        # 리밸런싱 플래그 리셋 (이번 리밸런싱 완료 후 다음 금요일까지 False 유지)
        self.selection_flag = False
        
        # ==================== 2단계: 기존 포지션 정리 ====================
        # 현재 포트폴리오에서 실제로 보유 중인 종목들을 확인
        # Portfolio는 QuantConnect의 포트폴리오 객체, 보유 종목과 수량 정보 포함
        invested: List[Symbol] = [x.Key for x in self.Portfolio if x.Value.Invested]
        
        # 기존 보유 종목 중 새로운 포트폴리오에 포함되지 않은 종목들을 청산
        for symbol in invested:
            # 새로운 롱 리스트와 숏 리스트 모두에 포함되지 않은 종목 확인
            if symbol not in self.long + self.short:
                # Liquidate(): QuantConnect API - 해당 종목의 모든 포지션을 시장가로 청산
                self.Liquidate(symbol)
                # 예: 기존에 보유했던 TSLA가 새로운 선택에서 제외되면 전량 매도

        # ==================== 3단계: 새로운 포지션 진입 ====================
        # enumerate()로 인덱스와 함께 롱/숏 포트폴리오를 순회
        # i=0: self.long (롱 포지션), i=1: self.short (숏 포지션)
        for i, portfolio in enumerate([self.long, self.short]):
            
            # 각 포트폴리오(롱 또는 숏)의 개별 종목들을 처리
            for symbol in portfolio:
                
                # 해당 종목의 데이터가 현재 data 슬라이스에 포함되어 있는지 확인
                # data[symbol]이 None이 아니고 유효한 가격 데이터가 있는지 체크
                if symbol in data and data[symbol]:
                    
                    # ==================== 4단계: 포지션 크기 계산 ====================
                    # 각 종목의 포트폴리오 가중치 계산 (동일 가중 방식)
                    # (-1)^i: i=0일 때 +1 (롱), i=1일 때 -1 (숏)
                    # len(portfolio): 해당 포트폴리오의 종목 수 (보통 10개)
                    weight = ((-1) ** i) / len(portfolio)
                    
                    # 가중치 계산 예시:
                    # 롱 포지션 (i=0): weight = (+1)^0 / 10 = +1/10 = +0.1 (10% 매수)
                    # 숏 포지션 (i=1): weight = (-1)^1 / 10 = -1/10 = -0.1 (10% 공매도)
                    
                    # ==================== 5단계: 실제 주문 실행 ====================
                    # SetHoldings(): QuantConnect API - 포트폴리오의 지정 비율만큼 해당 종목 매매
                    # 양수: 매수 (롱 포지션), 음수: 공매도 (숏 포지션)
                    self.SetHoldings(symbol, weight)
                    
                    # 예시: 포트폴리오 가치가 $100,000일 때
                    # AAPL에 weight=0.1 설정 → $10,000만큼 AAPL 매수
                    # TSLA에 weight=-0.1 설정 → $10,000만큼 TSLA 공매도

        # ==================== 6단계: 메모리 정리 ====================
        # 이번 리밸런싱에 사용된 롱/숏 종목 리스트를 초기화
        # 다음 주 금요일에 새로운 종목들로 다시 채워질 예정
        self.long.clear()   # 롱 포지션 리스트 비우기
        self.short.clear()  # 숏 포지션 리스트 비우기
        
        # 주의: 실제 포트폴리오 포지션은 유지됨 (self.Portfolio)
        # 여기서는 종목 선택 리스트만 비우는 것 (메모리 절약)
                
    def Selection(self) -> None:
        """
        매일 호출되는 스케줄 함수 - 리밸런싱 타이밍 제어
        
        매주 금요일(5일째)에 리밸런싱 플래그를 활성화하여
        다음 OnData에서 포트폴리오 재구성이 이루어지도록 함
        """
        # 5일째 (금요일)에 리밸런싱 실행
        if self.day == 5:
            self.selection_flag = True
        
        # 일일 카운터 증가
        self.day += 1
        # 5일을 넘으면 다시 1일로 리셋 (주간 사이클)
        if self.day > 5:
            self.day = 1
            
class SymbolData():
    """
    개별 종목의 가격 데이터를 관리하는 클래스
    
    롤링 윈도우를 사용하여 최근 N일간의 종가 데이터를 저장하고
    지정된 기간의 수익률을 계산하는 기능을 제공
    """
    
    def __init__(self, period:float) -> None:
        """
        SymbolData 초기화
        
        Args:
            period: 저장할 데이터 기간 (일수)
        """
        # 지정된 기간만큼의 종가 데이터를 저장하는 롤링 윈도우
        self._daily_close = RollingWindow[float](period)

    def update(self, close:float) -> None:
        """
        새로운 종가 데이터를 롤링 윈도우에 추가
        
        Args:
            close: 당일 종가
        """
        self._daily_close.Add(close)

    def is_ready(self) -> bool:
        """
        롤링 윈도우가 충분한 데이터로 채워졌는지 확인
        
        Returns:
            데이터가 충분히 쌓였으면 True, 아니면 False
        """
        return self._daily_close.IsReady
    
    def performance(self, period:int) -> float:
        """
        지정된 기간 동안의 수익률 계산
        
        Args:
            period: 수익률 계산 기간 (일수)
            
        Returns:
            수익률 (소수점, 예: 0.1 = 10% 수익)
            
        계산식: (현재가 / period일 전 가격) - 1
        """
        return self._daily_close[0] / self._daily_close[period] - 1

class CustomFeeModel(FeeModel):
    """
    커스텀 수수료 모델 클래스
    
    실제 거래 비용을 반영하기 위해 거래대금의 0.005%를 수수료로 부과
    이는 일반적인 온라인 브로커의 주식 거래 수수료 수준
    """
    
    def GetOrderFee(self, parameters):
        """
        주문 수수료 계산
        
        Args:
            parameters: 주문 파라미터 (가격, 수량 등 포함)
            
        Returns:
            계산된 수수료 정보
            
        수수료 = 주가 × 거래량 × 0.005%
        """
        fee = parameters.Security.Price * parameters.Order.AbsoluteQuantity * 0.00005
        return OrderFee(CashAmount(fee, "USD"))

# QuantConnect 기본 수수료 모델:
# - 플랫폼마다 다름
# - 때로는 비현실적으로 낮거나 높음
# - 일관성 없음

# CustomFeeModel 사용 이유:
# - 일관된 수수료 적용 (0.005%)
# - 예측 가능한 거래 비용
# - 실제 온라인 브로커 수준


# 수수료가 너무 낮으면:
# → 비현실적으로 높은 수익률
# → 실제 거래 시 실망스러운 결과

# 수수료가 너무 높으면:
# → 과도하게 보수적인 결과
# → 실행 가능한 전략도 포기할 수 있음

# CustomFeeModel:
# → 적당한 수준으로 현실적 평가

### 예시시
# 매주 금요일 거래:
# 포트폴리오 가치: $100,000
# 거래 대상: 20개 종목 × $5,000 = $100,000 거래

# CustomFeeModel 적용:
# 주간 거래 비용: $100,000 × 0.005% = $50
# 연간 거래 비용: $50 × 52주 = $2,600 (2.6%)

# 수수료 무시 시:
# 연간 수익률: 15%
# 수수료 반영 시: 15% - 2.6% = 12.4%
    
class NomuraFeeModel(FeeModel):
    def GetOrderFee(self, parameters):
        # 기본 거래 비용 요소들
        price = parameters.Security.Price
        quantity = parameters.Order.AbsoluteQuantity
        market_value = price * quantity
        
        # 1. 고정 수수료 (Brokerage Fee)
        fixed_fee = 1.0  # $1
        
        # 2. 비례 수수료 (Variable Fee)
        variable_rate = 0.0001  # 0.01%
        variable_fee = market_value * variable_rate
        
        # 3. 시장 임팩트 비용 (Market Impact) - 거래 규모에 따라
        if market_value > 1000000:  # $1M 이상
            impact_rate = 0.0005  # 0.05%
        elif market_value > 100000:  # $100K 이상
            impact_rate = 0.0002  # 0.02%
        else:
            impact_rate = 0.0001  # 0.01%
        
        impact_cost = market_value * impact_rate
        
        # 4. 총 거래 비용
        total_fee = fixed_fee + variable_fee + impact_cost
        
        return OrderFee(CashAmount(total_fee, "USD"))
    
    # 실제 거래 비용의 구성 요소:
    # 1. 기본 수수료 (Commission)
    # 2. 시장 임팩트 (Market Impact)    # 대량 거래 시 가격에 미치는 영향
    # 3. 비드-애스크 스프레드 (Spread)  # 호가 차이
    # 4. 유동성 프리미엄 (Liquidity)   # 유동성 부족 비용