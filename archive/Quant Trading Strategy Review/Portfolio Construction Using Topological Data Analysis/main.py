# region imports
from AlgorithmImports import *
from universe import TopologicalGraphUniverseSelectionModel
# endregion

# 시드 설정으로 재현 가능한 결과 보장 / Set seed for reproducible results
np.random.seed(0)

class TopologicalPortfolio(QCAlgorithm):
    """
    위상 데이터 분석(Topological Data Analysis)을 사용한 포트폴리오 구성 알고리즘
    Portfolio Construction Algorithm using Topological Data Analysis (TDA)
    
    이 알고리즘은 SPY ETF의 구성종목들을 대상으로:
    This algorithm targets SPY ETF constituents to:
    1. 위상학적 그래프를 구축하여 주식 간의 관계를 분석
       1. Build topological graphs to analyze relationships between stocks
    2. 클러스터링을 통해 유사한 주식들을 그룹화
       2. Group similar stocks through clustering
    3. 계층적 가중치 분배를 통해 포트폴리오를 구성
       3. Construct portfolio through hierarchical weight distribution
    """
    
    def initialize(self) -> None:
        """알고리즘 초기화 설정 / Algorithm initialization settings"""
        # 백테스팅 기간 설정 (5년간) / Set backtesting period (5 years)
        self.set_start_date(2020, 3, 25)
        self.set_end_date(2025, 3, 25)
        # 초기 자본금 설정 ($1,000,000) / Set initial capital ($1,000,000)
        self.set_cash(1000000)

        # SPY ETF를 벤치마크로 설정하여 성과 비교 / Set SPY ETF as benchmark for performance comparison
        spy = self.add_equity("SPY").symbol
        self.set_benchmark(spy)

        # 위상학적 구조 분석을 위한 히스토리 룩백 기간 (기본값: 150일) / History lookback period for topological structure analysis (default: 150 days)
        history_lookback = self.get_parameter("history_lookback", 150)
        # 위상학적 복합체 재구성 주기 (기본값: 125일) / Topological complex reconstruction period (default: 125 days)
        recalibrate_period = self.get_parameter("recalibrate_period", 125)
        
        # SPY 구성종목들을 대상으로 하는 위상학적 유니버스 선택 모델 생성 / Create topological universe selection model for SPY constituents
        self.universe_model = TopologicalGraphUniverseSelectionModel(
            spy,
            history_lookback,
            recalibrate_period,
            # 유니버스 필터 함수: 가중치가 있는 종목들 중 상위 200개 선택 / Universe filter function: select top 200 stocks with weights
            lambda u: [x.symbol for x in sorted(
                [x for x in u if x.weight], 
                key=lambda x: x.weight, 
                reverse=True
            )[:200]]
        )
        self.add_universe_selection(self.universe_model)

        # 매일 오전 9:31에 포트폴리오 리밸런싱 실행 / Execute portfolio rebalancing daily at 9:31 AM
        self.schedule.on(self.date_rules.every_day(spy), self.time_rules.at(9, 31), self.rebalance)

        # 유니버스 선택을 위한 1년간 워밍업 설정 / Set 1-year warm-up for universe selection
        self.set_warm_up(timedelta(365))

    def rebalance(self) -> None:
        """
        포트폴리오 리밸런싱 실행 / Execute portfolio rebalancing
        
        클러스터링된 종목들이 있을 경우: / When clustered symbols are available:
        1. 각 종목에 대한 가중치를 계산 / 1. Calculate weights for each stock
        2. 계산된 가중치로 포트폴리오를 재구성 / 2. Reconstruct portfolio with calculated weights
        """
        if self.universe_model.clustered_symbols:
            # 클러스터링된 종목들에 대한 가중치 분배 계산 / Calculate weight distribution for clustered symbols
            weights = self.weight_distribution(self.universe_model.clustered_symbols)
            # 계산된 가중치로 포트폴리오 재구성 (기존 포지션 청산 후) / Reconstruct portfolio with calculated weights (after liquidating existing positions)
            self.set_holdings([PortfolioTarget(symbol, weight) for symbol, weight in weights.items()], liquidate_existing_holdings=True)

    def weight_distribution(self, clustered_symbols):
        """
        계층적 클러스터 구조를 고려한 가중치 분배 계산 / Calculate weight distribution considering hierarchical cluster structure
        
        Args:
            clustered_symbols: 중첩된 리스트 형태의 클러스터링된 종목들 / Nested list of clustered symbols
            
        Returns:
            pd.Series: 각 종목에 대한 정규화된 가중치 / Normalized weights for each symbol
            
        Note:
            - 거대 클러스터와 작은 클러스터 간의 가중치 분배 / Weight distribution between giant and small clusters
            - 클러스터 내부에서의 가중치 분배 / Weight distribution within clusters
            - 아웃라이어는 투자하지 않음 / No investment in outliers
        """
        weights = {}
        
        def assign_weights(nested_list, level=1):
            """
            재귀적으로 중첩된 클러스터 구조에 가중치 할당 / Recursively assign weights to nested cluster structure
            
            Args:
                nested_list: 중첩된 클러스터 리스트 / Nested cluster list
                level: 현재 클러스터 레벨 (깊이에 따라 가중치 감소) / Current cluster level (weight decreases with depth)
            """
            num_elements = len(nested_list)
            if num_elements == 0:
                return
            # 각 요소에 동일한 가중치 할당 / Assign equal weight to each element
            weight_per_element = 1 / num_elements
            
            for item in nested_list:
                if isinstance(item, list):
                    # 중첩된 리스트인 경우 재귀적으로 처리 / Process nested lists recursively
                    assign_weights(item, level + 1)
                else:
                    # 개별 종목인 경우 가중치 할당 (레벨이 깊을수록 가중치 감소) / Assign weight to individual stocks (weight decreases with level depth)
                    weights[item] = weights.get(item, 0) + weight_per_element / (2 ** (level - 1))
        
        # 전체 클러스터 구조에 대해 가중치 계산 / Calculate weights for entire cluster structure
        assign_weights(clustered_symbols)
        # 가중치 정규화 (합이 1이 되도록) / Normalize weights (sum to 1)
        return pd.Series(weights) / sum(weights.values())