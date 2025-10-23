# region imports
from AlgorithmImports import *
from Selection.ETFConstituentsUniverseSelectionModel import ETFConstituentsUniverseSelectionModel
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from umap import UMAP
# endregion

# 시드 설정으로 재현 가능한 결과 보장 / Set seed for reproducible results
np.random.seed(0)

class TopologicalGraphUniverseSelectionModel(ETFConstituentsUniverseSelectionModel):
    """
    위상 데이터 분석을 사용한 ETF 구성종목 유니버스 선택 모델 / ETF Constituent Universe Selection Model using Topological Data Analysis
    
    이 클래스는 ETF의 구성종목들을 대상으로: / This class targets ETF constituents to:
    1. 위상학적 그래프를 구축하여 주식 간의 관계를 분석 / 1. Build topological graphs to analyze relationships between stocks
    2. 클러스터링을 통해 유사한 주식들을 그룹화 / 2. Group similar stocks through clustering
    3. 정기적으로 위상학적 구조를 재구성 / 3. Periodically reconstruct topological structure
    """
    
    def __init__(self, etf_symbol: Symbol, lookback_window: int = 250, recalibration_period: timedelta = None, universe_filter_func: Callable[list[ETFConstituentUniverse], list[Symbol]] = None) -> None:
        """
        위상학적 유니버스 선택 모델 초기화 / Initialize topological universe selection model
        
        Args:
            etf_symbol: 대상 ETF 심볼 / Target ETF symbol
            lookback_window: 히스토리 데이터 룩백 기간 (기본값: 250일) / History data lookback period (default: 250 days)
            recalibration_period: 위상학적 구조 재구성 주기 / Topological structure reconstruction period
            universe_filter_func: 유니버스 필터링 함수 / Universe filtering function
        """
        self._symbol = etf_symbol
        self.lookback_window = lookback_window
        self.recalibration_period = recalibration_period
        self.clustered_symbols = None  # 클러스터링된 종목들 저장 / Store clustered symbols
        super().__init__(etf_symbol, None, universe_filter_func)

    def create_universes(self, algorithm: QCAlgorithm) -> list[Universe]:
        """
        유니버스 생성 및 초기 위상학적 그래프 구축 스케줄링 / Universe creation and initial topological graph construction scheduling
        
        Args:
            algorithm: QuantConnect 알고리즘 인스턴스 / QuantConnect algorithm instance
            
        Returns:
            list[Universe]: 생성된 유니버스 리스트 / Generated universe list
        """
        # 부모 클래스의 유니버스 생성 메서드 호출 / Call parent class universe creation method
        universe_list = super().create_universes(algorithm)
        
        # 다음 시장 개장 시간 계산 / Calculate next market open time
        next_open = algorithm.securities[self._symbol].exchange.hours.get_next_market_open(algorithm.time, False)
        
        # 다음 시장 개장일 오전 9:31에 위상학적 그래프 구축 실행 / Execute topological graph construction at 9:31 AM on next market open day
        algorithm.schedule.on(
            algorithm.date_rules.on([next_open]),
            algorithm.time_rules.at(9, 31),
            lambda: self.get_graph_symbols(algorithm)
        )
        return universe_list

    def get_graph_symbols(self, algorithm: QCAlgorithm) -> None:
        """
        위상학적 그래프 구축 및 종목 클러스터링 실행 / Execute topological graph construction and symbol clustering
        
        Args:
            algorithm: QuantConnect 알고리즘 인스턴스 / QuantConnect algorithm instance
        """
        # 단체 복합체(Simplicial Complex) 구축 / Construct simplicial complex
        graph, symbol_list = self.construct_simplicial_complex(algorithm, self.lookback_window)
        
        # 유효한 종목이 있는 경우에만 클러스터링 수행 / Perform clustering only when valid symbols are available
        if len(symbol_list) > 0:
            self.clustered_symbols = self.clustering_symbols(graph, symbol_list)
        
        # 다음 위상학적 구조 재구성을 위한 스케줄 설정 / Set schedule for next topological structure reconstruction
        algorithm.schedule.on(
            algorithm.date_rules.on([algorithm.time + timedelta(self.recalibration_period)]),
            algorithm.time_rules.at(0, 1),
            lambda: self.get_graph_symbols(algorithm)
        )

    def construct_simplicial_complex(self, algorithm: QCAlgorithm, lookback_window: int) -> tuple[dict[str, object], list[Symbol]]:
        """
        단체 복합체(Simplicial Complex) 구축 / Construct simplicial complex
        
        주식 간의 관계를 위상학적 그래프로 모델링하는 과정: / Process of modeling stock relationships as topological graphs:
        1. 히스토리 데이터 수집 및 로그 수익률 계산 / 1. Collect historical data and calculate log returns
        2. 차원 축소 (PCA + UMAP) / 2. Dimensionality reduction (PCA + UMAP)
        3. 클러스터링 (DBSCAN) / 3. Clustering (DBSCAN)
        4. 위상학적 그래프 구축 / 4. Construct topological graph
        
        Args:
            algorithm: QuantConnect 알고리즘 인스턴스 / QuantConnect algorithm instance
            lookback_window: 히스토리 데이터 룩백 기간 / History data lookback period
            
        Returns:
            tuple: (위상학적 그래프, 종목 리스트) / (Topological graph, symbol list)
        """
        # 유니버스에 선택된 종목이 없는 경우 빈 결과 반환 / Return empty result if no symbols selected in universe
        if not self.universe.selected:
            return {}, []
        # self.universe.selected.shape = (200)
        
        # 주식 간의 관계를 분석하기 위한 히스토리 데이터 수집 / Collect historical data to analyze stock relationships
        prices = algorithm.history(self.universe.selected, lookback_window, Resolution.DAILY).unstack(0).close
        # prices.shape = (150, 200)
        
        # 일별 로그 수익률 계산 (주식 간 관계 분석을 위해) / Calculate daily log returns (for stock relationship analysis)
        log_returns = np.log(prices / prices.shift(1)).dropna().T
        # log_returns.shape = (200, 149)
        if log_returns.empty:
            return {}, []

        # Kepler Mapper 알고리즘 초기화 / Initialize Kepler Mapper algorithm
        mapper = km.KeplerMapper()
        
        # 데이터를 2차원 서브스페이스로 투영 (2가지 변환 방법 사용) / Project data into 2D subspace (using 2 transformation methods)
        # PCA: 분산을 최대한 유지하면서 노이즈 제거, 빠른 처리 속도 / PCA: Retain maximum variance while denoising, fast processing
        # UMAP: 비선형 관계를 잘 처리하고 지역적/전역적 구조를 모두 보존 / UMAP: Handles non-linear relationships well and preserves both local/global structures
        # MDS와 Isomap은 금융 데이터의 노이즈와 아웃라이어에 민감할 수 있어 제외 / MDS and Isomap excluded due to potential sensitivity to noise and outliers in financial data
        projected_data = mapper.fit_transform(
            log_returns, 
            projection=[
                PCA(n_components=0.8, random_state=1), 
                UMAP(n_components=1, random_state=1, n_jobs=-1)
            ]
        )
        # projected_data.shape = (200)
        
        # DBSCAN을 사용한 클러스터링 (노이즈 처리에 우수) / Clustering using DBSCAN (excellent for noise handling)
        # 상관관계 거리를 사용하여 포트폴리오 구성에 적합한 클러스터 형성 / Use correlation distance to form clusters suitable for portfolio construction
        graph = mapper.map(projected_data, log_returns, clusterer=DBSCAN(metric='correlation', n_jobs=-1))
        
        return graph, prices.columns

    def clustering_symbols(self, graph: dict[str, object], symbol_list: list[Symbol]) -> list[list[object]]:
        """
        위상학적 그래프에서 종목 클러스터링 수행 / Perform symbol clustering from topological graph
        
        연결된 구조들을 거대 클러스터로 그룹화하고, / Group connected structures into giant clusters and
        각 노드를 실제 종목 심볼로 변환하는 과정 / convert each node to actual symbol
        
        Args:
            graph: 위상학적 그래프 (노드와 링크 정보 포함) / Topological graph (including node and link information)
            symbol_list: 종목 심볼 리스트 / Symbol list
            
        Returns:
            list[list[object]]: 중첩된 리스트 형태의 클러스터링된 종목들 / Nested list of clustered symbols
        """
        # 연결된 구조들을 거대 클러스터로 그룹화 / Group connected structures into giant clusters
        linked_clusters = []
        
        # 그래프의 링크를 순회하며 연결된 클러스터들을 찾음 / Iterate through graph links to find connected clusters
        for x, y in graph['links'].items():
            isin = False
            # 기존 클러스터 중에 현재 링크의 노드가 포함된 클러스터가 있는지 확인 / Check if current link nodes are in existing clusters
            for i in range(len(linked_clusters)):
                if x in linked_clusters[i] or y in linked_clusters[i]:
                    # 기존 클러스터에 새로운 노드들을 추가 / Add new nodes to existing cluster
                    linked_clusters[i] = list(set(linked_clusters[i] + [x] + y))
                    isin = True
            if isin:
                continue
            # 새로운 클러스터 생성 / Create new cluster
            linked_clusters.append([x] + y)
        
        # 연결되지 않은 단일 노드들을 개별 클러스터로 추가 / Add unconnected single nodes as individual clusters
        linked_clusters += [[x] for x in graph['nodes'] if x not in [z for y in linked_clusters for z in y]]
        
        # 노드 인덱스를 실제 종목 심볼로 변환 / Convert node indices to actual symbol symbols
        return [[list([symbol_list[graph['nodes'][x]]][0]) for x in linked_cluster] for linked_cluster in linked_clusters]