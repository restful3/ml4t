`research.ipynb` 파일을 분석해서 설명드리겠습니다. 먼저 파일을 읽어보겠습니다.


[1 tool called]

파일이 잘렸네요. 전체 파일을 읽어보겠습니다.


[1 tool called]

파일이 더 길어 보이니 계속 읽어보겠습니다.


[1 tool called]

전체 파일을 다시 읽어보겠습니다.


[1 tool called]

파일이 짧네요. `research.html` 파일도 확인해보겠습니다.


[1 tool called]

HTML 파일이 너무 길어서 노트북의 실제 내용을 찾기 어렵네요. 노트북 파일의 구조를 다시 확인해보겠습니다.


[1 tool called]

노트북에 6개의 셀이 있네요. 전체 내용을 다시 읽어보겠습니다.


[1 tool called]

노트북의 나머지 셀들을 확인해보겠습니다.


[1 tool called]

이제 `research.ipynb` 파일의 전체 내용을 파악했습니다. 분석해드리겠습니다.

## research.ipynb 분석

이 Jupyter 노트북은 **위상 데이터 분석(Topological Data Analysis)을 사용한 포트폴리오 구성 연구**를 위한 실험적 분석 도구입니다. `main.py`와 `universe.py`에서 구현된 알고리즘의 이론적 배경과 시각화를 제공합니다.

### 노트북 구조:

#### **Cell 0: 헤더**
- QuantConnect 로고와 구분선
- 연구의 정체성을 나타내는 시각적 요소

#### **Cell 1: 라이브러리 임포트**
```python
import kmapper as km          # 위상 데이터 분석을 위한 Kepler Mapper
import networkx as nx         # 그래프 시각화
from persim import plot_diagrams, wasserstein  # 지속성 다이어그램
from ripser import Rips, ripser               # 지속성 호몰로지
from scipy.signal import find_peaks           # 신호 처리
from sklearn.cluster import DBSCAN            # 클러스터링
from sklearn.decomposition import PCA         # 차원 축소
from sklearn.manifold import TSNE             # 차원 축소
from statsmodels.graphics.tsaplots import plot_acf  # 시계열 분석
from statsmodels.tsa.stattools import acf     # 자기상관 함수
import umap                                   # 차원 축소
```

**주요 라이브러리들:**
- **위상 데이터 분석**: `kmapper`, `ripser`, `persim`
- **머신러닝**: `sklearn` (PCA, DBSCAN, TSNE)
- **그래프 분석**: `networkx`
- **시계열 분석**: `statsmodels`

#### **Cell 2: 데이터 수집**
```python
qb = QuantBook()  # QuantConnect 연구 환경
start_date = datetime(2020, 1, 1)
qb.set_start_date(start_date)

# QQQ ETF 구성종목들 수집
universe = qb.add_universe(qb.universe.etf("QQQ", Market.USA))
universe_history = qb.universe_history(universe, start_date-timedelta(1), start_date)
symbols = [x.symbol for x in universe_history.iloc[-1]]

# 200일간의 일별 가격 데이터 수집
prices = qb.history(symbols, 200, Resolution.DAILY).unstack(0).close
# 로그 수익률 계산 (주식 간 관계 분석용)
log_returns = np.log(prices / prices.shift(1)).dropna().T
```

**데이터 수집 과정:**
- **대상**: QQQ ETF 구성종목들 (SPY 대신 QQQ 사용)
- **기간**: 200일간의 일별 데이터
- **전처리**: 로그 수익률 계산 및 전치 (주식 간 관계 분석을 위해)

#### **Cell 3: 단체 복합체 구축**
```python
def construct_simplicial_complex(log_returns):
    mapper = km.KeplerMapper()
    # PCA + UMAP을 사용한 차원 축소
    projected_data = mapper.fit_transform(
        log_returns, 
        projection=[
            PCA(n_components=0.85, random_state=321), 
            umap.UMAP(n_components=1, random_state=124)
        ]
    )
    # DBSCAN을 사용한 클러스터링
    graph = mapper.map(projected_data, log_returns, clusterer=DBSCAN(metric='correlation', n_jobs=-1))
    return projected_data, graph
```

**알고리즘 파라미터:**
- **PCA**: 85% 분산 유지 (main.py의 80%와 약간 다름)
- **UMAP**: 1차원으로 축소
- **DBSCAN**: 상관관계 거리 사용

#### **Cell 4: 시각화**
```python
fig = plt.figure(figsize=(10, 6))

# 왼쪽: 상세한 로그 수익률 관계 그래프
ax = plt.subplot(121)
G = nx.from_dict_of_lists({x: [z for z in y if z != x] for y in graph['nodes'].values() for x in y})
pos = nx.spring_layout(G, k=0.2, iterations=15, seed=74)
# 클러스터별로 색상 구분하여 노드 그리기
for nodes, color in zip(list(graph['nodes'].values()), 
                       ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
                        "tab:pink", "tab:brown", "tab:cyan", "tab:grey", "yellow")):
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=25, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
plt.title("Detailed Graph of Log-Return Relation Between QQQ Constituents")

# 오른쪽: 단체 복합체 시각화
ax2 = fig.add_subplot(122)
nodes = graph["nodes"].keys()
edges = [[start, end] for start, ends in graph["links"].items() for end in ends]
g = nx.Graph()
g.add_nodes_from(nodes)
nx.set_node_attributes(g, dict(graph["nodes"]), "membership")
g.add_edges_from(edges)
pos = nx.spring_layout(g, k=1, iterations=30, seed=435)
nodes = nx.draw_networkx_nodes(g, node_size=300, pos=pos, ax=ax2)
nx.draw_networkx_edges(g, pos=pos, ax=ax2, arrows=False)
ax2.set_title("Simplicial Complexes")
```

**시각화 내용:**
1. **상세 그래프**: QQQ 구성종목들 간의 로그 수익률 관계를 클러스터별로 색상 구분하여 표시
2. **단체 복합체**: 위상학적 구조를 단순화하여 표시

#### **Cell 5: 빈 셀**
- 추가 분석이나 실험을 위한 공간

### 연구 노트북의 목적:

1. **알고리즘 검증**: `main.py`와 `universe.py`에서 구현된 알고리즘의 이론적 배경 검증
2. **시각적 분석**: 위상학적 그래프와 클러스터링 결과를 시각적으로 확인
3. **파라미터 튜닝**: 다양한 파라미터 조합 실험 (PCA 85% vs 80% 등)
4. **데이터 탐색**: QQQ ETF 구성종목들의 관계 분석

### main.py/universe.py와의 차이점:

1. **대상 ETF**: 연구에서는 QQQ, 실제 알고리즘에서는 SPY 사용
2. **파라미터**: 연구에서는 PCA 85%, 실제에서는 80% 사용
3. **목적**: 연구는 탐색적 분석, 실제 코드는 백테스팅용

이 노트북은 위상 데이터 분석을 사용한 포트폴리오 구성 방법론의 **연구 및 개발 단계**를 보여주며, 실제 알고리즘 구현 전에 이론적 검증과 시각적 분석을 수행하는 도구로 사용됩니다.