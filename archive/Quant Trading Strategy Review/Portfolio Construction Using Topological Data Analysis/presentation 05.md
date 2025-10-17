# KeplerMapper의 map 함수
KeplerMapper의 `map` 함수는 **데이터를 토폴로지적 맵(topological map)**으로 변환하는 핵심 함수입니다.
KeplerMapper는 **고차원 데이터의 구조를 시각화하는 라이브러리**로, 특히 비선형 데이터의 형태를 “연결망(graph)”으로 표현할 때 사용됩니다.
아래는 `map` 함수의 구조와 작동 원리를 단계별로 정리한 자세한 설명입니다.

---

## 🧩 기본 개념

`map` 함수는 다음 형태로 호출됩니다:

```python
graph = mapper.map(lens, X, cover, clusterer)
```

여기서:

* `X`: 원본 데이터 (numpy array 또는 pandas DataFrame)
* `lens`: 차원 축소 또는 projection 결과 (예: PCA, t-SNE, UMAP, 또는 단일 feature)
* `cover`: 커버링(covering) 파라미터 — 데이터의 overlapping 구간을 어떻게 나눌지 정의
* `clusterer`: 각 구간 내에서 클러스터링할 방법 (보통 DBSCAN, KMeans 등)

---

## ⚙️ 동작 과정 (4단계)

### 1️⃣ Lens 적용 (Projection)

먼저 `lens`를 통해 원본 데이터 `X`를 저차원으로 투사합니다.

* 예: `lens = sklearn.decomposition.PCA(n_components=2).fit_transform(X)`
* 결과적으로 각 데이터 포인트는 “렌즈 공간(lens space)”에서 좌표를 가지게 됩니다.

### 2️⃣ Covering (슬라이싱)

그다음 렌즈 공간을 겹치는 여러 구간(cover intervals)로 나눕니다.

* 예를 들어 2D 렌즈의 경우, 각 축을 일정 개수의 구간으로 나누고, `overlap` 비율만큼 겹치게 설정합니다.
* `cover` 인스턴스는 보통 다음과 같이 생성됩니다:

  ```python
  cover = km.Cover(n_cubes=10, perc_overlap=0.4)
  ```

  * `n_cubes`: 각 차원에서 몇 개의 구간으로 나눌지
  * `perc_overlap`: 각 구간이 얼마나 겹칠지 (0~1)

이 단계에서 **각 구간(=hyperrectangle)** 안에 들어가는 데이터 점들이 그룹화됩니다.

### 3️⃣ 클러스터링 (Local clustering)

각 구간 내부에서 선택한 `clusterer`로 클러스터링을 수행합니다.

* 예: `DBSCAN(eps=0.5, min_samples=3)`
* 각 클러스터는 노드(node)가 됩니다.
* 즉, 한 구간 내에서 여러 클러스터 → 여러 노드 생성.

### 4️⃣ 그래프 연결 (Graph construction)

겹치는 구간에 속하는 데이터 포인트가 서로 공유되면, 그 두 노드를 엣지(edge)로 연결합니다.
이로써 **데이터의 위상적 형태(topological shape)**를 나타내는 네트워크가 만들어집니다.

---

## 🧠 핵심 파라미터 정리

| 파라미터                     | 설명                        |
| ------------------------ | ------------------------- |
| `X`                      | 원본 데이터                    |
| `lens`                   | 투사된 데이터 (렌즈 공간)           |
| `cover`                  | 공간을 어떻게 나눌지 (커버링 방식)      |
| `clusterer`              | 각 구간 내의 클러스터링 알고리즘        |
| `precomputed`            | 거리 행렬을 미리 계산했을 경우 True 설정 |
| `remove_duplicate_nodes` | 중복 노드 제거 옵션               |



## 🧩 2️⃣ map 함수 내부 구조 개요

내부적으로 `map()`은 아래와 같은 순서로 lens를 사용합니다:

```python
for cube in cover:                      # 1️⃣ 렌즈 공간을 구간(cube)으로 나눔
    idx = points_in_cube(lens, cube)    # 2️⃣ 해당 구간에 속하는 데이터 포인트 인덱스 선택
    X_subset = X[idx]                   # 3️⃣ 원본 데이터 중 해당 포인트만 추출
    clusters = clusterer.fit(X_subset)  # 4️⃣ 원본 공간에서 클러스터링 수행
```

즉,

* **클러스터링은 lens 공간이 아니라 원본 데이터 X 위에서 수행되지만,**
* **어떤 데이터들을 클러스터링할지는 lens가 결정**합니다.

---

## 🧮 3️⃣ 구체적인 작동 예시

### 예시 설정

```python
cover = km.Cover(n_cubes=5, perc_overlap=0.3)
lens = PCA(2).fit_transform(X)
clusterer = DBSCAN(eps=0.5, min_samples=3)
graph = mapper.map(lens, X, cover=cover, clusterer=clusterer)
```

### 동작 원리

1. `lens`의 첫 번째 차원과 두 번째 차원을 각각 5개의 구간으로 나눕니다.
   (겹치는 구간도 포함)
2. 각 구간(cube)에 속하는 **데이터 인덱스 집합**을 찾습니다.

   * 예: 구간 A에는 `X[5], X[8], X[22], ...`가 포함
3. 해당 데이터 서브셋 `X_subset`에 대해 **클러스터링 수행**

   * 이때 clusterer는 **lens를 직접 쓰지 않고**, 원본 X의 feature를 이용.
   * 단지 "이 점들을 묶어서 분석하라"는 기준을 lens가 제공합니다.
4. 구간이 겹치는 부분의 클러스터들은 연결(edge)로 표시되어 위상 구조를 형성합니다.

---

## 🔄 요약하자면

| 단계       | lens의 역할                     | clustering에서의 영향           |
| -------- | ---------------------------- | -------------------------- |
| Cover 분할 | 렌즈 공간을 일정 간격으로 나누는 기준 제공     | 구간 경계 결정                   |
| 데이터 선택   | 각 구간(cube)에 속한 데이터 인덱스 결정    | 어떤 데이터들이 함께 클러스터링될지 결정     |
| 클러스터링    | (선택된 X 데이터에 대해) clusterer 실행 | 실제 클러스터링 대상이 lens 기반으로 제한됨 |
| 그래프 연결   | 겹치는 구간의 클러스터 연결              | 데이터 간 위상적 연결 형성            |

---

## 🎯 결론

* **lens는 클러스터링 입력을 직접 바꾸지 않는다.**
  → 클러스터링은 여전히 원본 `X` 데이터 공간에서 이루어집니다.

* 하지만 **lens는 어떤 샘플들이 함께 클러스터링될지를 결정하는 필터**로 작용합니다.
  즉, “데이터의 지역적 시야(local view)”를 만드는 역할을 하죠.


---

## ⚙️ 2️⃣ 기본 cover 설정값

| 파라미터           | 기본값      | 설명                                |
| -------------- | -------- | --------------------------------- |
| `n_cubes`      | 10       | 각 차원에서 커버링 구간 수 (즉, 10개의 구간으로 나눔) |
| `perc_overlap` | 0.2      | 구간 간 겹침 비율 (20%)                  |
| `limits`       | `None`   | 렌즈 공간 전체 범위를 자동으로 계산              |
| `kind`         | `'cube'` | 각 차원을 독립적으로 구간화 (사각형/큐브 형태)       |

따라서 `cover=None`이면, **기본적으로 10x10x... 큐브 커버링이 자동 설정**됩니다.

