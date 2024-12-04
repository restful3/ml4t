import numpy as np
from scipy.stats import norm


def simulate_gbm(s_0, mu, sigma, n_sims, T, N, random_seed=42, antithetic_var=False):
    """
    기하 브라운 운동(Geometric Brownian Motion)을 사용하여 주식 수익률을 시뮬레이션하는 함수입니다.
    
    매개변수:
    ------------
    s_0 : float
        초기 주가
    mu : float
        드리프트 계수 (평균 수익률)
    sigma : float
        변동성 계수 (표준편차)
    n_sims : int
        시뮬레이션 경로의 수
    T : float
        예측 기간의 길이 (년 단위)
    N : int
        예측 기간 내 시간 증분의 수
    random_seed : int
        재현성을 위한 랜덤 시드
    antithetic_var : bool
        분산 감소를 위해 대조변량법을 사용할지 여부

    반환값:
    -----------
    S_t : np.ndarray
        시뮬레이션 결과를 포함하는 행렬 (크기: n_sims x (N+1))
        행은 샘플 경로를, 열은 시간점을 나타냅니다.
    """
    
    np.random.seed(random_seed)
    
    # 시간 증분 계산
    dt = T/N
    
    # 브라운 운동 생성
    if antithetic_var:
        # 대조변량법 사용 시 절반의 경로만 생성하고 나머지는 부호를 바꿔 사용
        dW_ant = np.random.normal(scale = np.sqrt(dt), 
                                  size=(int(n_sims/2), N + 1))
        dW = np.concatenate((dW_ant, -dW_ant), axis=0)
    else: 
        # 일반적인 경우 모든 경로를 독립적으로 생성
        dW = np.random.normal(scale = np.sqrt(dt), 
                              size=(n_sims, N + 1))
  
    # 주가 과정의 진화 시뮬레이션
    S_t = s_0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * dW, axis=1)) 
    S_t[:, 0] = s_0  # 초기 주가 설정
    
    return S_t

def black_scholes_analytical(S_0, K, T, r, sigma, type="call"):
    """
    블랙-숄즈 모델의 해석적 형태를 사용하여 유럽식 옵션의 가격을 계산하는 함수입니다.
    
    매개변수:
    ------------
    S_0 : float
        초기 주가
    K : float
        행사가격
    T : float
        만기까지의 시간 (년 단위)
    r : float
        연간화된 무위험 이자율
    sigma : float
        주가 수익률의 표준편차 (변동성)
    type : str
        옵션의 종류. "call" 또는 "put" 중 하나여야 합니다.
    
    반환값:
    -----------
    option_premium : float
        블랙-숄즈 모델로 계산된 옵션의 프리미엄
    """
    
    # d1, d2 계산
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == "call":
        # 콜 옵션 가격 계산
        option_premium = (S_0 * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1))
    elif type == "put":
        # 풋 옵션 가격 계산
        option_premium = (K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S_0 * norm.cdf(-d1, 0, 1))
    else:
        raise ValueError("옵션 타입이 잘못되었습니다. 'call' 또는 'put'이어야 합니다!")

    return option_premium


def lsmc_american_option(S_0, K, T, N, r, sigma, n_sims, option_type, poly_degree, random_seed=42):
    """
    Longstaff와 Schwartz (2001)의 최소제곱 몬테카를로(Least Squares Monte Carlo) 알고리즘을 
    사용하여 미국식 옵션의 가격을 계산하는 함수입니다.
    
    매개변수:
    ------------
    S_0 : float
        초기 주가
    K : float
        행사가격
    T : float
        만기까지의 시간 (년 단위)
    N : int
        예측 기간 내 시간 증분의 수
    r : float
        연간화된 무위험 이자율
    sigma : float
        주가 수익률의 표준편차 (변동성)
    n_sims : int
        시뮬레이션할 경로의 수
    option_type : str
        옵션의 종류. "call" 또는 "put" 중 하나여야 합니다.
    poly_degree : int
        LSMC 알고리즘에서 사용할 다항식의 차수
    random_seed : int
        재현성을 위한 랜덤 시드
        
    반환값:
    -----------
    option_premium : float
        계산된 옵션의 프리미엄 
    """

    dt = T / N
    discount_factor = np.exp(-r * dt)

    # 기하 브라운 운동 시뮬레이션
    gbm_simulations = simulate_gbm(s_0=S_0, mu=r, sigma=sigma, 
                                   n_sims=n_sims, T=T, N=N,
                                   random_seed=random_seed)

    # 페이오프 행렬 계산
    if option_type == "call":
        payoff_matrix = np.maximum(
            gbm_simulations - K, np.zeros_like(gbm_simulations))
    elif option_type == "put":
        payoff_matrix = np.maximum(
            K - gbm_simulations, np.zeros_like(gbm_simulations))

    # 가치 행렬 초기화
    value_matrix = np.zeros_like(payoff_matrix)
    value_matrix[:, -1] = payoff_matrix[:, -1]

    # 역방향으로 옵션 가치 계산
    for t in range(N - 1, 0, -1):
        # 회귀 분석을 통한 계속 보유 가치 추정
        regression = np.polyfit(
            gbm_simulations[:, t], value_matrix[:, t + 1] * discount_factor, poly_degree)
        continuation_value = np.polyval(regression, gbm_simulations[:, t])
        
        # 즉시 행사 가치와 계속 보유 가치 비교
        value_matrix[:, t] = np.where(payoff_matrix[:, t] > continuation_value,
                                      payoff_matrix[:, t],
                                      value_matrix[:, t + 1] * discount_factor)

    # 옵션 프리미엄 계산
    option_premium = np.mean(value_matrix[:, 1] * discount_factor)
    return option_premium
