import backtrader as bt

class MyBuySell(bt.observers.BuySell):
    """
    매수/매도 지표의 시각화를 위한 커스텀 클래스입니다.
    
    backtrader의 기본 BuySell 클래스를 상속받아 매수/매도 시점의 마커 스타일을 커스터마이징합니다.
    - 매수 시점: 초록색 위쪽 삼각형(^) 마커
    - 매도 시점: 빨간색 아래쪽 삼각형(v) 마커
    """
    plotlines = dict(
        buy=dict(marker="^", markersize=8.0, color="green", fillstyle="full"),  # 매수 마커 설정
        sell=dict(marker="v", markersize=8.0, color="red", fillstyle="full")    # 매도 마커 설정
    )

def get_action_log_string(dir, action, price, size, asset=None, cost=None, 
                          commission=None, cash=None, open=None, close=None):
    """
    매수/매도 주문의 생성 및 실행에 대한 로그 문자열을 생성하는 헬퍼 함수입니다.

    매개변수:
        dir (str): 포지션의 방향을 나타내는 문자열입니다. "b"(매수) 또는 "s"(매도)의 값을 가질 수 있습니다.
        action (str): 주문 액션을 나타내는 문자열입니다. "e"(실행됨) 또는 "c"(생성됨)의 값을 가질 수 있습니다.
        price (float): 주문 가격입니다.
        size (float): 주문 수량입니다.
        asset (str): 자산의 이름입니다.
        cost (float, optional): 주문의 총 비용입니다. 기본값은 None입니다.
        commission (float, optional): 수수료입니다. 기본값은 None입니다.
        cash (float, optional): 현금 잔고입니다. 기본값은 None입니다.
        open (float, optional): 시가입니다. 기본값은 None입니다.
        close (float, optional): 종가입니다. 기본값은 None입니다.

    반환값:
        str: 로깅에 사용될 포맷된 문자열을 반환합니다.
    """
    # 매수/매도 방향을 나타내는 딕셔너리
    dir_dict = {
        "b": "BUY",
        "s": "SELL",
    }

    # 주문 상태를 나타내는 딕셔너리
    action_dict = {
        "e": "EXECUTED",
        "c": "CREATED"
    }

    # 기본 로그 문자열 생성 (가격과 수량 포함)
    str = (
        f"{dir_dict[dir]} {action_dict[action]} - "
        f"Price: {price:.2f}, Size: {size:.2f}"
    )
    
    # 자산 정보가 있는 경우 추가
    if asset is not None:
        str = str + f", Asset: {asset}"

    # 주문이 실행된 경우의 추가 정보
    if action == "e":
        if cost is not None:
            str = str + f", Cost: {cost:.2f}"
        if commission is not None:
            str = str + f", Commission: {commission:.2f}"
    # 주문이 생성된 경우의 추가 정보
    elif action == "c":
        if cash is not None:
            str = str + f", Cash: {cash:.2f}"
        if open is not None:
            str = str + f", Open: {open:.2f}"
        if close is not None:
            str = str + f", Close: {close:.2f}"

    return str

def get_result_log_string(gross, net):
    """
    Helper function for logging. Creates a string indicating the summary of
    an operation.

    Args:
        gross (float): the gross outcome
        net (float): the net outcome

    Returns:
        str: The string used for logging
    """
    str = f"OPERATION RESULT - Gross: {gross:.2f}, Net: {net:.2f}"
    return str
