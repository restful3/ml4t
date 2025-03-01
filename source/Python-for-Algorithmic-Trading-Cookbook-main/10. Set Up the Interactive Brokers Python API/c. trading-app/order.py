# IB API의 Order 클래스 임포트
from ibapi.order import Order

# 매수/매도 상수 정의
BUY = "BUY"
SELL = "SELL"


def market(action, quantity):
    """
    시장가 주문을 생성하는 함수
    
    시장가 주문은 현재 시장 가격으로 즉시 실행됩니다. 실행 속도를 우선시하여 
    빠른 체결이 가능하지만, 특정 가격을 보장하지는 않습니다. 가격보다 
    실행의 확실성이 중요한 경우에 사용됩니다.
    
    Args:
        action: 매수/매도 구분
        quantity: 주문 수량
    Returns:
        Order: IB API 시장가 주문 객체
    """
    order = Order()
    order.action = action
    order.orderType = "MKT"
    order.totalQuantity = quantity
    return order


def limit(action, quantity, limit_price):
    """
    지정가 주문을 생성하는 함수
    
    지정가 주문은 트레이더가 매수 시 지불할 수 있는 최대 가격 또는 
    매도 시 수용할 수 있는 최소 가격을 지정할 수 있게 해줍니다. 
    시장 가격이 제한 가격에 도달하거나 더 나은 가격일 경우에만 
    체결되므로 가격 통제는 가능하나 실행이 보장되지는 않습니다.
    
    Args:
        action: 매수/매도 구분
        quantity: 주문 수량
        limit_price: 지정가격
    Returns:
        Order: IB API 지정가 주문 객체
    """
    order = Order()
    order.action = action
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = limit_price
    return order


def stop(action, quantity, stop_price):
    """
    손절매 주문을 생성하는 함수
    
    손절매 주문은 지정된 스톱 가격에 도달하면 자동으로 주문을 실행하여
    손실을 제한합니다. 스톱 가격에 도달하면 시장가 주문으로 전환되어
    실행됩니다.
    
    Args:
        action: 매수/매도 구분
        quantity: 주문 수량
        stop_price: 스탑가격
    Returns:
        Order: IB API 스탑 주문 객체
    """
    order = Order()
    order.action = action
    order.orderType = "STP"
    order.auxPrice = stop_price
    order.totalQuantity = quantity
    return order
