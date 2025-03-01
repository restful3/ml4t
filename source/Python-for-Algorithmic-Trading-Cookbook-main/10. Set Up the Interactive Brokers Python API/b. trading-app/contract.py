# IB API의 Contract 클래스 임포트
from ibapi.contract import Contract


def future(symbol, exchange, contract_month):
    """
    선물 계약을 생성하는 함수
    
    Args:
        symbol: 선물 상품 심볼
        exchange: 거래소 
        contract_month: 만기월
    Returns:
        Contract: IB API 선물 계약 객체
    """
    contract = Contract()
    contract.symbol = symbol
    contract.exchange = exchange
    contract.lastTradeDateOrContractMonth = contract_month
    contract.secType = "FUT"

    return contract


def stock(symbol, exchange, currency):
    """
    주식 계약을 생성하는 함수
    
    Args:
        symbol: 주식 심볼
        exchange: 거래소
        currency: 통화
    Returns:
        Contract: IB API 주식 계약 객체
    """
    contract = Contract()
    contract.symbol = symbol
    contract.exchange = exchange
    contract.currency = currency
    contract.secType = "STK"

    return contract


def option(symbol, exchange, contract_month, strike, right):
    """
    옵션 계약을 생성하는 함수
    
    Args:
        symbol: 기초자산 심볼
        exchange: 거래소
        contract_month: 만기월
        strike: 행사가
        right: 권리(콜/풋)
    Returns:
        Contract: IB API 옵션 계약 객체
    """
    contract = Contract()
    contract.symbol = symbol
    contract.exchange = exchange
    contract.lastTradeDateOrContractMonth = contract_month
    contract.strike = strike
    contract.right = right
    contract.secType = "OPT"

    return contract
