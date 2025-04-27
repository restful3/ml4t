import threading

from ibapi.wrapper import EWrapper


class IBWrapper(EWrapper):
    def __init__(self):
        EWrapper.__init__(self)
        self.nextValidOrderId = None
        self.historical_data = {}
        self.streaming_data = {}
        self.market_data = {}
        self.stream_event = threading.Event()
        self.account_values = {}
        self.positions = {}
        self.account_pnl = {}
        self.portfolio_returns = None
        self.resolved_contract = None

    def nextValidId(self, order_id):
        super().nextValidId(order_id)
        self.nextValidOrderId = order_id

    def contractDetails(self, request_id, contract_details):
        self.resolved_contract = contract_details

    def historicalData(self, request_id, bar):
        bar_data = (
            bar.date,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        if request_id not in self.historical_data.keys():
            self.historical_data[request_id] = []
        self.historical_data[request_id].append(bar_data)

    def tickPrice(self, request_id, tick_type, price, attrib):
        if request_id not in self.market_data.keys():
            self.market_data[request_id] = {}

        self.market_data[request_id][tick_type] = float(price)

    def tickByTickBidAsk(
        self,
        request_id,
        time,
        bid_price,
        ask_price,
        bid_size,
        ask_size,
        tick_atrrib_last,
    ):
        tick_data = (
            time,
            bid_price,
            ask_price,
            bid_size,
            ask_size,
        )

        self.streaming_data[request_id] = tick_data
        self.stream_event.set()

    def orderStatus(
        self,
        order_id,
        status,
        filled,
        remaining,
        avg_fill_price,
        perm_id,
        parent_id,
        last_fill_price,
        client_id,
        why_held,
        mkt_cap_price,
    ):
        cursor = self.connection.cursor()
        query = "INSERT INTO order_status (order_id, client_id, status, filled, remaining, last_fill_price, avg_fill_price) VALUES (?, ?, ?, ?, ?, ?, ?)"
        values = (
            order_id,
            client_id,
            status,
            filled,
            remaining,
            last_fill_price,
            avg_fill_price,
        )
        cursor.execute(query, values)

    def openOrder(self, order_id, contract, order, order_state):
        cursor = self.connection.cursor()
        query = "INSERT INTO open_orders (order_id, symbol, sec_type, exhange, action, order_type, quantity, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        values = (
            order_id,
            contract.symbol,
            contract.secType,
            contract.exchange,
            order.action,
            order.orderType,
            order.totalQuantity,
            order_state.status,
        )
        cursor.execute(query, values)

    def execDetails(self, request_id, contract, execution):
        cursor = self.connection.cursor()
        query = "INSERT INTO trades (request_id, symbol, sec_type, currency, execution_id, order_id, quantity, last_liquidity) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        values = (
            request_id,
            contract.symbol,
            contract.secType,
            contract.currency,
            execution.execId,
            execution.orderId,
            execution.shares,
            execution.lastLiquidity,
        )
        cursor.execute(query, values)

    def updateAccountValue(self, key, val, currency, account):
        try:
            val_ = float(val)
        except:
            val_ = val
        self.account_values[key] = (val_, currency)

    def updatePortfolio(
        self,
        contract,
        position,
        market_price,
        market_value,
        average_cost,
        unrealized_pnl,
        realized_pnl,
        account_name,
    ):
        portfolio_data = {
            "contract": contract,
            "symbol": contract.symbol,
            "position": position,
            "market_price": market_price,
            "market_value": market_value,
            "average_cost": average_cost,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
        }

        self.positions[contract.symbol] = portfolio_data

    def pnl(self, request_id, daily_pnl, unrealized_pnl, realized_pnl):
        pnl_data = {
            "daily_pnl": daily_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
        }

        self.account_pnl[request_id] = pnl_data
