#region imports
from AlgorithmImports import *

from sklearn.tree import DecisionTreeRegressor
#endregion


class TradeCostEstimationAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how we can using machine learning to 
    reduce trading costs. If you run this algorithm with 
    `self._benchmark` enabled, the algorithm buys BTCUSDC at 12 AM and 
    then sells it at 1 AM everyday. If you run it with `self._benchmark` 
    disabled, the algorithm buys BTCUSDC at 12 AM and then but liquidates
    it at some point between 1 AM and 11:59 PM, when it predicts that 
    the liquidation costs are lower than usual. To predict the trading 
    costs, this algorithm uses a DecisionTreeRegressor model with the 
    following factors:
        - Absolute order quantity
        - Average true range
        - Average daily volume
        - Spread percent ((ask - bid) / bid)
        - Top of book size (in dollars)
    """

    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash('USDC', 10_000_000)

        self.settings.minimum_order_margin_portfolio_percentage = 0

        self._benchmark = False #True
        self._scan_for_exit = False

        # Add the asset.
        self._security = self.add_crypto("BTCUSDC", market=Market.BYBIT)
        self._security.set_slippage_model(SpreadSlippageModel())
        self._symbol = self._security.symbol

        # Schedule some events to trade.
        date_rule = self.date_rules.every_day()
        self.schedule.on(date_rule, self.time_rules.at(0, 0), self._entry)
        self.schedule.on(date_rule, self.time_rules.at(1, 0), self._exit)

        self._total_costs = 0
        self._quantity = 10
        self._costs = pd.Series(dtype=float)
        self._cost_sma = SimpleMovingAverage(
            self.get_parameter('cost_sma_period', 10)
        )
        self._order_fills = pd.DataFrame(
            columns=['fill_price', 'quantity', 'cost', 'tag']
        )

        if not self._benchmark:
            self._model = None 
            self.enable_automatic_indicator_warm_up = True
            self._atr = self.atr(
                self._symbol, self.get_parameter('atr_period', 14), 
                resolution=Resolution.DAILY
            )
            self._sma = self.sma(
                self._symbol, self.get_parameter('sma_period', 10), 
                Resolution.DAILY, Field.VOLUME
            )
            self.train(
                self.date_rules.month_start(), 
                self.time_rules.at(0, 0), 
                self._train
            )
            self._lookback_window = self.get_parameter('lookback_window', 100)
            self._factors = pd.DataFrame(
                columns=[
                    'abs_order_quantity', 'atr', 'avg_daily_volume', 
                    'spread_pct', 'top_of_book_size'
                ]
            )

    def _entry(self):
        self.market_order(self._symbol, self._quantity)

    def _exit(self):
        if self._benchmark:
            bid = self._security.bid_price
            ask = self._security.ask_price
            self.liquidate(tag=f"Bid: {bid}; Ask: {ask}")
            return
        self._scan_for_exit = True

    def _train(self):
        if self._factors.shape[0] < self._lookback_window:
            return
        if self._model is None:
            self._model = DecisionTreeRegressor(max_depth=10, random_state=0)
        self._model.fit(self._factors, self._costs)

    def _trim_samples(self):
        # Trim factors and labels that have fallen out of the lookback 
        # window.
        self._factors = self._factors.iloc[-self._lookback_window:]
        self._costs = self._costs.iloc[-self._lookback_window:]

    def on_data(self, data):
        if not self._scan_for_exit:
            return

        bid = self._security.bid_price
        ask = self._security.ask_price
        current_factors = [
            abs(self._quantity),           # Absolute order quantity
            self._atr.current.value,       # ATR
            self._sma.current.value,       # Average daily volume
            (ask - bid) / bid,     # Spread percent
            self._security.ask_size * ask  # Top of book size (in $)
        ]

        tag = "Hit time limit" 
        if self._model is None:
            tag = "ML model is not ready"
        # Check if we hit the time limit.
        elif data.time.time() < time(23, 59):  
            # Predict the trade cost.
            predicted_costs = self._model.predict([current_factors])[0]
            dollar_volume = self._quantity * bid
            predicted_costs_per_dollar = predicted_costs / dollar_volume

            # Don't trade until predicted costs are lower than the 
            # average costs we've seen.
            if predicted_costs_per_dollar >= self._cost_sma.current.value:
                return
            tag = (
                f'Predicted: {predicted_costs_per_dollar}; ' 
                + f'SMA: {self._cost_sma.Current.Value}'
            )

            # Plot the predicted costs.
            self.plot("Costs", "Predicted", predicted_costs)
            self.plot(
                "Costs Per Dollar", "Predicted", predicted_costs_per_dollar
            )
        
        self.plot("Model Readiness", "IsReady", int(self._model is not None))

        # Place the trade.
        self.market_order(self._symbol, -self._quantity, tag=tag)
        self._factors.loc[self.time] = current_factors
        self._scan_for_exit = False
        self._trim_samples()

    def on_order_event(self, order_event):
        if order_event.status == OrderStatus.FILLED:
            if order_event.quantity > 0:
                return
            t = self.time
            fill_price = order_event.fill_price
            dollar_volume = self._quantity * fill_price
            # Only use bid price because we are only training on the 
            # exit orders (sells, where they are filled at the bid).
            slippage_per_share = self._security.bid_price - fill_price
            cost = (
                order_event.order_fee.value.amount 
                + slippage_per_share * self._quantity 
            )
            self._total_costs += cost
            self._cost_sma.update(t, cost / dollar_volume)
            self._costs.loc[t] = cost
            self.plot("Costs", "Actual", cost)
            self.plot("Cumulative Costs", "Actual", self._total_costs)
            self.plot("Samples", "Count", len(self._costs))
            self.plot("Costs Per Dollar", "Actual", cost / dollar_volume)
            if self._cost_sma.is_ready:
                self.plot(
                    "Costs Per Dollar", "SMA(Actual)", 
                    self._cost_sma.current.value
                )
            self._order_fills.loc[t] = (
                fill_price, order_event.quantity, cost, order_event.ticket.tag
            )

    def on_end_of_algorithm(self):
        key = ("benchmark" if self._benchmark else "candidate") + "_order_fills"
        self._order_fills.to_csv(self.object_store.get_file_path(key))


class SpreadSlippageModel:
    def get_slippage_approximation(self, asset, order):
        # We are using the spread as a proxy for a high cost environment
        # for market orders.
        return asset.ask_price - asset.bid_price

