# region imports
from AlgorithmImports import *

from sklearn.linear_model import Ridge
# endregion


class InverseVolatilityRankAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates a way to use machine learning to form a 
    portfolio of Futures contracts where the weight of each contract is 
    the inverse of its expected future volatility. To forecast the
    future volatility, this strategy uses a ridge regression model and 
    the following factors: 
        - Volatility: Standard deviation of daily returns over the last 
          60 trading days
        - ATR: Average True Range over the last 60 trading days
        - Open interest
    """

    def initialize(self):
        self.set_start_date(2018, 12, 31)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000_000)

        self._std_period = self.get_parameter('std_months', 3) * 26
        self._atr_period = self.get_parameter('atr_months', 3) * 26
        self._training_set_duration = timedelta(
            self.get_parameter('training_set_duration', 365)
        )
        self._future_std_period = 6

        self._contracts = []
        tickers = [
            Futures.Indices.VIX, 
            Futures.Indices.SP_500_E_MINI,
            Futures.Indices.NASDAQ_100_E_MINI,
            Futures.Indices.DOW_30_E_MINI,
            Futures.Energy.BRENT_CRUDE,
            Futures.Energy.GASOLINE,
            Futures.Energy.HEATING_OIL,
            Futures.Energy.NATURAL_GAS,
            Futures.Grains.CORN,
            Futures.Grains.OATS,
            Futures.Grains.SOYBEANS,
            Futures.Grains.WHEAT
        ]
        for ticker in tickers:
            future = self.add_future(ticker, extended_market_hours=True)
            future.set_filter(lambda universe: universe.front_month())
        
        schedule_symbol = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        self.schedule.on(
            self.date_rules.week_start(schedule_symbol),
            self.time_rules.after_market_open(schedule_symbol, 1), 
            self._trade
        )

    def _trade(self):
        # Get the open interest factors.
        open_interest = self.history(
            OpenInterest, [c.symbol for c in self._contracts], 
            self._training_set_duration, fill_forward=False
        )
        open_interest.index = open_interest.index.droplevel(0)

        # Predict volatility over the next week for each security.
        expected_volatility_by_security = {}
        for security in self._contracts:
            symbol = security.symbol
            if symbol not in open_interest.index:
                continue
            # Get the factors.
            factors = pd.concat(
                [security.indicator_history, open_interest.loc[symbol]], 
                axis=1
            ).ffill().loc[security.indicator_history.index].dropna()
            if factors.empty:
                # The df can be empty if there is no open interest data 
                # for the asset (example: 
                # https://www.quantconnect.com/datasets/issue/16604).
                continue 
            # Get the labels.
            label = security.label_history

            # Align the factors and labels.
            idx = sorted(
                list(set(factors.index).intersection(set(label.index)))
            )
            # Ensure there are enough training samples.
            if len(idx) < 20:
                continue

            # Train the model.
            model = Ridge()
            model.fit(factors.loc[idx], label.loc[idx])

            # Predict the volatility over the next week.
            prediction = model.predict([factors.iloc[-1]])[0] 
            if prediction > 0:
                expected_volatility_by_security[security] = prediction
            self.plot("Predictions", security.symbol.canonical.value, prediction)

        # Calculate the portfolio weights and rebalance.
        portfolio_targets = []
        std_sum = sum(
            [
                1 / expected_vol 
                for expected_vol in expected_volatility_by_security.values()
            ]
        )
        for security, expected_vol in expected_volatility_by_security.items():
            weight = (
                3 
                / expected_vol 
                / std_sum 
                / security.symbol_properties.contract_multiplier
            )
            # The numerator `3` above scales the position size.
            # If it's set to 1, the algorithm only trades a few of 
            # the Futures in the universe because of the
            # minimum_order_margin_portfolio_percentage setting. 
            # If it's set too high, we run into margin calls. 
            # 3 is the middle-ground where the algorithm trades 
            # most of the Futures without having to set 
            # minimum_order_margin_portfolio_percentage to zero.
            portfolio_targets.append(PortfolioTarget(security.symbol, weight))
        self.set_holdings(portfolio_targets, True)

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            if security.symbol.is_canonical(): 
                continue  # Skip over continuous contracts.
            # Create the indicators.
            # We're using manual indicators here for demonstration
            # purposes. You can reduce the amount of code by using
            # the automatic indicators, but the following code
            # is an example of doing everything manually so you 
            # can have maximum control over the outcome.
            security.close_roc = RateOfChange(1)
            security.std_of_close_returns = IndicatorExtensions.of(
                StandardDeviation(self._std_period), 
                security.close_roc
            )
            security.atr = AverageTrueRange(self._atr_period)
            security.open_roc = RateOfChange(1)
            security.std_of_open_returns = IndicatorExtensions.of(
                StandardDeviation(self._future_std_period), security.open_roc
            )

            # Create some pandas objects to store the historical 
            # indicator values we'll use to train the ML.
            security.indicator_history = pd.DataFrame()
            security.label_history = pd.Series()

            # Create a consolidator to aggregate minute data into daily 
            # data for the indicators.
            security.consolidator = self.consolidate(
                security.symbol, Resolution.DAILY, self._consolidation_handler
            )
            
            # Warm up the indicators with historical data.
            warm_up_length = (
                max(self._std_period + 1, self._atr_period) 
                + self._training_set_duration.days
            )
            bars = self.history[TradeBar](
                security.symbol, warm_up_length, Resolution.DAILY
            )
            for bar in bars:
                security.consolidator.update(bar)

            self._contracts.append(security)

        for security in changes.removed_securities:
            # Remove the consolidator.
            self.subscription_manager.remove_consolidator(
                security.symbol, security.consolidator
            )
            # Reset the indicators.
            security.close_roc.reset()
            security.std_of_close_returns.reset()
            security.atr.reset()
            security.open_roc.reset()
            security.std_of_open_returns.reset()
            if security in self._contracts:
                self._contracts.remove(security)

    def _consolidation_handler(self, consolidated_bar):
        # Get the security object.
        security = self.securities[consolidated_bar.symbol]

        # Update the indicators and save their values.
        t = consolidated_bar.end_time
        if security.atr.update(consolidated_bar):
            security.indicator_history.loc[t, 'atr'] = \
                security.atr.current.value
        security.close_roc.update(t, consolidated_bar.close)
        if security.std_of_close_returns.is_ready:
            security.indicator_history.loc[t, 'std_of_close_returns'] = \
                security.std_of_close_returns.current.value
        security.open_roc.update(t, consolidated_bar.open)

        # Update the label history.
        if (security.std_of_open_returns.is_ready and 
            len(security.indicator_history.index) > self._future_std_period):
            security.label_history.loc[
                security.indicator_history.index[-self._future_std_period - 1]
            ] = security.std_of_open_returns.current.value
        
        # Trim the factor and label history.
        security.indicator_history = security.indicator_history[
            (security.indicator_history.index >= 
            self.time - self._training_set_duration)
        ]
        security.label_history = security.label_history[
            (security.label_history.index >= 
            self.time - self._training_set_duration)
        ]

