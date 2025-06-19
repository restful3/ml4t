# region imports
from AlgorithmImports import *

from sklearn.linear_model import LinearRegression
# endregion


class SplitEventsAlgorithm(QCAlgorithm):
    """
    This algorithm attempts to capitalize on the upcoming volatility 
    caused by stock splits. Specifically, it uses a multiple linear 
    regression model to predict the future return when a stock split is 
    about to occur. It places a trade in the same direction and then
    liquidates the position after a specified number of days.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)

        # Create a universe of stocks in a single sector (tech).
        self.universe_settings.resolution = Resolution.HOUR
        self.universe_settings.fill_forward = False
        self.universe_settings.asynchronous = True
        self.universe_settings.data_normalization_mode = DataNormalizationMode.Raw
        self._universe = self.add_universe(
            lambda fundamental: [
                x.symbol 
                for x in fundamental 
                if (x.asset_classification.morningstar_sector_code == 
                    MorningstarSectorCode.TECHNOLOGY)
            ]
        )

        # Set the parameters.
        self._max_open_trades = self.get_parameter('max_open_trades', 4)
        self._hold_duration = timedelta(self.get_parameter('hold_duration', 3))
        self._training_lookback = timedelta(
            self.get_parameter('training_lookback_years', 4) * 365
        )
        
        # Subscribe to the tech sector ETF and create an indicator to 
        # track the sector's momentum.
        self._sector_etf = self.add_equity(
            "XLK", self.universe_settings.resolution
        )
        self._sector_etf.roc = self.roc(
            self._sector_etf.symbol, 22, Resolution.DAILY
        )
        self._sector_etf.roc_history = pd.Series()
        self._sector_etf.roc.updated += self._update_event_handler
        bars = self.history[TradeBar](
            self._sector_etf.symbol, 
            self._training_lookback.days + self._sector_etf.roc.warm_up_period, 
            Resolution.DAILY
        )
        for bar in bars:
            self._sector_etf.roc.update(bar.end_time, bar.close)

        # Give equal exposure to each trade w/o using leverage (1 numerator).
        self._target_exposure_per_trade = 1 / self._max_open_trades
        self._trades_by_symbol = {}
        self._model = LinearRegression()
        self.train(
            self.date_rules.month_start(self._sector_etf.symbol),
            self.time_rules.midnight,
            self._train
        )

        self.schedule.on(
            self.date_rules.every_day(), 
            self.time_rules.midnight, 
            self._scan_for_trade_exits
        )

    def _update_event_handler(self, indicator, indicator_data_point):
        if not indicator.is_ready:
            return
        t = indicator_data_point.end_time
        self._sector_etf.roc_history.loc[t] = indicator_data_point.value
        # Trim off history that is older than 4 years.
        self._sector_etf.roc_history = self._sector_etf.roc_history[
            self._sector_etf.roc_history.index > t - self._training_lookback
        ]

    def _train(self):
        # Get the split and price data.
        splits = self.history[Split](
            self._universe.selected, self._training_lookback
        )
        assets_with_splits = set([])
        for splits_dict in splits:
            for symbol in splits_dict.keys():
                assets_with_splits.add(symbol)
        prices = self.history(
            list(assets_with_splits), self._training_lookback, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        )['open'].unstack(0)
        
        # Gather the training samples.
        samples = np.empty((0, 3))
        for splits_dict in splits:
            for symbol, split in splits_dict.items():
                if split.type == SplitType.SPLIT_OCCURRED:
                    continue
                t = split.end_time
                # Get entry price.
                entry_series = prices[symbol].loc[t < prices.index]
                if entry_series.empty or np.isnan(entry_series[0]):
                    continue
                entry_price = entry_series[0]
                # Get exit price.
                exit_series = prices[symbol].loc[
                    t + self._hold_duration < prices.index
                ]
                if exit_series.empty or np.isnan(exit_series[0]):
                    continue
                exit_price = exit_series[0]
                # Record the training sample.
                sector_roc = self._sector_etf.roc_history[
                    self._sector_etf.roc_history.index <= t
                ].iloc[-1]
                sample = np.array([
                    split.split_factor,
                    sector_roc,
                    (exit_price - entry_price) / entry_price
                ])
                samples = np.append(samples, [sample], axis=0)

        # Train the model.
        self.plot("Samples", "Count", len(samples))
        self._model.fit(samples[:, :2], samples[:, -1])

    def on_splits(self, splits):
        # Iterate through the split events.
        for symbol, split in splits.items():
            if symbol == self._sector_etf.symbol:
                continue
            
            # Check if it's a split warning and if we can open another trade.
            if (split.type == SplitType.WARNING and
                sum(
                    [len(trades) for trades in self._trades_by_symbol.values()]
                ) < self._max_open_trades):
                # Predict the future return.
                factors = [
                    split.split_factor, self._sector_etf.roc.current.value
                ]
                predicted_return = self._model.predict([factors])[0]
                self.log(f"{self.time};{str(symbol.id)};{predicted_return}")
                if predicted_return == 0:
                    continue

                # Open the trade.
                if symbol not in self._trades_by_symbol:
                    self._trades_by_symbol[symbol] = []
                quantity = self.calculate_order_quantity(
                    symbol, 
                    np.sign(predicted_return) * self._target_exposure_per_trade
                )
                if quantity == 0:
                    continue
                self._trades_by_symbol[symbol].append(
                    Trade(self, symbol, self._hold_duration, quantity)
                )

            # Check if the split occurred and we have an open order for 
            # the asset.
            elif (split.type == SplitType.SPLIT_OCCURRED and 
                symbol in self._trades_by_symbol):
                for trade in self._trades_by_symbol[symbol]:
                    trade.on_split_occurred(split)

    def _scan_for_trade_exits(self):
        closed_trades = []
        for trades in self._trades_by_symbol.values():
            closed_trades = []
            for i, trade in enumerate(trades):
                trade.scan(self)
                if trade.closed:
                    closed_trades.append(i)
        
            # Delete closed trades.
            for i in closed_trades[::-1]:
                del trades[i]


class Trade:
    def __init__(self, algorithm, symbol, hold_duration, quantity):
        self.closed = False
        self._symbol = symbol
        self._close_time = algorithm.time + hold_duration
        self._quantity = quantity
        algorithm.market_on_open_order(symbol, quantity)

    def on_split_occurred(self, split):
        self._quantity = int(self._quantity / split.split_factor)

    def scan(self, algorithm):
        if not self.closed and self._close_time <= algorithm.time:
            algorithm.market_on_open_order(self._symbol , -self._quantity)
            self.closed = True

