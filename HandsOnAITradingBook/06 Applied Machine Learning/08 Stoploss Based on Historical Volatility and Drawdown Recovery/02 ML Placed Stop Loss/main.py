# region imports
from AlgorithmImports import *

from sklearn.linear_model import Lasso
# endregion


class CaseOfTheMondaysAlgorithm(QCAlgorithm):
    """
    This algorithm is Part 2 of a 3 Part series. Similar to the 
    benchmark algorithm, this version allocates 100% to KO and places
    a stop market at 9:32 AM on the first trading day of each week. The 
    difference is that this version of the algorithm uses a Lasso 
    regression model to determine where to place the stop loss instead 
    of setting it a fixed percentage below the current price. The 
    factors we use as input to the model are:
        - VIX
        - Average true range of the last n months
        - Standard deviation of the last n months
    The label we train the model to predict is the rate of change from 
    the opening price of the week to the low price over the following 5
    trading days (1 week). When the algorithm places the stop loss, it 
    places it $0.0x below the predicted low price over the upcoming week
    so that it's only hit if the market is more volatile than the model 
    predicted. Just like Part 1, if stop loss isn't hit, the algorithm 
    automatically cancels it and liquidates the position with a market 
    on open order at the start of the next week.
    """

    def initialize(self):
        self.set_start_date(2018, 12, 31)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)
        self._security = self.add_equity(
            "KO", data_normalization_mode=DataNormalizationMode.RAW
        )
        self._symbol = self._security.symbol
        self._stop_loss_buffer = self.get_parameter('stop_loss_buffer', 0.01)
        
        # Create a DataFrame to store the factors and labels.
        self._samples = pd.DataFrame(
            columns=['vix', 'atr', 'std', 'weekly_low_return'],
            dtype=float
        )
        self._samples_lookback = timedelta(3 * 365)

        # Define the factors.
        period = 22 * self.get_parameter('indicator_lookback_months', 1)
        self._vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self._atr = AverageTrueRange(period, MovingAverageType.SIMPLE)
        self._std = StandardDeviation(period)
        
        # Warm up the factors.
        self._samples['vix'] = self.history(
            self._vix, self._samples_lookback, Resolution.DAILY
        ).loc[self._vix]['value']
        self._warm_up_samples()
        
        date_rule = self.date_rules.week_start(self._symbol)
        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(self._symbol, 2), 
            self._enter
        )
        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(self._symbol, -30), 
            self.liquidate
        )

        self._model = Lasso(alpha=10**(-self.get_parameter('alpha_exponent', 4)))

    def on_splits(self, splits):
        if splits[self._symbol].type == SplitType.SPLIT_OCCURRED:
            # Remove the consolidator. We'll need a new one
            # since we need to warm it up with the 
            # newly-adjusted data.
            self.subscription_manager.remove_consolidator(
                self._symbol, self._consolidator
            )
            self._warm_up_samples()

    def _warm_up_samples(self):
        # Setup the consolidator to generate the labels.
        self._consolidator = self.consolidate(
            self._symbol, Resolution.DAILY, self._consolidation_handler
        )
        
        # Reset the indicators.
        self._atr.reset()
        self._std.reset()
        
        # Clear the history.
        self._samples[['atr', 'std']] = np.nan
        self._trailing_bars = pd.DataFrame(columns=['open', 'low'])

        # Warm up the indicators and populate their history.
        warm_up_duration = self._samples_lookback + timedelta(
            2 * max(self._atr.warm_up_period, self._std.warm_up_period)
        )
        trade_bars = self.history[TradeBar](
            self._symbol, warm_up_duration, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        )
        for trade_bar in trade_bars:
            self._consolidator.update(trade_bar)

    def _consolidation_handler(self, consolidated_bar):
        t = consolidated_bar.end_time

        # Update indicators and the factor history.
        self._atr.update(consolidated_bar)
        self._std.update(t, consolidated_bar.close)
        if (self._atr.is_ready and 
            self._std.is_ready and 
            t in self._samples.index):
            # `t` is already in `self._samples` because
            # `on_data` added it.
            self._samples.loc[t, 'atr'] = self._atr.current.value
            self._samples.loc[t, 'std'] = self._std.current.value
            self._samples = self._samples[
                self._samples.index >= t - self._samples_lookback
            ]

        # Save the last week's worth of bars so we can calculate the 
        # training labels.
        self._trailing_bars.loc[t, :] = [
            consolidated_bar.open, consolidated_bar.low
        ]
        # Wait until we have a week's worth of data.
        if len(self._trailing_bars) < 6: 
            return
        self._trailing_bars = self._trailing_bars.iloc[1:]
        
        # Find the correct row in the `self._samples` DataFrame to 
        # assign the labels to.
        trade_open_time = self._trailing_bars.index[0] - timedelta(1)
        samples_timestamps = self._samples.index[
            self._samples.index <= trade_open_time
        ]
        if len(samples_timestamps) == 0:
            return
        samples_timestamp = samples_timestamps[-1]

        # Calculate the label, which is the return from the open price
        # on the day we start the trade to the low price over the 
        # following week. We use the low price because that is the
        # level where we place the stop loss.
        open_price = self._trailing_bars['open'][0]
        low_price = self._trailing_bars['low'].min()
        return_ = (low_price - open_price) / open_price
        self._samples.loc[samples_timestamp, 'weekly_low_return'] = return_

    def on_data(self, data):
        if self._vix in data:
            self._samples.loc[data.time, "vix"] = data[self._vix].value

    def _enter(self):
        # Train a model to predict the return from today's open to the 
        # low price of the upcoming week.
        training_samples = self._samples.dropna()
        self._model.fit(
            training_samples.iloc[:, :-1], 
            training_samples.iloc[:, -1]
        )

        prediction = self._model.predict(
            [self._samples.iloc[:, :-1].dropna().iloc[-1]]
        )[0]
        predicted_low_price = self._security.open * (1 + prediction)
        self.plot("Stop Loss", "Distance", 1 + prediction)

        # Place the entry order.
        quantity = self.calculate_order_quantity(self._symbol, 1)
        self.market_order(self._symbol, quantity)
        # Place the stop loss $0.0x below the predicted low price, so 
        # that the stop loss is only hit if the market is more volatile 
        # than we expected.
        self.stop_market_order(
            self._symbol, -quantity, 
            round(predicted_low_price - self._stop_loss_buffer, 2)
        )

