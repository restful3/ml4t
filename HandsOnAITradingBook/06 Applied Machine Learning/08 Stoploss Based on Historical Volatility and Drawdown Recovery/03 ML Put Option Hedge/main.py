# region imports
from AlgorithmImports import *

from sklearn.linear_model import Lasso
# endregion


class CaseOfTheMondaysAlgorithm(QCAlgorithm):
    """
    This algorithm is Part 2 of a 3-part series. Similar to the 
    benchmark algorithm, this version allocates 100% to KO at 
    9:32 AM on the first trading day of each week. However, instead
    of placing a stop market order, this version of the algorithm
    buys a put Option contract to hedge the position. This algorithm
    uses the same Lasso regression model as Part 2 to predict the 
    return from the start of the week to the low price throughout the
    week using the following factors:
        - VIX
        - Average true range of the last n months
        - Standard deviation of the last n months
    The difference is that instead of using the output of the to place
    the stop loss, we use the output to determine which strike price
    to use for the Option contract. If the Option is exercised, it closes
    the trade in the underlying stock. If it's not exercised, we sell it
    and the underlying shares at market open of the following week.
    """

    def initialize(self):
        self.set_start_date(2018, 12, 31)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)
 
        # Set the fee model.
        self.set_security_initializer(
            IBFeesSecurityInitializer(
                self.brokerage_model, 
                FuncSecuritySeeder(self.get_last_known_prices)
            )
        )

        # Subscribe to the underlying Equity.
        self._security = self.add_equity(
            "KO", data_normalization_mode=DataNormalizationMode.RAW
        )
        self._symbol = self._security.symbol
        
        # Subscribe to the put Option contracts.
        option = self.add_option(self._symbol)
        option.set_filter(
            lambda universe: universe.include_weeklys().puts_only()
                .strikes(-20, 0).expiration(0, 7)
        )

        # Create a DataFrame to store the factors and labels.
        self._samples = pd.DataFrame(
            columns=['vix', 'atr', 'std', 'weekly_low_return'],
            dtype=float
        )
        self._samples_lookback = timedelta(3 * 365)

        # Define the factors.
        self._vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self._atr = AverageTrueRange(22, MovingAverageType.SIMPLE)
        self._std = StandardDeviation(22)
        
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
            self._liquidate_if_possible
        )

        self._model = Lasso(alpha=10**(-self.get_parameter('alpha_exponent', 4)))

    def on_splits(self, splits):
        if splits[self._symbol].type == SplitType.SPLIT_OCCURRED:
            self._warm_up_samples()

    def _warm_up_samples(self):
        # Setup the consolidator to generate the labels.
        if hasattr(self, '_consolidator'):
            self.subscription_manager.remove_consolidator(
                self._symbol, self._consolidator
            )
        self._consolidator = self.consolidate(
            self._symbol, Resolution.DAILY, self._consolidation_handler
        )
        
        # Reset the indicators.
        self._atr.Reset()
        self._std.Reset()
        
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

        open_price = self._trailing_bars['open'][0]
        low_price = self._trailing_bars['low'].min()

        return_ = (low_price - open_price) / open_price
        self._samples.loc[samples_timestamp, 'weekly_low_return'] = return_

    def _liquidate_if_possible(self):
        self.liquidate(self._symbol)
        for symbol, security_holding in self.portfolio.items():
            # If it's an Option contract and we have no open orders for 
            # it, liquidate it.
            if (security_holding.type == SecurityType.OPTION and 
                not list(self.transactions.get_open_order_tickets(symbol))):
                self.liquidate(symbol)

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
            [self._samples.iloc[:, :3].dropna().iloc[-1]]
        )[0]
        predicted_low_price = self._security.open * (1 + prediction)
        self.plot("Stop Loss", "Distance", 1 + prediction)

        for chain in self.current_slice.option_chains.values():
            # Buy the underlying Equity.
            quantity = self.calculate_order_quantity(self._symbol, 1)
            self.market_order(self._symbol, quantity)

            # Select the put Contract.
            puts = [
                contract 
                for contract in chain 
                if contract.strike < predicted_low_price + contract.ask_price
            ]
            contract = sorted(puts, key=lambda contract: contract.strike)[-1]
            
            # Buy the put Contract.
            tag = f"Predicted weekly low price: {round(predicted_low_price, 2)}"
            self.market_order(contract.symbol, quantity // 100, tag=tag)


class IBFeesSecurityInitializer(BrokerageModelSecurityInitializer):

    def __init__(self, brokerage_model, security_seeder):
        super().__init__(brokerage_model, security_seeder)

    def initialize(self, security):
        super().initialize(security)
        security.set_fee_model(InteractiveBrokersFeeModel())

