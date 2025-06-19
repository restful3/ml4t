#region imports
from AlgorithmImports import *

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
#endregion


class MarkovModelAlgorithm(QCAlgorithm):
    """
    This algorithm is an extension of the previous version. Instead of 
    using Equity Options, this version of the strategy uses 
    European-style Index Options because they are cash-settled and 
    can't be exercised before the expiration date. Instead of using 
    SPY as the underlying asset, this algorithm uses the SPX index, 
    which is a more accurate representation of the S&P 500.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(1_000_000)

        self._index = self.add_index("SPX")
        self._index.hedge_contracts = []

        self._min_expiry = timedelta(self.get_parameter('min_expiry', 180))
        self._max_expiry = timedelta(self.get_parameter('max_expiry', 365))
        self._min_hold_period = timedelta(
            self.get_parameter('min_hold_period', 7)
        )
        option = self.add_index_option(self._index.symbol)
        option.set_filter(
            -1, 1, self._min_expiry + self._min_hold_period, self._max_expiry
        )
        self._option_symbol = option.symbol
        self._expiry = datetime.min
        
        self._lookback_period = timedelta(
            self.get_parameter('lookback_years', 3) * 365
        )

        # Create a trailing series of daily returns.
        self._daily_returns = pd.Series()
        roc = self.roc(self._index.symbol, 1, Resolution.DAILY)
        roc.updated += self._update_event_handler
        history = self.history[TradeBar](
            self._index.symbol, self._lookback_period + timedelta(7), 
            Resolution.DAILY
        )
        for bar in history:
            roc.update(bar.end_time, bar.close)

        self.schedule.on(
            self.date_rules.every_day(self._index.symbol),
            self.time_rules.after_market_open(self._index.symbol, 1),
            self._trade
        )
        self._previous_regime = None

    def _update_event_handler(self, indicator, indicator_data_point):
        if not indicator.is_ready:
            return
        t = indicator_data_point.end_time
        # Add the indicator value to the series of trailing returns.
        self._daily_returns.loc[t] = indicator_data_point.value
        # Trim the series to remove samples that have fallen out of the 
        # lookback window.
        self._daily_returns = self._daily_returns[
            t - self._daily_returns.index <= self._lookback_period
        ]

    def _trade(self):
        # Create the markov model.
        # `k_regimes` defines the number of regimes. In this case,
        # we want 2 regimes (high and low volatility).
        # `switching_variance` defines whether or not each regime
        # can have its own variance.
        model = MarkovRegression(
            self._daily_returns, k_regimes=2, switching_variance=True
        )
        
        # Get the current market regime (0 => low volatility; 1 => high 
        # volatility).
        regime = model.fit().smoothed_marginal_probabilities.values\
            .argmax(axis=1)[-1]
        self.plot('Regime', 'Volatility Class', regime)

        # Rebalance when the regime changes, when we aren't invested,
        # or when we need to rollover the contracts.
        if (regime != self._previous_regime or 
            not self.portfolio.invested or 
            self._expiry - self.time < self._min_expiry):
            # Close the previous straddle if there is one.
            self.liquidate()

            # Get the option chain.
            option_chains = self.current_slice.option_chains
            if self._option_symbol not in option_chains:
                return 
            chain = option_chains[self._option_symbol]
            min_expiry_date = self.time + self._min_expiry + self._min_hold_period
            expiries = [
                contract.expiry 
                for contract in chain 
                if contract.expiry >= min_expiry_date
            ]
            if not expiries:
                return

            # Define the straddle type. Low volatility regime => short 
            # straddle; High volatility regime => long straddle.
            if regime == 0:
                option_type = OptionStrategies.short_straddle
                expiry = min(expiries)
            else:
                option_type = OptionStrategies.straddle
                expiry = max(expiries)

            # Get the ATM strike price.
            strike = sorted(
                [contract for contract in chain if contract.expiry == expiry],
                key=lambda contract: abs(
                    chain.underlying.price - contract.strike
                )
            )[0].strike
            
            # Create the strategy. Low volatility regime => short 
            # straddle; High volaility regime => long straddle.
            option_strategy = option_type(
                self._option_symbol, strike, expiry
            )

            # Open the straddle trade.
            tickets = self.buy(option_strategy, 1)

            self._index.hedge_contracts = [t.symbol for t in tickets]
            self._expiry = expiry
        self._previous_regime = regime

    def on_order_event(self, order_event):
        # When one of the straddle legs are assigned/exercised,
        # the other leg is automatically liquidated so let's
        # remove the reference to each leg.
        if (order_event.status == OrderStatus.FILLED and 
            order_event.is_assignment):
            self._index.hedge_contracts = []
            self._expiry = datetime.min
