#region imports
from AlgorithmImports import *

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
#endregion


class MarkovModelAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to use a Markov-switching dynamic 
    regression model to detect 2 distinct regimes, a high volatility 
    regime and a low volatility regime. When the model detects a high 
    volatility regime, the algorithm allocates 100% of the portfolio to 
    TLT. When the model detects a low volatility regime, the algorithm 
    allocates 100% of the portfolio to SPY.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(1_000_000)
    
        self._spy = self.add_equity("SPY").symbol  # Low vol target
        self._tlt = self.add_equity("TLT").symbol  # High vol target
        
        self._lookback_period = timedelta(
            self.get_parameter('lookback_years', 3) * 365
        )

        # Create a trailing series of daily returns.
        self._daily_returns = pd.Series()
        roc = self.roc(self._spy, 1, Resolution.DAILY)
        roc.updated += self._update_event_handler
        history = self.history[TradeBar](
            self._spy, self._lookback_period + timedelta(7), Resolution.DAILY
        )
        for bar in history:
            roc.update(bar.end_time, bar.close)
        # Schedule trades.
        self.schedule.on(
            self.date_rules.every_day(self._spy),
            self.time_rules.after_market_open(self._spy, 1),
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

        # Rebalance when the regime changes.
        if regime != self._previous_regime:
            self.set_holdings(
                [
                    PortfolioTarget(self._tlt, regime), 
                    PortfolioTarget(self._spy, int(not regime))
                ]
            ) 
        self._previous_regime = regime

        
