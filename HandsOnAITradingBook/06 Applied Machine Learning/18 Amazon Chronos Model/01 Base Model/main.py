# region imports
from AlgorithmImports import *

import torch
from chronos import ChronosPipeline
from scipy.optimize import minimize
from transformers import set_seed 
# endregion

class HuggingFaceBaseModelDemo(QCAlgorithm):
    """
    This algorithm demonstrates how to use a pre-trained HuggingFace 
    model. It uses the "amazon/chronos-t5-tiny" model to forecast the 
    future equity curves of the 5 most liquid assets in the market,
    then it uses the SciPy package to find the portfolio weights
    that will maximize the future Sharpe ratio of the portfolio. 
    The portfolio is rebalanced every 3 months.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)

        # Disable the daily precise end time because of the time rules.
        self.settings.daily_precise_end_time = False
        self.settings.min_absolute_portfolio_target_percentage = 0

        # Enable reproducibility.
        set_seed(1, True)

        # Load the pre-trained model.
        self._pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )

        # Define the universe.
        spy = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        self.universe_settings.schedule.on(self.date_rules.month_start(spy))
        self.universe_settings.resolution = Resolution.DAILY
        self._universe = self.add_universe(
            self.universe.dollar_volume.top(
                self.get_parameter('universe_size', 5)
            )
        )

        # Define some trading parameters.
        self._lookback_period = timedelta(
            365 * self.get_parameter('lookback_years', 1)
        )
        self._prediction_length = 3*21  # Three months of trading days

        # Schedule rebalances.
        self._last_rebalance = datetime.min
        self.schedule.on(
            self.date_rules.month_start(spy, 1), 
            self.time_rules.midnight, 
            self._trade
        )

        # Add warm up so the algorithm trades on deployment.
        self.set_warmup(timedelta(31))

    def on_warmup_finished(self):
        # Trade right after warm up is done.
        self._trade()

    def _sharpe_ratio(
            self, weights, returns, risk_free_rate, trading_days_per_year=252):
        # Define how to calculate the Sharpe ratio so we can use
        # it to optimize the portfolio weights.

        # Calculate the annualized returns and covariance matrix.
        mean_returns = returns.mean() * trading_days_per_year 
        cov_matrix = returns.cov() * trading_days_per_year

        # Calculate the Sharpe ratio.
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        
        # Return negative Sharpe ratio because we minimize this
        # function in optimization.
        return -sharpe_ratio

    def _optimize_portfolio(self, equity_curves):
        returns = equity_curves.pct_change().dropna()
        num_assets = returns.shape[1]
        initial_guess = num_assets * [1. / num_assets,]
        # Find portfolio weights that mazimize the forward Sharpe
        # ratio.
        result = minimize(
            self._sharpe_ratio, 
            initial_guess, 
            args=(
                returns,
                self.risk_free_interest_rate_model.get_interest_rate(self.time)
            ), 
            method='SLSQP', 
            bounds=tuple((0, 1) for _ in range(num_assets)), 
            constraints=(
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
            )
        )    
        return result.x

    def _trade(self):
        # Don't rebalance during warm-up.
        if self.is_warming_up:
            return
        # Only rebalance on a quarterly basis.
        if self.time - self._last_rebalance < timedelta(80):
            return  
        self._last_rebalance = self.time

        symbols = list(self._universe.selected)

        # Get historical equity curves.
        history = self.history(symbols, self._lookback_period)['close'].unstack(0)
        
        # Forecast the future equity curves.
        all_forecasts = self._pipeline.predict(
            [
                torch.tensor(history[symbol].dropna()) 
                for symbol in symbols
            ], 
            self._prediction_length
        )
        
        # Take the median forecast for each asset.
        forecasts_df = pd.DataFrame(
            {
                symbol: np.quantile(
                    all_forecasts[i].numpy(), 0.5, axis=0   # 0.5 = median
                )
                for i, symbol in enumerate(symbols)
            }
        )

        # Find the weights that maximize the forward Sharpe 
        # ratio of the portfolio.
        optimal_weights = self._optimize_portfolio(forecasts_df)

        # Rebalance the portfolio.
        self.set_holdings(
            [
                PortfolioTarget(symbol, optimal_weights[i])
                for i, symbol in enumerate(symbols)
            ], 
            True
        )

    
