# region imports
from AlgorithmImports import *

from sklearn.decomposition import PCA
import statsmodels.api as sm
# endregion


class PCAStatArbitrageAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates a way to use principal component 
    analysis and linear regression to perform statistical arbitrage. 
    Statistical arbitrage strategies uses mean-reversion models to take 
    advantage of pricing inefficiencies between groups of correlated 
    securities. First, this implementation calculates the three 
    principal components. Second, it fits a linear regression model for 
    each security using the principal components as the independent
    variables and the asset's historical prices as the dependent 
    variable. Third, it derives the weight of each stock in the 
    portfolio based on its price deviation, which is measured by the 
    residual of the regression model.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(1_000_000)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        self._num_components = self.get_parameter("num_components", 3)
        self._lookback = self.get_parameter("lookback_days", 60)
        self._z_score_threshold = self.get_parameter("z_score_threshold", 1.5)
        self._universe_size = self.get_parameter("universe_size", 100)

        schedule_symbol = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.month_start(schedule_symbol)
        self.universe_settings.schedule.on(date_rule)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self._universe = self.add_universe(self._select_assets)

        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(schedule_symbol, 1), 
            self._trade
        )

        chart = Chart('Explained Variance Ratio')
        self.add_chart(chart)
        for i in range(self._num_components):
            chart.add_series(
                Series(f"Component {i}", SeriesType.LINE, "")
            )

    def _select_assets(self, fundamental):
        # Select the securities that have the most dollar volume and 
        # are above $5.
        return [
            f.symbol 
            for f in sorted(
                [f for f in fundamental if f.price > 5], 
                key=lambda f: f.dollar_volume
            )[-self._universe_size:]
        ]

    def _trade(self):
        tradeable_assets = [
            symbol 
            for symbol in self._universe.selected 
            if (self.securities[symbol].price and 
                symbol in self.current_slice.quote_bars)
        ]
        # Get historical data with the scaled raw normalization mode
        # so that the history is adjusted only for the splits and 
        # dividends that have occurred in the past.
        history = self.history(
            tradeable_assets, self._lookback, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        ).close.unstack(level=0)

        # Select the desired symbols and their weights for the portfolio
        # from the universe of symbols.
        weights = self._get_weights(history)
        
        # If the residual is greatly deviated from 0, enter the position
        # in the opposite way (mean reversion).
        self.set_holdings(
            [
                PortfolioTarget(symbol, -weight) 
                for symbol, weight in weights.items()
            ], 
            True
        )

    def _get_weights(self, history):
        """
        Get the finalized selected symbols and their weights according 
        to their level of deviation of the residuals from the linear 
        regression after PCA for each symbol.
        """
        # Get sample data for PCA (smooth it using np.log function).
        sample = np.log(history.dropna(axis=1))
        sample -= sample.mean()  # Center it column-wise

        # Fit the PCA model for sample data.
        model = PCA().fit(sample)

        # Plot the PCA results.
        for i in range(self._num_components):
            self.plot(
                'Explained Variance Ratio', f"Component {i}", 
                model.explained_variance_ratio_[i]
            )

        # Get the first n_components factors.
        factors = np.dot(sample, model.components_.T)[:,:self._num_components]

        # Add 1's to fit the linear regression (intercept).
        factors = sm.add_constant(factors)

        # Train Ordinary Least Squares linear model for each stock.
        model_by_ticker = {
            ticker: sm.OLS(sample[ticker], factors).fit() 
            for ticker in sample.columns
        }

        # Get the residuals from the linear regression after PCA for each stock.
        resids = pd.DataFrame(
            {ticker: model.resid for ticker, model in model_by_ticker.items()}
        )

        # Get the Z scores by standarizing the given pandas dataframe.
        # This is the residual of the most recent day.
        zscores = ((resids - resids.mean()) / resids.std()).iloc[-1] 

        # Get the stocks far from their mean (for mean reversion).
        selected = zscores[zscores < -self._z_score_threshold]

        # Return the weights for each selected stock.
        weights = selected * (1 / selected.abs().sum())
        return weights.sort_values()


