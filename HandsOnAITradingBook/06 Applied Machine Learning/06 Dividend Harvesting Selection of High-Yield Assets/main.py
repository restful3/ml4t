# region imports
from AlgorithmImports import *

from symboldata import SymbolData
# endregion


class DividendHarvestingAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to concentrate the portfolio in 
    assets that we think will produce the greatest yield from their next 
    dividend payment. Specifically, this algorithm uses a model decision
    tree regression model and the following factors to forecast future 
    dividend yields:
        - PE Ratio: The asset's PE ratio using the earning from the most
          recent financial report and the current price
        - Revenue growth: Growth in revenue between the last two 
          financial reports
        - Free cash flow percent: The ratio of free cash flow to 
          operating cash flow on the last financial report
        - Dividend payout ratio: The ratio of dividend paids to net 
          income during the last quarter
        - Current ratio: Current assets divided by current liabilities
    """
    _symbol_data_by_symbol = {}

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(1_000_000)
        self.set_security_initializer(
            BrokerageModelSecurityInitializer(
                self.brokerage_model, 
                FuncSecuritySeeder(self.get_last_known_prices)
            )
        )

        # Define the universe and settings.
        etf = Symbol.create("QQQ", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.month_start(etf)
        self.universe_settings.schedule.on(date_rule)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self.universe_settings.resolution = Resolution.HOUR
        self.universe_settings.extended_market_hours = True
        self._universe = self.add_universe(
            self.universe.etf(etf, universe_filter_func=self._select_assets)
        )
        self._universe_size = self.get_parameter('universe_size', 100)

        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(etf, -30), 
            self._trade
        )
        self._lookback = self.get_parameter('lookback_years', 5) * 365

        # Create a candlestick chart.
        chart = Chart('Model Training Results')
        chart.add_series(CandlestickSeries('R Squared', ''))
        self.add_chart(chart)

    def _select_assets(self, constituents):
        # Create SymbolData objects for each asset in the ETF.
        new_symbols = []
        for c in constituents:
            s = c.Symbol
            if s not in self._symbol_data_by_symbol:
                new_symbols.append(s)
                self._symbol_data_by_symbol[s] = SymbolData()
        
        # Warm-up the factors and labels for the new assets.
        # We need 5 years of data since we haven't collected
        # the factors and labels for these assets yet.
        self._update_factors(new_symbols, self._lookback)
        self._update_labels(new_symbols, self._lookback)
        
        # Update the factors and labels for the existing assets.
        # We only need the factors and labels of the last month
        # since this universe filter method runs once a month.
        existing_symbols = list(
            set(self._symbol_data_by_symbol.keys()) - set(new_symbols)
        )
        self._update_factors(existing_symbols, 31)
        self._update_labels(existing_symbols, 31)

        # Select the 100 largest assets in the ETF
        constituents = [c for c in constituents if c.weight]
        if not constituents:
            return Universe.UNCHANGED
        return [
            c.symbol 
            for c in sorted(constituents, key=lambda c: c.weight)[-self._universe_size:]
        ]

    def _update_labels(self, symbols, lookback_days):
        history = self.history[Dividend](symbols, timedelta(lookback_days))
        for dividends in history:
            for symbol, dividend in dividends.items():
                self._symbol_data_by_symbol[symbol].update_labels(dividend)

    def _update_factors(self, symbols, lookback_days):
        history = self.history[Fundamental](symbols, timedelta(lookback_days))
        for fundamental_dict in history:
            for symbol, asset_fundamentals in fundamental_dict.items():
                self._symbol_data_by_symbol[symbol].update_factors(asset_fundamentals)

    def _trade(self):
        # Predict the next dividend yield.
        r_squared_values = []
        prediction_by_symbol = {}
        for symbol in self._universe.selected:
            symbol_data = self._symbol_data_by_symbol[symbol]
            r_squared = symbol_data.train()
            if r_squared is None:
                continue
            r_squared_values.append(r_squared)
            prediction_by_symbol[symbol] = symbol_data.predict()

        # Plot the R Squared values.
        if r_squared_values:
            self.plot(
                'Model Training Results', 'R Squared', r_squared_values[0], 
                max(r_squared_values), min(r_squared_values), 
                r_squared_values[-1]
            )

        # Rebalance the portfolio so the weight of each asset is 
        # proportional to the expected dividend yield. i.e. Give greater
        # weight to assets we think will produce larger returns in 
        # dividends.
        prediction_sum = sum(prediction_by_symbol.values())
        portfolio_targets = [
            PortfolioTarget(symbol, prediction / prediction_sum) 
            for symbol, prediction in prediction_by_symbol.items()
        ]
        self.set_holdings(portfolio_targets, True)
        self.plot("Portfolio Size", "Count", len(portfolio_targets))

    def on_data(self, data):
        dividends_received = 0
        for symbol, dividend in data.dividends.items():
            security_holding = self.portfolio[symbol]
            if security_holding.invested:
                dividends_received += (
                    dividend.distribution * security_holding.quantity
                )
        self.plot("Dividends Received", "Value", dividends_received)

    def on_end_of_algorithm(self):
        # Log dividends received.
        self.log("Dividends received:")
        dividend_by_symbol = {
            security.symbol: security.holdings.total_dividends 
            for security in self.securities.total 
            if security.holdings.total_dividends
        }
        sorted_by_dividends_earned = sorted(
            dividend_by_symbol.items(), key=lambda x: x[1], reverse=True
        )
        for symbol, total_dividends in sorted_by_dividends_earned:
            self.log(f"- ${total_dividends} ({symbol.value})")
        self.log("-----------------")
        self.log(f"Total: ${sum(dividend_by_symbol.values())}")

        
