#region imports
from AlgorithmImports import *

from svmwavelet import SVMWavelet
#endregion


class SVMWaveletForecastingAlgorithm(QCAlgorithm):
    """
    This algorithm combines Support Vector Machines (SVM) and Wavelets 
    in an attempt to forecast the future price of Forex pairs. First, it
    decomposes the historical closing prices of each pair into 
    components using Wavelet decomposition. Second, it applies the SVM 
    to forecast one time-step ahead of each of the components. Lastly, 
    it recombines the components to get the aggregate forecast of the 
    SVM-Wavelet model.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)  
        self.set_end_date(2024, 4, 1)
        self.set_cash(1000000) 
        
        period = self.get_parameter("period", 152)
        self._leverage = self.get_parameter("leverage", 20)
        self._weight_threshold = self.get_parameter("weight_threshold", 0.005)

        self.set_benchmark(SecurityType.FOREX, "EURUSD")
        self.settings.minimum_order_margin_portfolio_percentage = 0

        for ticker in ["EURJPY", "GBPUSD", "AUDCAD", "NZDCHF"]:
            security = self.add_forex(ticker, leverage=self._leverage)

            # Create a RollingWindow to save trailing prices.
            security.window = RollingWindow[float](period)

            # Create and register a consolidator.
            consolidator = self.consolidate(
                security.symbol, Resolution.DAILY, TickType.QUOTE, 
                self._consolidation_handler
            )

            # Warm up the RollingWindow.
            history = self.history[QuoteBar](
                security.symbol, period, Resolution.DAILY
            )
            for bar in history:
                consolidator.update(bar)
        
        self._wavelet = SVMWavelet()
        
    def _consolidation_handler(self, bar):
        # Update the rolling window.
        security = self.securities[bar.symbol]
        security.window.add(bar.close)
        if self.is_warming_up: 
            return
        
        # Forecast and rebalance.
        prices = np.array(list(security.window))[::-1]
        forecasted_value = self._wavelet.forecast(prices)
        weight = (forecasted_value / bar.close) - 1
        if abs(weight) > self._weight_threshold:
            self.set_holdings(security.symbol, weight * self._leverage)

        # Plot the current state.
        ticker = security.symbol.value
        self.plot(ticker, "Price", bar.close)
        self.plot(ticker, "Forecast", forecasted_value)

