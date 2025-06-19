# region imports
from AlgorithmImports import *

from keras.saving import load_model
# endregion


class CNNPatternDetectionAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to load a trained keras model from 
    the Object Store and use its predictions to inform trading 
    decisions. Before you run this algorithm, run the cells in the 
    `research.ipynb` file.
    """
    _min_size = 25  # The model requires 25 data points as input.

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)
        self._security = self.add_forex("USDCAD", Resolution.DAILY)
        self._symbol = self._security.symbol

        self._max_size = self.get_parameter('max_size', 100)
        self._step_size = self.get_parameter('step_size', 10)
        self._confidence_threshold = self.get_parameter(
            'confidence_threshold', 0.5
        )  # 0.5 => 50%
        self._holding_period = timedelta(
            self.get_parameter('holding_period', 10)
        )

        self._model = load_model(
            self.object_store.get_file_path("head-and-shoulders-model.keras")
        )
        self._trailing_prices = pd.Series()
        self._liquidation_quantities = []

        chart = Chart('HS Patterns Detected')
        self.add_chart(chart)
        series = [
            Series('USDCAD Price', SeriesType.LINE, 0),
            Series('End of Pattern Detected', SeriesType.SCATTER, 0),
            Series("Window Length", SeriesType.SCATTER, 1)
        ]
        for s in series:
            chart.add_series(s)

    def on_data(self, data):
        t = self.time
        price = data[self._symbol].close
        # Plot the USDCAD price.
        self.plot('HS Patterns Detected', 'USDCAD Price', price)

        # Update the trailing window.
        self._trailing_prices.loc[t] = price
        self._trailing_prices = self._trailing_prices.iloc[-self._max_size:]

        # Calculate the order quantity.
        quantity = 0
        
        self._window_lengths_detected = []        
        for size in range(self._min_size, self._max_size + 1, self._step_size):
            # Ensure there are enough trailing data points to fill this 
            # window size.
            if len(self._trailing_prices) < size:
                continue

            window_trailing_prices = self._trailing_prices.iloc[-size:]
            # Downsample the trailing prices in this window to be 25 
            # data points.
            low_res_window = downsample(window_trailing_prices.values)
            # Standardize the downsampled trailing prices.
            factors = np.array(
                (
                    (low_res_window - low_res_window.mean()) 
                    / low_res_window.std()
                ).reshape(1, self._min_size, 1)
            )
            # Get the probability of the technical trading pattern in 
            # the downsampled and standardized window.
            prediction = self._model.predict(factors, verbose=0)[0][0]
            if prediction > self._confidence_threshold:
                self.log(
                    f"{t}: Pattern detected between "
                    + f"{window_trailing_prices.index[0]} and "
                    + f"{window_trailing_prices.index[-1]} with "
                    + f"{round(prediction * 100, 1)}% confidence."
                )
                
                self.plot(
                    "HS Patterns Detected", 'End of Pattern Detected', price
                )
                self._window_lengths_detected.append(size)
                quantity -= 10_000
        
        # Plot the window length values.
        for i in range(len(self._window_lengths_detected)):
            t = self.time + timedelta(seconds=i)
            self.schedule.on(
                self.date_rules.on(t.year, t.month, t.day),
                self.time_rules.at(t.hour, t.minute, t.second),
                lambda: self.plot(
                    "HS Patterns Detected", "Window Length", 
                    self._window_lengths_detected.pop(0)
                )
            )
        
        if quantity:
            self._cad_before_sell = self.portfolio.cash_book['CAD'].amount
            # Place the entry order.
            self.market_order(self._symbol, quantity)
            # Schedule the exit order.
            t_exit = t + self._holding_period
            self.schedule.on(
                self.date_rules.on(t_exit.year, t_exit.month, t_exit.day),  
                self.time_rules.at(t_exit.hour, t_exit.minute), 
                self._liquidate_position
            )

    def _liquidate_position(self):
        quantity = round(
            self._liquidation_quantities.pop(0) / self._security.ask_price
        )
        if quantity:
            self.market_order(self._symbol, quantity)

    def on_order_event(self, order_event):
        # When the entry order fills, record the amount of CAD
        # traded so that we can liquidate the correct amount later.
        if (order_event.status == OrderStatus.FILLED and 
            order_event.direction == OrderDirection.SELL):
            self._liquidation_quantities.append(
                self.portfolio.cash_book['CAD'].amount - self._cad_before_sell
            )


def downsample(values, num_points=25):
    if num_points == len(values):
        return values

    adj_values = []
    duplicates = int(2 * len(values) / num_points)
    if duplicates > 0:
        for x in values:
            for i in range(duplicates):
                adj_values.append(x)
    else:
        adj_values = values
    
    num_steps = num_points - 2
    step_size = int(len(adj_values) / num_steps)

    smoothed_data = [adj_values[0]]
    for i in range(num_steps):
        start_idx = i * step_size
        if i == num_steps - 1:
            end_idx = len(adj_values) - 1
        else:
            end_idx = (i + 1) * step_size - 1
        segment = np.array(adj_values[start_idx : end_idx+1])

        avg = sum(segment) / len(segment)
        smoothed_data.append(avg)
        
    smoothed_data.append(adj_values[-1])

    return np.array(smoothed_data)
