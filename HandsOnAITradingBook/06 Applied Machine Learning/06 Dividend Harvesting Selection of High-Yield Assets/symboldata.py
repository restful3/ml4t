#region imports
from AlgorithmImports import *

from sklearn.tree import DecisionTreeRegressor
#endregion


class SymbolData:

    def __init__(self, lookback_length=25, minimum_samples=10):
        self._lookback_length = lookback_length
        self._minimum_samples = minimum_samples
        
        # Create RollingWindow objects to store factors.
        self._factor_timestamps = []
        self._factor_names = [
            'pe_ratio', 'revenue_growth', 'free_cash_flow_pct', 
            'dividend_payout_ratio', 'current_ratio'
        ]
        self._factor_history = {
            factor_name: RollingWindow[float](lookback_length) 
            for factor_name in self._factor_names
        }
        self._last_file_date = datetime.min
        
        # Create RollingWindow objects to store labels.
        self._label_timestamps = []
        self._label_history = RollingWindow[float](lookback_length)

        # Define the ML model.
        self._model = DecisionTreeRegressor(random_state=0)

    def update_factors(self, fundamental):
        # Check if the company has released new financial reports.
        f = fundamental
        file_date = max(
            f.financial_statements.file_date.three_months, 
            f.financial_statements.file_date.twelve_months
        )
        if file_date <= self._last_file_date:
            return
        self._last_file_date = file_date

        # Record the date when the new factors were first available.
        self._factor_timestamps.append(f.end_time)
        self._factor_timestamps = self._factor_timestamps[
            -self._lookback_length:
        ]

        # Update the factor history.
        cf_statement = f.financial_statements.cash_flow_statement
        factor_values_by_name = {
            'pe_ratio': f.valuation_ratios.pe_ratio,
            'revenue_growth': f.operation_ratios.revenue_growth.three_months,
            'free_cash_flow_pct': (
                cf_statement.free_cash_flow.three_months 
                / cf_statement.operating_cash_flow.three_months
            ) if cf_statement.operating_cash_flow.three_months else 0,
            'dividend_payout_ratio': (
                -cf_statement.cash_dividends_paid.three_months / 
                f.financial_statements.income_statement.net_income.three_months 
            ) if f.financial_statements.income_statement.net_income.three_months 
                else 0,
            'current_ratio': f.operation_ratios.current_ratio.three_months
        }
        for factor_name, value in factor_values_by_name.items():
            self._factor_history[factor_name].add(value)
    
    def update_labels(self, dividend):
        t = dividend.end_time 
        if t in self._label_timestamps:
            return 
        self._label_timestamps.append(t)
        self._label_history.Add(dividend.distribution / dividend.reference_price)

    def train(self):
        if (len(self._factor_timestamps) < self._minimum_samples or 
            len(self._label_timestamps) < self._minimum_samples):
            return None

        # Align factors and labels. In order to maximize the training
        # samples, there are two cases to consider:
        #  Case 1: Consecutive factor samples and then a single label
        #          -> In this case, map each factor sample to the label.
        #          -> Ex:  X_1 ... X_2 ... y_1 => (X_1, y_1), (X_2, y_1) 
        #  Case 2: A single factor sample and then consecutive labels
        #          -> In this case, map the factor sample to each label.
        #          -> Ex:  X_1 ... y_1 ... y_2 => (X_1, y_1), (X_1, y_2)
        timestamps = []
        for i, label_timestamp in enumerate(self._label_timestamps):
            # Find all factor timestamps that came between this label 
            # and the previous label.
            previous_label_timestamp = self._label_timestamps[i-1] if i > 0 \
                else datetime.min
            factor_timestamps_between_labels = [
                t 
                for t in self._factor_timestamps 
                if previous_label_timestamp <= t < label_timestamp
            ]
            
            # Check if there were factor samples between this label and 
            # the previous label.
            if factor_timestamps_between_labels:
                # Pair this label with all of the factor samples that 
                # came between this label and the previous label.
                for factor_timestamp in factor_timestamps_between_labels:
                    timestamps.append((factor_timestamp, label_timestamp))
            else:
                # Pair this label with the most recent factor sample.
                earlier_factor_timestamps = [
                    t 
                    for t in self._factor_timestamps 
                    if t < label_timestamp
                ]
                if earlier_factor_timestamps:
                    factor_timestamp = max(earlier_factor_timestamps)
                    timestamps.append((factor_timestamp, label_timestamp))

        # Gather the training data (input: current factors; output: 
        # *next* dividend yield).
        X = np.ndarray((len(timestamps), len(self._factor_names)))
        y = []
        for row_index, (factor_time, label_time) in enumerate(timestamps):
            for column_index, factor_name in enumerate(self._factor_names):
                # Convert the list index into a RollingWindow index.
                rolling_window_idx = (
                    len(self._factor_timestamps) 
                    - self._factor_timestamps.index(factor_time) 
                    - 1
                )
                # Add the factor value to the X matrix.
                X[row_index, column_index] = self._factor_history[factor_name][
                    rolling_window_idx
                ]
            # Convert the list index into RollingWindow index.
            rolling_window_idx = (
                len(self._label_timestamps) 
                - self._label_timestamps.index(label_time) 
                - 1
            )
            # Add the label value to the y vector.
            y.append(self._label_history[rolling_window_idx])
        y = np.array(y)

        if len(y) < self._minimum_samples:
            return None

        # Train the model.
        self._model.fit(X, y)

        # Return the R Squared.
        return self._model.score(X, y)

    def predict(self):
        latest_factor_values = [
            self._factor_history[factor_name][0] 
            for factor_name in self._factor_names
        ]
        return self._model.predict([latest_factor_values])[0]

