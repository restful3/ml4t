# region imports
from AlgorithmImports import *

from QuantConnect.PredictNowNET import PredictNowClient
from QuantConnect.PredictNowNET.Models import *
from time import sleep
from datetime import datetime
# endregion


class PredictNowCPOAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to use PredictNow.ai to perform
    Conditional Portfolio Optimization (CPO). CPO utilizes the trailing
    asset returns and hundreds of other market features from PredictNow
    to determine weights that will maximize the future 1-month Sharpe 
    ratio of the portfolio. The algorithm rebalances at the beginning 
    of each month. To backtest this algorithm, first run the cells in
    the `research.ipynb` notebook.
    """

    def initialize(self):
        self.set_start_date(2020, 2, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100000)
        
        # Define the universe.
        tickers = [
            "TIP", "BWX", "EEM", "VGK", "IEF", "QQQ", "EWJ", "GLD", 
            "VTI", "VNQ", "TLT", "RWX", "SPY", "DBC", "REM", "SCZ"
        ]
        self._symbols = [self.add_equity(ticker).symbol for ticker in tickers]
        
        if self.live_mode:
            # Connect to PredictNow.
            self._client = PredictNowClient("jared@quantconnect.com")
            if not self._client.connected:
                self.quit(f'Could not connect to PredictNow')
            # Define some parameters.
            self._in_sample_backtest_duration = timedelta(
                self.get_parameter("in_sample_days", 365)
            )
            self._out_of_sample_backtest_duration = timedelta(
                self.get_parameter("out_of_sample_days", 365)
            )
        else:
            # Read the weights file from Object Store.
            self._weights_by_date = pd.read_json(
                self.object_store.Read("ETF_Weights_Test1.csv")
            )
        # Schedule training and trading sessions.
        self.train(
            self.date_rules.month_start(self._symbols[0].symbol), 
            self.time_rules.after_market_open(self._symbols[0].symbol, -10),
            self._rebalance
        )
        # Add a warm-up period to avoid errors on the first rebalance.
        self.set_warm_up(timedelta(7))
    
    def _rebalance(self):
        # Don't trade during warm-up.
        if self.is_warming_up:
            return
        # In live mode, get the weights from PredictNow.
        date = self.time.date()
        if self.live_mode:
            self._get_live_weights(date)

        # Create portfolio targets.
        targets = []
        for symbol, weight in self._weights_by_date[str(date)].items():
            self.log(f"Setting weight for {symbol.value} to {weight}")
            targets.append(PortfolioTarget(symbol, weight))

        # Rebalance the portfolio.
        self.set_holdings(targets, True)

    def _get_live_weights(self, date):
        self.log(f"Loading live weights for {date}")

        # Upload the returns file to PredictNow.
        # Note: the history request includes an extra 2 months of data
        # so that we can move the start and end dates of the in-sample 
        # and out-of-sample backtests so that they align with the start
        # and end of each month.
        self.debug(f"Uploading Returns file")
        returns_file_name = f"ETF_return_{str(date)}.csv"
        file_path = self.object_store.get_file_path(returns_file_name)
        returns = self.history(
            self._symbols, 
            (self._in_sample_backtest_duration 
            + self._out_of_sample_backtest_duration
            + timedelta(60)),
            Resolution.DAILY
        ).close.unstack(0).pct_change().dropna()
        returns.to_csv(file_path)
        self._client.upload_returns_file(file_path)
        self.debug(f"Uploaded file: {file_path}")

        # Create the asset weight constraints.
        content = "component,LB,UB"
        constraints_by_symbol = {
            Symbol.create(ticker, SecurityType.EQUITY, Market.USA).id: contraint
            for ticker, contraint in {
                "SPY": (0, 0.5),
                "QQQ": (0, 0.5),
                "VNQ": (0, 0.5),
                "REM": (0, 0.33),
                "IEF": (0, 0.5),
                "TLT": (0, 0.5)
            }.items()
        }
        
        # Upload the contraints file to PredictNow.
        self.debug(f"Uploading Contraints file")
        for symbol, boundaries in constraints_by_symbol.items():
            content += f'\n{symbol},{boundaries[0]},{boundaries[1]}'
        constraints_file_name = f"ETF_constraint_{str(date)}.csv"
        self.object_store.save(constraints_file_name, content)
        file_path = self.object_store.get_file_path(constraints_file_name)
        self.debug(f"Uploaded file: {file_path}")
        self._client.upload_constraint_file(file_path)
        
        # Define the portfolio parameters.
        portfolio_parameters = PortfolioParameters(
            name=f"Demo_Project_{str(date)}",
            returns_file=returns_file_name,
            constraint_file=constraints_file_name,
            max_cash=1.0,
            rebalancing_period_unit="month",
            rebalancing_period=1,
            rebalance_on="first",
            training_data_size=3,
            evaluation_metric="sharpe"
        )

        # Calculate the dates of the in- and out-of-sample backtests.
        oos_start_date, oos_end_date = self._get_start_and_end_dates(
            date, self._out_of_sample_backtest_duration
        )
        is_start_date, is_end_date = self._get_start_and_end_dates(
            oos_start_date-timedelta(1), self._in_sample_backtest_duration
        )

        # Run the in-sample backtest.
        self.debug("Running in-sample backtest")
        in_sample_result = self._client.run_in_sample_backtest(
            portfolio_parameters,
            training_start_date=is_start_date,
            training_end_date=is_end_date,
            sampling_proportion=0.3,
            debug="debug"
        )
        in_sample_job = self._client.get_job_for_id(in_sample_result.id)
        
        # Run the out-of-sample backtest.
        self.debug("Running out-of-sample backtest")
        out_of_sample_result = self._client.run_out_of_sample_backtest(
            portfolio_parameters,
            training_start_date=oos_start_date,
            training_end_date=oos_end_date,
            debug="debug"
        )
        out_of_sample_job = self._client.get_job_for_id(out_of_sample_result.id)
        
        # Wait until the backtests finish running.
        self.debug("Checking if the backtests are done")
        while (in_sample_job.status != "SUCCESS" or 
               out_of_sample_job.status != "SUCCESS"):
            in_sample_job = self._client.get_job_for_id(in_sample_result.id)
            out_of_sample_job = self._client.get_job_for_id(
                out_of_sample_result.id
            )
            self.debug(f"In Sample Job: {in_sample_job.status}")
            self.debug(f"Out of Sample Job: {out_of_sample_job.status}")
            sleep(60)

        # Run the live prediction.
        self.debug("Running Live Prediction")
        exchange_hours = self.securities[self._symbols[0]].exchange.hours
        live_prediction_result = self._client.run_live_prediction(
            portfolio_parameters,
            rebalance_date=date,
            next_rebalance_date=exchange_hours.get_next_market_open(
                Expiry.end_of_month(self.time), extended_market_hours=False
            ).date(),
            debug="debug"
        )
        live_job = self._client.get_job_for_id(live_prediction_result.id)
        
        # Wait until the live prediction job is done.
        self.debug("Checking Live prediction job status")
        while live_job.status != "SUCCESS":
            live_job = self._client.get_job_for_id(live_prediction_result.id)
            self.debug(f"Live Prediction status: {live_job.status}")
            sleep(60)

        # Get the prediction weights.
        self._weights_by_date = self._client.get_live_prediction_weights(
            portfolio_parameters, 
            rebalance_date=date,
            debug="debug"
        )

    def _get_start_and_end_dates(self, date, duration):
        start_date = end_date - duration
        start_date = datetime(start_date.year, start_date.month, 1)
        return start_date, end_date
