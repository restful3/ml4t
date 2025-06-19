# region imports
from AlgorithmImports import *

from sklearn.naive_bayes import GaussianNB
from dateutil.relativedelta import relativedelta
import pickle
# endregion


class GaussianNaiveBayesAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates one way to use Gaussian Naïve Bayes 
    (GNB) classifiers to forecast the daily returns of stocks in the 
    technology sector given the historical returns of the sector. In 
    this example, the classes in the model are: positive, negative, or 
    flat future return for a security. The features are the last 4 daily
    returns of all the universe constituents.
    """

    def initialize(self):
        self.set_start_date(2019, 1, 1)
        self.set_end_date(2024, 4, 1)
        self.set_cash(1_000_000)

        self._days_per_sample = self.get_parameter('days_per_sample', 4)
        self._samples = self.get_parameter('samples', 100)
        # `_holding_period` is calendar days, not trading days.
        self._holding_period = 30
        self._lookback = (
            self._days_per_sample
            + self._samples
            + self._holding_period 
            + 1
        )

        schedule_symbol = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.week_start(schedule_symbol)
        self.train(date_rule, self.time_rules.at(9, 0), self._train)
        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(schedule_symbol, 2), 
            self._trade
        )

        self._universe_size = self.get_parameter("universe_size", 10)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self.universe_settings.schedule.on(date_rule)
        self._universe = self.add_universe(self._select_assets)

        # Load pre-trained and pickled models from the Object Store.
        if self.live_mode:    
            self._models_by_symbol = {}
            self._key = 'gnb_models.pkl'
            if self.object_store.contains_key(self._key):
                self._models_by_symbol = pickle.loads(
                    self.object_store.read_bytes(self._key)
                )

    def _select_assets(self, fundamental):
        # Select the largest tech stocks.
        tech_stocks = [
            f 
            for f in fundamental 
            if (f.asset_classification.morningstar_sector_code == 
                MorningstarSectorCode.TECHNOLOGY)
        ]
        sorted_by_market_cap = sorted(tech_stocks, key=lambda x: x.market_cap)
        return [x.symbol for x in sorted_by_market_cap[-self._universe_size:]]

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            security.model = None
            self._set_up_consolidator(security)
            self._warm_up(security)
            
        for security in changes.removed_securities:
            self.subscription_manager.remove_consolidator(
                security.symbol, security.consolidator
            )

    def _set_up_consolidator(self, security):
        security.consolidator = self.consolidate(
            security.symbol, Resolution.DAILY, self._consolidation_handler
        )

    def _warm_up(self, security):
        security.roc_window = np.array([])
        security.previous_opens = pd.Series()
        security.labels_by_day = pd.Series()
        security.features_by_day = pd.DataFrame(
            {
                f'{security.Symbol.ID}_(t-{i})' : [] 
                for i in range(1, self._days_per_sample + 1)
            }
        )

        # Get historical prices.
        history = self.history(
            security.symbol, self._lookback, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        )
        if history.empty or 'close' not in history:
            self.log(f"Not enough history for {security.symbol} yet")
            return

        # Calculate the features.
        history = history.loc[security.symbol]
        history['open_close_return'] = (
            (history.close - history.open) / history.open
        )

        # Calculate the labels.        
        start = history.shift(-1).open
        end = history.shift(-22).open  # Trading days instead of calendar days.
        history['future_return'] = (end - start) / start

        for day, row in history.iterrows():
            security.previous_opens[day] = row.open
            
            # Update the features.
            if not self._update_features(security, day, row.open_close_return):
                continue

            # Update the labels.
            if not pd.isnull(row.future_return):
                security.labels_by_day[day] = np.sign(row.future_return)
                security.labels_by_day = security.labels_by_day[
                    -self._samples:
                ]
        security.previous_opens = security.previous_opens[
            -self._holding_period:
        ]

    def _consolidation_handler(self, bar):
        security = self.securities[bar.symbol]
        time = bar.end_time
        open_ = bar.open
        close = bar.close
        
        # Update the features.
        open_close_return = (close - open_) / open_
        if not self._update_features(security, time, open_close_return):
            return
        
        # Update the labels.
        open_days = security.previous_opens[
            (security.previous_opens.index <= 
            time - timedelta(self._holding_period))
        ]
        if len(open_days) == 0:
            return 
        open_day = open_days.index[-1]
        previous_open = security.previous_opens[open_day]
        open_open_return = (open_ - previous_open) / previous_open
        security.labels_by_day[open_day] = np.sign(open_open_return)
        security.labels_by_day = security.labels_by_day[-self._samples:]
            
        security.previous_opens.loc[time] = open_
        security.previous_opens = security.previous_opens[
            -self._holding_period:
        ]

    def _update_features(self, security, day, open_close_return):
        """
        Updates the training data features.
        
        Inputs
         - day
            Timestamp of when we're aware of the open_close_return
         - open_close_return
            Open to close intraday return
            
        Returns T/F, showing if the features are in place to start 
        updating the training labels.
        """
        security.roc_window = np.append(
            open_close_return, security.roc_window
        )[:self._days_per_sample]
        
        if len(security.roc_window) < self._days_per_sample: 
            return False
            
        security.features_by_day.loc[day] = security.roc_window
        security.features_by_day = security.features_by_day[
            -(self._samples + self._holding_period + 2):
        ]
        return True

    def _train(self):
        """
        Trains the Gaussian Naive Bayes classifier model.
        """
        features = pd.DataFrame()
        labels_by_symbol = {}

        self._tradable_securities = []
        for symbol in self._universe.selected:
            security = self.securities[symbol]
            if self._is_ready(security):
                self._tradable_securities.append(security)
                features = pd.concat(
                    [features, security.features_by_day], 
                    axis=1
                )
                labels_by_symbol[symbol] = security.labels_by_day
        
        # The first and last row can have NaNs because this `_train` 
        # method fires when the universe changes, which is before the 
        # consolidated bars close. Let's remove them.
        features.dropna(inplace=True) 

        # Find the intersection of the indices for the features and 
        # labels. 
        idx = set([t for t in features.index])
        for i, (symbol, labels) in enumerate(labels_by_symbol.items()):
            a = set([t for t in labels.index])
            idx &= a
        idx = sorted(list(idx))
        
        for security in self._tradable_securities:
            symbol = security.symbol
            security.model = GaussianNB().fit(
                features.loc[idx], labels_by_symbol[symbol].loc[idx]
            )
            # If we're live trading, save the trained model in case the
            # algorithm is stopped and re-deployed before the 
            # `_trade` method runs.
            if self.live_mode:
                key = str(symbol.id)
                self._models_by_symbol[key] = pickle.dumps(security.model)

    def _is_ready(self, security):
        return (
            security.features_by_day.shape[0] == 
            self._samples + self._holding_period + 2
        )

    def on_splits(self, splits):
        for symbol, split in splits.items():
            if split.type != SplitType.SPLIT_OCCURRED:
                continue
            security = self.securities[symbol]
            # Reset the consolidator and warm-up the factors
            # and labels.
            self.subscription_manager.remove_consolidator(
                symbol, security.consolidator
            )
            self._set_up_consolidator(security)
            self._warm_up(security)

    def _trade(self):
        # Get the features.
        features = [[]]
        for security in self._tradable_securities:
            features[0].extend(security.features_by_day.iloc[-1].values)
        
        # Select the assets that the model predicts will have
        # positive returns.
        long_symbols = []
        for security in self._tradable_securities:
            # If the live algorithm hasn't called _train yet, load 
            # the trained model from the Object Store.
            key = str(security.symbol.id)
            if (self.live_mode and 
                not hasattr(security, 'model') and
                key in self._models_by_symbol):
                security.model = pickle.loads(self._models_by_symbol[key])
            # If the model predicts 1, save this asset to long.
            if security.model.predict(features) == 1:
                long_symbols.append(security.symbol)
        if len(long_symbols) == 0:
            return 
        # Rebalance the portfolio.
        weight = 1 / len(long_symbols)
        self.set_holdings(
            [PortfolioTarget(symbol, weight) for symbol in long_symbols], 
            True
        )

        # Plot the results.
        self.plot(
            "Trades", "Tradable Securities", len(self._tradable_securities)
        )
        self.plot("Trades", "Target Assets", len(long_symbols))

    def on_end_of_algorithm(self):
        if self.live_mode:
            self.object_store.save_bytes(
                self._key, pickle.dumps(self._models_by_symbol)
            )
