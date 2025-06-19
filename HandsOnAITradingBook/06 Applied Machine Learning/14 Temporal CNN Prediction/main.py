# region imports
from AlgorithmImports import *

import math

from temporalcnn import TemporalCNN, Direction, factor_names
# endregion


class TemporalCNNPredictionAlgorithm(QCAlgorithm):
    """
    This algorithm demonstration one way to apply Deep Learning 
    Classification in an attempt to forecast the movement of future 
    stock prices. Specifically, the strategy uses a Temporal 
    Convolutional Neural Network model to predict the direction of 
    future prices based on several samples of trailing OHLCV data.
    """

    def initialize(self):
        self.set_start_date(2018, 12, 31)  
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)

        self._training_samples = self.get_parameter("training_samples", 500)
        self._universe_size = self.get_parameter("universe_size", 3)

        etf = Symbol.create("QQQ", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.week_start(etf)
        self.universe_settings.schedule.on(date_rule)
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self.universe_settings.asynchronous = True
        self._universe = self.add_universe(
            self.universe.etf(etf, universe_filter_func=self._select_assets)
        )

        self.train(date_rule, self.time_rules.at(9, 0), self._update_models)
        self.schedule.on(
            date_rule, self.time_rules.after_market_open(etf, 2), self._trade
        )

        # Load pre-trained and serialized models from the Object Store.
        if self.live_mode:
            self._models_by_symbol = {}
            self._key = 'cnn_models'
            if self.object_store.contains_key(self._key):
                self._models_by_symbol = json.loads(
                    self.object_store.read(self._key)
                )

    def _select_assets(self, constituents):
        # Select the assets with the largest weight in the ETF.
        constituents = [c for c in constituents if c.weight]
        if constituents: 
            return [
                c.symbol 
                for c in sorted(
                    constituents, key=lambda c: c.weight
                )[-self._universe_size:]
            ]
        return Universe.UNCHANGED

    def _trade(self):
        # Get predictions for all the assets.
        weight_by_symbol = {}
        for symbol in self._universe.selected:
            security = self.securities[symbol]
            symbol_df = security.history.tail(15)
            prediction, confidence = security.cnn.predict(symbol_df)
            if (prediction != Direction.STATIONARY and 
                not math.isnan(confidence) and 
                confidence > .55):
                factor = (-1 if prediction == Direction.DOWN else 1)
                weight_by_symbol[security.symbol] = factor * confidence
            self.plot("Confidence", str(security.symbol.id), confidence)
        
        # Calculate portfolio weights and rebalance.
        weight_sum = sum([abs(x) for x in weight_by_symbol.values()])
        weight_factor = 1 if weight_sum <= 1 else 1 / weight_sum
        portfolio_targets = [
            PortfolioTarget(symbol, weight * weight_factor) 
            for symbol, weight in weight_by_symbol.items()
        ]
        self.set_holdings(portfolio_targets, True)

    def on_splits(self, splits):
        for symbol, split in splits.items():
            if split.type == SplitType.SPLIT_OCCURRED:
                self._initialize_security(self.securities[symbol])

    def _initialize_security(self, security):
        symbol = security.symbol
        # Remove the old consolidator if there was one.
        if hasattr(security, 'consolidator'):
            self.subscription_manager.remove_consolidator(
                symbol, security.consolidator
            )
        # Add a new consolidator that adds bars to the asset history.
        security.consolidator = self.consolidate(
            symbol, Resolution.DAILY, self._consolidation_handler
        )
        # Warm up the history dataframe.
        security.history = self.history(
            symbol, self._training_samples, Resolution.DAILY, 
            data_normalization_mode=DataNormalizationMode.SCALED_RAW
        ).loc[symbol][factor_names]

    def _consolidation_handler(self, bar):
        security = self.securities[bar.symbol]
        security.history.loc[bar.end_time] = (
            bar.open, bar.high, bar.low, bar.close, bar.volume
        )
        security.history = security.history.iloc[-self._training_samples:]

    def on_securities_changed(self, changes):
        for security in changes.added_securities:
            serialized_model = None
            # If we're live trading and the algorithm was
            # re-deployed before `_update_models` runs, load
            # the pre-trained model from the last deployment.
            if self.live_mode:
                serialized_model = self._models_by_symbol.get(
                    str(security.symbol.id), None
                )
            security.cnn = TemporalCNN(serialized_model)
            self._initialize_security(security)
            
        for security in changes.removed_securities:
            self.subscription_manager.remove_consolidator(
                security.symbol, security.consolidator
            )

    def _update_models(self):
        for symbol in self._universe.selected:
            security = self.securities[symbol]            
            model_json = security.cnn.train(security.history)
            if self.live_mode:
                self._models_by_symbol[str(symbol.id)] = model_json

    def on_end_of_algorithm(self):
        if self.live_mode:
            self.object_store.save(
                self._key, json.dumps(self._models_by_symbol)
            )

            
