# region imports
from AlgorithmImports import *

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, set_seed
from pathlib import Path
# endregion

class FinbertBaseModelAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to use a pre-trained HuggingFace 
    model. It uses the "ProsusAI/finbert" model to determine the 
    sentiment of the latest news releases for an asset. At the start
    of every month, the universe first selects the top 10 most liquid 
    assets and then narrows the selection down to just the most 
    volatile asset. When it's time for the monthly rebalance, the
    algorithm calculates an aggregated sentiment score for all the 
    news releases over the last 10 days. If the model determines the
    sentiment is more positive than negative, the algorithm enters
    a long position. Otherwise, it enters a short position.
    """

    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2023, 1, 1)
        self.set_cash(100_000)

        # Define the universe.
        spy = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.schedule.on(self.date_rules.month_start(spy))
        # The universe selection function returns a list of Symbol
        # objects to define the universe constituents. The `history`
        # method returns a DataFrame where the first index level is
        # the Symbol and the second index is the time of the 
        # price sample.
        self._universe = self.add_universe(
            lambda fundamental: [
                self.history(
                    [
                        f.symbol 
                        for f in sorted(
                            fundamental, key=lambda f: f.dollar_volume
                        )[-10:]
                    ], 
                    timedelta(365), Resolution.DAILY
                )['close'].unstack(0).pct_change().iloc[1:].std().idxmax()
            ]
        )
        # Enable reproducibility.
        set_seed(1, True)
        
        # Load the tokenizer and the model
        model_path = "ProsusAI/finbert"
        self._tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        self._model = TFBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        # Schedule rebalances.
        self._last_rebalance_time = datetime.min
        self.schedule.on(
            self.date_rules.month_start(spy, 1),
            self.time_rules.midnight,
            self._trade
        )
        # Add a warm-up period.
        self.set_warm_up(timedelta(30))

    def on_warmup_finished(self):
        self._trade()

    def on_securities_changed(self, changes):
        for security in changes.removed_securities:
            self.remove_security(security.dataset_symbol)
        for security in changes.added_securities:
            security.dataset_symbol = self.add_data(TiingoNews, security.symbol).symbol

    def _trade(self):
        if self.is_warming_up or self.time - self._last_rebalance_time < timedelta(14):
            return

        # Get the target security.
        security = self.securities[list(self._universe.selected)[0]]

        # Get the latest news articles.
        articles = self.history[TiingoNews](security.dataset_symbol, 10, Resolution.DAILY)
        article_text = [article.description for article in articles]
        if not article_text:
            return

        # Prepare the input sentences.
        inputs = self._tokenizer(article_text, padding=True, truncation=True, return_tensors='tf')

        # Get the model outputs.
        outputs = self._model(**inputs)

        # Apply softmax to the outputs to get probabilities.
        scores = tf.nn.softmax(outputs.logits, axis=-1).numpy()
        self.log(f"{str(scores)}")
        scores = self._aggregate_sentiment_scores(scores)
        
        self.plot("Sentiment Probability", "Negative", scores[0])
        self.plot("Sentiment Probability", "Neutral", scores[1])
        self.plot("Sentiment Probability", "Positive", scores[2])

        # Rebalance the portfolio.
        weight = 1 if scores[2] > scores[0] else -0.25
        self.set_holdings(security.symbol, weight, True)
        self._last_rebalance_time = self.time

    def _aggregate_sentiment_scores(self, sentiment_scores):
        n = sentiment_scores.shape[0]
        
        # Generate exponentially increasing weights.
        weights = np.exp(np.linspace(0, 1, n))
        
        # Normalize weights to sum to 1.
        weights /= weights.sum()
        
        # Apply weights to sentiment scores.
        weighted_scores = sentiment_scores * weights[:, np.newaxis]
        
        # Aggregate weighted scores by summing them.
        aggregated_scores = weighted_scores.sum(axis=0)
        
        return aggregated_scores
