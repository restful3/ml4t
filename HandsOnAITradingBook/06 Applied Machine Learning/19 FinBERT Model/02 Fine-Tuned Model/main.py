# region imports
from AlgorithmImports import *

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, set_seed
from pathlib import Path
from datasets import Dataset
import pytz
# endregion

class FinbertBaseModelAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to fine-tune the "ProsusAI/finbert" 
    model from HuggingFace to determine the sentiment of the latest news
    releases for an asset. At the start of every month, the universe 
    first selects the top 10 most liquid assets and then narrows the 
    selection down to just the most volatile asset. When it's time 
    fine-tune the model at the start of the month, the algorithm 
    gets the news articles that were released for the universe of assets
    over the last 30 days. The label that we use to fine-tune the model
    is the assets return between two consecutive news releases. The
    algorithm uses the same rebalancing logic as the previous example.
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
        # Schedule rebalances.
        self._last_rebalance_time = datetime.min
        self.schedule.on(
            self.date_rules.month_start(spy, 1),
            self.time_rules.midnight,
            self._trade
        )
        # Add a warm-up period.
        self.set_warm_up(timedelta(30))
        # Define the model and some settings.
        self._model_name = "ProsusAI/finbert"
        self._tokenizer = BertTokenizer.from_pretrained(self._model_name) 

    def on_warmup_finished(self):
        self._trade()

    def on_securities_changed(self, changes):
        for security in changes.removed_securities:
            self.remove_security(security.dataset_symbol)
        for security in changes.added_securities:
            security.dataset_symbol = self.add_data(
                TiingoNews, security.symbol
            ).symbol

    def _trade(self):
        if (self.is_warming_up or 
            self.time - self._last_rebalance_time < timedelta(14)):
            return

        # Get the target security.
        security = self.securities[list(self._universe.selected)[0]]

        # Get samples to fine-tune the model
        samples = pd.DataFrame(columns=['text', 'label'])
        news_history = self.history(security.dataset_symbol, 30, Resolution.DAILY)
        if news_history.empty:
            return
        news_history = news_history.loc[security.dataset_symbol]['description']
        asset_history = self.history(
            security.symbol, timedelta(30), Resolution.SECOND
        ).loc[security.symbol]['close']
        for i in range(len(news_history.index)-1):
            # Get factor (article description).
            factor = news_history.iloc[i]
            if not factor:
                continue

            # Get the label (the market reaction to the news, for now).
            release_time = self._convert_to_eastern(news_history.index[i])
            next_release_time = self._convert_to_eastern(news_history.index[i+1])
            reaction_period = asset_history[
                (asset_history.index > release_time) &
                (asset_history.index < next_release_time + timedelta(seconds=1))
            ]
            if reaction_period.empty:
                continue
            label = (
                (reaction_period.iloc[-1] - reaction_period.iloc[0]) 
                / reaction_period.iloc[0]
            )
            
            # Save the training sample.
            samples.loc[len(samples), :] = [factor, label]

        samples = samples.iloc[-100:]
        
        if samples.shape[0] < 10:
            self.liquidate()
            return
        
        # Classify the market reaction into positive/negative/neutral.
        # 75% of the most negative labels => class 0 (negative)
        # 75% of the most positive labels => class 2 (positive)
        # Remaining labels                => class 1 (netural)
        sorted_samples = samples.sort_values(
            by='label', ascending=False
        ).reset_index(drop=True)
        percent_signed = 0.75
        positive_cutoff = (
            int(percent_signed 
            * len(sorted_samples[sorted_samples.label > 0]))
        )
        negative_cutoff = (
            len(sorted_samples) 
            - int(percent_signed * len(sorted_samples[sorted_samples.label < 0]))
        )
        sorted_samples.loc[
            list(range(negative_cutoff, len(sorted_samples))), 'label'
        ] = 0
        sorted_samples.loc[
            list(range(positive_cutoff, negative_cutoff)), 'label'
        ] = 1
        sorted_samples.loc[list(range(0, positive_cutoff)), 'label'] = 2       

        # Load the pre-trained model.
        model = TFBertForSequenceClassification.from_pretrained(
            self._model_name, num_labels=3, from_pt=True
        )
        # Compile the model.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        # Create the training dataset.
        dataset = Dataset.from_pandas(sorted_samples)
        dataset = dataset.map(
            lambda sample: self._tokenizer(
                sample['text'], padding='max_length', truncation=True
            )
        )
        dataset = model.prepare_tf_dataset(
            dataset, shuffle=True, tokenizer=self._tokenizer
        )
        # Train the model.
        model.fit(dataset, epochs=2)
        # Prepare the input sentences.
        inputs = self._tokenizer(
            list(samples['text'].values), padding=True, truncation=True, 
            return_tensors='tf'
        )

        # Get the model outputs.
        outputs = model(**inputs) 

        # Apply softmax to the outputs to get probabilities.
        scores = tf.nn.softmax(outputs.logits, axis=-1).numpy()
        scores = self._aggregate_sentiment_scores(scores)
        
        self.plot("Sentiment Probability", "Negative", scores[0])
        self.plot("Sentiment Probability", "Neutral", scores[1])
        self.plot("Sentiment Probability", "Positive", scores[2])

        # Rebalance.
        weight = 1 if scores[2] > scores[0] else -0.25
        self.set_holdings(security.symbol, weight, True)
        self._last_rebalance_time = self.time

    def _convert_to_eastern(self, dt):
        return dt.astimezone(pytz.timezone('US/Eastern')).replace(tzinfo=None)

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
