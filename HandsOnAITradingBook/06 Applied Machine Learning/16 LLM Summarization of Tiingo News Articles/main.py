# region imports
from AlgorithmImports import *
# endregion


class LLMSummarizationAlgorithm(QCAlgorithm):
    """
    This algorithm demonstrates how to load in the sentiment scores from
    the Object Store and use them to inform trading decisions. Before 
    you can run this algorithm, run the cells in the `research.ipynb` 
    file.
    """

    def initialize(self):
        self.set_start_date(2023, 11, 1)
        self.set_end_date(2024, 3, 1)
        self.set_cash(100_000)

        self._tsla = self.add_equity("TSLA")
        self._dataset_symbol = self.add_data(
            TiingoNewsSentiment, "TiingoNewsSentiment", Resolution.HOUR
        ).symbol
        self._roc = self.roc(self._dataset_symbol, 2)

        self.set_benchmark(self._tsla.symbol)

    def on_data(self, data):
        # Get the current sentiment.
        if self._dataset_symbol not in data:
            return
        sentiment = data[self._dataset_symbol].value
        
        # If the market isn't open right now, do nothing.
        if not self.is_market_open(self._tsla.symbol):
            return

        self.plot(
            "Sentiment", "Change in OpenAI Sentiment", self._roc.current.value
        )

        # If sentiment is flat/increasing and not already long, long.
        # The condition to buy here doesn't include `sentiment >= 0`
        # because excluding it allows the algorithm to buy when 
        # sentiment is down but relatively OK/good (since sentiment
        # is flat/increasing). If you wait for sentiment to be 
        # positive, it'll to late to benefit from the reversion
        # in price upwards following a crash from negative news. 
        if self._roc.current.value >= 0 and not self._tsla.holdings.is_long:
            self.set_holdings(self._tsla.symbol, 1)
        # If sentiment is negative and sentiment is decreasing and not 
        # already short, short.
        elif (sentiment < 0 and 
            self._roc.current.value < 0 and 
            not self._tsla.holdings.is_short):
            self.set_holdings(self._tsla.symbol, -1)


class TiingoNewsSentiment(PythonData):

    def get_source(self, config, date, is_live):
        return SubscriptionDataSource(
            f"tiingo-{date.strftime('%Y-%m-%d')}.csv", 
            SubscriptionTransportMedium.OBJECT_STORE, 
            FileFormat.CSV
        )

    def reader(self, config, line, date, is_live):
        # Skip the header line.
        if line[0] == ",": 
            return None
        
        # Parse the CSV line into a list.
        data = line.split(',')

        # Construct the new sentiment datapoint.
        t = TiingoNewsSentiment()
        t.symbol = config.symbol
        t.time = date.replace(hour=int(data[0]), minute=0, second=0)
        t.end_time = t.time + timedelta(hours=1)
        t.value = float(data[1])
        t["sentiment"] = t.value
        t["volume"] = float(data[2])

        return t
