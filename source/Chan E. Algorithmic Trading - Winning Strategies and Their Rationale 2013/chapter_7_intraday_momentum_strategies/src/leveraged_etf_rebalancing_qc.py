# Chan Ch7 - Leveraged ETF Rebalancing Momentum Strategy
# Based on: Ernest Chan, "Algorithmic Trading" Chapter 7
# QuantConnect Project: chan-ch7-leveraged-etf-rebalancing (ID: 29188753)
#
# Logic:
#   - At 15:45 ET, compute return from previous close to current price
#   - If return > +2%: buy DRN (3x leveraged REIT ETF)
#   - If return < -2%: short DRN
#   - Exit all positions at 15:59 ET (market close)

from AlgorithmImports import *


class LeveragedETFRebalancingAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2011, 10, 1)
        self.set_end_date(2013, 12, 31)
        self.set_cash(100000)

        # DRN: Direxion Daily Real Estate Bull 3X Shares
        self.drn = self.add_equity("DRN", Resolution.MINUTE)
        self.drn.set_data_normalization_mode(DataNormalizationMode.RAW)

        self.entry_threshold = 0.02  # +/- 2%
        self.previous_close = None

        # Schedule: entry check at 15:45 ET (15 min before close)
        self.schedule.on(
            self.date_rules.every_day("DRN"),
            self.time_rules.before_market_close("DRN", 15),
            self.check_entry
        )

        # Schedule: exit at 15:59 ET (1 min before close)
        self.schedule.on(
            self.date_rules.every_day("DRN"),
            self.time_rules.before_market_close("DRN", 1),
            self.exit_positions
        )

    def check_entry(self):
        if self.previous_close is None or self.previous_close == 0:
            return

        if not self.drn.has_data:
            return

        current_price = self.drn.price

        ret = (current_price - self.previous_close) / self.previous_close

        if ret > self.entry_threshold:
            self.set_holdings("DRN", 1.0)
            self.debug(f"{self.time} LONG DRN | ret={ret:.4f} | prev_close={self.previous_close:.2f} | price={current_price:.2f}")
        elif ret < -self.entry_threshold:
            self.set_holdings("DRN", -1.0)
            self.debug(f"{self.time} SHORT DRN | ret={ret:.4f} | prev_close={self.previous_close:.2f} | price={current_price:.2f}")

    def exit_positions(self):
        if self.portfolio.invested:
            self.liquidate("DRN")
            self.debug(f"{self.time} EXIT all positions")

    def on_data(self, data):
        pass

    def on_end_of_day(self, symbol):
        # Record close price for next day's signal calculation
        if symbol == self.drn.symbol and self.drn.has_data:
            self.previous_close = self.drn.price
        self.plot("Strategy", "Portfolio Value", self.portfolio.total_portfolio_value)
