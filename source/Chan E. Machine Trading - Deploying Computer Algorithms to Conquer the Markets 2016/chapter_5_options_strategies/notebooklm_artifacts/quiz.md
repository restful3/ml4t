# Options Quiz

## Question 1
Why does the author argue that algorithmic options traders should primarily focus on delta-neutral strategies rather than strategies with high delta exposure?

- [ ] Delta-neutral strategies are the only ones where the Black-Scholes equation remains mathematically valid.
- [x] The transaction costs for obtaining delta exposure are significantly lower when trading stocks directly due to higher liquidity.
- [ ] It is mathematically impossible to backtest delta-heavy strategies across a large portfolio of options.
- [ ] Delta-neutral strategies are immune to the effects of negative theta and time decay.

**Hint:** Consider the comparative costs of executing trades in different asset classes.

## Question 2
In the context of trading volatility without options, why is simply 'holding' a short position in VX futures over a long period technically impossible?

- [x] VX futures contracts have a monthly expiration and must be continuously rolled to the next nearby contract.
- [ ] The CBOE prohibits holding short volatility positions for more than one fiscal quarter.
- [ ] The margin requirements for VX futures increase exponentially the longer a position is held.
- [ ] VX futures lose all correlation with the VIX index once they enter the final week before expiration.

**Hint:** Think about the structural difference between a stock and a derivative contract with a 'tenor'.

## Question 3
According to the GARCH model prediction results, what was the observed relationship between the sign of the change in realized volatility ($RV$) and the sign of the change in $VXX$?

- [ ] They move in the same direction over 90% of the time, confirming $VXX$ as a perfect proxy for daily realized volatility.
- [ ] There is no statistical correlation between the two, making daily GARCH predictions useless for trading $VXX$.
- [x] They moved in the same direction only about 35% of the time, suggesting realized volatility often moves opposite to implied volatility proxies.
- [ ] They only correlate during extreme market crashes (tail events) and are random otherwise.

**Hint:** Recall why the author suggested a 'reverse' strategy for trading $VXX$ based on GARCH signals.

## Question 4
In a Gamma Scalping strategy, what is the primary purpose of holding a long straddle or strangle position?

- [ ] To generate profit through the accumulation of daily theta.
- [x] To provide a hedge against extreme underlying movements that would otherwise cause catastrophic losses in the mean-reversion component.
- [ ] To ensure the portfolio remains vega-neutral throughout the trading period.
- [ ] To capture the 'jump' in implied volatility that typically occurs during quiet market periods.

**Hint:** Consider the 'Black Swan' risk associated with pure mean-reversion strategies.

## Question 5
Why is 'Dispersion Trading' often referred to as 'Correlation Trading'?

- [ ] It involves trading the statistical correlation between an option's strike price and its time-to-maturity.
- [x] By shorting index options and buying component options, the trader is effectively shorting the implied correlation of the index constituents.
- [ ] The strategy requires a 1.0 correlation between the stock portfolio and the S&P 500 index to be profitable.
- [ ] It uses the correlation between theta and vega to identify mispriced 'ladders'.

**Hint:** Think about what happens to the price of an index option when all its components start moving in the same direction.

## Question 6
When implementing the 'Cross-Sectional Mean Reversion of Implied Volatility' strategy, why might out-of-the-money (OTM) options produce higher 'paper' returns compared to at-the-money (ATM) options?

- [ ] OTM options have much lower transaction costs and bid-ask spreads.
- [x] OTM options possess higher embedded leverage, magnifying the percentage P&L relative to the option's market value.
- [ ] The GARCH model is specifically optimized to predict the movement of OTM strike prices.
- [ ] Put-call parity is only applicable to OTM options, providing an arbitrage opportunity.

**Hint:** Consider the formula for leverage: $Leverage = \frac{\Delta \times S}{Premium}$.

## Question 7
In the provided dispersion trading example, what criteria are used to select the 50 stocks from the S&P 500 index?

- [ ] The stocks with the highest historical realized volatility over the past 30 days.
- [x] The stocks whose straddles have the highest (least negative) theta.
- [ ] The stocks with the lowest bid-ask spread to ensure efficient daily rebalancing.
- [ ] The stocks with the highest market capitalization to mirror the index weighting.

**Hint:** The strategy aims to reduce the 'cost of carry' associated with long options positions.

## Question 8
What is the primary reason the author expresses caution regarding backtest results for intraday options strategies, even when they appear highly profitable?

- [ ] Option prices are non-linear, making standard standard deviation calculations invalid.
- [x] The impact of wide bid-ask spreads can easily negate theoretical profits when using market orders for execution.
- [ ] The Black-Scholes model fails to account for dividends, which are the main driver of options returns.
- [ ] Historical data for options Greeks is notoriously inaccurate before the year 2015.

**Hint:** Consider what happens to a trade's profit if you have to pay the 'ask' price and sell at the 'bid' price.

## Question 9
Why does the VIX index itself not suffer from time decay (negative theta), while an investment in $VXX$ typically does?

- [ ] $VXX$ is a levered instrument, and all leverage naturally incurs theta decay.
- [x] The VIX is a calculated index that constantly 'replaces' aging options with new ones to maintain a fixed 30-day maturity.
- [ ] The CBOE applies a mathematical 'theta-correction' to the VIX formula daily.
- [ ] $VXX$ is based on European options which have higher theta than the American options in the VIX.

**Hint:** Think about the difference between a static portfolio of options and a rolling weighted average.

## Question 10
In the $GARCH(p, q)$ model defined by the equation $\sigma^2_t = \omega + \sum_{i=1}^p \alpha_i \sigma^2_{t-i} + \sum_{i=1}^q \beta_i r^2_{t-i}$, what does the term $r^2_{t-i}$ represent?

- [ ] The risk-free rate of return at a specific time lag.
- [x] The past actual squared log returns of the underlying price series.
- [ ] The residual error of the linear regression between delta and gamma.
- [ ] The implied volatility of a straddle at time $t-i$.

**Hint:** This term relates to the historical observations of price movements.
