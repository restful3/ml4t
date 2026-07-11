# Trading Quiz

## Question 1
According to the source material, why is it considered best practice to use the exact same computer program for both backtesting and live order execution?

- [ ] It eliminates the need for separate historical and live market data APIs.
- [x] It ensures the trading logic being executed is identical to the model that was validated during the research phase.
- [ ] It allows the program to automatically adjust for survivorship bias in real-time trading.
- [ ] It significantly reduces latency by bypassing the Broker API during the backtesting phase.

**Hint:** Consider the risks associated with translating a mathematical model from one programming language or environment to another.

## Question 2
A trader uses consolidated closing prices for a daily strategy backtest. Why might this lead to 'inflated' performance results according to the text?

- [x] Consolidated prices include data from multiple exchanges that may not reflect the primary auction price available at the close.
- [ ] Consolidated feeds are inherently slower, introducing a look-ahead bias into the backtest results.
- [ ] Consolidated prices fail to account for stock splits and dividends automatically.
- [ ] These prices are typically derived from mid-prices rather than the best bid and offer (BBO).

**Hint:** Think about the difference between a general market price and the specific price reached during an exchange's primary auction.

## Question 3
Why does the author recommend using the Calmar ratio over the MAR ratio for evaluating strategy performance?

- [ ] The Calmar ratio utilizes a rolling standard deviation to account for fat-tailed distributions.
- [x] The MAR ratio is overly sensitive to the length of the backtest because maximum drawdown tends to increase over time.
- [ ] The Calmar ratio assumes a Gaussian distribution of returns, making it more mathematically robust.
- [ ] The MAR ratio cannot be used for strategies that incorporate leverage resize on account equity.

**Hint:** Consider how the maximum drawdown of a strategy behaves as you increase the historical window of the data set.

## Question 4
When calculating the relationship between mean net returns ($m$) and mean log returns ($\mu$) in continuous time, which formula accurately represents the approximation according to equation 1.1?

- [x] $\mu \approx m - \frac{s^2}{2}$
- [ ] $\mu \approx m + \frac{s^2}{2}$
- [ ] $m \approx \mu - \frac{s^2}{2}$
- [ ] $\mu \approx \frac{m}{s^2}$

**Hint:** Recall the 'variance drag' concept from Ito's Lemma discussed in the context of continuous finance.

## Question 5
What is a primary danger of utilizing the Kelly optimal leverage formula ($F^{*} = C^{-1}M$) in a real-world trading environment?

- [ ] The formula requires an infinite time horizon to achieve the tangency portfolio.
- [x] It assumes returns follow a Gaussian distribution, which fails to account for 'fat-tail' events that can wipe out equity.
- [ ] It minimizes the variance of returns rather than maximizing the compound annual growth rate.
- [ ] It is mathematically incompatible with the Markowitz efficient frontier.

**Hint:** Consider the statistical assumptions regarding return distributions and the presence of extreme market moves like 'Black Monday'.

## Question 6
Why might a trader prefer a 'minimum variance portfolio' over the 'tangency portfolio' despite the latter theoretically maximizing the Sharpe ratio?

- [x] Expected returns are notoriously difficult to predict, while covariances are more stable and easier to estimate from historical data.
- [ ] The minimum variance portfolio is the only allocation that achieves the Kelly optimal compound growth rate.
- [ ] Minimum variance portfolios automatically incorporate tail-risk protections like maximum drawdown limits.
- [ ] The tangency portfolio requires all assets to have a Gaussian distribution, whereas minimum variance does not.

**Hint:** Think about which input in the portfolio optimization process (returns vs. covariances) is more susceptible to estimation error.

## Question 7
According to the text, how can a risk parity portfolio contribute to a market 'contagion' during a crisis?

- [ ] It forces traders to increase leverage when volatility decreases, creating an asset bubble.
- [x] The strategy targets volatility; as market panic increases volatility, the funds are forced to liquidate positions simultaneously, driving prices lower.
- [ ] It relies on the CLS bank for settlement, which can become a single point of failure during a currency collapse.
- [ ] Risk parity funds only invest in low-volatility stocks, which lack the liquidity to handle large withdrawals.

**Hint:** Consider the relationship between rising market volatility and the mandate to maintain a constant risk level per asset.

## Question 8
If an individual trader is concerned about being personally liable for negative equity in a highly levered account, what structural solution does the author suggest?

- [ ] Utilizing a broker that offers SIPC insurance for commodities futures accounts.
- [x] Trading through a limited liability vehicle such as an LLC or a corporation.
- [ ] Restricting all trades to currencies that are settled through the CLS bank.
- [ ] Relying on 'payment for order flow' rebates to offset potential margin calls.

**Hint:** Think about the legal distinction between personal assets and business entities in the event of a total investment loss.

## Question 9
Which programming language is described as having the highest 'IDE polish' and speed while also allowing compilation to C/C++, making it a strong candidate for quantitative trading?

- [ ] R
- [x] Python
- [ ] MATLAB
- [ ] Java

**Hint:** The author provides a specific ranking table for MATLAB, R, and Python across several features.

## Question 10
For a trader focusing on illiquid out-of-the-money options, why is the 'last trade price' a potentially misleading metric for a backtest?

- [ ] Options contracts are subject to survivorship bias that last trade prices cannot capture.
- [x] The last trade may have occurred hours before the close, whereas the bid-ask quote reflects the actual price at which one could trade at the close.
- [ ] Last trade prices do not include the Greeks or implied volatility surfaces.
- [ ] Implied volatility interpolation often overestimates the value of out-of-the-money options.

**Hint:** Consider the frequency of trading for obscure contracts and how that impacts the relevance of the most recent transaction price.
