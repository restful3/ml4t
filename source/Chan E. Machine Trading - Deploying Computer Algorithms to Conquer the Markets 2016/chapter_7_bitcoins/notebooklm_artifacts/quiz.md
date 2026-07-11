# Bitcoin Quiz

## Question 1
According to the text, why are fundamental factors like Federal Reserve interest rate decisions described as 'contemporaneous' rather than 'predictive' for short-term bitcoin trading?

- [x] They occur simultaneously with market movements and can only explain price shifts after they happen.
- [ ] Bitcoin is fundamentally decoupled from global macroeconomic events like interest rate changes.
- [ ] Fundamental data is released too slowly to be captured by the one-minute bar data used in the analysis.
- [ ] Technical analysis tools like Bollinger bands are inherently more accurate than fundamental models in all time frames.

**Hint:** Consider the distinction between a cause that can be anticipated and an event that happens at the same time as the price change.

## Question 2
When trading the currency pair BTC.CNY, which of the following is true based on the 'alphabetical mnemonic' mentioned in the source?

- [x] BTC is the base currency because 'B' comes before 'C' in the alphabet.
- [ ] CNY is the base currency because it represents the quote of the exchange rate.
- [ ] The pair represents the number of bitcoins needed to buy one unit of Chinese Yuan.
- [ ] The alphabetical rule is only used when trading against the US Dollar (USD).

**Hint:** Think about the relationship between the symbols 'B' and 'Q' and their order in the alphabet.

## Question 3
How does the risk profile of BTC.USD compare to traditional assets like SPY (S&P 500 ETF) based on the data provided for 2014-2015?

- [x] BTC.USD exhibits significantly higher annualized volatility and higher kurtosis than SPY.
- [ ] The maximum drawdown for BTC.USD is lower than that of SPY because it trades 24/7.
- [ ] BTC.USD and SPY share similar tail risks, as indicated by their near-identical kurtosis values.
- [ ] The volatility of BTC.USD is actually lower than MXN.USD because it is a more liquid global asset.

**Hint:** Refer to the statistical measures of 'tail risk' and 'annualized price fluctuations' in the provided comparison table.

## Question 4
An Augmented Dickey-Fuller (ADF) test on BTC.USD one-minute midprices returned a statistic of $-3$ with a $95\%$ critical value of $-2.9$. What is the primary implication for a quantitative trader?

- [x] The price series is stationary and likely to reward mean-reversion strategies.
- [ ] The price series follows a random walk, suggesting a trend-following approach is necessary.
- [ ] The data is non-stationary, meaning an $AR(p)$ model cannot be applied to the levels.
- [ ] The negative statistic indicates that the price is in a long-term downward trend.

**Hint:** Compare the test statistic to the critical value and recall what 'rejecting the null hypothesis' means for price behavior.

## Question 5
The $AR(16)$ model applied to BTC.USD resulted in an extremely high CAGR of $40,000$. Why does the author caution that this return is 'not realistic'?

- [x] The backtest assumes that limit orders are always filled instantaneously at the midprice.
- [ ] The model suffered from look-ahead bias by using the test set for parameter estimation.
- [ ] The $AR(16)$ model is mathematically inferior to the $ARMA(3, 7)$ model for all currency pairs.
- [ ] Bitcoin exchanges do not allow for the high-frequency trades required by autoregressive models.

**Hint:** Think about the difference between a theoretical 'midprice' and the actual prices at which a trader can buy or sell.

## Question 6
In the Bollinger band strategy described, what specific condition triggers the liquidation of a long position?

- [x] The price reverts to the current moving average.
- [ ] The price touches the upper band ($MA + k \times MSTD$).
- [ ] The moving standard deviation ($MSTD$) exceeds a predetermined threshold.
- [ ] A fixed time-based exit occurs after 60 one-minute bars.

**Hint:** The strategy is based on mean reversion; look for the point representing the 'mean' in the Bollinger band setup.

## Question 7
Which machine learning technique failed to produce positive returns for BTC.USD, despite having worked previously for SPY?

- [x] Support Vector Machine (SVM).
- [ ] Feedforward Neural Networks.
- [ ] Regression tree algorithm with bagging.
- [ ] Autoregressive Integrated Moving Average (ARIMA).

**Hint:** Consider the algorithm that the author explicitly warned does not always work across different asset classes.

## Question 8
How is 'order flow' calculated in the context of the BitStamp trade tick data strategy?

- [x] It is the sum of transaction volumes, signed positively for buyer-initiated trades and negatively for seller-initiated trades.
- [ ] It is the difference between the current bid price and the current ask price.
- [ ] It is the total number of trades occurring within a microsecond regardless of direction.
- [ ] It is the ratio of limit orders to market orders currently sitting in the order book.

**Hint:** Focus on the term 'signed transaction volume' and the role of the 'aggressor tag'.

## Question 9
What is the primary reason the market allows a persistent price difference (crossed market) between exchanges like Bitfinex and btc-e?

- [x] The market assesses a higher credit risk for the exchange with lower prices.
- [ ] Regulation NMS Rule 610 prevents arbitrage between these specific bitcoin exchanges.
- [ ] The 20 bps commission fee is higher than the typical US$5.00 price difference.
- [ ] Bitcoin cannot be transferred between exchanges, making physical arbitrage impossible.

**Hint:** Recall the discussion on 'credit risk' and the historical frequency of exchange hacks.

## Question 10
What significant risk factor mentioned by Johansson and Tjernstrom (2014) is unique to the bitcoin exchange landscape compared to traditional equity exchanges?

- [x] $45\%$ of bitcoin exchanges fail due to thefts and hacks, often losing investor deposits.
- [ ] Bitcoin exchanges are legally required to close for 48 hours every weekend.
- [ ] High kurtosis in returns causes exchanges to automatically liquidate all accounts daily.
- [ ] The absence of any trading volume on top exchanges leads to permanent illiquidity.

**Hint:** Consider the safety of funds held on the platforms rather than the volatility of the asset itself.
