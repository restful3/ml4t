# Factor Quiz

## Question 1
According to the source material, why do factor returns generally persist over long periods while 'alpha' tends to diminish?

- [x] Factor returns are associated with undiversifiable risks that many investors are unwilling to bear.
- [ ] Factor models are kept secret by institutional funds to prevent the decay of returns.
- [ ] Factor returns represent riskless arbitrage opportunities that are only accessible to large-scale traders.
- [ ] The capacity of factor models is much smaller than alpha-based strategies, preventing overcrowding.

**Hint:** Consider the relationship between risk, diversification, and the secrecy of trading strategies.

## Question 2
In the context of a time-series factor model, what are the characteristic dimensions of the factors and the factor loadings ($\beta_{i}$)?

- [x] Factors typically have dimensions of returns, while factor loadings are dimensionless.
- [ ] Both factors and factor loadings are expressed as percentages to maintain consistency across different asset classes.
- [ ] Factors are dimensionless indicators, while factor loadings have dimensions of volatility.
- [ ] Factor loadings are time-varying time-series data, while factors are static constants for each stock.

**Hint:** Think about which component represents a portfolio's return and which represents a regression coefficient.

## Question 3
The descriptive factor model $Return(t, s) - r_{F} = \alpha(s) + \beta_{1}(s) * Factor_{1}(t) + \dots + \epsilon(t, s)$ is primarily used for which of the following purposes in quantitative finance?

- [ ] Predicting the next period's stock returns based on current factor values.
- [x] Risk management, performance attribution, and understanding exposure to systematic risks.
- [ ] Identifying specific mispriced assets for high-frequency arbitrage opportunities.
- [ ] Determining the optimal number of stocks required to diversify away factor risk.

**Hint:** Analyze the time indices in the equation and what they imply about the relationship between factors and returns.

## Question 4
What is the primary difference in the observability of components between time-series and cross-sectional factor models?

- [x] In time-series models, factors are observable and loadings are estimated; in cross-sectional models, loadings are observable and factors are estimated.
- [ ] In time-series models, loadings are directly extracted from financial statements, whereas factors are derived via PCA.
- [ ] Cross-sectional models require historical price data to observe factors, while time-series models do not.
- [ ] Both factors and loadings are unobservable in time-series models and must be derived statistically.

**Hint:** Consider whether you start with the market return (Factor) or a company's P/E ratio (Loading).

## Question 5
In the study of the two-factor model using log(ROE) and log(BM), what was a significant finding regarding its application to S&P 500 (SPX) stocks compared to small-cap stocks?

- [ ] The model performed significantly better on SPX stocks due to higher liquidity and data transparency.
- [ ] Market-neutral portfolios using these factors generated consistent alpha in large-cap universes.
- [x] The ROE factor is a more reliable predictor than BM, and performance was notably stronger in small-cap stocks.
- [ ] Combining log(ROE) and log(BM) eliminated the need for market hedging in large-cap portfolios.

**Hint:** Recall the 'fly in the ointment' discussed regarding large-cap vs. small-cap predictive power.

## Question 6
Why do researchers such as Bali, Hu, and Murray (2015) argue that stocks with high implied kurtosis should have higher expected returns?

- [ ] High kurtosis indicates a higher probability of positive surprises that option traders are front-running.
- [x] Investors are averse to tail risks and demand a higher risk premium to hold stocks with high extreme-event probability.
- [ ] Options with high kurtosis are cheaper to hedge, allowing traders to bid up the underlying stock price.
- [ ] Kurtosis is a proxy for high liquidity, which reduces the required rate of return for institutional investors.

**Hint:** Focus on investor psychology regarding 'tail risk' and the compensation required for bearing it.

## Question 7
Which specific metric of short interest was found to have a more reliable negative factor loading with respect to future returns in the author's backtest?

- [ ] Short Interest Ratio (SIR), defined as shares borrowed divided by total shares outstanding.
- [x] Days to Cover (DTC), defined as shares borrowed divided by average daily trading volume.
- [ ] The weekly change in the total number of short positions across the entire SPX index.
- [ ] The rebate rate paid by stock lenders to short sellers in the securities lending market.

**Hint:** Distinguish between the metric that uses 'shares outstanding' and the one that uses 'daily trading volume'.

## Question 8
When utilizing Principal Component Analysis (PCA) to derive statistical factors for predicting returns, what does the first principal component typically represent?

- [ ] A momentum factor capturing stocks with the highest recent price acceleration.
- [x] The market return, characterized by positive factor loadings for nearly all component stocks.
- [ ] The statistical noise ($\epsilon$) that cannot be captured by fundamental financial models.
- [ ] The mean reversion factor of the covariance matrix's smallest eigenvalues.

**Hint:** Think about the most common 'driver' that causes most stocks to move together in the same direction.

## Question 9
How does the 'ranking methodology' used in multisorting provide a more robust trading model compared to a standard linear regression for predicting returns?

- [ ] It automatically adjusts for the time-decay of alpha by rebalancing more frequently.
- [x] It is immune to 'high-leverage points' or outliers that can distort regression coefficients.
- [ ] It requires significantly less historical data to reach a statistically significant conclusion.
- [ ] It utilizes the Spearman rank correlation to eliminate the need for calculating a covariance matrix.

**Hint:** Consider how an extreme data point (outlier) affects a line-of-best-fit versus a sorted list.

## Question 10
What is the primary reason provided for why some factor returns, such as the SMB (Small Minus Big) factor, have decreased in magnitude over time?

- [ ] The underlying companies have grown too large for the factor to be applicable.
- [x] Investors perceive that the associated factor risks have decreased, leading to a lower required risk premium.
- [ ] The increase in market efficiency has allowed alpha-seekers to arbitrage the factor to zero.
- [ ] The government has implemented regulations that limit the leverage available to small-cap investors.

**Hint:** Think about the relationship between the perception of risk and the 'reward' investors demand for it.
