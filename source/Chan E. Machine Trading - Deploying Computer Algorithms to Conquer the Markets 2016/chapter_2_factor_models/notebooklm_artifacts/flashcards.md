# Factor Flashcards

## Card 1

**Front:** What is the primary reason investors might avoid a simple buy-and-hold strategy of the S&P 500 index despite long-term profitability?

**Back:** Market risk and the associated long periods of drawdowns that may force inopportune liquidations.

---

## Card 2

**Front:** Define 'alpha' in the context of quantitative trading returns.

**Back:** Returns that are not known to be associated with common factor risks and can be reduced to zero through diversification.

---

## Card 3

**Front:** How does the diversifiability of factor risk differ from that of alpha?

**Back:** Alpha can be diversified away by adding more stocks, while factor risk remains unchanged regardless of portfolio size.

---

## Card 4

**Front:** Why do factor returns generally remain 'alive and well' year after year compared to alpha?

**Back:** Factors are associated with undiversifiable risks and sharp drawdowns, discouraging many investors from exploiting them.

---

## Card 5

**Front:** In the context of linear models, what does the term 'factor model' specifically refer to?

**Back:** Simple linear models where the predictors in the regression are the identified factors.

---

## Card 6

**Front:** What is the industry term for regression coefficients of factors in a linear model?

**Back:** Smart betas.

---

## Card 7

**Front:** How is a 'dumb' beta defined in equity modeling?

**Back:** The regression coefficient between the return of a specific stock and the return of the market index.

---

## Card 8

**Front:** Which type of factor varies from time to time but remains constant across different stocks or assets?

**Back:** Time-series factors.

---

## Card 9

**Front:** What is a 'hedge portfolio' in the context of factor modeling?

**Back:** A long-short portfolio whose returns directly represent a time-series factor.

---

## Card 10

**Front:** Define 'factor loading' (or factor exposure) in a time-series model.

**Back:** The regression coefficient that represents a stock's return response to a specific factor.

---

## Card 11

**Front:** Identify the units typically associated with factor loadings in time-series models.

**Back:** Factor loadings are dimensionless.

---

## Card 12

**Front:** In the descriptive factor model $Return(t, s) - r_F = \alpha(s) + \beta_1(s) * Factor_1(t) + \epsilon(t, s)$, what does $\epsilon$ represent?

**Back:** White noise or idiosyncratic risk that can be diversified away cross-sectionally and serially.

---

## Card 13

**Front:** What is the primary limitation of a contemporaneous (descriptive) factor model for active trading?

**Back:** It explains current returns rather than predicting future returns, making it primarily useful for risk management.

---

## Card 14

**Front:** How is a descriptive factor model converted into a predictive factor model?

**Back:** By regressing the return at time $t+1$ against the factors observed at time $t$.

---

## Card 15

**Front:** Aside from return prediction, what are the two main uses of factor models in large funds?

**Back:** Risk management and performance attribution.

---

## Card 16

**Front:** Why is performance attribution important for fund managers regarding incentive fees?

**Back:** Investors may refuse to pay performance fees on returns that could have been easily obtained via cheap smart-beta ETFs.

---

## Card 17

**Front:** What does the factor acronym HML stand for?

**Back:** High Minus Low (High book-to-market stocks minus Low book-to-market stocks).

---

## Card 18

**Front:** What does the factor acronym SMB stand for?

**Back:** Small Minus Big (Small market-capitalization stocks minus Big market-capitalization stocks).

---

## Card 19

**Front:** What is the UMD (or WML) factor in momentum strategies?

**Back:** Up Minus Down (or Winners Minus Losers), representing the return of past winners minus past losers.

---

## Card 20

**Front:** Between 1965 and 2011, which factor historically provided a higher return: HML or UMD?

**Back:** UMD (approx. 3000%) provided a significantly higher return than HML (approx. 900%).

---

## Card 21

**Front:** Name three macroeconomic variables that can serve as time-series factors.

**Back:** GDP growth, Consumer Price Index (CPI) change, and Volatility (VIX) change.

---

## Card 22

**Front:** What was the result of using Fama-French factors to predict next-day returns for S&P 500 stocks?

**Back:** The model worked well in-sample but generated negative returns out-of-sample.

---

## Card 23

**Front:** Contrast time-series factors and cross-sectional factors regarding their observability.

**Back:** Time-series factors are directly observable as portfolio returns, while cross-sectional factors (loadings) are observable as stock characteristics like P/E ratios.

---

## Card 24

**Front:** In a cross-sectional factor model, how are the time-series factor returns determined?

**Back:** By regressing stock returns against the observed factor loadings (characteristics) at a specific point in time.

---

## Card 25

**Front:** What is a major risk when including a large number of fundamental factor loadings (e.g., 112 factors) in a model?

**Back:** Severe overfitting, leading to high in-sample performance but poor or negative out-of-sample performance.

---

## Card 26

**Front:** In the Chattopadhyay, Lyle, and Wang (2015) two-factor model, what are the two primary predictors?

**Back:** Log of Return-on-Equity (ROE) and the log of Book-to-Market ratio (BM).

---

## Card 27

**Front:** Formula: Return on Equity ($ROE$) as defined for the two-factor model.

**Back:** $ROE(i, s) = 1 + X(i, s) / Book(i - 1, s)$, where $X$ is quarterly net income.

---

## Card 28

**Front:** According to research, which factor is a more reliable predictor of future returns: ROE or BM?

**Back:** Return-on-Equity (ROE).

---

## Card 29

**Front:** What 'pitfall' was identified regarding the ROE strategy's performance on S&P 500 stocks between 2010 and 2013?

**Back:** The returns were largely driven by the strategy's net long exposure during a bull market and disappeared once hedged with SPY.

---

## Card 30

**Front:** How does the efficacy of value factors (like ROE or BM) typically vary between large-cap (SPX) and small-cap (SML) stocks?

**Back:** Value factors are often useless for predicting returns of large-cap stocks but remain effective for small-cap stocks.

---

## Card 31

**Front:** According to market folklore and research, which group of traders is often considered 'smarter' or better informed than stock traders?

**Back:** Options traders.

---

## Card 32

**Front:** How is 'implied skewness' typically measured using option implied volatilities?

**Back:** The difference between the implied volatilities of out-of-the-money (OTM) calls and OTM puts.

---

## Card 33

**Front:** In the context of option factors, what does 'implied kurtosis' measure?

**Back:** The difference between the sum of OTM call/put implied volatilities and the sum of at-the-money (ATM) call/put implied volatilities.

---

## Card 34

**Front:** What is the relationship between implied volatility and expected stock returns based on the CAPM?

**Back:** Expected returns are positively related to implied volatility because higher risks require higher compensation.

---

## Card 35

**Front:** Why do stocks with high implied skewness tend to have higher expected returns?

**Back:** Informed traders with positive expectations buy OTM calls or sell OTM puts, driving up the implied skewness.

---

## Card 36

**Front:** What happened to the 'Implied Moments' strategy when applied strictly to S&P 500 stocks out-of-sample?

**Back:** It produced almost exactly zero return, failing to predict returns for large-cap stocks.

---

## Card 37

**Front:** How does a large increase in call implied volatility over the previous month typically affect future stock returns?

**Back:** It tends to predict higher future returns.

---

## Card 38

**Front:** What does a significant deviation from put-call parity suggest about future stock returns?

**Back:** Stocks where call implied volatilities are much higher than put implied volatilities tend to have higher future returns.

---

## Card 39

**Front:** What is the 'volatility smirk' (or skew) factor used by Zhang, Zhao, and Xing?

**Back:** The difference between OTM put and ATM call implied volatilities.

---

## Card 40

**Front:** According to research on the volatility smirk, do stocks in the top quintile of skew overperform or underperform?

**Back:** They underperform those in the bottom quintile.

---

## Card 41

**Front:** How is the change in the VIX index used as a time-series factor for individual stocks?

**Back:** Stocks that react positively to market volatility increases (hedges) tend to have lower future returns because investors overpay for them.

---

## Card 42

**Front:** Define 'Short Interest Ratio' (SIR).

**Back:** The number of shares borrowed for shorting divided by the total shares outstanding.

---

## Card 43

**Front:** Which short interest metric proved to be a better predictor of negative returns: SIR or Days to Cover (DTC)?

**Back:** Days to Cover (DTC).

---

## Card 44

**Front:** What is the formula for 'Days to Cover' (DTC)?

**Back:** The number of shares borrowed for shorting divided by the average daily trading volume.

---

## Card 45

**Front:** What is the general finding regarding the relationship between a stock's liquidity and its future returns?

**Back:** Less liquid stocks typically offer higher future returns (liquidity premium).

---

## Card 46

**Front:** How did the liquidity factor perform on S&P 500 stocks compared to the broader market in recent tests?

**Back:** The result was opposite to the broad market; most liquid SPX stocks outperformed the least liquid ones.

---

## Card 47

**Front:** What are 'statistical factors' in a factor model?

**Back:** Factors derived purely from the statistical properties of stock returns (like a covariance matrix) rather than fundamental data.

---

## Card 48

**Front:** Name the mathematical technique commonly used to extract statistical factors from a return covariance matrix.

**Back:** Principal Component Analysis (PCA).

---

## Card 49

**Front:** In PCA factor models, what property is assigned to the first principal component?

**Back:** It has the largest variance and is typically interpreted as the market return factor.

---

## Card 50

**Front:** Why might statistical factors be more useful than fundamental factors for intraday trading?

**Back:** Fundamental factors (like P/E or ROE) do not change intraday, whereas statistical factors capture rapidly evolving price dynamics.

---

## Card 51

**Front:** What was the out-of-sample CAGR of the statistical factor model applied to SPX stocks from 2007 to 2013?

**Back:** 15.6%.

---

## Card 52

**Front:** What is the first step in applying PCA to stock returns for a predictive model?

**Back:** Cleansing data to eliminate stocks with missing (NaN) returns during the lookback period.

---

## Card 53

**Front:** What is a standard cure for collinearity when combining multiple factors in a regression model?

**Back:** Systematically reducing factors using criteria such as AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion).

---

## Card 54

**Front:** Why is the 'ranking methodology' often more robust than using precise predicted returns from a linear regression?

**Back:** Ranking is immune to outliers (high-leverage points) that can distort regression coefficients.

---

## Card 55

**Front:** Concept: Multisorting

**Back:** Definition: Sorting stocks sequentially by multiple factor loadings to narrow down a portfolio to the most desirable candidates.

---

## Card 56

**Front:** What is the benefit of performing factor sorts within specific industry groups or market cap quantiles?

**Back:** It creates a portfolio that is neutral to those specific time-series factor risks (e.g., industry-neutral or size-neutral).

---

## Card 57

**Front:** Which specific factor return has decreased over time because investors perceive small-cap stocks as no riskier than large-cap stocks?

**Back:** The SMB (Size) factor.

---

## Card 58

**Front:** What is 'Smart Beta'?

**Back:** A term for regression coefficients (loadings) for factors in a simple linear model.

---

## Card 59

**Front:** Under PCA, how are the factor loadings ($\beta_i$) determined relative to the covariance matrix?

**Back:** They are the eigenvectors of the covariance matrix, sorted by the magnitude of their eigenvalues.

---

## Card 60

**Front:** Why does alpha typically diminish once discovered?

**Back:** It has limited capacity and more people discovering it leads to increased competition and thinner margins.

---

## Card 61

**Front:** In the time-series model equation, what is the role of the risk-free rate ($r_F$)?

**Back:** It is subtracted from the stock return to isolate the 'excess return' driven by factors.

---

## Card 62

**Front:** If a strategy loads positively to volatility change, how should a trader hedge that specific factor risk?

**Back:** By owning another portfolio or strategy that loads negatively to volatility change.

---

## Card 63

**Front:** What does a high $R^2$ in a training set (e.g., 0.96) often signal in financial predictive modeling?

**Back:** Overfitting, which likely results in poor out-of-sample performance.

---

## Card 64

**Front:** How does the 'insurance industry' provide evidence for the validity of factor-based investing?

**Back:** Its longevity demonstrates that selling options (volatility factor) generates positive long-term returns despite sharp drawdowns.

---

## Card 65

**Front:** True or False: Time-series factors vary across different stocks.

**Back:** False; time-series factors are the same for all assets, but their loadings ($\\beta$) vary.

---

## Card 66

**Front:** True or False: Cross-sectional factor loadings are directly observable stock characteristics.

**Back:** True.

---

## Card 67

**Front:** What is the 'long-value short-growth' portfolio often called in empirical finance?

**Back:** The HML (High Minus Low) hedge portfolio.

---

## Card 68

**Front:** In a statistical factor model, what do the eigenvalues of the covariance matrix represent?

**Back:** The variance captured by each corresponding principal component (statistical factor).

---

## Card 69

**Front:** In Example 2.1, what lookback period was used for the training set to estimate factor loadings?

**Back:** The first half of the data set (static) or approximately 3 years.

---

## Card 70

**Front:** According to the text, why might the SMB factor return have decreased recently?

**Back:** Investors believe small-cap stocks are no more risky than large-cap stocks if sufficiently diversified.

---

## Card 71

**Front:** Which factor model implementer uses 'magic formula' investing based on Greenblatt's two-factor model?

**Back:** www.magicformulainvesting.com

---

## Card 72

**Front:** What is the primary characteristic of 'White Noise' in factor residuals?

**Back:** It is uncorrelated both serially (in time) and cross-sectionally (across stocks).

---

## Card 73

**Front:** In the context of factor risk, what happened to the value-growth strategy during the dot-com bubble?

**Back:** It suffered significant losses as growth stocks dramatically outperformed value stocks.

---

## Card 74

**Front:** What is the dimensionality of the time-series factor 'Market Return'?

**Back:** Return (dollar divided by time).

---

## Card 75

**Front:** Why is the use of mid-quotes at market close preferred over actual closing prices in backtests?

**Back:** To avoid effects of widened bid-ask spreads at the market close.

---

## Card 76

**Front:** In Example 2.2, how many fundamental factor loadings were tested for the quarterly return prediction?

**Back:** 27 size-independent loadings.

---

## Card 77

**Front:** What is the 'moneyness' of an option?

**Back:** The ratio of the strike price to the current stock price.

---

## Card 78

**Front:** For the 'volatility smirk' research, what was the defined moneyness range for an out-of-the-money (OTM) put?

**Back:** Between 0.8 and 0.95.

---

## Card 79

**Front:** In the predictive equation $Return(t+1, s) - r_F = \alpha(s) + \beta_1(s) * Factor_1(t) + ...$, what is the assumption about $\alpha$?

**Back:** It is set as a constant in time ($\\alpha(t, s) = \alpha(s)$).

---

## Card 80

**Front:** If you buy the top 50 stocks by predicted return and short the bottom 50, what is the 'topN' parameter?

**Back:** The number of stocks held on each side of the long-short portfolio (e.g., 50).

---

## Card 81

**Front:** According to Kaplan (2014), what is the efficacy of value factors for S&P 500 stocks?

**Back:** They are useless for predicting returns of SPX stocks.

---

## Card 82

**Front:** Why might a trader choose to trade factor models despite their 'lack of sex appeal' compared to alpha?

**Back:** They are more sociable/sharable since the returns are tied to known risks and hard to steal or arbitrage away.

---
