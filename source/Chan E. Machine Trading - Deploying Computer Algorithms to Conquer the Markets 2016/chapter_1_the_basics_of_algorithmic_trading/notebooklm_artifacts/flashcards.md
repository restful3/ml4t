# Trading Flashcards

## Card 1

**Front:** According to the textbook, what is the best way to ensure a live trading model is identical to the one that was backtested?

**Back:** Using the exact same computer program for both generating backtest results and submitting live orders.

---

## Card 2

**Front:** Which historical market data vendor provides automatic adjustment for splits and dividends and powers Yahoo! Finance?

**Back:** CSI (csidata.com)

---

## Card 3

**Front:** Why do many professional futures traders prefer to backtest using original contract prices instead of back-adjusted continuous contracts?

**Back:** Back-adjusted prices depend on a specific roll method and may contain look-ahead bias.

---

## Card 4

**Front:** What is the primary function of Quandl.com in the context of market data?

**Back:** It acts as a consolidator of data from many different vendors, accessible via API.

---

## Card 5

**Front:** Which database is recommended for academic researchers who need survivorship-bias-free data and best bid/offer (BBO) prices at the close?

**Back:** CRSP (Center for Research in Security Prices)

---

## Card 6

**Front:** What is the benefit of using primary exchange auction prices for backtesting daily strategies?

**Back:** It avoids the performance inflation that can occur when using consolidated closing prices.

---

## Card 7

**Front:** What specific advantage does a Bloomberg terminal offer for event-driven strategies?

**Back:** It provides superior and timely breaking news coverage on obscure stocks.

---

## Card 8

**Front:** In options historical data, why is it important to use closing bid-ask quotes rather than the last trade price?

**Back:** Infrequently traded options may have a last trade price that is unrepresentative of the actual tradable price at the close.

---

## Card 9

**Front:** What is an 'implied volatility surface' in the context of options data?

**Back:** An interpolation of implied volatilities from actual options to estimate values for non-existent strikes and tenors.

---

## Card 10

**Front:** Which platform allows users to rent intraday option prices more cheaply than buying daily data from other vendors?

**Back:** QuantGo.com

---

## Card 11

**Front:** What is the industry standard database for analyst earnings estimates?

**Back:** Thomson Reuters' IBES

---

## Card 12

**Front:** What makes Estimize unique compared to traditional earnings estimate sources?

**Back:** It is a crowd-sourced platform whose contributors can often out-forecast sell-side analysts.

---

## Card 13

**Front:** What is 'elementized news'?

**Back:** Machine-readable news feeds that allow programs to easily capture keywords, categories, and sentiment.

---

## Card 14

**Front:** Which news vendor provides 'impact scores' that indicate a move is coming without specifying direction?

**Back:** AcquireMedia (NewsEdge database)

---

## Card 15

**Front:** In intraday trading, what is the generally accepted threshold for 'low latency' data feed providers?

**Back:** Latency below $10$ ms.

---

## Card 16

**Front:** What is the primary drawback of developing a strategy in one language (like R) and executing it in another (like C++)?

**Back:** It is bug-prone and difficult to ensure both programs encapsulate the exact same trading logic.

---

## Card 17

**Front:** Which MATLAB toolbox allows for direct connection to broker APIs for data and order submission?

**Back:** The Trading Toolbox

---

## Card 18

**Front:** What is the main disadvantage of R compared to MATLAB or Python in quantitative trading?

**Back:** It is the slowest of the three and cannot be easily compiled into C or C++ for speed.

---

## Card 19

**Front:** Which Python library is specifically mentioned as being similar to R for data analysis?

**Back:** pandas

---

## Card 20

**Front:** What are REPL (Read-Eval-Print-Loop) languages particularly useful for in trading?

**Back:** Shortening the research cycle by allowing quick prototyping and modification without compilation.

---

## Card 21

**Front:** Why is object-oriented design (OOD) sometimes avoided in scripting languages for automated execution?

**Back:** OOD can make scripting languages run too slowly for efficient live execution.

---

## Card 22

**Front:** What practice involves a broker forwarding orders to a specific market maker for a rebate?

**Back:** Payment for order flow

---

## Card 23

**Front:** What is 'Direct Market Access' (DMA)?

**Back:** A service that allows traders to bypass the broker's routing preferences and potentially pocket exchange rebates themselves.

---

## Card 24

**Front:** How do some FX brokers earn money without charging commissions?

**Back:** By widening the bid-offer spread on currency pairs.

---

## Card 25

**Front:** Which organization provides up to $\$250,000$ in cash insurance for US securities accounts?

**Back:** The Securities Investor Protection Corporation (SIPC)

---

## Card 26

**Front:** What is the main regulatory risk for a commodities futures trader regarding their cash balance?

**Back:** Futures accounts are not covered by SIPC insurance if the broker goes bankrupt.

---

## Card 27

**Front:** What is 'Herstatt risk' (or settlement risk) in FX trading?

**Back:** The risk that one party delivers their currency but the counterparty collapses before delivering the other leg.

---

## Card 28

**Front:** How does the CLS bank eliminate settlement risk in the FX market?

**Back:** By facilitating simultaneous settlement of both legs of a currency transaction through accounts with central banks.

---

## Card 29

**Front:** How can a trader protect personal assets from 'negative equity' in a leveraged account?

**Back:** Trading through a limited liability vehicle like an LLC, LP, or corporation.

---

## Card 30

**Front:** How does CAGR differ from average annualized return?

**Back:** CAGR assumes profits are compounded and reinvested while maintaining constant leverage.

---

## Card 31

**Front:** What should the leverage be set to in a backtest to ensure meaningful performance comparison?

**Back:** A leverage of $1$.

---

## Card 32

**Front:** What is the formula for Kelly optimal leverage?

**Back:** $\text{Optimal leverage} = \frac{\text{Mean of Excess Returns}}{\text{Variance of Excess Returns}}$

---

## Card 33

**Front:** What happens to CAGR if a trader applies leverage higher than the Kelly optimal leverage?

**Back:** The CAGR will begin to decrease and eventually becomes guaranteed to reach $-100\%$.

---

## Card 34

**Front:** Why is the Sharpe ratio considered a limited risk-adjusted measure?

**Back:** It assumes a Gaussian distribution of returns and only accounts for volatility, ignoring fat tails.

---

## Card 35

**Front:** What is the definition of the Calmar ratio?

**Back:** CAGR divided by the absolute value of the maximum drawdown over the most recent three years.

---

## Card 36

**Front:** How does the MAR ratio differ from the Calmar ratio?

**Back:** The MAR ratio uses the maximum drawdown over the entire history instead of a fixed three-year period.

---

## Card 37

**Front:** What is the Markowitz portfolio optimization objective?

**Back:** To maximize expected return for a constant variance, or minimize variance for a constant expected return.

---

## Card 38

**Front:** What is the mathematical relationship between the mean log return $\mu$ and the mean net return $m$?

**Back:** $\mu \approx m - \frac{s^2}{2}$, where $s$ is the standard deviation.

---

## Card 39

**Front:** In portfolio theory, what does the 'Efficient Frontier' represent?

**Back:** A curve plotting the minimum variance possible for every level of expected return.

---

## Card 40

**Front:** What is the 'tangency portfolio'?

**Back:** The specific portfolio on the efficient frontier that maximizes the Sharpe ratio.

---

## Card 41

**Front:** Why is maximizing the expected compound growth rate equivalent to maximizing the Sharpe ratio?

**Back:** Both lead to the same normalized capital allocation ($C^{-1}M$) when assuming an infinite horizon.

---

## Card 42

**Front:** What is a 'minimum variance portfolio'?

**Back:** An allocation that minimizes the variance of returns using only covariance estimates, without needing expected return estimates.

---

## Card 43

**Front:** What is the core principle of a 'risk parity portfolio'?

**Back:** Allocating capital such that each asset contributes an equal amount of risk (usually inverse to its volatility).

---

## Card 44

**Front:** Why can risk parity strategies lead to market 'contagion'?

**Back:** If volatility increases, all funds using the strategy are forced to liquidate simultaneously, driving prices lower.

---

## Card 45

**Front:** As an alternative to volatility, what measure does the author suggest for risk parity allocation to better capture tail risk?

**Back:** Maximum drawdown.

---

## Card 46

**Front:** What are 'consolidated' closing prices, and why can they be problematic for backtesting?

**Back:** They are aggregate prices across all exchanges, which may not represent the actual tradable price at the primary exchange close.

---

## Card 47

**Front:** Which MATLAB toolboxes are recommended by the author for quantitative trading (in order of importance)?

**Back:** 1. Statistics and Machine Learning; 2. Econometrics; 3. Financial Instruments.

---

## Card 48

**Front:** What is the primary risk of using the Kelly formula in the presence of 'Black Monday' type events?

**Back:** The formula assumes Gaussian returns; a single extreme 'fat tail' event can wipe out an account at Kelly leverage.

---

## Card 49

**Front:** Term: Survivorship-bias-free data

**Back:** Definition: A dataset that includes historical prices for securities that have been delisted or went bankrupt.

---

## Card 50

**Front:** In the context of the Kelly formula, what does 'excess returns' refer to?

**Back:** Returns achieved above the risk-free rate.

---

## Card 51

**Front:** Process: How is the tangency portfolio visually identified on a graph?

**Back:** It is the point where a line from the origin (risk-free rate) is tangent to the efficient frontier.

---

## Card 52

**Front:** According to equation 1.1, if a stock has a high standard deviation, how does its mean log return compare to its mean net return?

**Back:** The mean log return will be significantly lower than the mean net return.

---

## Card 53

**Front:** What does a zero Lagrange multiplier imply in the context of Sharpe ratio optimization?

**Back:** It implies that the constant leverage constraint does not affect the optimal relative weights of the assets.

---

## Card 54

**Front:** Which specific platform is cited as a good example of an integrated backtesting and trading system for US equities using Python?

**Back:** Quantopian.com

---

## Card 55

**Front:** What is 'excess cash' in a commodities account?

**Back:** The cash balance that exceeds the margin requirement for existing positions.

---

## Card 56

**Front:** Why can net returns never truly have a Gaussian distribution?

**Back:** Net returns are bounded at $-100\%$ (price cannot go below zero), whereas a Gaussian distribution extends to negative infinity.

---

## Card 57

**Front:** What is 'Expected Shortfall' (ES) used for?

**Back:** It is a risk measure that serves as a stand-in for volatility without relying on the Gaussian assumption.

---

## Card 58

**Front:** Formula: What is the optimal capital allocation vector $F^*$ in matrix notation?

**Back:** $F^* = C^{-1}M$, where $C$ is the covariance matrix and $M$ is the vector of expected returns.

---

## Card 59

**Front:** What is the maximum total SIPC coverage for a combination of securities and cash?

**Back:** $\$500,000$

---

## Card 60

**Front:** Why is the use of 'best bid and offer' (BBO) midprices preferred over trade prices for certain strategies?

**Back:** Trade prices can bounce between bid and ask, creating artificial volatility and inflated backtest results.

---

## Card 61

**Front:** Which database provides geographical locations of oil tankers to help predict short-term supply?

**Back:** Thomson Reuters' Eikon platform.

---

## Card 62

**Front:** What is the specific risk of having a 'personal' account when trading with high leverage?

**Back:** The trader is personally liable for negative equity, meaning losses can exceed the initial investment.

---

## Card 63

**Front:** What is the common name for the method of maximizing the geometric mean return (compound growth)?

**Back:** The Kelly Criterion.

---

## Card 64

**Front:** Why are estimates of covariances generally more reliable than estimates of expected returns?

**Back:** Covariance estimates improve with larger sample sizes, whereas expected return estimates do not.

---

## Card 65

**Front:** How can a trader automate the sweep of excess cash into an insured account?

**Back:** By using a broker that is both a securities broker and a Futures Commission Merchant (FCM).

---

## Card 66

**Front:** Which platform is designed specifically for high-frequency trading and level 2 quote requirements?

**Back:** Lime Brokerage's Strategy Studio.

---

## Card 67

**Front:** Concept: Fat Tails

**Back:** Definition: The phenomenon where extreme market moves occur more frequently than predicted by a normal distribution.

---

## Card 68

**Front:** How is the Sharpe ratio defined for a portfolio in matrix notation?

**Back:** $\frac{F^T M}{(F^T C F)^{1/2}}$

---

## Card 69

**Front:** What is the primary benefit of Python's 'pandas' and 'rpy2' packages for quants?

**Back:** They provide data analysis tools similar to R and allow Python to access all R packages.

---

## Card 70

**Front:** What is the advantage of using Microsoft's Visual Studio or PyCharm for Python development?

**Back:** They provide a polished Integrated Development Environment (IDE) for more productive debugging.

---

## Card 71

**Front:** According to the text, what is the 'minimum variance portfolio's' performance characteristic in the last 20 years?

**Back:** Low volatility stocks (min variance) have consistently outperformed high volatility stocks.

---

## Card 72

**Front:** Which vendor is known for culling short interest details from stock lenders around the Street?

**Back:** SunGard’s Astec database.

---

## Card 73

**Front:** What happens to the approximation $\mu \approx m - s^2/2$ as the time subperiods become smaller?

**Back:** The approximation becomes more exact, approaching continuous time.

---

## Card 74

**Front:** Which Python package facilitates compilation to C/C++ for increased performance?

**Back:** Numba (or NumbaPro for parallel processing).

---
