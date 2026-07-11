# Bitcoin Flashcards

## Card 1

**Front:** In short-term bitcoin trading, why are fundamental events like Fed interest rate decisions considered 'contemporaneous factors' rather than predictive ones?

**Back:** They can only be used to explain movements after the fact rather than predicting them in advance.

---

## Card 2

**Front:** What is the broad definition of 'technical analysis' used in the text?

**Back:** Predictive techniques that only require prices and volumes as input.

---

## Card 3

**Front:** In the currency pair notation B.Q, which currency is the 'base' currency?

**Back:** The first symbol (B).

---

## Card 4

**Front:** Which currency is the most traded pair with bitcoin besides the USD?

**Back:** CNY (Chinese Yuan).

---

## Card 5

**Front:** Why is trading BTC.CNY considered impractical for traders living outside of China or Hong Kong?

**Back:** The CNY is not freely exchangeable and its rate is regulated by the government.

---

## Card 6

**Front:** Unlike traditional stock markets, what are the trading hours for bitcoin?

**Back:** Bitcoin trades 24/7.

---

## Card 7

**Front:** How does the volatility of BTC.USD compare to MXN.USD according to Table 7.1?

**Back:** BTC.USD has significantly higher annualized volatility ($67\%$ vs $16\%$).

---

## Card 8

**Front:** What does the 'kurtosis' metric measure in the context of bitcoin returns?

**Back:** Tail risks (the likelihood of extreme price movements).

---

## Card 9

**Front:** According to Johansson and Tjernstrom (2014), what percentage of bitcoin exchanges fail due to thefts and hacks?

**Back:** $45\%$.

---

## Card 10

**Front:** What is the primary credit risk associated with trading bitcoin on exchanges?

**Back:** Exchange failure due to hacks or thefts resulting in the loss of investor deposits.

---

## Card 11

**Front:** What statistical test is used to confirm if a BTC.USD price series is stationary and mean-reverting?

**Back:** The Augmented Dickey-Fuller (ADF) test.

---

## Card 12

**Front:** Fitting BTC.USD one-minute bar data to an $AR(p)$ model resulted in what specific value for $p$?

**Back:** $p = 16$.

---

## Card 13

**Front:** In the $AR(16)$ model for BTC.USD, what was the calculated value of the first autoregressive coefficient $\phi_1$?

**Back:** $\phi_1 = 0.685$.

---

## Card 14

**Front:** What does a $\phi_1$ value of $0.685$ in an $AR(16)$ model suggest about the BTC.USD time series?

**Back:** The time series is strongly mean-reverting.

---

## Card 15

**Front:** Why was the $40,000\%$ CAGR result for the $AR(16)$ strategy considered unrealistic?

**Back:** It assumed limit orders at midprice are always filled instantaneously.

---

## Card 16

**Front:** What were the optimal $p$ and $q$ parameters found for the $ARMA(p, q)$ model on BTC.USD?

**Back:** $p = 3$ and $q = 7$.

---

## Card 17

**Front:** In a Bollinger band strategy, under what condition would a trader 'buy' a unit of BTC.USD?

**Back:** When the midprice is $k$ moving standard deviations below the moving average.

---

## Card 18

**Front:** In a Bollinger band strategy, when should a trader exit a position?

**Back:** When the price mean-reverts to the current moving average.

---

## Card 19

**Front:** Formula: What is the calculation for the 'Upper band' in a Bollinger band strategy?

**Back:** $Upper Band = MA + k \times MSTD$.

---

## Card 20

**Front:** Formula: What is the calculation for the 'Lower band' in a Bollinger band strategy?

**Back:** $Lower Band = MA - k \times MSTD$.

---

## Card 21

**Front:** To avoid learning 'spurious effects' like bid-ask bounce in AI models, what price data should be used?

**Back:** Midprices.

---

## Card 22

**Front:** Which machine learning technique mentioned in the text resulted in negative returns for both train and test sets on BTC.USD?

**Back:** Support Vector Machine (SVM).

---

## Card 23

**Front:** How was the feedforward neural network trained to improve the reliability of predicted returns?

**Back:** By training 100 networks with different initial guesses and averaging their predictions.

---

## Card 24

**Front:** What is the definition of 'order flow'?

**Back:** Signed transaction volume (positive for buy market orders, negative for sell market orders).

---

## Card 25

**Front:** What is an 'aggressor tag' in a bitcoin exchange data feed?

**Back:** A label specifying whether a trade resulted from a buy or sell market order.

---

## Card 26

**Front:** In the order flow strategy, what triggers the exit of a long position?

**Back:** When the one-minute order flow becomes zero or changes to the opposite sign.

---

## Card 27

**Front:** What is the primary transaction cost concern when executing a high-frequency order flow strategy?

**Back:** The bid-ask spread.

---

## Card 28

**Front:** Concept: Cross-exchange arbitrage

**Back:** Definition: Buying an instrument on one exchange and selling it on another where the bid price is higher than the ask price.

---

## Card 29

**Front:** Why is cross-exchange arbitrage rare in the US stock market?

**Back:** SEC Regulation NMS Rule 610 prevents exchanges from displaying quotes that cross other exchanges.

---

## Card 30

**Front:** What extra risk must be hedged when performing cross-border cross-exchange arbitrage (e.g., IBM on NYSE vs LSE)?

**Back:** Currency risk (exchange rate fluctuations).

---

## Card 31

**Front:** Beyond commissions, what is a significant cost when regularly moving funds between bitcoin exchanges for arbitrage?

**Back:** Withdrawal fees (often around $1\%$).

---

## Card 32

**Front:** If one exchange consistently has lower quotes than others for the same bitcoin pair, what is a likely reason?

**Back:** The market perceives that exchange to have a higher credit risk or lower credit rating.

---

## Card 33

**Front:** In the $AR(16)$ model logic, if the prediction is for a price increase, the trader should _____.

**Back:** Buy

---

## Card 34

**Front:** The process of using multiple regression trees to improve prediction accuracy is known as _____.

**Back:** Bagging (Bootstrap Aggregating)

---

## Card 35

**Front:** True or False: Bitcoin volatility is lower than the volatility of the S&P 500 (SPY).

**Back:** False (Bitcoin volatility is much higher).

---

## Card 36

**Front:** The 'quote' currency in the pair BTC.USD is _____.

**Back:** USD

---

## Card 37

**Front:** What specific data timestamping is required for high-frequency order flow strategies?

**Back:** Microseconds.

---

## Card 38

**Front:** What does a negative 'worst daily move' of $-24\%$ for BTC.USD indicate about its risk profile?

**Back:** High tail risk/kurtosis.

---

## Card 39

**Front:** In the Bollinger band code, the variable 'lookback' refers to the _____.

**Back:** Time window used to calculate the moving average and standard deviation.

---

## Card 40

**Front:** In the order flow strategy, the 'entryThreshold' represents the _____.

**Back:** Minimum net signed volume required within the lookback period to trigger a trade.

---

## Card 41

**Front:** Why is backtesting with 'Trade and Quote' (TAQ) data superior to simplified event-driven backtests?

**Back:** It accounts for the actual bid-ask spread and liquidity available at each microsecond.

---

## Card 42

**Front:** The mnemonic for currency pairs states that the symbol ahead alphabetically is usually the _____ currency.

**Back:** Base

---

## Card 43

**Front:** What platform is mentioned for viewing the latest exchange rates and volumes of various bitcoin exchanges?

**Back:** bitcoincharts.com

---

## Card 44

**Front:** In the provided $AR(16)$ example, the test set for predictions was from _____ to _____.

**Back:** September 3, 2014 to January 15, 2015.

---

## Card 45

**Front:** What was the result of applying the $ARMA(3, 7)$ model compared to the $AR(16)$ model on the test set?

**Back:** The CAGR was much lower ($3.9$ vs $40,000$).

---

## Card 46

**Front:** Which specific exchange was mentioned as being hacked in August 2016?

**Back:** Bitfinex.

---

## Card 47

**Front:** What is the 'smartcumsum' function likely used for in order flow analysis?

**Back:** Calculating the cumulative sum of signed trade sizes to determine net order flow.

---

## Card 48

**Front:** What is the significance of the $24/7$ trading nature of bitcoin for quantitative models?

**Back:** It eliminates 'weekend gaps' and allows for continuous time-series analysis.

---

## Card 49

**Front:** In a mean-reverting strategy, 'stationary' means the price series tends to _____.

**Back:** Return to a long-term mean or constant level over time.

---

## Card 50

**Front:** If the ADF test statistic is less than the critical value (e.g., $-3 < -2.9$), we _____ the null hypothesis of a unit root.

**Back:** Reject

---

## Card 51

**Front:** What is the role of the 'entryZscore' in the Bollinger band strategy?

**Back:** It determines how many standard deviations the price must deviate from the mean to trigger an entry.

---

## Card 52

**Front:** According to the text, what is the 'ideal playground' for a quantitative analyst due to its immunity to fundamental factors?

**Back:** Currencies and bitcoins.

---

## Card 53

**Front:** Which currency pair is used as a proxy for emerging market assets in the risk comparison table?

**Back:** MXN.USD (Mexican Peso vs US Dollar).

---

## Card 54

**Front:** A situation where an exchange fails and takes investor deposits is categorized as _____ risk.

**Back:** Credit

---

## Card 55

**Front:** In the order flow strategy backtest, a profit of $0.097$ per trade was found. If the bid-ask spread is $0.12$, what is the transaction cost per trade?

**Back:** $0.06$ (half the spread).

---

## Card 56

**Front:** If the order flow is positive, it indicates a prevalence of _____ market orders.

**Back:** Buy

---

## Card 57

**Front:** The 'moving standard deviation' in a Bollinger band strategy is calculated on _____ rather than returns.

**Back:** Prices

---

## Card 58

**Front:** What does 'p = 16' represent in an $AR(p)$ model?

**Back:** The model uses 16 previous time steps (lags) to predict the current value.

---

## Card 59

**Front:** In the $ARMA(3,7)$ model, what does 'q = 7' represent?

**Back:** The model includes a moving average of the error terms from the 7 previous time steps.

---

## Card 60

**Front:** What is the primary risk of holding a long position on an exchange with lower-than-average BTC prices?

**Back:** Credit risk (the risk that the exchange might fail or be unable to process withdrawals).

---

## Card 61

**Front:** Why must order flow be updated even if no trades occur in a given time bar?

**Back:** Because the 'past one minute' window continuously moves, potentially dropping old trades from the calculation.

---

## Card 62

**Front:** How does the text define a 'signed' transaction volume?

**Back:** Volume assigned a positive or negative sign based on whether the 'aggressor' was a buyer or a seller.

---

## Card 63

**Front:** What is the standard error reported for the $\phi_1$ coefficient in the BTC $AR(16)$ model?

**Back:** $0.0001$.

---

## Card 64

**Front:** In the code `idx=find( dn <= dn(t)-lookback/60/60/24)`, what is being calculated?

**Back:** The index of trades that occurred exactly one 'lookback' period ago.

---

## Card 65

**Front:** The maximum drawdown for BTC.USD in the analyzed period was _____.

**Back:** $-79\%$.

---

## Card 66

**Front:** In the AI section, why were returns from the regression tree and neural networks described as 'astounding' but 'subject to caveats'?

**Back:** They likely did not account for transaction costs and execution slippage.

---

## Card 67

**Front:** What is the primary difference between $AR(p)$ and $ARMA(p, q)$ models?

**Back:** $ARMA$ adds a Moving Average ($MA$) component ($q$) to the Autoregressive ($AR$) component ($p$).

---

## Card 68

**Front:** Why is credit risk particularly high for bitcoin compared to traditional currencies?

**Back:** The lack of regulation and the historical frequency of exchange hacks/failures ($45\%$).

---

## Card 69

**Front:** If BTC.USD is trading at $239.19$ on Bitfinex and $233.54$ on btc-e, the market is said to be _____.

**Back:** Crossed

---

## Card 70

**Front:** Which exchange was identified as the primary one for BTC.CNY trading in the text?

**Back:** BTC China.

---

## Card 71

**Front:** The term 'bid-ask bounce' refers to _____.

**Back:** Artificial price volatility caused by trades alternating between the bid and ask prices.

---

## Card 72

**Front:** What is 'onshore' Chinese currency called?

**Back:** CNY

---

## Card 73

**Front:** What is 'offshore' Chinese currency called?

**Back:** CNH

---

## Card 74

**Front:** The strategy of averaging predictions from 100 neural networks is used to reduce the impact of _____.

**Back:** Random initial guesses of network parameters.

---

## Card 75

**Front:** In the Bollinger band strategy, what is 'MA' an abbreviation for?

**Back:** Moving Average.

---

## Card 76

**Front:** What is 'CAGR'?

**Back:** Compound Annual Growth Rate.

---

## Card 77

**Front:** Which bitcoin exchange uses the suffix '.USD' and was mentioned as a top exchange alongside Bitfinex and itBit?

**Back:** BitStamp.

---

## Card 78

**Front:** In MATLAB, what is the equivalent of the `adf` function in the Econometrics Toolbox?

**Back:** `adftest`

---

## Card 79

**Front:** How is kurtosis annualized according to the endnotes?

**Back:** By assuming it scales linearly with time.

---

## Card 80

**Front:** True or False: Technical analysis indicators like RSI and Bollinger bands are considered 'broad' technical analysis because they only use price and volume.

**Back:** True.

---
