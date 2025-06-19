
[<img src="https://github.com/user-attachments/assets/0c3a338b-95e6-4432-8160-e0dce5b01a32">](https://qnt.co/book-amazon)

# Master the art of AI-driven algorithmic trading strategies through hands-on examples, in-depth insights, and step-by-step guidance

Hands-On AI Trading with Python, QuantConnect, and AWS explores **real-world applications of AI technologies in algorithmic trading**. It provides practical examples with complete code, allowing readers to understand and expand their AI toolbelt.

<br/>

![getting-started-banner](https://github.com/user-attachments/assets/b00363eb-0c2c-47aa-be24-f7109dfbba95)

<br/>

To use the repo open the ```main.py``` and ```research.ipynb``` files from an example folder and copy/paste the code into your project’s files on QuantConnect Cloud / Local Platform.

<br/>

**If you prefer to use the [LEAN CLI](https://www.lean.io/cli/), follow these steps:**

1. Clone the repo.
2. Run bash command to move the ```main.py```/```research.ipynb``` to a project in your [organization workspace](https://www.quantconnect.com/docs/v2/lean-cli/initialization/organization-workspaces).
3. ```cd``` to the organization workspace.
4. [Run the backtest](https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-backtest) / [Open the notebook](https://www.quantconnect.com/docs/v2/lean-cli/api-reference/lean-research) on your local machine, or push the project to QuantConnect Cloud.

<br/>

**Note:** Running the projects locally require the local machine to have the required [datasets](https://www.quantconnect.com/datasets/).

<br/>

**Using the backtestlib functionality:**

In order to use the [backtestlib](https://github.com/QuantConnect/HandsOnAITradingBook/blob/master/00%20Libraries/backtestlib/backtestlib.py) functionality:

1. Create a library called ```backtestlib```.
2. add the library to your project.

The process to create and add a library depends on whether you use [Cloud Platform](https://www.quantconnect.com/docs/v2/cloud-platform/projects/shared-libraries#02-Create-Libraries), [Local Platform](https://www.quantconnect.com/docs/v2/local-platform/projects/shared-libraries#02-Create-Libraries), or the [CLI](https://www.quantconnect.com/docs/v2/lean-cli/projects/libraries/project-libraries#02-Create-Libraries).

<br/>

![examples-banner](https://github.com/user-attachments/assets/852a9678-9655-4d2a-83ea-1f45d79353d8)

<br/>

Follow the steps above to use these examples:

<br/>

[ML Trend Scanning with MLFinlab](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/01%20ML%20Trend%20Scanning%20with%20MLFinlab)
<br/>
Uses MLFinLab's trend scanning package to detect price trends (up/down/no-trend) for timing Bitcoin trades.
<br/><br/>

[Factor Preprocessing Techniques for Regime Detection](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/02%20Factor%20Preprocessing%20Techniques%20for%20Regime%20Detection)
<br/>
Applies different preprocessing techniques and PCA on market factors to predict SPY's weekly returns using a multiclass random forest model.
<br/><br/>

[Reversion vs. Trending - Strategy Selection by Classification](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/03%20Reversion%20vs%20Trending%20-%20Strategy%20Selection%20by%20Classification)
<br/>
Uses neural networks to predict whether the next trading day will favor momentum or reversion risk exposure by analyzing volatility indicators.
<br/><br/>

[Alpha by Hidden Markov Models](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/04%20Alpha%20by%20Hidden%20Markov%20Models)
<br/>
Employs Hidden Markov Models to predict market volatility regimes and allocate funds between different ETFs and options accordingly.
<br/><br/>

[FX SVM Wavelet Forecasting](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/05%20FX%20SVM%20Wavelet%20Forecasting)
<br/>
Uses Support Vector Machines (SVM) and wavelets to predict forex pair prices, where wavelets decompose price data into components and SVM predicts each component separately.
<br/><br/>

[Dividend Harvesting Selection of High-Yield Assets](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/06%20Dividend%20Harvesting%20Selection%20of%20High-Yield%20Assets)
<br/>
Uses a decision tree regression model to predict future dividend yields based on financial ratios to build a high-yield portfolio.
<br/><br/>

[Effect of Positive-Negative Splits](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/07%20Effect%20of%20Positive-Negative%20Splits)
<br/>
Utilizes a multiple linear regression model to estimate future returns when stock splits are imminent and trades accordingly.
<br/><br/>

[Stoploss Based on Historical Volatility and Drawdown Recovery](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/08%20Stoploss%20Based%20on%20Historical%20Volatility%20and%20Drawdown%20Recovery)
<br/>
Uses regression models to dynamically adjust stop-loss levels based on market conditions.
<br/><br/>

[ML Trading Pairs Selection](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/09%20ML%20Trading%20Pairs%20Selection)
<br/>
Demonstrates using PCA and clustering techniques to identify potential pairs for statistical arbitrage trading. It first applies PCA to transform standardized stock returns into principal components, then uses the OPTICS clustering algorithm and various statistical tests (cointegration, Hurst exponent, half-life) to select optimal trading pairs.
<br/><br/>

[Stock Selection through Clustering Fundamental Data](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/10%20Stock%20Selection%20through%20Clustering%20Fundamental%20Data)
<br/>
Uses PCA and learning-to-rank algorithms to predict relative performance of stocks based on fundamental data.
<br/><br/>

[Inverse Volatility Rank and Allocate to Future Contracts](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/11%20Inverse%20Volatility%20Rank%20and%20Allocate%20to%20Future%20Contracts)
<br/>
Applies ridge regression to predict volatility and allocate futures contracts inversely proportional to their expected volatility.
<br/><br/>

[Trading Costs Optimization](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/12%20Trading%20Costs%20Optimization)
<br/>
Uses a ```DecisionTreeRegressor``` to predict trading costs and optimize trade execution timing.
<br/><br/>

[PCA Statistical Arbitrage Mean Reversion](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/13%20PCA%20Statistical%20Arbitrage%20Mean%20Reversion)
<br/>
Applies PCA and linear regression for statistical arbitrage to exploit price differences between related securities.
<br/><br/>

[Temporal CNN Prediction](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/14%20Temporal%20CNN%20Prediction)
<br/>
Uses a temporal CNN to predict the direction of future stock prices based on OHLCV data.
<br/><br/>

[Gaussian Classifier for Direction Prediction](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/15%20Gaussian%20Classifier%20for%20Direction%20Prediction)
<br/>
Employs Gaussian Naive Bayes classifiers to predict daily returns of technology stocks.
<br/><br/>

[LLM Summarization of Tiingo News Articles](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/16%20LLM%20Summarization%20of%20Tiingo%20News%20Articles)
<br/>
Uses OpenAI's GPT-4 to analyze sentiment from news articles for trading decisions.
<br/><br/>

[Head Shoulders Pattern Matching with CNN](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/17%20Head%20Shoulders%20Pattern%20Matching%20with%20CNN)
<br/>
Uses a one-dimensional CNN to detect head-and-shoulders patterns and trade forex accordingly.
<br/><br/>

[Amazon Chronos Model](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/18%20Amazon%20Chronos%20Model)
<br/>
Utilizes Amazon's Chronos model to forecast future price paths and optimize portfolio weights.
<br/><br/>

[FinBERT Model](https://github.com/QuantConnect/HandsOnAITradingBook/tree/master/06%20Applied%20Machine%20Learning/19%20FinBERT%20Model)
<br/>
Applies the FinBERT language model to assess news sentiment and make trading decisions based on aggregate sentiment scores.
<br/><br/>
<br/><br/>

[<img src="https://github.com/user-attachments/assets/37828761-5a04-4a25-b9f1-c30a796f74ce">](https://qnt.co/book-amazon)

