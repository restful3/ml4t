# CHAPTER 7 Special Topics in Quantitative Trading

The first six chapters of this book covered most of the basic knowledge needed to research, develop, and execute your own quantitative strategy. This chapter explains important themes in quantitative trading in more detail. These themes form the bases of statistical arbitrage trading, and most quantitative traders are conversant in some if not most of these topics. They are also very helpful in informing our intuition about trading.

I will describe the two basic categories of trading strategies: mean-reverting versus momentum strategies. Periods of mean-reverting and trending behaviors are examples of what some traders call regimes, and the different regimes require different strategies, or at least different parameters of the same strategy. Mean-reverting strategies derive their mathematical justification from the concepts of stationarity and cointegration of time series, which I will cover next. Following that, I will discuss a novel application of machine learning to adapt the parameters of a trading strategy to different regimes that we call Conditional Parameter Optimization (CPO). Then I will describe a theory that many hedge funds use to manage large portfolios and one that has caused much turmoil in their performances: namely, factor models. Other categories of strategies that traders frequently discuss are seasonal trading and highfrequency strategies. All trading strategies require a way to exit their positions; I will describe the different logical ways to do this. Finally, I ponder the question of how to best enhance the returns of a strategy: through higher leverage or trading higher-beta stocks?

### MEAN-REVERTING VERSUS MOMENTUM STRATEGIES

Trading strategies can be profitable only if securities prices are either mean-reverting or trending. Otherwise, they are random-walking, and trading will be futile. If you believe that prices are mean reverting and that they are currently low relative to some reference price, you should buy now and plan to sell higher later. However, if you believe the prices are trending and that they are currently low, you should (short) sell now and plan to buy at an even lower price later. The opposite is true if you believe prices are high.

Academic research has indicated that stock prices are on average very close to random walking. However, this does not mean that under certain special conditions, they cannot exhibit some degree of mean reversion or trending behavior. Furthermore, at any given time, stock prices can be both mean reverting and trending, depending on the time horizon you are interested in. Constructing a trading strategy is essentially a matter of determining if the prices under certain conditions and for a certain time horizon will be mean reverting or trending, and what the initial reference price should be at any given time. (When the prices are trending, they are also said to have “momentum,” and thus the corresponding trading strategy is often called a momentum strategy.)

Reversion of the price of a single stock from a temporary deviation from its mean price level back to its mean is called time-series mean reversion, which doesn't happen often. (See, however, the strategy example in the next section on regime change and parameter optimization, which describes an apparently successful attempt to adapt a mean reversion strategy to the changing daily regimes for an ETF.) Mean reversion of the spread of a pair of stocks, or a portfolio of stocks, back to its mean level is called cross-sectional mean reversion, and it happens much more often.

I have already described a trading strategy based on the mean reversion of a pair of stocks (ETFs, to be precise) in Example 3.6. As for the mean reversion of a long–short portfolio of stocks, financial researchers (Khandani and Lo, 2007) have constructed a very simple shortterm mean reversal model that is profitable (before transaction costs) over many years. Of course, whether the mean reversion is strong enough and consistent enough such that we can trade profitably after factoring in transaction costs is another matter, and it is up to you, the trader, to find those special circumstances when it is strong and consistent.

Though cross-sectional mean reversion is quite prevalent, backtesting a profitable meanreverting strategy can be quite perilous.

Many historical financial databases contain errors in price quotes. Any such error tends to artificially inflate the performance of mean-reverting strategies. It is easy to see why: a meanreverting strategy will buy on a fictitious quote that is much lower than some moving average and sell on the next correct quote that is in line with the moving average and thus makes a profit. One must make sure the data is thoroughly cleansed of such fictitious quotes before one can completely trust your backtesting performance on a mean-reverting strategy.

Survivorship bias also affects the backtesting of mean-reverting strategies disproportionately, as I discussed in Chapter 3. Stocks that went through extreme price actions are likely to be either acquired (the prices went very high) or went bankrupt (the prices went to zeros). A meanreverting strategy will short the former and buy the latter, losing money in both cases. However, these stocks may not appear at all in your historical database if it has survivorship bias, thus artificially inflating your backtest performance. You can look up Table 3.1 to find out which database has survivorship bias.

Momentum can be generated by the slow diffusion of information—as more people become aware of certain news, more people decide to buy or sell a stock, thereby driving the price in the same direction. I suggested earlier that stock prices may exhibit momentum when the expected earnings have changed. This can happen when a company announces its quarterly earnings, and investors either gradually become aware of this announcement or they react to this change by incrementally executing a large order (so as to minimize market impact). And indeed, this leads to a momentum strategy called post earnings announcement drift, or PEAD. (For a particularly useful article with lots of references on this strategy, look up   
quantlogic.blogspot.com/2006/03/pocket-phd-post-earning-announcment.html.) Essentially, this strategy recommends that you buy a stock when its earnings exceed expectations and short a stock when it falls short. More generally, many news announcements have the potential of altering expectations of a stock's future earnings, and therefore have the potential to trigger a trending period. As to what kind of news will trigger this, and how long the trending period will last, it is again up to you to find out.

Besides the slow diffusion of information, momentum can be caused by the incremental execution of a large order due to the liquidity needs or private investment decisions of a large investor. This cause probably accounts for more instances of short-term momentum than any other causes. With the advent of increasingly sophisticated execution algorithms adopted by the large brokerages, it is, however, increasingly difficult to ascertain whether a large order is behind the observed momentum.

Momentum can also be generated by the herdlike behavior of investors: investors interpret the (possibly random and meaningless) buying or selling decisions of others as the sole justifications of their own trading decisions. As Yale economist Robert Schiller said in the New York Times (Schiller, 2008), nobody has all the information they need in order to make a fully informed financial decision. One has to rely on the judgment of others. There is, however, no sure way to discern the quality of the judgment of others. More problematically, people make their financial decisions at different times, not meeting at a town hall and reaching a consensus once and for all. The first person who paid a high price for a house is “informing” the others that houses are good investments, which leads another person to make the same decision, and so on. Thus, a possibly erroneous decision by the first buyer is propagated as “information” to a herd of others.

Unfortunately, momentum regimes generated by these two causes (private liquidity needs and herdlike behavior) have highly unpredictable time horizons. How could you know how big an order an institution needs to execute incrementally? How do you predict when the “herd” is large enough to form a stampede? Where is the infamous tipping point? If we do not have a reliable way to estimate these time horizons, we cannot execute a momentum trade profitably based on these phenomena. In a later section on regime switch, I will examine some attempts to predict these tipping or “turning” points.

There are other causes of momentum that are more predictable: the persistence of roll returns in futures markets, and forced sale and purchases of securities due to risk management or portfolio rebalancing. Both these causes are explored in detail in my second book (Chan, 2013).

There is one last contrast between mean-reverting and momentum strategies that is worth pondering. What are the effects of increasing competition from traders with the same strategies? For mean-reverting strategies, the effect typically is the gradual elimination of any arbitrage opportunity, and thus gradually diminishing returns down to zero. When the number of arbitrage opportunities has been reduced to almost zero, the mean-reverting strategy is subject to the risk that an increasing percentage of trading signals are actually due to fundamental changes in stocks' valuation and thus is not going to mean revert. For momentum strategies, the effect of competition is often the diminishing of the time horizon over which the trend will continue. As news disseminates at a faster rate and as more traders take advantage of this trend earlier on, the equilibrium price will be reached sooner. Any trade entered after this equilibrium price is reached will be unprofitable.

### REGIME CHANGE AND CONDITIONAL PARAMETER OPTIMIZATION

The concept of regimes is most basic to financial markets. What else are “bull” and “bear” markets if not regimes? The desire to predict regime changes is also as old as financial markets themselves.

If our attempts to predict the switching from a bull to a bear market were even slightly successful, we could focus our discussion to this one type of switching and call it a day. If only it were that easy. The difficulty with predicting this type of switching encourages researchers to look more broadly at other types of regime switching in the financial markets, hoping to find some that may be more amenable to existing statistical tools.

I have already described two regime changes that are due to changes in market and regulatory structures: decimalization of stock prices in 2003 and the elimination of the short-sale plus-tick rule in 2007. (See Chapter 5 for details.) These regime changes are preannounced by the government, so no predictions of the shifts are necessary, though few people can predict the exact consequences of the regulatory changes.

Some of the other most common financial or economic regimes studied are inflationary vs. recessionary regimes, high- vs. low-volatility regimes, and mean-reverting vs. trending regimes. A more recent regime change may be the rise of retail call options buyers who drove up “meme” stocks’ prices to the stratosphere starting in 2020, due to promotion at the r/WallStreetBets forum at Reddit.com (Kochkodin, 2021). (Those of us who have witnessed the dotcom bubble in 1999 have seen this movie before.) Many well-respected hedge funds (e.g., Melvin Capital) have been brought to their knees due to such regime changes.

Regime changes sometimes necessitate a complete change of trading strategy (e.g. trading a mean-reverting instead of momentum strategy). Other times, traders just need to change the parameters of their existing trading strategy. Traders typically adapt their parameters by optimizing them on a moving (or ever expanding) lookback period, but this conventional method is usually too slow in reacting to a rapidly changing market environment. I have come up with a novel way of adapting the parameters of a trading strategy based on machine learning that I call Conditional Parameter Optimization (CPO). This allows traders to adapt new parameters as frequently as they like—perhaps for every single trade.

CPO uses machine learning to place orders optimally based on changing market conditions in any market. Traders in these markets typically already possess a basic trading strategy that decides the timing, pricing, type, and/or size of such orders. This trading strategy will usually have a small number of adjustable parameters (trading parameters) that are often optimized based on a fixed historical data set (train set). Alternatively, they may be periodically reoptimized using an expanding or continuously updated train set. (The latter is often called Walk Forward Optimization.) In either case, this conventional optimization procedure can be called Unconditional Parameter Optimization, as the trading parameters do not respond to rapidly changing market conditions. Even though they may be optimal on average (where the average is taken over by the historical train set), they may not be optimal under every market condition. Even though we may update the train set to update the parameters, the changes in parameter values are typically small since the changes to the train set from one day to the next are necessarily small. Ideally, we would like trading parameters that are much more sensitive to the market conditions and yet are trained on a large enough amount of data.

To address this adaptability problem, we apply a supervised machine learning algorithm (specifically, random forest with boosting) to learn from a large predictor (feature) set that captures various aspects of the prevailing market conditions, together with specific values of the trading parameters, to predict the outcome of the trading strategy. (An example outcome is the strategy's future one-day return.) Once such a machine-learning model is trained to predict the outcome, we can apply it to live trading by feeding in the features that represent the latest market conditions as well as various combinations of the trading parameters. The set of parameters that results in the optimal predicted outcome (e.g., the highest future one-day return) will be selected as optimal, and will be adopted for the trading strategy for the next period. The trader can make such predictions and adjust the trading strategy as frequently as needed to respond to rapidly changing market conditions. The frequency and magnitude of such adjustments is no longer constrained by the large amount of historical data required for robust optimization using conventional unconditional optimization.

In Example 7.1, I illustrate how we apply CPO using PredictNow.ai's financial machine learning API to adapt the parameters of a Bollinger Band-based mean reversion strategy on GLD (the gold ETF) and obtain superior results.

### Example 7.1: Conditional Parameter Optimization applied to an ETF trading strategy

(This example is reproduced from a blog post on predictnow.ai/blog.)

To illustrate the CPO technique, we describe below an example trading strategy on an ETF.

This strategy uses the lead-lag relationship between the GLD and GDX ETFs using 1- minute bars from January 1, 2006, until December 31, 2020, splitting it $80 \% / 2 0 \%$ between train/test periods. The trading strategy has 3 trading parameters: the hedge ratio (GDX_weight), entry threshold (entry_threshold), and a moving lookback window (lookback). The spread is defined as

$$
S p r e a d ( t ) = G L D_{-} c l o s e ( t ) - G D X \_ c l o s e ( t ) \times G D X \_ w e i g h t .
$$

e may enter a trade for GLD at time $t$ , and exit it at time $t + 1$ minute, hopefully realizin profit. We want to optimize the three trading parameters on a $5 \times 1 0 \times 8$ grid. The grid efined as follows:

$$
\begin{array}{r l} & { \mathrm { { G D X \_ w e i g h t } = \{ 2 , 2 . 5 , 3 , 3 . 5 , 4 \} } } \\ & { \mathrm { { e n t r y \_ t h r e s h o l d = \{ 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 7 , 1 , 1 . 2 5 , 1 . 5 , 2 , 2 , 5 \} } } } \\ & { \mathrm { { l o o k b a c k } = \{ 3 0 , 6 0 , 9 0 , 1 2 0 , 1 8 0 , 2 4 0 , 3 6 0 , 7 2 0 \} } } \end{array}
$$

To be clear, even though we are using GLD and GDX prices and functions of these prices to make trading decisions, we only trade GLD, unlike the typical long-short pair trading setup.

Every minute we compute Spread(t) in equation $\mathbf { \Psi } ( \underline { { 1 } } )$ , and compute its “Bollinger Bands,” conventionally defined as

$$
Z_{-} s c o r e ( t ) = \frac { S p r e a d ( t ) - S p r e a d \_ E M A ( t ) } { \sqrt { S p r e a d \_ V A R ( t ) } }
$$

where Spread_EMA is the exponential moving average of the Spread, and Spread_VAR is its exponential moving variance (see the endnote for their conventional definitions).

Similar to a typical mean-reverting strategy using Bollinger Bands, we trade into a new GLD position based on these rules:

a. Buy GLD if Z_score $<$ -entry_threshold (resulting in long position).   
b. Short GLD if Z_score $>$ entry_threshold (resulting in short position).   
c. Liquidate long position if Z_score $>$ exit_threshold.   
d. Liquidate short position if Z_score $<$ -exit_threshold.

exit_threshold can be anywhere between entry_threshold and –entry_threhold. After optimization in the train set, we set exit_threshold $. = - 0 . 6^{*}$ entry_threshold and keep that relationship fixed when we vary entry_threshold in our future (unconditional or conditional) parameter optimizations. We trade the strategy on 1-minute bars between 9:30 and 15:59 ET, and liquidate any position at 16:00. For each combination of our three trading parameters, we record the daily return of the resulting intraday strategy and form a time series of daily strategy returns, to be used as labels for our machine learning step in CPOs. Note that since the trading strategy may execute multiple round trips per day before forced liquidation at the market close, this daily strategy return is the sum of such roundtrip returns.

### Unconditional vs. Conditional Parameter Optimizations

In conventional, unconditional, parameter optimization, we select the three trading parameters (GDX_weight, entry threshold, and lookback) that maximize cumulative insample return over the three-dimensional parameter grid using exhaustive search. (Gradient-based optimization did not work due to multiple local maxima.) We use that fixed set of three optimal trading parameters to specify the strategy out-of-sample on the test set.

With conditional, parameter optimization, the set of trading parameters used each day depends on a predictive machine-learning model trained on the train set. This model will predict the future one-day return of our trading strategy, given the trading parameters and other market conditions. Since the trading parameters can be varied at will (i.e., they are control variables), we can predict a different future return for many sets of trading parameters each day, and select the optimal set that predicts the highest future return. That optimal parameter set will be used for the trading strategy for the next day. This step is taken after the current day's market close and before the market open of the next day.

In addition to the three trading parameters, the predictors (or “features”) for input to our machine learning model are eight technical indicators obtained from the Technical Analysis Python library: Bollinger Bands Z-score, Money Flow, Force Index, Donchian Channel, Average True Range, Awesome Oscillator, and Average Directional Index. We choose these indicators to represent the market conditions. Each indicator actually produces ${ \bf 2 } \times { \bf 7 }$ features, since we apply them to each of the ETFs GLD and GDX price series, and each was computed using seven different lookback windows: 50,100, 200, 400, 800, 1600, and 3200 minutes. (Note: This is not the same as the trading parameter “lookback” described earlier.) Hence, there are a total of $3 + 8 \times 2 \times 7 = 1 1 5$ features used in predicting the future one-day return of the strategy. But because there are $5 \times 1 0 \times 8 = 4 0 0$ combinations of the three trading parameters, each trading day comes with 400 rows of training data that looks something like the table below (labels are not displayed):

<table><tr><td rowspan=1 colspan=1>GDX_ weight</td><td rowspan=1 colspan=1>entry_hreshold</td><td rowspan=1 colspan=1>lookback</td><td rowspan=1 colspan=1>GLD(50)</td><td rowspan=1 colspan=1>Z-score-GDX(50)</td><td rowspan=1 colspan=1>Money-Flow-GLD(50)</td><td rowspan=1 colspan=1>Money-Flow-GDX(50)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>…•</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>•</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>•….</td></tr><tr><td rowspan=1 colspan=1>…</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>240</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>……•</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>360</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>...</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>720</td><td rowspan=1 colspan=1>0.123</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1></td></tr></table>

After the machine learning model is trained, we can use it for live predictions and trading. Each trading day after the market closes, we prepare an input vector, which is structured like one row of the table above, populated with one particular set of the trading parameters and the current values of the technical indicators, and use the machine learning model to predict the trading strategy's return on the next day. We do that 400 times, varying the trading parameters, but obviously not the technical indicators’ values, and find out which trading parameter set predicts the highest return. We adopt that optimal set for the trading strategy next day. In mathematical terms,

where is the predictive function available from our machine learning website predictnow.ai's API, which uses random forest with boosting as the training algorithm. The sample Python Jupyter notebook code fragment for training a model and using it for predictions is displayed here. (The code won't work unless you sign up for a trial with predictnow.ai.)

$\#$ TO BEGIN ANY WORK WITH PREDICTNOW.AI CLIENT, WE START BY IMPORTING AND CREATING A CLASS INSTANCE   
from predictnow.pdapi import PredictNowClient   
import pandas as pd   
api_key $=$ "%KeyProvidedToEachOfOurSubscriber"   
api_host $=$ "http://12.34.567.890:1000" # our SaaS server   
username $=$ "helloWorld"   
email $=$ "helloWorld@yourmail.com"   
client $=$ PredictNowClient(api_host,api_key)   
$\#$ You will need to edit this input dataset file path and labelname!   
file_path $=$ 'my_amazing_features.xlsx'   
labelname $=$ 'Next_day_strategy_return'   
import os   
$\#$ FANTASTIC JOB! NOW YOUR PREDICTNOW.AI CLIENT HAS BEEN SETUP.   
### For classification problems   
#params $=$ {'timeseries': 'yes', 'type': 'classification', 'feature_selection': 'shap', 'analysis': 'none', 'boost': 'gbdt', 'testsize': '0.2', 'weights': 'no', 'eda': 'yes', 'prob_calib': 'no', 'mode': 'train'}   
$\#$ For regression problems, suitable for CPO   
params $=$ {'timeseries': 'yes', 'type': 'regression', 'feature_selection': 'none', 'analysis': 'none', 'boost': 'gbdt', 'testsize': '0.2', 'weights': 'no', 'eda': 'yes', 'prob_calib': 'no', 'mode': 'train'}   
$\#$ LET'S CREATE THE MODEL BY SENDING THE PARAMETERS TO PREDICTNOW.AI   
response $=$ client.create_model(   
username $=$ username, # only letters, numbers, or underscores   
model_name $=$ "test1",   
params=params,   
)   
### LET'S LOAD UP THE FILE TO PANDAS IN THE LOCAL ENVIRONMENT   
from pandas import read_csv # If you have the Excel file, replace read_csv with   
read_excel   
from pandas import read_excel   
df $=$ read_excel(file_path, engine $=$ "openpyxl") # Same here   
df.name $=$ "testdataframe" # Optional, but recommended   
response $=$ client.train( model_name $=$ "test1", input_df $\mathop { \bf { \ : \underline { { \ : \cdot } } } } =$ df, label $=$ labelname, username $=$ username, email $=$ email, return_output ${ } = { }$ False,   
)   
print("FANTASTIC! YOUR FIRST-EVER MODEL TRAINING AT PREDICTNOW.AI HAS BEEN   
COMPLETED!")   
print(response)   
$\#$ User can now examine the train/test sets results from the model by calling the   
getresult function (and providing the name of the model that resides on   
Predictnow.ai server   
status $=$ client.getstatus(username $=$ username, train_id=response["train_id"])   
if status["state"] $= =$ "COMPLETED": response $=$ client.getresult(   
model_name $=$ "test1", username $=$ username, ) import pandas as pd predicted_targets_cv $=$ pd.read_json(response. predicted_targets_cv) print("predicted_targets_cv") print(predicted_targets_cv) predicted_targets_test $=$ pd.read_json(response. predicted_targets_test) print("predicted_targets_test") print(predicted_targets_test) performance:metrics $=$ pd.read_json(response. performance:metrics) print("performance:metrics") print(performance:metrics)   
### # Now we can make LIVE predictions for many combinations of the parameters by   
populating many rows in the example_input_live.csv file with these parameter   
combinations   
if status["state"] $= =$ "COMPLETED": df $=$ read_csv("example_input_live.csv") # Input data for live prediction df.name $=$ "myfirstpredictname" # optional, but recommended # Making live predictions response $=$ client.predict( model_name $=$ "test1", input_df $\mathop { \bf { \bar { \mathbf { \Lambda } } } }$ df, username $=$ username, eda $=$ "yes", prob_calib=params["prob_calib"], ) # FOR LIVE PREDICTION: (remember labels and probabilities each can have many   
rows corresponding to many combinations of parameters y_pred $=$ pd.read_json(response.labels) print("THE LABELS") print(labels)

An example output labels file from this step looks like this:

<table><tr><td rowspan=1 colspan=1>Date</td><td rowspan=1 colspan=1>pred_target</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_30_0.2 20218132334_ 0.011875</td><td rowspan=1 colspan=1>0.011875</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_60_0.2 20218132344_ 0.012139</td><td rowspan=1 colspan=1>0.01213</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_90_0.2 20218132354 0.012139</td><td rowspan=1 colspan=1>0.01213</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_120_0.2 20218132364 0.012975</td><td rowspan=1 colspan=1>0.012975</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_180_0.2 20218132374 0.012975</td><td rowspan=1 colspan=1>0.012975</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_240_0.2 20218132384 0.012975</td><td rowspan=1 colspan=1>0.012975</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_360_0.2 20218132394 0.012975</td><td rowspan=1 colspan=1>0.012975</td></tr><tr><td rowspan=1 colspan=1>2020-12-24 2.5_720_0.2 20218132404 0.012975</td><td rowspan=1 colspan=1>0.012975</td></tr></table>

where 2.5_30_0.2 is one parameter combination, and 2.5_60_0.2 is another.

It is important to understand that unlike a naïve application of machine learning to predict GLD's one-day return using technical indicators, we are using machine learning to predict the return of a trading strategy applied to GLD given a set of trading parameters, and using those predictions to optimize these parameters on a daily basis. The naïve approach is less likely to succeed because everybody is trying to predict GLD's (i.e., gold's) returns and inviting arbitrage activities, but nobody (until they read this book!) is predicting the returns of this particular GLD trading strategy. Furthermore, many traders do not like using machine learning as a black box to predict returns. In CPO, the trader's own strategy is making the actual predictions. Machine learning is merely used to optimize the parameters of this trading strategy. This provides for a much greater degree of transparency and interpretability.

### Performance Comparisons

We compare out-of-sample test set performance of Unconditional vs. Conditional Parameter Optimization on the last three years of data ending on December 31, 2020, and find the cumulative three-year return to be $7 3 \%$ and $83 \%$ , respectively. All other metrics are improved using CPO. The comparable equity curves can be found in Figure 7.1.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Unconditional Optimization Conditional Optimization</td><td rowspan=1 colspan=1>Unconditional Optimization Conditional Optimization</td></tr><tr><td rowspan=1 colspan=1>Annual Return 17.29%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>19.77%</td></tr><tr><td rowspan=1 colspan=1>Sharpe Ratio</td><td rowspan=1 colspan=1>1.947</td><td rowspan=1 colspan=1>2.325</td></tr><tr><td rowspan=1 colspan=1>Calmar Ratio</td><td rowspan=1 colspan=1>0.984</td><td rowspan=1 colspan=1>1.454</td></tr></table>

![](images/efaf0a4c65a74c3e8a257d4801516d353aa5eb3ce40088bab9d880d6ef566c29.jpg)  
FIGURE 7.1 Cumulative returns of strategies based on Conditional vs. Unconditional Parameter Optimization.

### Endnote: Definitions of and

$$
\begin{array}{l} { \displaystyle S p r e a d_{-} E M A ( t + 1 ) = \frac { 2 } { l o o k b a c k_{-} p e r i o d } S p r e a d ( t + 1 ) + } \\ { \displaystyle \left( 1 - \frac { 2 } { l o o k b a c k_{-} p e r i o d } \right) S p r e a d_{-} E M A ( t ) , } \\ { \displaystyle S p r e a d_{-} V A R ( 1 ) = \left( S p r e a d ( 1 ) - S p r e a d ( 0 ) \right) ^ { 2 } , } \\ { \displaystyle S p r e a d \ V A R ( t + 1 ) = \frac { 2 } { 2 } \left( S p r e a d ( t + 1 ) - S p r e a d \ E M A ( t + 1 ) \right) ^ { 2 } + } \end{array}
$$

### STATIONARITY AND COINTEGRATION

A time series is stationary if it never drifts farther and farther away from its initial value. In technical terms, stationary time series are “integrated of order zero,” or I(0) (Alexander, 2001). It is obvious that if the price series of a security is stationary, it would be a great candidate for a mean-reversion strategy. Unfortunately, most stock price series are not stationary—they exhibit a geometric random walk that gets them farther and farther away from their starting (i.e., initial public offering) values. However, you can often find a pair of stocks such that if you long one and short the other, the market value of the pair is stationary. If this is the case, then the two individual time series are said to be cointegrated. They are so described because a linear combination of them is integrated of order zero. Typically, two stocks that form a cointegrating pair are from the same industry group. Traders have long been familiar with this so-called pairtrading strategy. They buy the pair portfolio when the spread of the stock prices formed by these pairs is low, and sell/short the pair when the spread is high—in other words, a classic meanreverting strategy.

![](images/8092c79eb6adf666c4ecd368dc9dee63b4d3f9e515a3d59137dd208d06546078.jpg)  
FIGURE 7.2 A stationary time series formed by the spread between GLD and GDX.

An example of a pair of cointegrating price series is the gold ETF GLD versus the gold miners ETF, GDX, which I discussed in Example $3 . 6$ . If we form a portfolio with long 1 share of GLD and short 1.631 share of GDX, the prices of the portfolio form a stationary time series (see Figure 7.2). The exact number of shares of GLD and GDX can be determined by a regression fit of the two component time series (see Example 7.2). Note that just like Example 3.6, I have used only the first 252 data points as the training set for this regression.

### Example 7.2: How to Form a Good Cointegrating (and Mean-Reverting) Pair of Stocks

As I explained in the main text, if you are long one security and short another one in the same industry group and in the right proportion, sometimes the combination (or "spread") becomes a stationary series. A stationary series is an excellent candidate for a meanreverting strategy. This example teaches you how to use a free MATLAB package, downloadable at www.spatial-econometrics.com, to determine if two price series are cointegrated and, if so, how to find the optimal hedge ratio (i.e., the number of shares of the second security versus one share of the first security).

The main method used to test for cointegration is called the cointegrating augmented Dickey-Fuller test, hence the function name cadf. A detailed description of this method can be found in the manual, also available on the same website mentioned earlier.

### Using MATLAB

The following program is available online as epchan.com/book/example7_2.m:

$\%$ make sure previously defined variables are erased.   
clear;   
$\%$ read a spreadsheet named "GLD.xls" into MATLAB.   
[num, txt] $\mathsf { \Omega }_{1} = \mathtt{x} \mathtt { \bot }$ sread('GLD');   
$\%$ the first column (starting from the second row) is % the trading days in format mm/dd/yyyy.   
tday1 $=$ txt(2:end, 1);   
% convert the format into yyyymmdd.   
tday1 $=$ ..   
datestr(datenum(tday1, 'mm/dd/yyyy'), 'yyyymmdd');   
$\%$ convert the date strings first into cell arrays and $\%$ then into numeric format.   
tday1 $=$ str2double(cellstr(tday1));   
$\%$ the last column contains the adjusted close prices.   
adjcls1 $=$ num(:, end);   
$\%$ read a spreadsheet named "GDX.xls" into MATLAB.   
[num2, txt2] $=$ xlsread('GDX');   
$\%$ the first column (starting from the second row) is $\%$ the trading days in format mm/dd/yyyy.   
tday2 $: =$ txt2(2:end, 1);   
% convert the format into yyyymmdd.   
tday2 $: =$ ..   
datestr(datenum(tday2, 'mm/dd/yyyy'), 'yyyymmdd');   
$\%$ convert the date strings first into cell arrays and $\%$ then into numeric format.   
tday2 $=$ str2double(cellstr(tday2));   
adjcls2 $=$ num2(:, end);   
$\%$ find all the days when either GLD or GDX has data.   
tday $=$ union(tday1, tday2);   
[foo idx idx1] $=$ intersect(tday, tday1);   
$\%$ combining the two price series adjcls $=$ NaN(length(tday), 2);   
adjcls(idx, 1) $= a$ djcls1(idx1);   
[foo idx idx2] $=$ intersect(tday, tday2);   
adjcls(idx, 2) $=$ adjcls2(idx2);   
$\%$ days where any one price is missing baddata $\equiv$ find(any(\~isfinite(adjcls), 2));   
tday(baddata) $=$ [];   
adjcls(baddata,:) $= [ ~ ]$ ;   
trainset $^ { = 1 }$ :252; % define indices for training set vnames $=$ strvcat('GLD', 'GDX'); adjcls ${ }_{,} = { }$ adjcls(trainset, :);   
tday $=$ tday(trainset, :);   
$\%$ run cointegration check using   
$\%$ augmented Dickey-Fuller test   
res $=$ cadf(adjcls(:, 1), adjcls(:, 2), 0, 1);   
prt(res, vnames);   
% Output from cadf function:   
% Augmented DF test for co-integration variables:   
GLD,GDX   
$\%$ CADF t-statistic # of lags AR(1) estimate   
$\%$ -3.18156477 1 -0.070038   
$\%$   
$\%$ 1% Crit Value 5% Crit Value 10% Crit Value   
$\%$ -3.924 -3.380 -3.082   
$\%$ The t-statistic of -3.18 which is in between the $5 \%$ Crit Value of -3.38 $\%$ and the $10 \%$ Crit Value of -3.08 means that there is a better than $90 \%$   
$\%$ probability that these 2 time series are cointegrated.results $=$ ols(adjcls(:, 1), adjcls(:, 2));   
hedgeRatio $=$ results.beta   
$_ { z = }$ results.resid;   
$\%$ A hedgeRatio of 1.6766 was found.   
$\%$ I.e. $\mathtt { G L D = 1 . 6 7 6 6^{\star} G D X + \mu_{\Sigma} }$ , where z can be   
$\%$ interpreted as the   
$\%$ spread GLD-1. $6 7 6 6^{\star} \mathsf { G D X }$ and should be stationary.   
$\%$ This should produce a chart similar to Figure 7.2.   
plot(z);

### Using Python

The following program is available as epchan.com/book/example7_2.ipynb:

How to Form a Good Cointegrating (and Mean-Reverting) Pair of Stocks   
import numpy as np   
import pandas as pd   
import matplotlib.pyplot as plt   
from statsmodels.tsa.stattools import coint   
from statsmodels.api import OLS   
df1=pd.read_excel('GLD.xls')   
df2=pd.read_excel('GDX.xls')   
df=pd.merge(df1, df2, on ${ } = { }$ 'Date', suffixes $=$ ('_GLD', '_GDX'))   
df.set_index('Date', inplace $=$ True)   
df.sort_index(inplace $=$ True)   
trainset ${ } = { }$ np.arange(0, 252)   
df=df.iloc[trainset,] Run cointegration (Engle-Granger) test   
coint_t, pvalue, crit_value $=$ coint(df['Adj Close_GLD'], df['Adj Close_GDX'])   
(coint_t, pvalue, crit_value) # abs(t-stat) $>$ critical value at 95%. pvalue says   
probability of null hypothesis (of no cointegration) is only 1.8%   
(-2.3591268376687244, 0.3444494880427884, array([-3.94060523, -3.36058133, -3.06139039]))   
Determine hedge ratio   
model ${ \bf \Phi } . = { \bf { \Phi } }$ OLS(df['Adj Close_GLD'], df['Adj Close_GDX'])   
results $=$ model.fit()   
hedgeRatio $=$ results.params   
hedgeRatio   
Adj Close_GDX 1.631009 dtype: float64   
spread $=$ GLD - hedgeRatio\*GDX   
spread $\equiv$ df['Adj Close_GLD']-hedgeRatio[0]\*df['Adj Close_GDX']   
plt.plot(spread)

You may notice that the Python code's Engle-Granger test generates a $t { \cdot }$ -statistic of $- 2 . 4$ , whose absolute value is less than the $90 \%$ critical value, indicating that the two series are not cointegrating. This contradicts the results of the MATLAB cadf test. Which should we trust? Let me just say that Python's libraries are free and come with no guarantees on accuracy nor correctness, whereas MATLAB employs a staff of numerous PhD computer scientists and statisticians.

### Using R

You can download the R code as example7_2.R.

### Need the zoo package for its na.locf function   
install.packages('zoo')   
### Need the CADFtest package for its CADFtest function   
install.packages('CADFtest')   
library('zoo')   
library('CADFtest')   
data1 <- read.delim("GLD.txt") # Tab-delimited   
data_sort1 <- data1[order(as.Date(data1[,1], '%m/%d/%Y')),] # sort in ascending order of dates (1st column of data)   
tday1 <- as.integer(format(as.Date(data_sort1[,1], '%m/%d/%Y'), '%Y%m%d')) adjcls1 <- data_sort1[,ncol(data_sort1)]   
data2 <- read.delim("GDX.txt") # Tab-delimited   
data_sort2 <- data2[order(as.Date(data2[,1], '%m/%d/%Y')),] # sort in ascending order of dates (1st column of data)   
tday2 <- as.integer(format(as.Date(data_sort2[,1], '%m/%d/%Y'), '%Y%m%d')) adjcls2 <- data_sort2[,ncol(data_sort2)]   
### find the intersection of the two data sets   
tday <- intersect(tday1, tday2)   
adjcls1 <- adjcls1[tday1 %in% tday]   
adjcls2 <- adjcls2[tday2 %in% tday]   
### CADFtest cannot have NaN values in input   
adjcls1 <- zoo::na.locf(adjcls1)   
adjcls2 <- zoo::na.locf(adjcls2)   
mydata <- list(GLD $=$ adjcls1, GDX $\equiv$ adjcls2);   
trainset <- 1:252   
res <- CADFtest(model $=$ GLD\~GDX, data $=$ mydata, type $=$ "drift", max.lag. ${ \tt X } { = } 1$ ,   
subset ${ } = { }$ trainset)   
summary(res) # As the following input shows, p-value is about 0.005; hence we can reject null hypothesis of no cointegration at 99.5% level.   
### Covariate Augmented DF test   
### CADF test   
### t-test statistic: -3.240868894   
### estimated rho^2: 0.260414676   
### p-value: 0.004975155   
### Max lag of the diff. dependent variable: 1.000000000   
### Max lag of the stationary covariate(s): 1.000000000   
### Max lead of the stationary covariate(s): 0.000000000   
### dynlm(formula $=$ formula(model), start $=$ obs.1, end $=$ obs.T)   
Residuals:   
Min 1Q Median 3Q Max   
-2.70728 -0.26235 0.00595 0.29684 1.47164   
### Coefficients:   
### Estimate Std. Error t value Pr(>|t|)   
### (Intercept) -0.07570 0.29814 -0.254 0.79970   
### L(y, 1) -0.03817 0.01178 -3.241 0.00498 \*\*   
L(d(y), 1) 0.08542 0.03077 2.776 0.00578 \*\*   
L(X, 0) 0.75428 0.02802 26.919 < 2e-16 \*\*\*   
L(X, 1) -0.68942 0.03161 -21.812 < 2e-16 \*\*\*

The R code's Engle-Granger test generates a $t \cdot$ -statistic of $- 3 . 2$ , which rejects the null hypothesis that the pair is not cointegrating. This corroborates the MATLAB cadf test, while repudiating the Python's result. Moral of the story: Do not trust Python's statistics and econometrics packages.

In case you think that any two stocks in the same industry group would be cointegrating, here is a counterexample: KO (Coca-Cola) versus PEP (Pepsi). The same cointegration test as used in Example 7.1 tells us that there is a less than 90 percent probability that they are cointegrated. (You should try it yourself and then compare with my program   
epchan.com/book/example7_3.m.) If you use linear regression to find the best fit between KO and PEP, the plot of the time series will resemble Figure 7.3.

If a price series (of a stock, a pair of stocks, or, in general, a portfolio of stocks) is stationary, then a mean-reverting strategy is guaranteed to be profitable, as long as the stationarity persists into the future (which is by no means guaranteed). However, the converse is not true. You don't necessarily need a stationary price series in order to have a successful mean-reverting strategy. Even a nonstationary price series can have many short-term reversal opportunities that one can exploit, as many traders have discovered.

![](images/9c873f1a4eac593ee3d472f6a1f74f39f3cea9ee0da54777f500cc524c239285.jpg)  
FIGURE 7.3 A nonstationary time series formed by the spread between KO and PEP.

Many pair traders are unfamiliar with the concepts of stationarity and cointegration. But most of them are familiar with correlation, which superficially seems to mean the same thing as cointegration. Actually, they are quite different. Correlation between two price series actually refers to the correlations of their returns over some time horizon (for concreteness, let's say a day). If two stocks are positively correlated, there is a good chance that their prices will move in the same direction most days. However, having a positive correlation does not say anything about the long-term behavior of the two stocks. In particular, it doesn't guarantee that the stock prices will not grow farther and farther apart in the long run, even if they do move in the same direction most days. However, if two stocks were cointegrated and remain so in the future, their prices (weighted appropriately) will be unlikely to diverge. Yet their daily (or weekly, or any other time horizon) returns may be quite uncorrelated.

As an artificial example of two stocks, A and B, that are cointegrated but not correlated, see Figure 7.4. Stock B clearly doesn't move in any correlated fashion with stock A: Some days they move in the same direction, other days the opposite. Most days, stock B doesn't move at all. But notice that the spread in stock prices between A and B always returns to about $\$ 1$ after a while.

![](images/f039a920a9c5250ea6cd52bbc746f8ad1dac9edb4f1afb56b553769cfea01103.jpg)  
FIGURE 7.4 Cointegration is not correlation. Stocks A and B are cointegrated but not correlated.

Can we find a real-life example of this phenomenon? Well, KO versus PEP is one. In the program example7_3.m, I have shown that they do not cointegrate. If, however, you test their daily returns for correlation, you will find that their correlation of 0.4849 is indeed statistically significant. The correlation test is presented at the end of the example7_3.m program and shown here in Example 7.3.

### Example 7.3: Testing the Cointegration versus Correlation Properties between KO and PEP

The cointegration test for KO and PEP is the same as that for GDX and GLD in Example 7.2, so it won't be repeated here. (It is available from epchan.com/book/example7_3.m.) The cointegration result shows that the $t \cdot$ -statistic for the augmented Dicky-Fuller test is – 2.14, larger than the 10 percent critical value of $- 3 . 0 3 8 $ , meaning that there is a less than 90 percent probability that these two time series are cointegrated.

The following code fragment, however, tests for correlation between the two time series:

### Using MATLAB

You can download the MATLAB code as example7_3.m.

$\%$ A test for correlation. dailyReturns $=$ (adjcls-lag1(adjcls))./lag1(adjcls); [R,P] $=$ corrcoef(dailyReturns(2:end,:)); $\%$ R $=$ $\%$   
$\%$ 1.0000 0.4849   
$\%$ 0.4849 1.0000 $\%$ $\%$ P = $\%$   
$\%$ 1 0   
0 1 $\%$ The P value of 0 indicates that the two time series are significantly correlated.

### Using Python

You can download the Python Jupyter notebook as example7_3.ipynb.

How to Form a Good Cointegrating (and Mean-Reverting) Pair of Stocks   
import numpy as np   
import pandas as pd   
import matplotlib.pyplot as plt   
from statsmodels.tsa.stattools import coint   
from statsmodels.api import OLS   
from scipy.stats import pearsonr   
df1=pd.read_excel('KO.xls')   
df2=pd.read_excel('PEP.xls')   
df=pd.merge(df1, df2, on ${ } = { }$ 'Date', suffixes $=$ ('_KO', '_PEP'))   
df.set_index('Date', inplace $=$ True)   
df.sort_index(inplace $=$ True)   
Run cointegration (Engle-Granger) test   
coint_t, pvalue, crit_value $=$ coint(df['Adj Close_KO'], df['Adj Close_PEP'])   
(coint_t, pvalue, crit_value) # abs(t-stat) $<$ critical value at 90%. pvalue says   
probability of null hypothesis (of no cointegration) is 73%   
(-1.5815517041517178,   
0.7286134576473527,   
array([-3.89783854, -3.33691006, -3.04499143])) Determine hedge ratio   
model $=$ OLS(df['Adj Close_KO'], df['Adj Close_PEP'])   
results=model.fit()   
hedgeRatio $=$ results.params hedgeRatio   
Adj Close_PEP 1.011409   
dtype: float64   
spread $=$ KO - hedgeRatio\*PEP   
spread $\equiv$ df['Adj Close_KO']-hedgeRatio[0]\*df['Adj Close_PEP']   
plt.plot(spread) # Figure 7.2   
[<matplotlib.lines.Line2D at 0x2728e431b00>]   
png   
png   
Correlation test   
dailyret ${ } = { }$ df.loc[:, ('Adj Close_KO', 'Adj Close_PEP')].pct_change()   
dailyret.corr()   
Adj Close_KO   
Adj Close_PEP   
Adj Close_KO   
1.000000   
0.484924   
Adj Close_PEP   
0.484924   
1.000000   
dailyret_clean ${ } = { }$ dailyret.dropna()   
pearsonr(dailyret_clean.iloc[:,0], dailyret_clean.iloc[:,1]) # first output correlation coefficient, second output is pvalue.   
(0.4849239439370571, 0.0)

### Using R

You can download the R code as example7_3.R.

### Need the zoo package for its na.locf function# install.packages('zoo')   
### install.packages('CADFtest')   
library('zoo')   
library('CADFtest')   
source('calculateReturns.R') data1 <- read.delim("KO.txt") # Tab-delimited   
data_sort1 <- data1[order(as.Date(data1[,1], '%m/%d/%Y')),] # sort in ascending order of dates (1st column of data)   
tday1 <- as.integer(format(as.Date(data_sort1[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls1 <- data_sort1[,ncol(data_sort1)]   
data2 <- read.delim("PEP.txt") # Tab-delimited   
data_sort2 <- data2[order(as.Date(data2[,1], '%m/%d/%Y')),] # sort in ascending order of dates (1st column of data)   
tday2 <- as.integer(format(as.Date(data_sort2[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls2 <- data_sort2[,ncol(data_sort2)]   
### find the intersection of the two data sets   
tday <- intersect(tday1, tday2)   
adjcls1 <- adjcls1[tday1 %in% tday]   
adjcls2 <- adjcls2[tday2 %in% tday]   
### CADFtest cannot have NaN values in input   
adjcls1 <- zoo::na.locf(adjcls1)   
adjcls2 <- zoo::na.locf(adjcls2)   
mydata <- list( $\operatorname { K O } =$ adjcls1, PEP $^ { \circ } =$ adjcls2);   
res <- CADFtest(model $=$ KO\~PEP, data $=$ mydata, type $=$ "drift", max.lag. ${ \tt X } { = } 1$ )   
summary(res) # As the following input shows, p-value is about 0.16, hence we cannot reject null hypothesis.   
### Covariate Augmented DF test   
### CADF test   
### t-test statistic: -2.2255225   
### estimated rho^2: 0.8249085   
### p-value: 0.1612782

![](images/91ff06b2039dd466267c9753352b52f0385ef6452e54542a8f5bab679c36c558.jpg)

Stationarity is not limited to the spread between stocks: it can also be found in certain currency rates. For example, the Canadian dollar/Australian dollar (CAD/AUD) cross-currency rate is quite stationary, both being commodities currencies. Numerous pairs of futures as well as fixedincome instruments can be found to be cointegrating as well. (The simplest examples of cointegrating futures pairs are calendar spreads: long and short futures contracts of the same underlying commodity but different expiration months. Similarly for fixed-income instruments, one can long and short bonds by the same issuer but of different maturities.)

### FACTOR MODELS

Financial commentators often say something like this: “The current market favors value stocks,” “The market is focusing on earnings growth,” or, “Investors are paying attention to inflation numbers.” How do we quantify these and other common drivers of returns?

There is a well-known framework in quantitative finance called factor models (also known as arbitrage pricing theory [APT]) that attempts to capture the different drivers of returns such as earnings growth rates, interest rate, or the market capitalization of a company. These drivers are called factors. Mathematically, we can write the excess returns (returns minus risk-free rate) $R$ of $N$ stocks as

$$
R = X b + u
$$

where $X$ is an $N \times F$ matrix of factor exposures (also known as factor loadings), $^ { b }$ is an $F$ vector of factor returns, and $u$ an $N$ vector of specific returns. (Every one of these quantities is time dependent, but I suppress this explicit dependence for simplicity.)

The terms factor exposure, factor return, and specific return are commonly used in quantitative finance, and it is well worth our effort to understand their meanings.

Let's focus on a specific category of factors called time-series factors—they are returns on specially constructed long-short portfolios called hedge portfolios, explained as follows. These factor returns are the common drivers of stock returns, and are therefore independent of a particular stock, but they do vary over time (hence, they are call time-series factors).

Factor exposures are the sensitivities of a stock to each of these common drivers. Any part of a stock's return that cannot be explained by these common factor returns is deemed a specific return (i.e., specific to a stock and essentially regarded as just random noise within the APT framework). Each stock's specific return is assumed to be uncorrelated to another stock's.

Let's illustrate these using a simple time-series factor model called the Fama-French Three-Factor model (Fama and French, 1992). This model postulates that the excess return of a stock depends linearly on only three factors:

1. The return of the market (the market factor).   
2. The return of a hedge portfolio that longs small (based on market capitalization) stocks and shorts big stocks. This is the SMB, or small-minus-big, factor.   
3. The return of a hedge portfolio that longs high book-to-price-ratio (or “cheap”) stocks and shorts low book-to-price-ratio (or “expensive”) stocks. This is the HML, or high-minus-low, factor.

More intuitively, the SMB factor measures whether the market favors small-cap stocks. It usually does, except for the last three year as of this writing. The HML factor measures whether the market favors value stocks. It usually does, except for 8 of the last 12 years as of this writing (Phillips, 2020)!

The factor exposures of stock are the sensitivity (regression coefficient) of its returns with respect to the factor returns: its beta (i.e., its sensitivity to the market index), its sensitivity to SMB, and its sensitivity to HML. Factor exposures are obviously different for each stock. A small-cap stock has a positive exposure to SMB, while a growth stock has a negative exposure to HML. (Factor exposures are often normalized such that the average of the factor exposures within a universe of stocks is zero, and the standard deviation is 1.)

To find the factor exposures of a stock, run a linear regression of its historical returns against the Fama-French factors, as in Equation 7.1. (You can download the historical Fama-French factors from mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.) Note this regression is contemporaneous, not predictive—you can't use these factor returns to predict a stock's next day's return, unless you can predict the next day's SMB and HML.

There are other stock factors, such as the price-to-earnings ratio or dividend yield of a stock, where we can directly observe the factor exposures of each stock (e.g., the price-to-earnings factor exposure of AAPL is just AAPL's $\mathrm { P / E }$ ratio!). These have been called cross-sectional factors, because we have to estimate the factor return of a single period by regressing all the stocks’ returns against their factor exposures. For example, if we want to estimate the $\mathrm { P / E }$ factor return, the $X$ variable of this regression is a matrix of different stocks’ $\mathrm { P / E }$ ratios, while the $Y$ variable is a vector of the corresponding returns of those stocks in the calendar quarter during which the earnings were announced. Note that this regression is also contemporaneous, not predictive—you can't use these factor returns to predict a stock's next quarter, unless you can predict their next quarter's $\mathrm { P / E }$ .

The Fama-French model has no monopoly on the choice of time-series factors. In fact, you can construct as many factors as creativity and rationality allow. For example, some people have constructed the WML (winners-minus-losers) factor, which is a momentum factor that measures the return of a hedge portfolio that longs stocks that previously had positive returns and short stocks that previously had negative returns. There are even more choices for crosssectional factors. For example, you can choose return on equity as a factor exposure. You can choose any number of other economic, fundamental, or technical time-series or cross-sectional factors. Whether the factor exposures you have chosen are sensible will determine whether the factor model explains the excess returns of the stocks adequately. If the factor exposures (and consequently the model as a whole) are poorly chosen, the linear regression fit will produce specific returns of significant sizes, and the $R^{2}$ statistic of the fit will be small. According to experts (Grinold and Kahn, 1999), the $R^{2}$ statistic of a good factor model with monthly returns of 1,000 stocks and 50 factors is typically about 30 to 40 percent.

Since these factor models are contemporaneous—that is, given historical returns and factor exposures, we can compute the factor returns of those same historical periods—what good are they for trading? It turns out that often factor returns are more stable than individual stock returns—they exhibit stronger serial autocorrelations than individual stock's returns. In other words, they have momentum. You can therefore assume that their values remain unchanged from the current period (known from the regression fit) to the next time period. If this is the case, then, of course, you can also predict the excess returns, as long as the factor exposures are well chosen and therefore the time-varying specific returns are not significant.

Let me clarify one point of potential confusion. Even though I stated that factor models can be useful as a predictive model (and therefore for trading) only if we assume the factor returns have momentum, it does not mean that factor models cannot capture mean reversion of stock returns. You can, in fact, construct a factor exposure that captures mean reversion, such as the negative of the previous period return. If stock returns are indeed mean reverting, then the corresponding factor return will be positive.

If you are interested in building a trading model based on fundamental factors, there are a number of vendors from whom you can obtain historical factor data:

S&P Capital IQ: spgglobal.com/marketintelligence   
Compustat: www.compustat.com   
MSCI: www.msci.com   
Sharadar: sharadar.com (This is the most affordable source.)   
A very comprehensive and technical introduction to factor models can be found in Ruppert (2015).

### Example 7.4: Principal Component Analysis as an Example of the Factor Model

The examples of factor exposures I described above are typically economic (e.g., outperformance of value stocks), fundamental (e.g., book-to-price ratio), or technical (e.g., previous period's return). However, there is one kind of factor model that relies on nothing more than historical returns to construct. These are the so-called statistical factors, obtained using methods such as the principal component analysis (PCA).

If we use PCA to construct the statistical factor exposures and factor returns, we must assume that the factor exposures are constant (time independent) over the estimation period. (This rules out factors that represent mean reversion or momentum, since these factor exposures depend on the prior period returns.) In a sense, this is more similar to the time-series factors such as SMB than the cross sectional factors such as $\mathrm { P / E }$ , because the factor exposures of time-series factors are also assumed to be constant over a long lookback period. However, unlike time-series factors, the statistical factors are unobservable, and unlike cross-sectional factor exposures, the statistical factor exposures are also unobservable. More importantly, we assume that the factor returns are uncorrelated; that is to say, their covariance matrix $\langle b b^{T} \rangle$ is diagonal. If we use the eigenvectors of the covariance matrix $\langle R R^{T} \rangle$ as the columns of the matrix X in the APT equation $R = X b +$ $u$ , we will find via elementary linear algebra that $\langle b b^{T} \rangle$ is indeed diagonal; and furthermore, the eigenvalues of $\langle R R^{T} \rangle$ are none other than the variances of the factor returns $^ { b }$ . But of course, there is no point to use factor analysis if the number of factors is the same as the number of stocks—typically, we can just pick the eigenvectors with the top few eigenvalues to form the matrix $\mathbf{X}$ . The number of eigenvectors to pick is a parameter that you can adjust to optimize your trading model.

In the following programs, I illustrate a possible trading strategy applying PCA to S&P 600 small-cap stocks. It is a strategy based on the assumption that factor returns have momentum: They remain constant from the current time period to the next. Hence, we can buy the stocks with the highest expected returns based on these factors, and short the ones with the lowest expected returns. The average annualized return of this strategy is only $2 \%$ (MATLAB) to $4 \%$ (Python and R), and only when we assume no transaction costs. (The difference in returns among the programs are essentially round off errors.)

### Using MATLAB

You can download the MATLAB code as example7_4.m.

clear;   
lookback $= 2 5 2$ ; $\%$ use lookback days as estimation (training) period for determining factor exposures.   
numFactors $= 5$ ;   
topN $= 5 0$ ; % for trading strategy, long stocks with topN expected 1-day returns.   
onewaytcost ${ } = 0$ /10000; load('IJR_20080114.mat');   
$\%$ test on SP600 smallcap stocks. (This MATLAB binary input file contains tday, stocks, op, hi, lo, cl arrays.

mycls $=$ fillMissingData(cl);

positionsTable $=$ zeros(size(cl));

dailyret ${ } = { }$ (mycls-backshift(1, mycls))./backshift(1, mycls); % note the rows of   
dailyret are the observations at different time periods   
end_index $=$ length(tday);   
for t=lookback+2:end_index $\mathrm{R} =$ dailyret(t-lookback:t-1,:)'; % here the columns of R are the different   
observations. hasData $=$ find(all(isfinite(R), 2)); $\%$ avoid any stocks with missing returns $\mathrm{R} { = } \mathrm{R}$ (hasData, :); [PCALoadings,PCAScores,PCAVar] $=$ pca(R); $\begin{array}{r l} { \mathrm { ~ X ~ } } & { { } = } \end{array}$ PCAScores(:,1:numFactors); $\begin{array}{r l} { \boldsymbol { \nabla } } & { { } = } \end{array}$ dailyret(t, hasData)'; Xreg $=$ [ones(size(X, 1), 1) X]; [b,sigma] $=$ mvregress(Xreg,y); $\sf { p r e d } \ = \ \sf { X r e g } { \star } \mathrm{b} .$ ; Rexp $=$ sum(pred,2); % Rexp is the expected return for next period assuming factor   
returns remain constant. [foo idxSort] $=$ sort(Rexp, 'ascend'); positionsTable(t, hasData(idxSort(1:topN))) $= - 1$ ; $\%$ short topN stocks with lowest   
expected returns positionsTable(t, hasData(idxSort(end-topN $^ { + 1 }$ :end))) $^ { = 1 }$ ; $\%$ buy topN stocks with   
highest expected returns   
end   
ret $=$ smartsum(backshift(1, positionsTable).\*dailyret-onewaytcost\*abs(positionsTable  
backshift(1, positionsTable)), 2)./smartsum(abs(backshift(1, positionsTable)), 2);   
% compute daily returns of trading strategy   
fprintf(1, 'AvgAnnRet $\begin{array}{r} { { \bf \Pi } = \frac { 9 } { 9 } } \end{array}$ f Sharpe $= \%$ f\n', smartmean(ret,1) $^ { \star } 2 5 2$ ,   
sqrt(252)\*smartmean(ret,1)/smartstd(ret,1));   
$\%$ AvgAnnRet ${ } = 0$ .020205 Sharpe ${ } = 0$ .211120

This program made use of a function mvregress for multivariate linear regression with possible missing or NaN values in the input matrix. Using this function, the computation time is under a minute. Otherwise, it may take hours.

### Using Python

You can download the Python code as example7_4.py.

### Principal Component Analysis as an Example of Factor Model   
import math   
import numpy as np   
import pandas as pd   
from numpy.linalg import eig   
from numpy.linalg import eigh   
#from statsmodels.api import OLS   
import statsmodels.api as sm   
from sklearn.linear_model import LinearRegression   
from sklearn.multioutput import MultiOutputRegressor   
from sklearn.decomposition import PCA   
from sklearn.preprocessing import StandardScaler   
from sklearn import linear_model   
from sklearn.linear_model import Ridge   
import time   
lookback $= 2 5 2$ # training period for factor exposure   
numFactors $= 5$   
topN $= 5 0$ # for trading strategy, long stocks with topN exepcted 1-day returns   
df=pd.read_table('IJR_20080114.txt')   
df['Date'] $=$ df['Date'].astype('int')   
df.set_index('Date', inplace $=$ True)   
df.sort_index(inplace $=$ True)   
df.fillna(method='ffill', inplace $=$ True)   
dailyret ${ } = { }$ df.pct_change() # note the rows of dailyret are the observations at   
different time periods   
positionsTabl $\circleddash$ np.zeros(df.shape)   
end_index $=$ df.shape[0]   
#end_index $=$ lookback $^ +$ 10   
for t in np.arange(lookback $^ { + 1 }$ ,end_index): $\mathrm{R} =$ dailyret.iloc[t-lookback+1:t,].T # here the columns of R are the different   
observations. hasData $= \mathtt{np}$ .where(R.notna().all(axis $^ { = 1 }$ ))[0] R.dropna(inplace $=$ True) # avoid any stocks with missing returns pca $=$ PCA() $\begin{array}{r l} { \mathrm { ~ X ~ } } & { { } = } \end{array}$ pca.fit_transform(R.T)[:, :numFactors] $\begin{array}{r l} { \mathrm { ~ X ~ } } & { { } = } \end{array}$ sm.add_constant(X) $\mathrm { y 1 ~ = ~ \mathbb{R} . T }$ $_ { \textrm { C l f } } =$   
MultiOutputRegressor(LinearRegression(fit_intercept $=$ False),n_jobs $; = 4$ ).fit(X, y1) Rexp $=$ np.sum(clf.predict(X),axis ${ } = 0$ ) $\mathrm{R} =$ dailyret.iloc[t-lookback+1:t+1,].T # here the columns of R are the different   
observations.

idxSort $\mathbf { \tau } = \mathbf { \tau }$ Rexp.argsort()

positionsTable[t, hasData[idxSort[np.arange(0, topN) $] ] ] = - 1$   
### positionsTable[t, hasData[idxSort[np.arange(-topN,0) $] ] ] = ]$   
positionsTable[t, hasData[idxSort[np.arange(-topN, -1) $] \ ] \ ] \ = \ 1$   
capital $=$ np.nansum(np.array(abs(pd.DataFrame(positionsTable)).shift()), axis $^ { * = 1 }$ )   
positionsTable[capita $\scriptstyle - = = 0$ ,] $] = 0$   
capital[capital $= = 0 1 = 1$   
ret $=$ np.nansum(np.array(pd.DataFrame(positionsTable).shift()) $\star_{\mathrm{np} }$ .array(dailyret),   
axis $^ { * = 1 }$ )/capital   
avgret $\equiv .$ np.nanmean(ret) $^ { \star } 2 5 2$   
avgstdev $=$ np.nanstd(ret) $\star$ math.sqrt(252)   
Sharpe $=$ avgret/avgstdev   
print(avgret)   
print(avgstdev)   
print(Sharpe)   
#0.04052422056844459   
#0.07002908500498846   
#0.5786769963588398

### Using R

You can download the R code as example7_4.R.

rm(list=ls()) # clear workspace   
backshift <- function(mylag, x) {   
rbind(matrix(NaN, mylag, ncol(x)), as.matrix(x[1:(nrow(x)-mylag),]))   
}   
#install.packages('pls')   
library("pls")   
library('zoo')   
source('calculateReturns.R')   
#source('backshift.R')   
lookback <- 252 # use lookback days as estimation (training) period for determining factor exposures.   
numFactors <- 5   
topN <- 50 # for trading strategy, long stocks with topN expected 1-day returns.   
data1 <- read.csv("IJR_20080114.csv") # Tab-delimited   
cl <- data.matrix(data1[, 2:ncol(data1)])   
cl[ is.nan(cl) ] <- NA   
tday $< -$ data.matrix(data1[, 1])   
mycls <- na.fill(cl, type $=$ "locf", nan ${ = } \mathrm{N} \mathbf { \mathbb{A} }$ , fill $=$ NA)   
end_loop <- nrow(mycls)   
positionsTable $< -$ matrix(0, end_loop, ncol(mycls))   
dailyret <- calculateReturns(mycls, 1)   
dailyret[is.nan(dailyret)] <- 0   
dailyret <- dailyret[1:end_loop,]   
for (it in (lookback $^ { + 2 }$ ):end_loop) { R <- dailyret[(it-lookback+2):it,] hasData $< -$ which(complete.cases(t(R))) $\mathrm { ~ \mathsf ~ { ~ R ~ } ~ } < - \mathrm { ~ \mathsf ~ { ~ R ~ } ~ } [$ [, hasData ] PCA $< -$ prcomp(t(R)) X <- t(PCA\$x[1:numFactors,]) Rexp <- rep(0,ncol(R)) for (s in 1:ncol(R)){ reg_result <- lm(R[,s] \~ X ) pred <- predict(reg_result) pred[is.nan(pred)] <- 0 Rexp[s] <- sum(pred) } result <- sort(Rexp, index.return ${ } = { }$ TRUE) positionsTable[it, hasData[result\$ix[1:topN]] ] = -1   
positionsTable[it, hasData[result\$ix[(length(result\$ix)-topN-  
1):length(result\$ix)]] ] = 1   
}   
capital $< -$ rowSums(abs(backshift(1, positionsTable)), na.rm $=$ TRUE, dims $\ c = ~ 1$ )   
ret <- rowSums(backshift(1, positionsTable)\*dailyret, na.rm $=$ TRUE, dims $=$   
1)/capital   
avgret $< -$ $2 5 2^{\star}$ mean(ret, na.rm $=$ TRUE)   
avgstd $< -$ sqrt(252)\*sd(ret, na.rm $=$ TRUE)   
Sharpe $=$ avgret/avgstd   
print(avgret)   
print(avgstd)   
print(Sharpe)   
#0.04052422056844459   
#0.07002908500498846   
#0.5786769963588398

How good are the performances of factor models in real trading? Naturally, it mostly depends on which factor model we are looking at. But one can make a general observation that factor models that are dominated by fundamental and macroeconomic factors have one major drawback—they depend on the fact that investors persist in using the same metric to value companies. This is just another way of saying that the factor returns must have momentum for factor models to work.

For example, even though the value (HML) factor returns are usually positive, there are periods of time when investors prefer growth stocks, such as during the internet bubble in the late 1990s, in 2007, and quite recently from 2017 to 2020. As The Economist noted, one reason growth stocks were back in favor in 2007 is the simple fact that their price premium over value stocks had narrowed significantly (Economist, 2007a). Another reason is that as the US economy slowed, investors increasingly opted for companies that still managed to generate increasing earnings instead of those that were hurt by the recessionary economy. In 2020, Covid-19 caused many sectors of the economy to slump, but not for technology companies, as consumers and businesses moved much of their activities to online only.

Therefore, it is not uncommon for factor models to experience steep drawdown during the times when investors' valuation methods shift, even if only for a short duration. But then, this

problem is common to practically any trading model that holds stocks overnight.

### WHAT IS YOUR EXIT STRATEGY?

While entry signals are very specific to each trading strategy, there isn't usually much variety in the way exit signals are generated. They are based on one of these:

A fixed holding period A target price or profit cap The latest entry signals A stop price

A fixed holding period is the default exit strategy for any trading strategy, whether it is a momentum model, a reversal model, or some kind of seasonal trading strategy, which can be either momentum or reversal based (more on this later). I said before that one of the ways momentum is generated is the slow diffusion of information. In this case, the process has a finite lifetime. The average value of this finite lifetime determines the optimal holding period, which can usually be discovered in a backtest.

One word of caution on determining the optimal holding period of a momentum model: As I said before, this optimal period typically decreases due to the increasing speed of the diffusion of information and the increasing number of traders who catch on to this trading opportunity. Hence, a momentum model that has worked well with a holding period equal to a week in the backtest period may work only with a one-day holding period now. Worse, the whole strategy may become unprofitable a year into the future. Also, using a backtest of the trading strategy to determine holding period can be fraught with data-snooping bias, since the number of historical trades may be limited. Unfortunately, for a momentum strategy where the trades are triggered by news or events, there are no other alternatives. For a mean-reverting strategy, however, there is a more statistically robust way to determine the optimal holding period that does not depend on the limited number of actual trades.

The mean reversion of a time series can be modeled by an equation called the Ornstein-Uhlenbeck formula (Uhlenbeck and Ornstein, 1930). Let's say we denote the mean-reverting spread (long market value minus short market value) of a pair of stocks as $z ( t )$ . Then we can write

$$
d z ( t ) = - \theta \bigl ( z ( t ) - \mu \bigr ) d t + d W
$$

where $\mu$ is the mean value of the prices over time and $d W$ is simply some random Gaussian noise. Given a time series of the daily spread values, we can easily find $\theta$ (and $\mu { \dot { } }$ ) by performing a linear regression fit of the daily change in the spread $d z$ against the spread itself. Mathematicians tell us that the average value of $z ( t )$ follows an exponential decay to its mean $\mu$ , and the half-life of this exponential decay is equal to $l n ( 2 ) / \theta$ , which is the expected time it takes for the spread to revert to half its initial deviation from the mean. This half-life can be used to determine the optimal holding period for a mean-reverting position. Since we can make use of the entire time series to find the best estimate of $\theta$ , and not just on the days where a trade was triggered, the estimate for the half-life is much more robust than can be obtained directly from a trading model. In Example 7.5, I demonstrate this method of estimating the half-life of mean reversion using our favorite spread between GLD and GDX.

### Example 7.5: Calculation of the Half-Life of a Mean-Reverting Time Series

We can use the mean-reverting spread between GLD and GDX in Example 7.2 to illustrate the calculation of the half-life of its mean reversion.

### Using MATLAB

The MATLAB code is available as example7_5.m. (The first part of the program is the same as example7_2.m.)

$\begin{array}{r l} { \frac { \circ } { \circ } } & { { } = = = } \end{array}$ Insert example7_2.m in the beginning here $= = = =$ prevz $=$ backshift(1, z); % z at a previous time-step $\mathtt{d} z = z$ -prevz;   
d $\ z \ ( 1 ) = [ \ ]$ ;   
prevz $( \tilde { \bot } ) = [ \tilde { \Big ] }$ ;   
$\%$ assumes dz $=$ theta\*(z-mean(z))dt+w,   
$\%$ where w is error term   
results $=$ ols(dz, prevz-mean(prevz));theta ${ } = { }$ results.beta; halfli 7.8390 $_ { -- 1 0 9 }$ (2)/theta   
$\%$ $=$   
$\%$

The program finds that the half-life for mean reversion of the GLD-GDX is about 10 days, which is approximately how long you should expect to hold this spread before it becomes profitable.

### Using Python

The Python code is available as example7_5.ipynb. (The first part of the program is the same as example7_2.ipynb.)

Calculation of the Half-Life of a Mean-Reverting Time Series   
import numpy as np   
import pandas as pd   
import matplotlib.pyplot as plt   
from statsmodels.tsa.stattools import coint   
from statsmodels.api import OLS   
df1=pd.read_excel('GLD.xls')   
df2=pd.read_excel('GDX.xls')   
df=pd.merge(df1, df2, on ${ } = { }$ 'Date', suffixes $=$ ('_GLD', '_GDX'))   
df.set_index('Date', inplace $=$ True)   
df.sort_index(inplace $=$ True)   
Run cointegration (Engle-Granger) test   
coint_t, pvalue, crit_value $=$ coint(df['Adj Close_GLD'], df['Adj Close_GDX'])   
(coint_t, pvalue, crit_value) # abs(t-stat) > critical value at 95%. pvalue says   
probability of null hypothesis (of no cointegration) is only 1.8%   
(-3.6981160763300593, 0.018427835409537425,   
array([-3.92518794, -3.35208799, -3.05551324]))   
Determine hedge ratio   
model ${ \bf \Phi } . = { \bf { \Phi } }$ OLS(df['Adj Close_GLD'], df['Adj Close_GDX'])   
results=model.fit()   
hedgeRatio $=$ results.params   
hedgeRatio   
Adj Close_GDX 1.639523   
dtype: float64   
$z =$ GLD - hedgeRatio\*GDX   
$_ { z = }$ df['Adj Close_GLD']-hedgeRatio[0]\*df['Adj Close_GDX']   
plt.plot(z)   
prev $z = z$ .shift()   
$\mathtt{d} z = z$ -prevz   
$\mathtt{d} z { = } \mathtt{d} z \left[ 1 : , \ \right]$   
prevz $=$ prevz[1:,]   
model2 $=$ OLS(dz, prevz-np.mean(prevz))   
results $? =$ model2.fit()   
theta $. ^ { = }$ results2.params   
theta   
x1 -0.088423 dtype: float64   
halflife $=$ -np.log(2)/theta   
halflife   
x1 7.839031 dtype: float64

### Using R

The R code is available as example7_5.R. (The first part of the program is the same as example7_2.R.)

source(backshift.R')   
data1 $< -$ read.delim("GLD.txt") # Tab-delimited   
data_sort1 $< -$ data1[order(as.Date(data1[,1], '%m/%d/%Y')),] # sort in ascending   
order of dates (1st column of data)   
tday1 $< -$ as.integer(format(as.Date(data_sort1[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls1 <- data_sort1[,ncol(data_sort1)]   
data2 <- read.delim("GDX.txt") # Tab-delimited   
data_sort2 <- data2[order(as.Date(data2[,1], '%m/%d/%Y')),] # sort in ascending   
order of dates (1st column of data)   
tday2 <- as.integer(format(as.Date(data_sort2[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls2 <- data_sort2[,ncol(data_sort2)]   
### find the intersection of the two data sets   
tday <- intersect(tday1, tday2)   
adjcls1 <- adjcls1[tday1 %in% tday]   
adjcls2 <- adjcls2[tday2 %in% tday]   
### determines the hedge ratio on the trainset   
result <- lm(adjcls1 \~ 0 + adjcls2 )   
hedgeRatio <- coef(result) # 1.64   
spread <- adjcls1-hedgeRatio\*adjcls2 # spread $=$ GLD - hedgeRatio\*GDX   
prevSpread <- backshift(1, as.matrix(spread))   
prevSpread <- prevSpread - mean(prevSpread, na.rm $=$ TRUE)   
deltaSpread <- c(NaN, diff(spread)) # Change in spread from t-1 to t   
result2 $< -$ lm(deltaSpread \~ 0 $^ +$ prevSpread )   
theta $< -$ coef(result2)   
halflife <- -log(2)/theta # 7.839031

If you believe that your price series is mean reverting, then you also have a ready-made target price—the mean value of the historical prices of the security, or $\mu$ in the Ornstein-Uhlenbeck formula. This target price can be used together with the half-life as exit signals (exit when either criterion is met).

Target prices can also be used in the case of momentum models if you have a fundamental valuation model of a company. But as fundamental valuation is at best an inexact science, target prices are not as easily justified in momentum models as in mean-reverting models. If it were that easy to profit using target prices based on fundamental valuation, all investors have to do is to check out stock analysts' reports every day to make their investment decisions.

Suppose you are running a trading model, and you entered into a position based on its signal. Sometime later, you run this model again. If you find that the sign of this latest signal is opposite to your original position (e.g., the latest signal is “buy” when you have an existing short position), then you have two choices. Either you simply use the latest signal to exit the existing position and become flat or you can exit the existing position and then enter into an opposite position. Either way, you have used a new, more recent entry signal as an exit signal for your existing position. This is a common way to generate exit signals when a trading model can be run in shorter intervals than the optimal holding period.

Notice that this strategy of exiting a position based on running an entry model also tells us whether a stop-loss strategy is recommended. In a momentum model, when a more recent entry signal is opposite to an existing position, it means that the direction of momentum has changed, and thus a loss (or more precisely, a drawdown) in your position has been incurred. Exiting this position now is almost akin to a stop loss. However, rather than imposing an arbitrary stop-loss price and thus introducing an extra adjustable parameter, which invites data-snooping bias, exiting based on the most recent entry signal is clearly justified based on the rationale for the momentum model.

Consider a parallel situation when we are running a reversal model. If an existing position has incurred a loss, running the reversal model again will simply generate a new signal with the same sign. Thus, a reversal model for entry signals will never recommend a stop loss. (On the contrary, it can recommend a target price or profit cap when the reversal has gone so far as to hit the opposite entry threshold.) And, indeed, it is much more reasonable to exit a position recommended by a mean-reversal model based on holding period or profit cap than stop loss, as a stop loss in this case often means you are exiting at the worst possible time. (The only exception is when you believe that you have suddenly entered into a momentum regime because of recent news.)

### SEASONAL TRADING STRATEGIES

This type of trading strategy is also called the calendar effect. Generally, these strategies recommend that you buy or sell certain securities at a fixed date of every year, and close the position at another fixed date. These strategies have been applied to both equity and commodity futures markets. However, from my own experience, much of the seasonality in equity markets has weakened or even disappeared in recent years, perhaps due to the widespread knowledge of this trading opportunity, whereas some seasonal trades in commodity futures are still profitable.

The most famous seasonal trade in equities is called the January effect. There are actually many versions of this trade. One version states that small-cap stocks that had the worst returns in the previous calendar year will have higher returns in January than small-cap stocks that had the best returns (Singal, 2006). The rationale for this is that investors like to sell their losers in December to benefit from tax losses, which creates additional downward pressure on their prices. When this pressure disappeared in January, the prices recovered somewhat. This strategy did not work in 2006–2007, but worked wonderfully in January 2008, which was a spectacular month for mean-reversal strategies. (That January was the one that saw a major trading scandal at Société Générale, which indirectly may have caused the Federal Reserve to have an emergency 75-basis-point rate cut before the market opened. The turmoil slaughtered many momentum strategies, but mean-reverting strategies benefited greatly from the initial severe downturn and then dramatic rescue by the Fed.) The codes for backtesting this January effect strategy are given in Example 7.6.

### Example 7.6: Backtesting the January Effect

Here are the codes to compute the returns of a strategy applied to S&P 600 small-cap stocks based on the January effect.

### Using MATLAB

The MATLAB codes can be found at epchan.com/book/example7_6.m, and the input data is also available there.

clear;   
load('IJR_20080131');   
onewaytcost ${ } = 0$ .0005; % 5bp one way transaction cost   
year $\tt S =$ year(datetime(tday, 'ConvertFrom', 'yyyymmdd'));   
months $=$ month(datetime(tday, 'ConvertFrom', 'yyyymmdd'));   
nextdayyear $=$ fwdshift(1, years);   
nextdaymonth ${ . } = { }$ fwdshift(1, months);   
lastdayofDec $=$ find(month $\mathtt{S} = = 1 2$ & nextdaymonth $\scriptstyle = = 1$ );   
lastdayofJan ${ } = { }$ find(month $\tt{3} = \tt { = } 1$ & nextdaymonth $\ c = = 2$ );   
% lastdayofDec starts in 2004,   
$\%$ so remove 2004 from lastdayofJan   
lastdayofJan $( \tilde { \bot } ) = [ \tilde { \Big ] }$ ;% Ensure each lastdayofJan date   
after each   
$\%$ lastdayofDec date   
assert(all(tday(lastdayofJan) $>$ tday(lastdayofDec)));   
eoy $\underline { { \underline { { \mathbf { \Pi } } } } } =$ find(years $\sim =$ nextdayyear); % End Of Year indices   
eoy(end) $= [ ~ ]$ ; % last index is not End of Year   
$\%$ Ensure eoy dates match lastdayofDec dates   
assert(all(tday(eoy) $= =$ tday(lastdayofDec)));   
annret $=$ ..   
(cl(eoy(2:end),:)-cl(eoy(1:end-1),:))./..   
cl(eoy(1:end-1),:); % annual returns   
janret $=$ ..   
(cl(lastdayofJan(2:end),:)-   
cl(lastdayofDec(2:end),:))./cl(lastdayofDec(2:end),:);   
% January returns   
for $\scriptstyle \mathtt{Y} = 1$ :size(annret, 1) % pick those stocks with valid annual returns hasData ${ } = { }$ .. find(isfinite(annret(y,:))); % sort stocks based on prior year's returns [foo sortidx] $=$ sort(annret(y, hasData), 'ascend'); % buy stocks with lowest decile of returns, % and vice versa for highest decile   
topN $=$ round(length(hasData)/10); % portfolio returns portRet ${ } = { }$ .. (smartmean(janret(y, hasData(sortidx(1:topN))))-.. smartmean(janret(y, hasData(.. sortidx(end-topN $^ { + 1 }$ :end)))))/2-2\*onewaytcost; fprintf(1,'Last holding date %i: Portfolio return=%7.4f\n', tday(lastdayofJan $( \mathbb{Y}^{+ 1} )$ ), portRet);   
end   
$\%$ These should be the output   
$\%$ Last holding date 20060131: Portfolio return $=$ -0.0244   
% Last holding date 20070131: Portfolio return $=$ -0.0068   
$\%$ Last holding date 20080131: Portfolio return $=$ 0.0881

This program uses a number of utility programs. The first one is the assert function, which is very useful for ensuring the program is working as expected.

function assert(pred, str)   
$\%$ ASSERT Raise an error if the predicate is not true. $\%$ assert(pred, string)   
if nargin $< 2$ , str $=$ ''; end   
if \~pred   
$\begin{array}{r l} { \mathsf { s } } & { { } = } \end{array}$ sprintf('assertion violated: %s', str);   
error(s);   
end

The second one is the fwdshift function, which works in the opposite way to the lag1 function: It shifts the time series one step forward.

function $\mathrm{y} =$ fwdshift(day,x)   
assert( $\mathtt { d a y > = 0 }$ );   
y=[x(day $^ { + 1 }$ :end,:,:); ..   
NaN\*ones(day,size(x,2), size(x, 3))];

### Using Python

The Python codes can be found at epchan.com/book/example7_ $\underline { { 6 . \mathrm{py} } } .$ , and the input data is also available there.

### Backtesting the January Effect   
import numpy as np   
import pandas as pd   
onewaytcost ${ } = 0$ .0005   
df=pd.read_table('IJR_20080131.txt')   
df['Date'] $=$ df['Date'].round().astype('int')   
df['Date'] $=$ pd.to_datetime(df['Date'], format $=$ '%Y%m%d')   
df.set_index('Date', inplace $=$ True)   
eoyPrice $: =$ df.resample('Y').last()[0:-1] # End of December prices. Need to remove last date because it isn't really end of year   
annret ${ } = { }$ eoyPrice.pct_change().iloc[1:,:] # first row has NaN   
eojPrice $=$ df.resample('BA-JAN').last()[1:-1] # End of January prices. Need to remove first date to match the years in lastdayofDec. Need to remove last date because it isn't really end of January.   
janret ${ } = { }$ (eojPrice.values-eoyPrice.values)/eoyPrice.values   
janret=janret[1:,] # match number of rows in annret   
for y in range(len(annret)):   
hasData $=$ np.where(np.isfinite(annret.iloc[y, :]))[0]   
sortidx $=$ np.argsort(annret.iloc[y, hasData])   
topN $\mathop { \bf { \ : \equiv } }$ np.round(len(hasData)/10)   
portRet $=$ (np.nanmean(janret[y, hasData[sortidx.iloc[np.arange(0, topN)]]])- np.nanmean(janret[y, hasData[sortidx.iloc[np.arange(-topN $^ { + 1 }$ , -1)]]]))/2-   
$2^{\star}$ onewaytcost # portfolio returns   
print("Last holding date %s: Portfolio return $= \%$ f" $\%$ (eojPrice.index $\mathbb{I} \mathbb{Y}^{+} \mathbb{1} .$ ], portRet))   
#Last holding date 2006-01-31 00:00:00: Portfolio return $=$ -0.023853   
#Last holding date 2007-01-31 00:00:00: Portfolio return $=$ -0.003641   
#Last holding date 2008-01-31 00:00:00: Portfolio return ${ } = 0$ .088486

### Using R

The R codes can be found at epchan.com/book/example7_6.R, and the input data is also available there.

### install.packages('lubridate')   
library('lubridate')   
source('calculateReturns.R')   
source('fwdshift.R')   
onewaytcost <- 5/10000 # 5bps one way transaction cost   
data1 <- read.delim("IJR_20080131.txt") # Tab-delimited   
cl <- data.matrix(data1[, 2:ncol(data1)])   
tday <- ymd(data.matrix(data1[, 1])) # dates in lubridate format   
years $< -$ year(tday)   
months <- month(tday)   
years <- as.matrix(years, length(years), 1)   
months <- as.matrix(months, length(months), 1)   
nextdayyear $< -$ fwdshift(1, years)   
nextdaymonth $< -$ fwdshift(1, months)   
eom $< -$ which(months! $=$ nextdaymonth) # End of month indices.   
eoy $< -$ which(years! $=$ nextdayyear) # End Of Year indices. Note that in R, 2008!=NaN returns FALSE whereas in Matlab $2 0 0 8 \sim = \mathrm{NaN}$ returns TRUE   
annret $< -$ calculateReturns(cl[eoy,], 1) # annual returns   
annret $< -$ annret[-1,]   
monret <- calculateReturns(cl[eom,], 1) # monthly returns   
janret <- monret[months[eom] $= = 1$ ,] # January returns   
janret $< -$ janret[-(1:2),] # First January does not have preceding year   
exitDay $< -$ tday[months $= = 1$ & nextdaymonth $\Longrightarrow 2$ ] # Last day of Janurary   
exitDay $< -$ exitDay[-(c(1))] # Exclude first January   
for (y in 1:nrow(annret)) {   
hasData $< -$ which(is.finite(annret[y,])) # pick those stocks with valid annual returns   
sortidx <- order(annret[y, hasData]) # sort stocks based on prior year's returns topN $< -$ round(length(hasData)/10) # buy stocks with lowest decile of returns, and vice versa for highest decile   
portRet <- (sum(janret[y, hasData[sortidx[1:topN]]], na. $\tt r m =$ TRUE)-sum(janret[y, hasData[sortidx[(length(sortidx)-topN $^ { + 1 }$ ):length(sortidx)]]], na.rm $\cdot^{=}$ TRUE))/2/topN-2\*onewaytcost # portfolio returns   
msg $< -$ sprintf('Last holding date %s: Portfolio return $= \% 7$ .4f\n',   
as.character(exitDay $\cdot Y^{+ 1 ]}$ ), portRet)   
cat(msg)   
}   
### Last holding date 2006-01-31: Portfolio return $=$ -0.0244   
### Last holding date 2007-01-31: Portfolio return $=$ -0.0068   
### Last holding date 2008-01-31: Portfolio return $=$ 0.0881

Does this seasonal stock strategy still work? I will leave it as an out-of-sample exercise for the reader.

Another seasonal strategy in equities was proposed more recently (Heston and Sadka, 2007; available at lcb1.uoregon.edu/rcg/seminars/seasonal072604.pdf). This strategy is very simple: each month, buy a number of stocks that performed the best in the same month a year earlier, and short the same number of stocks that performed poorest in that month a year earlier. The average annual return before 2002 was more than 13 percent before transaction costs. However, I have found that this effect has disappeared since then, as you can check for yourself in Example 7.7. (See the readers' comments to my blog post   
epchan.blogspot.com/2007/11/seasonal-trades-in-stocks.html.)

### Example 7.7: Backtesting a Year-on-Year Seasonal Trending Strategy

Here are the codes for the year-on-year seasonal trending strategy I quoted earlier. Note that the data contains survivorship bias, as it is based on the S&P 500 index on November 23, 2007.

### Using MATLAB

The source code can be downloaded from epchan.com/book/example7_7.m. The data is also available at that site.

$\%$   
$\%$ written by:   
clear;   
load('SPX_20071123', 'tday', 'stocks', 'cl');   
monthEnds $=$ find(isLastTradingDayOfMonth(tday)); % find the indices of the days that   
are at month ends.   
tday $=$ tday(monthEnds);   
$_ \mathrm{c1} { = }_{\mathrm{C} } 1$ (monthEnds, :);   
monthlyRet ${ } = { }$ (cl-lag1(cl))./lag1(cl);   
positions $=$ zeros(size(monthlyRet));   
for $\mathtt{m} = \mathtt{l} \mathtt{4}$ :size(monthlyRet, 1) [monthlyRetSorted sortIndex] $=$ sort(monthlyRet(m-12, :)); badData ${ } = { }$ find(\~isfinite(monthlyRet $( \mathtt{m} - 1 2$ , :)) | \~isfinite(cl(monthEnds(m-1), :))); sortIndex ${ . } = { }$ setdiff(sortIndex, badData, 'stable'); topN $\mathop { \bf { \doteq } }$ floor(length(sortIndex)/10); $\%$ take top decile of stocks as longs, bottom   
decile as shorts positions $( \mathfrak{m} - 1$ , sortIndex(1:topN)) $= - 1$ ; positions $( \mathtt{m} - 1$ , sortIndex(end-topN $^ { + 1 }$ :end)) $) = 1$ ;   
end   
ret $=$ smartsum(lag1(positions).\*monthlyRet, 2)./smartsum(abs(lag1(positions)), 2);   
ret $( 1 : 1 3 ) = [ ]$ ;   
avgannret $= 1 2^{\star}$ smartmean(ret);   
sharpe $=$ sqrt(12) $\star$ smartmean(ret)/smartstd(ret);   
fprintf(1, 'Avg ann return $1 = \% 7$ .4f Sharpe rati $) = \frac { 0 } { 0 } 7$ .4f\n', avgannret, sharpe);   
$\%$ Output should be   
% Avg ann return $= - 0$ .0129 Sharpe ratio $\scriptstyle \longmapsto - 0$ .1243

This program contains a few utility functions. The first one is LastTradingDayOfMonth, which returns a logical array of 1s and 0s, indicating whether a month in a trading-date array is the last trading day of a month.

Another is the backshift function, which is like the lag1 function except that one can shift any arbitrary number of periods instead of just 1.

function y=backshift(day,x)   
$\%$ y=backshift(day,x)   
assert( $\mathtt { d a y > = 0 }$ );   
$\mathrm { _ { y = } }$ [NaN(day,size $\left( \mathbf{x} , 2 \right)$ , size(x, 3));x(1:end-day,:,:)];

You can try the most recent five years instead of the entire data period, and you will find that the average returns are even worse.

### Using Python

### Backtesting a Year-on-Year Seasonal Trending Strategy   
import numpy as np   
import pandas as pd   
df=pd.read_table('SPX_20071123.txt')   
df['Date'] $=$ df['Date'].round().astype('int')   
df['Date'] $=$ pd.to_datetime(df['Date'], format $=$ '%Y%m%d')   
df.set_index('Date', inplace $=$ True)   
eomPrice $: =$ df.resample('M').last()[:-1] # End of month prices. Need to remove last   
date because it isn't really end of January.   
monthlyRet ${ } = { }$ eomPrice.pct_change(1, fill_method $=$ None)   
positions $=$ np.zeros(monthlyRet.shape)   
for m in range(13, monthlyRet.shape[0]): hasData $=$ np.where(np.isfinite(monthlyRet.iloc[m-12, :]))[0] sortidx $=$ np.argsort(monthlyRet.iloc[m-12, hasData]) badData $=$ np.where(np.logical_not(np.isfinite(monthlyRet.iloc[m-1,   
hasData[sortidx]])))[0] # these are indices sortidx.drop(sortidx.index[badData], inplace $=$ True) topN $\mathop { \bf { \ : \equiv } }$ np.floor(len(sortidx)/10).astype('int') positions $[ \mathfrak{m} - 1$ , hasData[sortidx.values[np.arange(0, topN) $] ] ] = - 1$ positions $[ \mathfrak{m} - 1$ , hasData[sortidx.values[np.arange(-topN,0)]]] $^ { = 1 }$   
capital ${ \bf \Phi } . = { \bf { \Phi } }$ np.nansum(np.array(pd.DataFrame(abs(positions)).shift()), axi $\mathsf { S } = 1$ )   
capital[capital $= = 0 1 = 1$   
ret $=$ np.nansum(np.array(pd.DataFrame(positions).shift()) $\star_{\Pi \mathbb{P} }$ .array(monthlyRet),   
axi $\gimel = 1$ )/capital   
ret $=$ np.delete(ret, np.arange(13))   
avgret $= \mathtt{np}$ .nanmean(ret) $\star 1 2$   
sharpe $=$ np.sqrt(12) $\star_{\Pi \mathbb{P} }$ .nanmean(ret)/np.nanstd(ret)   
print('Avg ann return $\ c = \%$ f Sharpe ratio $\begin{array}{r} { = \frac { 9 } { c } } \end{array}$ %f' % (avgret, sharpe))   
#Avg ann return $= - 0$ .012679 Sharpe ratio $= - 0$ .122247

### Using R

### The source code can be downloaded as example7_7.R.

### Need the lubridate package for its dates handling install.packages('lubridate')

![](images/442de66f140250f6cee66a6bd27c83cd7c2bde70d66109aceed6dadca83d55a6.jpg)

In contrast to equity seasonal strategies, commodity futures' seasonal strategies are alive and well. That is perhaps because seasonal demand for certain commodities is driven by “real” economic needs rather than speculations.

One of the most intuitive commodity seasonal trades is the gasoline future trade: Simply buy the gasoline future contract that expires in May near the middle of April, and sell it by the end of April. This trade has been profitable for 19 of the last 21 years, as of 2015, the last 9 of which are out of sample (see the sidebar for details). It appears that one can always depend on approaching summer driving seasons in North America to drive up gasoline futures prices in the spring.

### A SEASONAL TRADE IN GASOLINE FUTURES

Whenever the summer driving season comes up, it should not surprise us that gasoline futures prices will be rising seasonally. The only question for the trader is: which month contract to buy, and to hold for what period? After scanning the literature, the best trade I have found so far is one where we buy one May contract of RB (the unleaded gasoline futures trading on the New York Mercantile Exchange [NYMEX]) at the close of April 13 (or the following trading day if it is a holiday), and sell it at the close of April 25 (or the previous trading day if it is a holiday). Historically, we would have realized a profit every year since 1995. Here is the annual profit and loss (P&L) and maximum drawdown (measured from day 1, the entry point) experienced by this position (the 2007–2015 numbers are out-of-sample results):

<table><tr><td rowspan=1 colspan=1>Year</td><td rowspan=1 colspan=1>P&L in $</td><td rowspan=1 colspan=1>P&amp;L in $ Maximum Drawdown in $</td></tr><tr><td rowspan=1 colspan=1>1995</td><td rowspan=1 colspan=1>1,037</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>1996</td><td rowspan=1 colspan=1>1,638</td><td rowspan=1 colspan=1>-2,226</td></tr><tr><td rowspan=1 colspan=1>1997</td><td rowspan=1 colspan=1>227</td><td rowspan=1 colspan=1>-664</td></tr><tr><td rowspan=1 colspan=1>1998</td><td rowspan=1 colspan=1>118</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>1999</td><td rowspan=1 colspan=1>197</td><td rowspan=1 colspan=1>-588</td></tr><tr><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>735</td><td rowspan=1 colspan=1>-1,198</td></tr><tr><td rowspan=1 colspan=1>2001</td><td rowspan=1 colspan=1>1,562</td><td rowspan=1 colspan=1>-304</td></tr><tr><td rowspan=1 colspan=1>2002</td><td rowspan=1 colspan=1>315</td><td rowspan=1 colspan=1>-935</td></tr><tr><td rowspan=1 colspan=1>2003</td><td rowspan=1 colspan=1>1,449</td><td rowspan=1 colspan=1>-2,300</td></tr><tr><td rowspan=1 colspan=1>2004</td><td rowspan=1 colspan=1>361</td><td rowspan=1 colspan=1>-1,819</td></tr><tr><td rowspan=1 colspan=1>2005</td><td rowspan=1 colspan=1>6,985</td><td rowspan=1 colspan=1>-830</td></tr><tr><td rowspan=1 colspan=1>2006</td><td rowspan=1 colspan=1>890</td><td rowspan=1 colspan=1>-4,150</td></tr><tr><td rowspan=1 colspan=1>2007*</td><td rowspan=1 colspan=1>4,322</td><td rowspan=1 colspan=1>-5,279</td></tr><tr><td rowspan=1 colspan=1>2008*</td><td rowspan=1 colspan=1>9,740</td><td rowspan=1 colspan=1>-1,156</td></tr><tr><td rowspan=1 colspan=1>2009*</td><td rowspan=1 colspan=1>-890</td><td rowspan=1 colspan=1>-4,167</td></tr><tr><td rowspan=1 colspan=1>2010*</td><td rowspan=1 colspan=1>1,840</td><td rowspan=1 colspan=1>-3,251</td></tr><tr><td rowspan=1 colspan=1>2011*</td><td rowspan=1 colspan=1>3,381</td><td rowspan=1 colspan=1>-2,298</td></tr><tr><td rowspan=1 colspan=1>2012*</td><td rowspan=1 colspan=1>-7,997</td><td rowspan=1 colspan=1>-8,742</td></tr><tr><td rowspan=1 colspan=1>2013*</td><td rowspan=1 colspan=1>2,276</td><td rowspan=1 colspan=1>-2,573</td></tr><tr><td rowspan=1 colspan=1>2014</td><td rowspan=1 colspan=1>1,541</td><td rowspan=1 colspan=1>-814</td></tr><tr><td rowspan=1 colspan=1>2015*</td><td rowspan=1 colspan=1>8,539</td><td rowspan=1 colspan=1>-1,753</td></tr></table>

\*  Out-of-sample results.

For those who desire less risk, you can buy the mini gasoline futures QU at NYMEX, which trade at half the size of RB, though it is illiquid.

This research has been inspired by the monthly seasonal trades published by Paul Kavanaugh who published at PFGBest. Even though the trades were profitable, the futures broker PFGBest was shut down in 2012 since it embezzled client money. You can read up on this and other seasonal futures patterns in Fielden (2006) or Toepke (2004.)

Besides demand for gasoline, natural gas demand also goes up as summer approaches due to increasing demand from power generators to provide electricity for air conditioning. Hence, another commodity seasonal trade that has been profitable for 13 consecutive years as of this writing is the natural gas trade: Buy the natural gas futures contract that expires in June near the end of February, and sell it by the middle of April. (Again, see sidebar for details.)

### A SEASONAL TRADE IN NATURAL GAS FUTURES

The summer season is also when natural gas demand goes up due to the increasing demand from power generators to provide electricity for air conditioning. This suggests a seasonal trade in natural gas where we long a June contract of NYMEX natural gas futures (Symbol: NG) at the close of February 25 (or the following trading day if it is a holiday), and exit this position on April 15 (or the previous trading day if it is a holiday). This trade has been profitable for 14 consecutive years at of this writing. Here is the annual P&L and maximum drawdown of this trade. (The 2007–2015 numbers are out-of-sample results):

<table><tr><td rowspan=1 colspan=1>Year</td><td rowspan=1 colspan=1>P&amp;L in $</td><td rowspan=1 colspan=1> Maximum Drawdown in $</td></tr><tr><td rowspan=1 colspan=1>1995</td><td rowspan=1 colspan=1>1,970</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>1996</td><td rowspan=1 colspan=1>3,090</td><td rowspan=1 colspan=1>-630</td></tr><tr><td rowspan=1 colspan=1>1997</td><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>-430</td></tr><tr><td rowspan=1 colspan=1>1998</td><td rowspan=1 colspan=1>2,150</td><td rowspan=1 colspan=1>-1,420</td></tr><tr><td rowspan=1 colspan=1>1999</td><td rowspan=1 colspan=1>4,340</td><td rowspan=1 colspan=1>-370</td></tr><tr><td rowspan=1 colspan=1>2000</td><td rowspan=1 colspan=1>4,360</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>2001</td><td rowspan=1 colspan=1>2,730</td><td rowspan=1 colspan=1>-1,650</td></tr><tr><td rowspan=1 colspan=1>2002</td><td rowspan=1 colspan=1>9,860</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>2003</td><td rowspan=1 colspan=1>2,000</td><td rowspan=1 colspan=1>-5,550</td></tr><tr><td rowspan=1 colspan=1>2004</td><td rowspan=1 colspan=1>5,430</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>2005</td><td rowspan=1 colspan=1>2,380</td><td rowspan=1 colspan=1>-230</td></tr><tr><td rowspan=1 colspan=1>2006</td><td rowspan=1 colspan=1>2,250</td><td rowspan=1 colspan=1>-1,750</td></tr><tr><td rowspan=1 colspan=1>2007</td><td rowspan=1 colspan=1>800</td><td rowspan=1 colspan=1>-7,470</td></tr><tr><td rowspan=1 colspan=1>2008</td><td rowspan=1 colspan=1>10,137</td><td rowspan=1 colspan=1>-1,604</td></tr><tr><td rowspan=1 colspan=1>2009*</td><td rowspan=1 colspan=1>-4,240</td><td rowspan=1 colspan=1>-8,013</td></tr><tr><td rowspan=1 colspan=1>2010*</td><td rowspan=1 colspan=1>-8,360</td><td rowspan=1 colspan=1>-10,657</td></tr><tr><td rowspan=1 colspan=1>2011*</td><td rowspan=1 colspan=1>1,310</td><td rowspan=1 colspan=1>-3,874</td></tr><tr><td rowspan=1 colspan=1>2012*</td><td rowspan=1 colspan=1>-7,180</td><td rowspan=1 colspan=1>-8,070</td></tr><tr><td rowspan=1 colspan=1>2013</td><td rowspan=1 colspan=1>5,950</td><td rowspan=1 colspan=1>-1,219</td></tr><tr><td rowspan=1 colspan=1>2014*</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>-3,168</td></tr><tr><td rowspan=1 colspan=1>2015*</td><td rowspan=1 colspan=1>-2,770</td><td rowspan=1 colspan=1>-4,325</td></tr><tr><td rowspan=1 colspan=1>2016*</td><td rowspan=1 colspan=1>530</td><td rowspan=1 colspan=1>-1,166</td></tr></table>

\*  Out-of-sample results.

Unlike the gasoline trade, this natural gas trade didn't hold up as well out-of-sample. Natural gas futures are notoriously volatile, and we have seen big trading losses for hedge funds (e.g., Amaranth Advisors, $\mathrm{loss} = \$ 6$ billion) and major banks (e.g., Bank of Montreal, $\mathrm{loss} = \$ 450$ million). Therefore, one should be cautious if one wants to try out this trade— perhaps at reduced capital using the mini QG futures at half the size of the full NG contract.

Commodity futures seasonal trades do suffer from one drawback, despite their consistent profitability: they typically occur only once a year; therefore, it is hard to tell whether the backtest performance is a result of data-snooping bias (which is why I especially marked the more recent years as out-of-sample). As usual, one way to alleviate this problem is to try somewhat different entry and exit dates to see if the profitability holds up. In addition, one should consider only those trades where the seasonality makes some economic sense. The gasoline and natural gas trades amply satisfy these criteria.

### HIGH-FREQUENCY TRADING STRATEGIES

In general, if a high Sharpe ratio is the goal of your trading strategy (as it should be, given what I said in Chapter 6), then you should be trading at high frequencies, rather than holding stocks overnight.

What are high-frequency trading strategies, and why do they have superior Sharpe ratios? Many experts in high-frequency trading would not regard any strategy that holds positions for more than a few seconds as high frequency, but here I would take a more pedestrian approach and include any strategy that does not hold a position overnight. Many of the early high-frequency strategies were applied to the foreign exchange market, and then later on to the futures market, because of their abundance of liquidity. In the last decade, however, with the increasing liquidity in the equity market, the availability of historical tick database for stocks, and mushrooming computing power, these types of strategies have become widespread for stock trading as well (Lewis, 2014).

The reason why these strategies have high Sharpe ratio is simple: Based on the “law of large numbers,” the more bets you can place, the smaller the percent deviation from the mean return you will experience. With high-frequency trading, one can potentially place hundreds if not thousands of bets all in one day. Therefore, provided the strategy is sound and generates positive mean return, you can expect the day-to-day deviation from this return to be minimal. With this high Sharpe ratio, one can increase the leverage to a much higher level than longerterm strategies can, and this high leverage in turn boosts the return-on-equity of the strategy to often stratospheric levels.

Of course, the law of large numbers does not explain why a particular high-frequency strategy has positive mean return in the first place. In fact, it is impossible to explain in general why high-frequency strategies are often profitable, as there are as many such strategies as there are fund managers. Some of them are mean reverting, while others are trend following. Some are market-neutral pair traders, while others are long-only directional traders. In general, though, these strategies aim to exploit tiny inefficiencies in the market or to provide temporary liquidity needs for a small fee. Unlike betting on macroeconomic trends or company fundamentals where the market environment can experience upheavals during the lifetime of a trade, such inefficiencies and need for liquidity persist day to day, allowing consistent daily profits to be made. Furthermore, high-frequency strategies typically trade securities in modest sizes. Without large positions to unwind, risk management for high-frequency portfolios is fairly easy: “Deleveraging” can be done very quickly in the face of losses, and certainly one can stop trading and be completely in cash when the going gets truly rough. The worst that can happen as these strategies become more popular is a slow death as a result of gradually diminishing returns. Sudden drastic losses are not likely, nor are contagious losses across multiple accounts.

Though successful high-frequency strategies have such numerous merits, it is not easy to backtest such strategies when the average holding period decreases to minutes or even seconds. Transaction costs are of paramount importance in testing such strategies. Without incorporating transactions, the simplest strategies may seem to work at high frequencies. As a consequence, just having high-frequency data with last prices is not sufficient—data with bid, ask, and last quotes is needed to find out the profitability of executing on the bid versus the ask. Sometimes, we may even need historical order book information for backtesting. Quite often, the only true test for such strategies is to run it in real-time unless one has an extremely sophisticated simulator.

Backtesting is only a small part of the game in high-frequency trading. High-speed execution may account for a large part of the actual profits or losses. Professional high-frequency trading firms have been writing their strategies in C instead of other, more user-friendly languages, and locating their servers next to the exchange or a major Internet backbone to reduce the microsecond delays. So even though the Sharpe ratio is appealing and the returns astronomical, truly high-frequency trading is not by any means easy for an independent trader to achieve in the beginning. But there is no reason not to work toward this goal gradually as expertise and resources accrue.

### IS IT BETTER TO HAVE A HIGH-LEVERAGE VERSUS A HIGH-BETA PORTFOLIO?

In Chapter 6, I discussed the optimal leverage to apply to a portfolio based on the Kelly formula. In the section on factor models earlier in this chapter, I discussed the Fama-French Three-Factor model, which suggests that return of a portfolio (or a stock) is proportional to its beta (if we hold the market capitalization and book value of its stocks fixed). In other words, you can increase return on a portfolio by either increasing its leverage or increasing its beta (by selecting high-beta stocks.) Both ways seem commonsensical. In fact, it is clear that given a low-beta portfolio and a high-beta portfolio, it is easy to apply a higher leverage on the low-beta portfolio so as to increase its beta to match that of the high-beta portfolio. And assuming that the stocks of two portfolios have the same average market capitalizations and book values, the average returns of the two will also be the same (ignoring specific returns, which will decrease in importance as long as we increase the number of stocks in the portfolios), according to the Fama-French model. So should we be indifferent to which portfolio to own?

The answer is no. Recall in Chapter 6 that the long-term compounded growth rate of a portfolio, if we use the Kelly leverage, is proportional to the Sharpe ratio squared, and not to the average return. So if the two hypothetical portfolios have the same average return, then we would prefer the one that has the smaller risk or standard deviation. Empirical studies have found that a portfolio that consists of low-beta stocks generally has lower risk and thus a higher Sharpe ratio.

For example, in a paper titled “Risk Parity Portfolios” (not publicly distributed), Dr. Edward Qian at PanAgora Asset Management argued that a typical 60–40 asset allocation between stocks and bonds is not optimal because it is overweighted with risky assets (stocks in this case). Instead, to achieve a higher Sharpe ratio while maintaining the same risk level as the 60–40 portfolio, Dr. Qian recommended a 23–77 allocation while leveraging the entire portfolio by 1.8.

Somehow, the market is chronically underpricing high-beta stocks. Hence, given a choice between a portfolio of high-beta stocks and a portfolio of low-beta stocks, we should prefer the low-beta ones, which we can then leverage up to achieve the maximum compounded growth rate.

There is one usual caveat, however. All this is based on the Gaussian assumption of return distributions. (See discussions in Chapter 6 on this issue.) Since the actual returns distributions have fat tails, one should be quite wary of using too much leverage on normally low-beta stocks.

### SUMMARY

This book has been largely about a particular type of quantitative trading called statistical arbitrage in the investment industry. Despite this fancy name, statistical arbitrage is actually far simpler than trading derivatives (e.g., options) or fixed-income instruments, both conceptually and mathematically. I have described a large part of the statistical arbitrageur's standard arsenal: mean reversion and momentum, regime switching, stationarity and cointegration, arbitrage pricing theory or factor model, seasonal trading models, and, finally, high-frequency trading.

Some of the important points to note can be summarized here:

Mean-reverting regimes are more prevalent than trending regimes.   
There are some tricky data issues involved with backtesting mean-reversion strategies: Outlier quotes and survivorship bias are among them.   
Trending regimes are usually triggered by the diffusion of new information, the execution of a large institutional order, or “herding” behavior.   
Competition between traders tends to reduce the number of mean-reverting trading opportunities.   
Competition between traders tends to reduce the optimal holding period of a momentum trade.   
Trading parameters for each day or even each trade can be optimized using a machinelearning-based method we called CPO.   
A stationary price series is ideal for a mean-reversion trade.   
Two or more nonstationary price series can be combined to form a stationary one if they are “cointegrating.”   
Cointegration and correlation are different things: Cointegration is about the long-term behavior of the prices of two or more stocks, while correlation is about the short-term behavior of their returns.   
Factor models, or arbitrage pricing theory, are commonly used for modeling how fundamental factors affect stock returns linearly.   
One of the most well-known factor models is the Fama-French Three-Factor model, which postulates that stock returns are proportional to their beta and book-to-price ratio, and negatively to their market capitalizations.   
Factor models typically have a relatively long holding period and long drawdowns due to regime switches.   
Exit signals should be created differently for mean-reversion versus momentum strategies.   
Estimation of the optimal holding period of a mean-reverting strategy can be quite robust, due to the Ornstein-Uhlenbeck formula.   
Estimation of the optimal holding period of a momentum strategy can be error prone due to the small number of signals.   
Stop loss can be suitable for momentum strategies but not reversal strategies.   
Seasonal trading strategies for stocks (i.e., calendar effect) have become unprofitable in recent years.   
Seasonal trading strategies for commodity futures continue to be profitable.   
High-frequency trading strategies rely on the “law of large numbers” for their high Sharpe ratios.   
High-frequency trading strategies typically generate the highest long-term compounded growth due to their high Sharpe ratios.   
High-frequency trading strategies are very difficult to backtest and very technology-reliant for their execution.   
Holding a highly leveraged portfolio of low-beta stocks should generate higher long-term compounded growth than holding an unleveraged portfolio of high-beta stocks.

Most statistical arbitrage trading strategies are some combination of these effects or models: Whether they are profitable or not is more of an issue of where and when to apply them than whether they are theoretically correct.

## REFERENCES

Alexander, Carol. 2001. Market Models: A Guide to Financial Data Analysis. West Sussex: John Wiley & Sons Ltd.   
Chan, Ernest. 2013. Algorithmic Trading: Winning Strategies and Their Rationale. Wiley.   
Economist . 2007a. “This Year's Model.” December 13. www.economist.com/finance/displaystory.cfm?story_id=10286619.   
Fama, Eugene, and Kenneth French. 1992. “The Cross-Section of Expected Stock Returns.” Journal of Finance XLVII (2): 427–465.   
Fielden, Sandy. 2006. “Seasonal Surprises.” Energy Risk, September. https://www.risk.net/infrastructure/1523742/seasonal-surprises.   
Grinold, Richard, and Ronald Kahn. 1999. Active Portfolio Management. New York: McGraw-Hill.   
Heston, Steven, and Ronnie Sadka. 2007. “Seasonality in the Cross-Section of Expected Stock Returns.” AFA 2006 Boston Meetings Paper, July. lcb1.uoregon.edu/rcg/seminars/seasonal072604.pdf.   
Khandani, Amir, and Andrew Lo. 2007. “What Happened to the Quants in August 2007?” Preprint. web.mit.edu/alo/www/Papers/august07.pdf.   
Kochkodin, Brandon. 2021. “How WallStreetBets Pushed GameStop Shares to the Moon.” www.bloomberg.com/news/articles/2021-01-25/how-wallstreetbets-pushed-gamestopshares-to-the-moon?sref=MqSE4VuP.   
Lewis, Michael. 2014. Flash Boys. W.W. Norton.   
Phillips, Daniel. 2020. “Investment Strategy Commentary: Value Stocks: Trapped or Spring-Loaded?” Northern Trust, September 24. https://www.northerntrust.com/canada/insightsresearch/2020/investment-management/value-stocks.   
Singal, Vijay. 2006. Beyond the Random Walk. Oxford University Press, USA.   
Schiller, Robert. 2008. “Economic View; How a Bubble Stayed under the Radar.” New York Times, March 2. www.nytimes.com/2008/03/02/business/02view.html? ex=1362286800&en=da9e48989b6f937a&ei=5124&partner=permalink&exprod=permalink.   
Toepke, Jerry. 2004. “Fill 'Er Up! Benefit from Seasonal Price Patterns in Energy Futures.” Stocks, Futures and Options Magazine (3) (March 3). www.sfomag.com/issuedetail.asp? MonthNameID $=$ March&yearID=2004.   
Uhlenbeck, George, and Leonard Ornstein. 1930. “On the Theory of Brownian Motion.” Physical Review 36: 823–841.