# Interday Momentum Strategies

There are four main causes of momentum:

- 1. For futures, the persistence of roll returns, especially of their signs.
- 2. The slow diffusion, analysis, and acceptance of new information.
- 3. The forced sales or purchases of assets of various type of funds.
- 4. Market manipulation by high-frequency traders.

We will be discussing trading strategies that take advantage of each cause of momentum in this and the next chapter. In particular, roll returns of futures, which featured prominently in the last chapter, will again take center stage. Myriad futures strategies can be constructed out of the persistence of the sign of roll returns.

Researchers sometimes classify momentum in asset prices into two types: time series momentum and cross-sectional momentum, just as we classified mean reversion into two corresponding types in Chapter 2 (Moskowitz, Yao, and Pedersen, 2010). Time series momentum is very simple and intuitive: past returns of a price series are positively correlated with future returns. Cross-sectional momentum refers to the relative performance of a price series in relation to other price series: a price series with returns that outperformed other price series will likely keep doing so in the future and

vice versa. We will examine examples of both types in momentum in futures and stocks.

The strategies I describe in this chapter tend to hold positions for multiple days, which is why I call them "interday" momentum strategies. I will consider the intraday, higher-frequency momentum strategies in the next chapter. The reason for this distinction is that many interday momentum strategies suff er from a recently discovered weakness, while intraday momentum strategies are less aff ected by it. I will highlight this weakness in this chapter, and also discuss the very diff erent properties of momentum strategies versus their mean-reverting counterparts, as well as their pros and cons.

## ■ **Tests for Time Series Momentum**

Before we delve into the diff erent causes of momentum, we should fi rst see how we can measure momentum, or more specifi cally, time series momentum. Time series momentum of a price series means that past returns are positively correlated with future returns. It follows that we can just calculate the correlation coeffi cient of the returns together with its *p*-value (which represents the probability for the null hypothesis of no correlation). One feature of computing the correlation coeffi cient is that we have to pick a specifi c time lag for the returns. Sometimes, the most positive correlations are between returns of diff erent lags. For example, 1-day returns might show negative correlations, while the correlation between past 20-day return with the future 40-day return might be very positive. We should fi nd the optimal pair of past and future periods that gives the highest positive correlation and use that as our look-back and holding period for our momentum strategy.

Alternatively, we can also test for the correlations between the signs of past and future returns. This is appropriate when all we want to know is that an up move will be followed by another up move, and we don't care whether the magnitudes of the moves are similar.

If we are interested instead in fi nding out whether there is long-term trending behavior in the time series without regard to specifi c time frames, we can calculate the Hurst exponent together with the Variance Ratio test to rule out the null hypothesis of random walk. These tests were described in Chapter 2 for the detection of mean reversion, but they can just as well be used as momentum tests.

![](_page_2_Figure_2.jpeg)

**FIGURE 6.1** Nonoverlapping Periods for Correlation Calculations

 I will illustrate the use of these tests below using the two-year Treasury note future TU trading on the Chicago Mercantile Exchange (CME) as an example. The correlation coeffi cient and its *p*-value can be computed using the MATLAB function *corrcoef*, while the Hurst exponent and Variance Ratio test are, as before, given by *genhurst* and *vratiotest*.

In computing the correlations of pairs of returns resulting from diff erent look-back and holding periods, we must take care not to use overlapping data. If look-back is greater than the holding period, we have to shift forward by the holding period to generate a new returns pair. If the holding period is greater than the look-back, we have to shift forward by the lookback period. This is illustrated in Figure 6.1.

The top two bars in Figure 6.1 are for the case where look-back is greater than the holding period. The top bar represents the data set that forms the fi rst returns pair, and the second bar from the top represents the data set that forms the second independent returns pair. The bottom two bars are for the case where the look-back is smaller than the holding. The code is listed below (and available for download as *TU\_mom.m*).

#### **Finding Correlations between Returns of Different Time Frames**

**BOX 6.1**

```
% Correlation tests
for lookback=[1 5 10 25 60 120 250]
 for holddays=[1 5 10 25 60 120 250]
 ret_lag=(cl-backshift(lookback, cl)) ...
 ./backshift(lookback, cl);
 ret_fut=(fwdshift(holddays, cl)-cl)./cl;
 badDates=any([isnan(ret_lag) isnan(ret_fut)], 2);
```

(Continued )

```
ret_lag(badDates) = [];
ret_fut(badDates) = [];

if (lookback >= holddays)
        indepSet = [1:lookback:length(ret_lag)];
else
        indepSet = [1:holddays:length(ret_lag)];
end

ret_lag=ret_lag(indepSet);
ret_fut=ret_fut(indepSet);

[cc, pval] = corrcoef(ret_lag, ret_fut);
fprintf(1, 'lookback=%3i holddays=%3i cc=%7.4f ...
        pval=%6.4f\n', lookback, holddays, cc(1, 2), ...
        pval(1, 2));
end
end
```

If we shift the data forward by one day, we will get a slightly different set of returns for computing our correlations. For simplicity, I have only tested correlation of one among many possible sets of returns, but because of the large overlap of data between two different sets of returns, the results are unlikely to be greatly different. Some of the more significant results are tabulated in Table 6.1.

We see that there is a compromise between the correlation coefficient and the p-value. The following (look-back, holding days) pairs offer some of the best compromises: (60, 10), (60, 25), (250, 10), (250, 25), (250, 60), (250, 120). Of course, from a trading point of view, we prefer as short a holding period as possible as those tend to generate the best Sharpe ratios.

I have also tested the correlations between the signs of past and future returns instead, and the results are not very different from Table 6.1. I found the best pair candidates in that case are (60, 10), (250, 10), and (250, 25).

In contrast, we found that the Hurst exponent is 0.44, while the Variance Ratio test failed to reject the hypothesis that this is a random walk.

How are these two conflicting results reconciled? As we show in the correlation tests, this time series (as with many other financial time series) exhibits momentum and mean reversion at different time frames. The Variance Ratio test is unable to test the specific time frames where the correlations might be stronger than average.

| TABLE 6.1 | Correlations between TU Returns of Different Time Frames |         |         |  |  |  |
|-----------|----------------------------------------------------------|---------|---------|--|--|--|
| Look-back | Holding days<br>Correlation coefficient                  |         | p-value |  |  |  |
| 25        | 1                                                        | –0.0140 | 0.5353  |  |  |  |
| 25        | 5                                                        | 0.0319  | 0.5276  |  |  |  |
| 25        | 10                                                       | 0.1219  | 0.0880  |  |  |  |
| 25        | 25                                                       | 0.1955  | 0.0863  |  |  |  |
| 25        | 60                                                       | 0.2333  | 0.0411  |  |  |  |
| 25        | 120                                                      | 0.1482  | 0.2045  |  |  |  |
| 25        | 250                                                      | 0.2620  | 0.0297  |  |  |  |
| 60        | 1                                                        | 0.0313  | 0.1686  |  |  |  |
| 60        | 5                                                        | 0.0799  | 0.1168  |  |  |  |
| 60        | 10                                                       | 0.1718  | 0.0169  |  |  |  |
| 60        | 25                                                       | 0.2592  | 0.0228  |  |  |  |
| 60        | 60                                                       | 0.2162  | 0.2346  |  |  |  |
| 60        | 120                                                      | –0.0331 | 0.8598  |  |  |  |
| 60        | 250                                                      | 0.3137  | 0.0974  |  |  |  |
| 120       | 1                                                        | 0.0222  | 0.3355  |  |  |  |
| 120       | 5                                                        | 0.0565  | 0.2750  |  |  |  |
| 120       | 10                                                       | 0.0955  | 0.1934  |  |  |  |
| 120       | 25                                                       | 0.1456  | 0.2126  |  |  |  |
| 120       | 60                                                       | –0.0192 | 0.9182  |  |  |  |
| 120       | 120                                                      | 0.2081  | 0.4567  |  |  |  |
| 120       | 250                                                      | 0.4072  | 0.1484  |  |  |  |
| 250       | 1                                                        | 0.0411  | 0.0857  |  |  |  |
| 250       | 5                                                        | 0.1068  | 0.0462  |  |  |  |
| 250       | 10                                                       | 0.1784  | 0.0185  |  |  |  |
| 250       | 25                                                       | 0.2719  | 0.0238  |  |  |  |
| 250       | 60                                                       | 0.4245  | 0.0217  |  |  |  |
| 250       | 120                                                      | 0.5112  | 0.0617  |  |  |  |
| 250       | 250                                                      | 0.4873  | 0.3269  |  |  |  |

## ■ **Time Series Strategies**

For a certain future, if we fi nd that the correlation coeffi cient between a past return of a certain look-back and a future return of a certain holding period is high, and the *p*-value is small, we can proceed to see if a profi table momentum strategy can be found using this set of optimal time periods. Since Table 6.1 shows us that for TU, the 250-25-days pairs of returns have a correlation coeffi cient of 0.27 with a *p*-value of 0.02, we will pick this look-back and holding period. We take our cue for a simple time series momentum strategy from a paper by Moskowitz, Yao, and Pedersen: simply buy (sell) the future if it has a positive (negative) 12-month return, and hold the position for 1 month (Moskowitz, Yao, and Pedersen, 2012). We will modify one detail of the original strategy: Instead of making a trading decision every month, we will make it every day, each day investing only one twenty-fi fth of the total capital.

#### **Example 6.1: TU Momentum Strategy**

This code assumes the closing prices are contained in a *T* × 1 array cl. This code is contained in *TU\_mom.m*.

```
lookback=250;
holddays=25;
longs=cl > backshift(lookback, cl) ;
shorts=cl < backshift(lookback, cl) ;
pos=zeros(length(cl), 1);
for h=0:holddays-1
 long_lag=backshift(h, longs);
 long_lag(isnan(long_lag))=false;
 long_lag=logical(long_lag);
 short_lag=backshift(h, shorts);
 short_lag(isnan(short_lag))=false;
 short_lag=logical(short_lag);
 pos(long_lag)=pos(long_lag)+1;
 pos(short_lag)=pos(short_lag)-1;
end
ret=(backshift(1, pos).*(cl-lag(cl))./lag(cl))/holddays;
```

From June 1, 2004, to May 11, 2012, the Sharpe ratio is a respectable 1. The annual percentage rate (APR) of 1.7 percent may seem low, but our return is calculated based on the notional value of the contract, which is

![](_page_6_Figure_2.jpeg)

**FIGURE 6.2** Equity Curve of TU Momentum Strategy

about \$200,000. Margin requirement for this contract is only about \$400. So you can certainly employ a reasonable amount of leverage to boost return, though one must also contend with the maximum drawdown of 2.5 percent. The equity curve also looks quite attractive (see Figure 6.2).

This simple strategy can be applied to all kinds of futures contracts, with diff erent optimal look-back periods and the holding days. The results for three futures we considered are listed in Table 6.2.

Why do many futures returns exhibit serial correlations? And why do these serial correlations occur only at a fairly long time scale? The explanation lies in the roll return component of the total return of futures we discussed in Chapter 5. Typically, the sign of roll returns does not vary very often. In other words, the futures stay in contango or backwardation over long periods of time. The spot returns, however, can vary very rapidly in both sign and magnitude. So if we hold a future over a long period of time, and if the average roll returns dominate the average total returns, we will fi nd serial correlation of total returns. This explanation certainly makes sense for BR, HG, and TU, since from Table 5.1 we can see that they all have

| TABLE 6.2 | Time Series Momentum Strategies for Various Futures |              |       |              |              |  |
|-----------|-----------------------------------------------------|--------------|-------|--------------|--------------|--|
| Symbol    | Look-back                                           | Holding days | APR   | Sharpe ratio | Max drawdown |  |
| BR (CME)  | 100                                                 | 10           | 17.7% | 1.09         | –14.8%       |  |
| HG (CME)  | 40                                                  | 40           | 18.0% | 1.05         | –24.0%       |  |
| TU (CBOT) | 250                                                 | 25           | 1.7%  | 1.04         | –2.5%        |  |

roll returns that are bigger in magnitude than their spot returns. (I haven't found the reason why it doesn't work for C, despite its having the largest roll return magnitude compared to its average spot return, but maybe you can!)

If we accept the explanation that the time series momentum of futures is due to the persistence of the signs of the roll returns, then we can devise a cleaner and potentially better momentum signal than the lagged total return. We can use the lagged roll return as a signal instead, and go long when this return is higher than some threshold, go short when this return is lower than the negative of that threshold, and exit any existing position otherwise. Applying this revised strategy on TU with a threshold of an annualized roll return of 3 percent yields a higher APR of 2.5 percent and Sharpe ratio of 2.1 from January 2, 2009, to August 13, 2012, with a reduced maximum drawdown of 1.1 percent.

There are many other possible entry signals besides the simple "sign of return" indicator. For example, we can buy when the price reaches a new *N*-day high, when the price exceeds the *N*-day moving average or exponential moving average, when the price exceeds the upper Bollinger band, or when the number of up days exceeds the number of down days in a moving period.

There is also a classic momentum strategy called the Alexander Filter, which tells us to buy when the daily return moves up at least *x* percent, and then sell and go short if the price moves down at least *x* percent from a subsequent high (Fama and Blume, 1966).

Sometimes, the combination of mean-reverting and momentum rules may work better than each strategy by itself. One example strategy on CL is this: buy at the market close if the price is lower than that of 30 days ago and is higher than that of 40 days ago; vice versa for shorts. If neither the buy nor the sell condition is satisfi ed, fl atten any existing position. The APR is 12 percent, with a Sharpe ratio of 1.1. Adding a mean-reverting fi lter to the momentum strategy in Example 6.1 will add IBX (MEFF), KT (NYMEX), SXF (DE), US (CBOT), CD (CME), NG (NYMEX), and W (CME) to Table 6.2, and it will also improve the returns and Sharpe ratios of the existing contracts in that table.

In fact, if you don't want to construct your own time series momentum strategy, there is a ready-made index that is composed of 24 futures: the Standard & Poor's (S&P) Diversifi ed Trends Indicator (DTI). The essential strategy behind this index is that we will long a future if it is above its exponential moving average, and short it if it is below, with monthly rebalancing. (For details, you can visit [www.standardandpoors.com.\) Th](http://www.standardandpoors.com)ere is a mutual fund (RYMFX) as well as an exchange-traded fund (WDTI) that tracks this index. Michael Dever computed that the Sharpe ratio of this index was 1.3 with a maximum drawdown of –16.6 percent from January 1988 to December 2010 (Dever, 2011). (This may be compared to the S&P 500 index SPX, which has a Sharpe ratio of 0.61 and a maximum drawdown of –50.96 percent over the same period, according to the author.) However, in common with many other momentum strategies, its performance is poor since the 2008 fi nancial crisis, a point that will be taken up later.

Since there aren't many trades in the relatively limited amount of test data that we used due to the substantial holding periods, there is a risk of data-snooping bias in these results. The real test for the strategy is, as always, in true out-of-sample testing.

#### ■ **Extracting Roll Returns through Future versus ETF Arbitrage**

If futures' total returns = spot returns + roll returns, then an obvious way to extract roll return is buy the underlying asset and short the futures, if the roll return is negative (i.e., under contango); and vice versa if the roll return is positive (i.e., under backwardation). This will work as long as the sign of the roll return does not change quickly, as it usually doesn't. This arbitrage strategy is also likely to result in a shorter holding period and a lower risk than the buy-and-hold strategy discussed in the previous section, since in that strategy we needed to hold the future for a long time before the noisy spot return can be averaged out.

However, the logistics of buying and especially shorting the underlying asset is not simple, unless an exchange-traded fund (ETF) exists that holds the asset. Such ETFs can be found for many precious metals. For example, GLD actually owns physical gold, and thus tracks the gold spot price very closely. Gold futures have a negative roll return of –4.9 percent annualized from December 1982 to May 2004. A backtest shows that holding a long position in GLD and a short position in GC yields an annualized return of 1.9 percent and a maximum drawdown of 0.8 percent from August 3, 2007, to August 2, 2010. This might seem attractive, given that one can apply a leverage of 5 or 6 and get a decent return with reasonable risk, but in reality it is not. Remember that in contrast to owning futures, owning GLD actually incurs fi nancing cost, which is not very diff erent from 1.9 percent over the backtest period! So the excess return of this strategy is close to zero.

(The astute reader might notice another caveat of our quick backtest of GC versus GLD: the settlement or closing prices of GC are recorded at 1:30 p.m. ET, while those of GLD are recorded at 4:00 p.m. ET. This asynchronicity is a pitfall that I mentioned in Chapter 1. However, it doesn't matter to us in this case because the trading signals are generated based on GC closing prices alone.)

If we try to look outside of precious metals ETFs to fi nd such arbitrage opportunities, we will be stumped. There are no ETFs that hold other physical commodities as opposed to commodities futures, due to the substantial storage costs of those commodities. However, there is a less exact form of arbitrage that allows us to extract the roll returns. ETFs containing commodities producing companies often cointegrate with the spot price of those commodities, since these commodities form a substantial part of their assets. So we can use these ETFs as proxy for the spot price and use them to extract the roll returns of the corresponding futures.

One good example is the arbitrage between the energy sector ETF XLE and the WTI crude oil futures CL. Since XLE and CL have diff erent closing times, it is easier to study the arbitrage between XLE and the ETF USO instead, which contains nothing but front month contracts of CL. The strategy is simple:

- Short USO and long XLE whenever CL is in contango.
- Long USO and short XLE whenever CL is in backwardation.

The APR is a very respectable 16 percent from April 26, 2006, to April 9, 2012, with a Sharpe ratio of about 1. I have plotted the cumulative returns curve in Figure 6.3.

![](_page_9_Figure_7.jpeg)

**FIGURE 6.3** Cumulative Returns of XLE-USO Arbitrage

What about a future whose underlying is not a traded commodity? VX is an example of such a future: It is very expensive to maintain a basket of options that replicate the underlying VIX index, and no ETF sponsors have been foolish enough to do that. But, again, we do not need to fi nd an instrument that tracks the spot price exactly—we just need to fi nd one that has a high correlation (or anti-correlation) with the spot return. In the case of VIX, the familiar ETF SPY fi ts the bill. Because the S&P E-mini future ES has insignifi cant roll return (about 1 percent annualized), it has almost the same returns as the underlying asset. Because it is certainly easier to trade futures than an ETF, we will investigate the performance of our earlier arbitrage strategy using ES instead.

# **Volatility Futures versus Equity Index Futures: Redux**

VX is a natural choice if we want to extract roll returns: its roll returns can be as low as –50 percent annualized. At the same time, it is highly anticorrelated with ES, with a correlation coeffi cient of daily returns reaching –75 percent. In Chapter 5, we used the cointegration between VX and ES to develop a profi table mean-reverting strategy. Here, we will make use of the large roll return magnitude of VX, the small roll return magnitude of ES, and the anticorrelation of VX and ES to develop a momentum strategy. This strategy was proposed by Simon and Campasano (2012):

- 1. If the price of the front contract of VX is higher than that of VIX by 0.1 point (contango) times the number of trading days untill settlement, short 0.3906 front contracts of VX and short 1 front contract of ES. Hold for one day.
- 2. If the price of the front contract of VX is lower than that of VIX by 0.1 point (backwardation) times the number of trading days untill settlement, buy 0.3906 front contracts of VX and buy 1 front contract of ES. Hold for one day.

Recall that if the front contract price is higher than the spot price, the roll return is negative (see Figure 5.3). So the diff erence in price between VIX and VX divided by the time to maturity is the roll return, and we buy VX if the roll return is positive. Why didn't we use the procedure in Example 5.3 where we use the slope of the futures log forward curve to compute the roll return here? That is because Equation 5.7 doesn't work for VX, and therefore the VX forward prices do not fall on a straight line, as explained in Chapter 5.

![](_page_11_Figure_2.jpeg)

**FIGURE 6.4** Cumulative Returns of VX-ES Roll Returns Strategy

Notice that the hedge ratio of this strategy is slightly diff erent from that reported by Simon and Campasano: It is based on the regression fi t between the VX versus ES prices in Equation 5.11, not between their returns as in the original paper. The settlement is the day after the contracts expire. The APR for July 29, 2010, to May 7, 2012 (this period was not used for hedge ratio determination) is 6.9 percent, with a Sharpe ratio of 1. The cumulative return chart is displayed in Figure 6.4. You can fi nd the MATLAB code for this strategy in *VX\_ES\_rollreturn.m* on my website.

# ■ **Cross-Sectional Strategies**

There is a third way to extract the often large roll returns in futures besides buying and holding or arbitraging against the underlying asset (or against an instrument correlated with the underlying asset). This third way is a crosssectional strategy: We can just buy a portfolio of futures in backwardation, and simultaneously short a portfolio of futures in contango. The hope is that the returns of the spot prices cancel each other out (a not unreasonable expectation if we believe commodities' spot prices are positively correlated with economic growth or some other macroeconomic indices), and we are left with the favorable roll returns. Daniel and Moskowitz described just such a simple "cross-sectional" momentum strategy that is almost a mirror image of the linear long-short mean-reverting stock model proposed by Khandani and Lo described in Chapter 3, albeit one with a much longer look-back and holding period (Daniel and Moskowitz, 2011).

![](_page_12_Figure_2.jpeg)

**FIGURE 6.5** Cumulative Returns of Cross-Sectional Futures Momentum Strategy

A simplifi ed version of the strategy is to rank the 12-month return (or 252 trading days in our program below) of a group of 52 physical commodities *every day*, and buy and hold the future with the highest return for 1 month (or 25 trading days) while short and hold the future with the lowest return for the same period. I tested this strategy from June 1, 2005, to December 31, 2007, and the APR is an excellent 18 percent with a Sharpe ratio of 1.37. The cumulative returns are plotted in Figure 6.5. Unfortunately, this model performed very negatively from January 2, 2008, to December 31, 2009, with an APR of –33 percent, though its performance recovered afterwards. The fi nancial crisis of 2008–2009 ruined this momentum strategy, just like it did many others, including the S&P DTI indicator mentioned before.

Daniel and Moskowitz have also found that this same strategy worked for the universe of world stock indices, currencies, international stocks, and U.S. stocks—in other words, practically everything under the sun. Obviously, cross-sectional momentum in currencies and stocks can no longer be explained by the persistence of the sign of roll returns. We might attribute that to the serial correlation in world economic or interest rate growth in the currency case, and the slow diff usion, analysis, and acceptance of new information in the stock case.

Applying this strategy to U.S. stocks, we can buy and hold stocks within the top decile of 12-month lagged returns for a month, and vice versa for the bottom decile. I illustrate the strategy in Example 6.2.

#### **Example 6.2: Cross-Sectional Momentum Strategy for Stocks**

This code assumes the close prices are contained in *T* × *N* array *cl*, where *T* is the number of trading days, and *N* is the number of the stocks in S&P 500. It makes use of utilities functions *smartsum* and *backshift,* available from [http://epchan.com/book2.](http://epchan.com/book2) The code itself can be downloaded as *kentdaniel.m*.

```
lookback=252;
holddays=25;
topN=50;
ret=(cl- backshift(lookback,cl))./backshift(lookback,cl); 
 % daily returns
longs=false(size(ret));
shorts=false(size(ret));
positions=zeros(size(ret));
for t=lookback+1:length(tday)
 [foo idx]=sort(ret(t, :), 'ascend');
 nodata=find(isnan(ret(t, :)));
 idx=setdiff(idx, nodata, 'stable');
 longs(t, idx(end-topN+1:end))=true;
 shorts(t, idx(1:topN))=true;
end
for h=0:holddays-1
 long_lag=backshift(h, longs);
 long_lag(isnan(long_lag))=false;
 long_lag=logical(long_lag);
 short_lag=backshift(h, shorts);
 short_lag(isnan(short_lag))=false;
 short_lag=logical(short_lag);
 positions(long_lag)=positions(long_lag)+1;
 positions(short_lag)=positions(short_lag)-1;
end
dailyret=smartsum(backshift(1, positions).*(cl-lag(cl)) ...
 ./ lag(cl), 2)/(2*topN)/holddays;
dailyret(isnan(dailyret))=0;
```

#### **Example 6.2 (***Continued***)**

 The APR from May 15, 2007, to December 31, 2007, is 37 percent with a Sharpe ratio of 4.1. The cumulative returns are shown in Figure 6.6. (Daniel and Moskowitz found an annualized average return of 16.7 percent and a Sharpe ratio of 0.83 from 1947 to 2007.) However, the APR from January 2, 2008, to December 31, 2009, is a miserable –30 percent. The fi nancial crisis of 2008–2009 also ruined this momentum strategy. The return after 2009 did stabilize, though it hasn't returned to its former high level yet.

Just as in the case of the cross-sectional mean reversion strategy discussed in Chapter 4, instead of ranking stocks by their lagged returns, we can rank them by many other variables, or "factors," as they are usually called. While we wrote *total return* = *spot return* + *roll return* for futures, we can write *total return* = *market return* + *factor returns* for stocks. A cross-sectional portfolio of stocks, whether mean reverting or momentum based, will eliminate the market return component, and its returns will be driven solely by the factors. These factors may be fundamental, such as earnings growth or bookto-price ratio, or some linear combination thereof. Or they may be statistical factors that are derived from, for example, Principal Component Analysis (PCA) as described in *Quantitative Trading* (Chan, 2009). All these factors

![](_page_14_Figure_5.jpeg)

**FIGURE 6.6** Cumulative Returns of Cross-Sectional Stock Momentum Strategy

with the possible exception of PCA tend to change slowly, so using them to rank stocks will result in as long holding periods as the cross-sectional models I discussed in this section.

While we are on the subject of factors, it bears mentioning that a factor model can be applied to a cross-sectional portfolio of futures as well. In this case, we can fi nd macroeconomic factors such as gross domestic product (GDP) growth or infl ation rate and correlate them with the returns of each futures instrument, or we can again employ PCA.

In recent years, with the advance of computer natural language processing and understanding capability, there is one other factor that has come into use. This is the so-called news sentiment score, our next topic.

# **News Sentiment as a Fundamental Factor**

With the advent of machine-readable, or "elementized," newsfeeds, it is now possible to programmatically capture all the news items on a company, not just those that fi t neatly into one of the narrow categories such as earnings announcements or merger and acquisition (M&A) activities. Furthermore, natural language processing algorithms are now advanced enough to analyze the textual information contained in these news items, and assign a "sentiment score" to each news article that is indicative of its price impact on a stock, and an aggregation of these sentiment scores from multiple news articles from a certain period was found to be predictive of its future return. For example, Hafez and Xie, using RavenPack's Sentiment Index, found that buying a portfolio of stocks with positive sentiment change and shorting one with negative sentiment change results in an APR from 52 percent to 156 percent and Sharpe ratios from 3.9 to 5.3 before transaction costs, depending on how many stocks are included in the portfolios (Hafez and Xie, 2012). The success of these cross-sectional strategies also demonstrates very neatly that the slow diff usion of news is the cause of stock momentum.

There are other vendors besides RavenPack that provide news sentiments on stocks. Examples include Recorded Future, thestocksonar.com, and Thomson Reuters News Analytics. They diff er on the scope of their news coverage and also on the algorithm they use to generate the sentiment score. If you believe your own sentiment algorithm is better than theirs, you can subscribe directly to an elementized news feed instead and apply your algorithm to it. I mentioned before that Newsware off ers a low-cost version of this type of news feeds, but off erings with lower latency and better coverage are provided by Bloomberg Event-Driven Trading, Dow Jones Elementized News Feeds, and Thomson Reuters Machine Readable News.

Beyond such very reasonable use of news sentiment as a factor for cross-sectional momentum trading, there has also been research that suggested the general "mood" of society as revealed in the content of Twitter feeds is predictive of the market index itself (Bollen, Mao, and Zeng, 2010). In fact, a multimillion-dollar hedge fund was launched to implement this outland-ish idea (Bryant, 2010), though the validity of the research itself was under attack (*Buy the Hype*, 2012).

# **Mutual Funds Asset Fire Sale and Forced Purchases**

Researchers Coval and Stafford (2007) found that mutual funds experiencing large redemptions are likely to reduce or eliminate their existing stock positions. This is no surprise since mutual funds are typically close to fully invested, with very little cash reserves. More interestingly, funds experiencing large capital inflows tend to increase their existing stock positions rather than using the additional capital to invest in other stocks, perhaps because new investment ideas do not come by easily. Stocks disproportionately held by poorly performing mutual funds facing redemptions therefore experience negative returns. Furthermore, this asset "fire sale" by poorly performing mutual funds is contagious. Since the fire sale depresses the stock prices, they suppress the performance of other funds holding those stocks, too, causing further redemptions at those funds. The same situation occurs in reverse for stocks disproportionately held by superbly performing mutual funds with large capital inflows. Hence, momentum in both directions for the commonly held stocks can be ignited.

(This ignition of price momentum due to order flow is actually a rather general phenomenon, and it happens at even the shortest time scale. We find more details on that in the context of high-frequency trading in Chapter 7.)

A factor can be constructed to measure the selling (buying) pressure on a stock based on the net percentage of funds holding them that experienced redemptions (inflows). More precisely,

$$= \frac{\sum_{j} (Buy(j,i,t) | flow(j,t) > 5\%) - \sum_{j} (Sell(j,i,t) | flow(j,t) < -5\%)}{\sum_{j} Own(j,i,t-1)}$$
(6.1)

where *PRESSURE*(*i*, *t*) is the factor for stock *i* at the end of quarter *t*, *Buy*( *j*, *i*, *t*) = 1 if fund *j* increased its holding in stock *i* during the quarter *t* and if the fund experienced infl ows greater than 5 percent of its net asset value (NAV) ("*fl ow*( *j*, *t*) > 5%"), and zero otherwise. *Sell*( *j*, *i*, *t*) is similarly defi ned for decreases in holdings, and ∑*<sup>j</sup> Own*( *j*, *i*, *t* − 1) is the total number of mutual funds holding stock *i* the beginning of quarter *t*.

Note that the *PRESSURE* variable does not take into account the size (NAV) of the fund, as *Buy* is a binary variable. One wonders whether weighing *Buy* by NAV will give better results.

Coval and Staff ord found that a market-neutral portfolio formed based on shorting stocks with highest selling pressure (bottom decile of *PRESSURE* ranking) and buying stocks with the highest (top decile of *PRESSURE* ranking) buying pressure generates annualized returns of about 17 percent before transaction costs. (Since data on stock holdings are available generally on a quarterly basis only, our portfolio is updated quarterly as well.)

Furthermore, capital fl ows into and out of mutual funds can be predicted with good accuracy based on their past performance and capital fl ows, a refl ection of the herdlike behavior of retail investors. Based on this prediction, we can also predict the future value of the pressure factor noted above. In other words, we can front-run the mutual funds in our selling (buying) of the stocks they currently own. This front-running strategy generates another 17 percent annualized return before transaction costs.

Finally, since these stocks experience such selling and buying pressures due to liquidity-driven reasons, and suff er suppression or elevation of their prices through no fault or merit on their own, their stock prices often mean-revert after the mutual fund selling or buying pressure is over. Indeed, buying stocks that experienced the most selling pressure in the *t* − 4 up to the *t* − 1 quarters, and vice versa, generates another 7 percent annualized returns.

Combining all three strategies (momentum, front running, and mean reverting) generates a total return of about 41 percent before transaction costs. However, the slippage component of the transaction costs is likely to be signifi cant because we may experience delays in getting mutual fund holdings information at the end of a quarter. In addition, the implementation of this strategy is not for the faint-of-heart: clean and accurate mutual holdings and returns data have to be purchased from the Center for Research in Security Prices (CRSP) at a cost of about \$10,000 per year of data.

Mutual funds are not the only type of funds that can induce momentum in stocks due to forced asset sales and purchases. In Chapter 7, we will discover that index funds and levered ETFs ignite similar momentum as well. In fact, forced asset sales and purchases by hedge funds can also lead to momentum in stocks, and that caused the August 2007 quant funds meltdown, as I explain in Chapter 8.

## ■ **Pros and Cons of Momentum Strategies**

Momentum strategies, especially interday momentum strategies, often have diametrically opposite reward and risk characteristics in comparison to mean reverting strategies. We will compare their pros and cons in this section.

Let's start with the cons. In my own trading experience, I have often found that it is harder to create profi table momentum strategies, and those that are profi table tend to have lower Sharpe ratios than mean-reversal strategies. There are two reasons for this.

First, as we have seen so far, many established momentum strategies have long look-back and holding periods. So clearly the number of independent trading signals is few and far in between. (We may rebalance a momentum portfolio every day, but that doesn't make the trading signals more independent.) Fewer trading signals naturally lead to lower Sharpe ratio. Example: The linear mean reversion model for S&P 500 stocks described in Chapter 4 relies on the short-term cross-sectional mean reversion properties of stocks, and the holding period is less than a day. It has a high Sharpe ratio of 4.7. For the same universe of stocks, the opposite cross-sectional momentum strategy described earlier in this chapter has a holding period of 25 days, and though it performed similarly well pre-2008, the performance collapsed during the fi nancial crisis years.

Secondly, research by Daniel and Moskowitz on "momentum crashes" indicates that momentum strategies for futures or stocks tend to perform miserably for several years after a fi nancial crisis (Daniel and Moskowitz, 2011). We can see that easily from a plot of the S&P DTI index (Figure 6.7). As of this writing, it has suff ered a drawdown of –25.9 percent since December 5, 2008. Similarly, cross-sectional momentum in stocks also vanished during the aftermath of the stock market crash in 2008–2009, and is replaced by strong mean reversion. We still don't know how long this mean reversion regime will last: After the stock market crash of 1929, a representative momentum strategy did not return to its high watermark for more than 30 years! The cause of this crash is mainly due to the strong rebound of short positions following a market crisis.

![](_page_19_Figure_1.jpeg)

**FIGURE 6.7** The S&P DTI Index

Third, and this relates mostly to the shorter-term news-driven momentum that we will talk about in the next chapter, the duration over which momentum remains in force gets progressively shorter as more traders catch on to it. For example, price momentum driven by earnings announcements used to last several days. Now it lasts barely until the market closes. This is quite understandable if we view price momentum as generated by the slow diff usion of information. As more traders learn about the information faster and earlier, the diff usion—and thus, momentum—also ends sooner. This of course creates a problem for the momentum trader, since we may have to constantly shorten our holding period, yet there is no predictable schedule for doing so.

Lest you think that we should just give up on momentum strategies, let's look at the list of pros for momentum strategies. Such lists usually start with the ease of risk management. To see why, we observe that there are two common types of exit strategies for momentum strategies: time-based and stop loss. All the momentum strategies I have discussed so far involve only timebased exits. We specify a holding period, and we exit a position when we reached that holding period. But we can also impose a stop loss as the exit condition, or maybe as an additional exit condition. Stop losses are perfectly consistent with momentum strategies. If momentum has changed direction, we should enter into the opposite position. Since the original position would have been losing, and now we have exited it, this new entry signal eff ectively served as a stop loss. In contrast, stop losses are not consistent with meanreverting strategies, because they contradict mean reversion strategies' entry signals. (This point will be taken up again in Chapter 8.) Because of either a time-based exit or a stop loss, the loss of a momentum position is always limited. In contrast, we can incur an enormous drawdown with just one position due to a mean-reverting strategy. (This is not to say that the cumulative loss of successive losing positions due to a momentum strategy won't bankrupt us!)

Not only do momentum strategies survive risks well, they can thrive in them (though we have seen how poorly they did in the *aftermath* of risky events). For mean-reverting strategies, their upside is limited by their natural profi t cap (set as the "mean" to which the prices revert), but their downside can be unlimited. For momentum strategies, their upside is unlimited (unless one arbitrarily imposes a profi t cap, which is ill-advised), while their downside is limited. The more often "black swan" events occur, the more likely that a momentum strategy will benefi t from them. The thicker the tails of the returns distribution curve, or the higher its kurtosis, the better that market is for momentum strategies. (Remember the simulation in Example 1.1? We simulated a returns series with the same kurtosis as the futures series for TU but with no serial autocorrelations. We found that it can still generate the same returns as our TU momentum strategy in 12 percent of the random realizations!)

Finally, as most futures and currencies exhibit momentum, momentum strategies allow us to truly diversify our risks across diff erent asset classes and countries. Adding momentum strategies to a portfolio of meanreverting strategies allows us to achieve higher Sharpe ratios and smaller drawdowns than either type of strategy alone.

#### **KEY POINTS**

- Time-series momentum refers to the positive correlation of a price series' past and future returns.
- Cross-sectional momentum refers to the positive correlation of a price series' past and future relative returns, in relation to that of other price series in a portfolio.
- Futures exhibit time series momentum mainly because of the persistence of the sign of roll returns.
- If you are able to fi nd an instrument (e.g., an ETF or another future) that cointegrates or correlates with the spot price or return of a commodity, you can extract the roll return of the commodity future by shorting that instrument during backwardation, or buying that instrument during contango.
- Portfolios of futures or stocks often exhibit cross-sectional momentum: a simple ranking algorithm based on returns would work.
- Profi table strategies on news sentiment momentum show that the slow diffusion of news is a cause for stock price momentum.
- The contagion of forced asset sales and purchases among mutual funds contributes to stock price momentum.
- Momentum models thrive on "black swan" events and the positive kurtosis of the returns distribution curve.