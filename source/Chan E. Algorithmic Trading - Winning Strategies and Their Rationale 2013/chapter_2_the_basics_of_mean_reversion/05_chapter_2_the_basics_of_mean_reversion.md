# The Basics of Mean Reversion

Whether we realize it or not, nature is filled with examples of mean reversion. Figure 2.1 shows the water level of the Nile from 622 AD to 1284 AD, clearly a mean-reverting time series. Mean reversion is equally prevalent in the social sciences. Daniel Kahneman cited a famous example: the "Sports Illustrated jinx," which is the claim that "an athlete whose picture appears on the cover of the magazine is doomed to perform poorly the following season" (Kahneman, 2011). The scientific reason is that an athlete's performance can be thought of as randomly distributed around a mean, so an exceptionally good performance one year (which puts the athlete on the cover of Sports Illustrated) is very likely to be followed by performances that are closer to the average.

Is mean reversion also prevalent in financial price series? If so, our lives as traders would be very simple and profitable! All we need to do is to buy low (when the price is below the mean), wait for reversion to the mean price, and then sell at this higher price, all day long. Alas, most price series are not mean reverting, but are geometric random walks. The *returns*, not the prices, are the ones that usually randomly distribute around a mean of zero. Unfortunately, we cannot trade on the mean reversion of returns. (One should not confuse mean reversion of returns with anti-serial-correlation of returns, which we can definitely trade on. But anti-serial-correlation of returns is the same as the mean reversion of prices.) Those few price series that are found to be mean reverting are called *stationary*, and in this chapter we will describe the statistical tests (ADF test and the Hurst exponent and Variance Ratio test) for stationarity. There are not too many prefabricated

![](_page_1_Figure_2.jpeg)

**FIGURE 2.1** Minimum Water Levels of the Nile River, 622–1284 ad

prices series that are stationary. By *prefabricated* I meant those price series that represent assets traded in the public exchanges or markets.

Fortunately, we can manufacture many more mean-reverting price series than there are traded assets because we can often combine two or more individual price series that are not mean reverting into a portfolio whose net market value (i.e., price) is mean reverting. Those price series that can be combined this way are called cointegrating, and we will describe the statistical tests (CADF test and Johansen test) for cointegration, too. Also, as a by-product of the Johansen test, we can determine the exact weightings of each asset in order to create a mean reverting portfolio. Because of this possibility of artifi cially creating stationary portfolios, there are numerous opportunities available for mean reversion traders.

As an illustration of how easy it is to profi t from mean-reverting price series, I will also describe a simple linear trading strategy, a strategy that is truly "parameterless."

One clarifi cation: The type of mean reversion we will look at in this chapter may be called *time series* mean reversion because the prices are supposed to be reverting to a mean determined by its own historical prices. The tests and trading strategies that I depict in this chapter are all tailored to time series mean reversion. There is another kind of mean reversion, called "crosssectional" mean reversion. Cross-sectional mean reversion means that the cumulative returns of the instruments in a basket will revert to the cumulative return of the basket. This also implies that the short-term *relative* returns of the instruments are serially anticorrelated. (Relative return of an instrument is the return of that instrument minus the return of the basket.) Since this phenomenon occurs most often for stock baskets, we will discuss how to take advantage of it in Chapter 4 when we discuss mean-reverting strategies for stocks and ETFs.

# ■ **Mean Reversion and Stationarity**

Mean reversion and stationarity are two equivalent ways of looking at the same type of price series, but these two ways give rise to two diff erent statistical tests for such series.

The mathematical description of a mean-reverting price series is that the change of the price series in the next period is proportional to the diff erence between the mean price and the current price. This gives rise to the ADF test, which tests whether we can reject the null hypothesis that the proportionality constant is zero.

However, the mathematical description of a stationary price series is that the variance of the log of the prices increases slower than that of a geometric random walk. That is, their variance is a sublinear function of time, rather than a linear function, as in the case of a geometric random walk. This sublinear function is usually approximated by $\tau^2 H$, where $\tau$ is the time separating two price measurements, and $H$ is the so-called Hurst exponent, which is less than 0.5 if the price series is indeed stationary (and equal to 0.5 if the price series is a geometric random walk). The Variance Ratio test can be used to see whether we can reject the null hypothesis that the Hurst exponent is actually 0.5.

Note that stationarity is somewhat of a misnomer: It doesn't mean that the prices are necessarily range bound, with a variance that is independent of time and thus a Hurst exponent of zero. It merely means that the variance increases slower than normal diff usion.

A clear mathematical exposition of the ADF and Variance Ratio tests can be found in Walter Beckert's course notes (Beckert, 2011). Here, we are interested only in their applications to practical trading strategies.

# **Augmented Dickey-Fuller Test**

If a price series is mean reverting, then the current price level will tell us something about what the price's next move will be: If the price level is higher than the mean, the next move will be a downward move; if the price level is lower than the mean, the next move will be an upward move. The ADF test is based on just this observation.

We can describe the price changes using a linear model:

$$\Delta y(t) = \lambda y(t-1) + \mu + \beta t + \alpha_1 \Delta y(t-1) + \dots + \alpha_k \Delta y(t-k) + \epsilon_t \qquad (2.1)$$

where $\Delta y(t) \equiv y(t) - y(t - 1)$, $\Delta y(t - 1) \equiv y(t - 1) - y(t - 2)$, and so on. The ADF test will fi nd out if $\lambda = 0$. If the hypothesis $\lambda = 0$ can be rejected, that means the next move $\Delta y(t)$ depends on the current level $y(t - 1)$, and therefore it is not a random walk. The test statistic is the regression coeffi cient $\lambda$ (with $y(t - 1)$ as the independent variable and $\Delta y(t)$ as the dependent variable) divided by the standard error of the regression fi t: $\lambda/\text{SE}(\lambda)$. The statisticians Dickey and Fuller have kindly found out for us the distribution of this test statistic and tabulated the critical values for us, so we can look up for any value of $\lambda/\text{SE}(\lambda)$ whether the hypothesis can be rejected at, say, the 95 percent probability level.

Notice that since we expect mean regression, λ/SE(λ) has to be negative, and it has to be more negative than the critical value for the hypothesis to be rejected. The critical values themselves depend on the sample size and whether we assume that the price series has a non-zero mean −μ/λ or a steady drift −β*t*/λ. In practical trading, the constant drift in price, if any, tends to be of a much smaller magnitude than the daily fl uctuations in price. So for simplicity we will assume this drift term to be zero (β = 0).

In Example 2.1, we apply the ADF test to a currency rate series USD.CAD.

#### **Example 2.1: Using ADF Test for Mean Reversion**

The ADF test is available as a MATLAB Econometrics function *adftest*, or from the open-source MATLAB package spatial-econometrics.com's *adf* function. We will use *adf* below, and my code is available for download as *stationarityTests.m* from [http://epchan.com/book2.](http://epchan.com/book2)

(After you have downloaded the spatial-econometrics.com's jplv7 folder to your computer, remember to add all the subfolders of this package to your MATLAB path before using it.)

## **Example 2.1 (***Continued***)**

The *adf* function has three inputs. The fi rst is the price series in ascending order of time (chronological order is important). The second is a parameter indicating whether we should assume the off set μ and whether the drift β in Equation 2.1 should be zero. We should assume the off set is nonzero, since the mean price toward which the prices revert is seldom zero. We should, however, assume the drift is zero, because the constant drift in price tends to be of a much smaller magnitude than the daily fl uctuations in price. These considerations mean that the second parameter should be 0 (by the package designer's convention). The third input is the lag *k*. You can start by trying *k* = 0, but often only by setting *k* = 1 can we reject the null hypothesis, meaning that the change in prices often does have serial correlations. We will try the test on the exchange rate USD.CAD (how many Canadian dollars in exchange for one U.S. dollar). We assume that the daily prices at 17:00 ET are stored in a MATLAB array γ. The data fi le is that of one-minute bars, but we will just extract the end-of-day prices at 17:00 ET. Sampling the data at intraday frequency will not increase the statistical signifi cance of the ADF test. We can see from Figure 2.2 that it does not look very stationary.

![](_page_4_Figure_4.jpeg)

**FIGURE 2.2** USD.CAD Price Series

#### **Example 2.1 (***Continued***)**

And indeed, you should fi nd that the ADF test statistic is about −1.84, but the critical value at the 90 percent level is −2.594, so we can't reject the hypothesis that λ is zero. In other words, we can't show that USD.CAD is stationary, which perhaps is not surprising, given that the Canadian dollar is known as a commodity currency, while the U.S. dollar is not. But note that λ is negative, which indicates the price series is at least not trending.

```
results=adf(y, 0, 1); 
prt(results);
% Augmented DF test for unit root variable: variable 1 
% ADF t-statistic # of lags AR(1) estimate 
% -1.840744 1 0.994120 
% 
% 1% Crit Value 5% Crit Value 10% Crit Value 
% -3.458 -2.871 -2.594
```

# **Hurst Exponent and Variance Ratio Test**

Intuitively speaking, a "stationary" price series means that the prices diff use from its initial value more slowly than a geometric random walk would. Mathematically, we can determine the nature of the price series by measuring this speed of diff usion. The speed of diff usion can be characterized by the variance

$$\text{Var}(\tau) = \langle |z(t+\tau) - z(t)|^2 \rangle \qquad (2.2)$$

where $z$ is the log prices ($z = \text{log}(y)$), $\tau$ is an arbitrary time lag, and $\langle \dots \rangle$ is an average over all $t$'s. For a geometric random walk, we know that

$$\langle |z(t+\tau) - z(t)|^2 \rangle \sim \tau \qquad (2.3)$$

The $\sim$ means that this relationship turns into an equality with some proportionality constant for large $\tau$, but it may deviate from a straight line for small $\tau$. But if the (log) price series is mean reverting or trending (i.e., has positive correlations between sequential price moves), Equation 2.3 won't hold. Instead, we can write:

$$\langle |z(t+\tau) - z(t)|^2 \rangle \sim \tau^{2H} \qquad (2.4)$$

where we have defined the Hurst exponent $H$. For a price series exhibiting geometric random walk, $H = 0.5$. But for a mean-reverting series, $H < 0.5$, and for a trending series, $H > 0.5$. As $H$ decreases toward zero, the price series is more mean reverting, and as $H$ increases toward 1, the price series is increasingly trending; thus, $H$ serves also as an indicator for the degree of mean reversion or trendiness.

In Example 2.2, we computed the Hurst exponent for the same currency rate series USD.CAD that we used in the previous section using the MATLAB code. It generates an H of 0.49, which suggests that the price series is weakly mean reverting.

#### **Example 2.2: Computing the Hurst Exponent**

Using the same USD.CAD price series in the previous example, we now compute the Hurst exponent using a function called *genhurst* we can download from MATLAB Central (www.mathworks.com /matlabcentral/fileexchange/30076-generalized-hurst-exponent). This function computes a generalized version of the Hurst exponent defined by $\langle |z(t+\tau)-z(t)|^{2q} \rangle \sim \tau^{2H(q)}$, where $q$ is an arbitrary number. But here we are only interested in $q=2$, which we specify as the second input parameter to *genhurst*.

```
H=genhurst(log(y), 2);
```

If we apply this function to USD.CAD, we get H = 0.49, indicating that it may be weakly mean reverting.

Because of finite sample size, we need to know the statistical significance and MacKinlay of an estimated value of H to be sure whether we can reject the null hypothesis that H is really 0.5. This hypothesis test is provided by the Variance Ratio test (Lo, 2001).

The Variance Ratio Test simply tests whether

$$\frac{\text{Var}(z(t)-z(t-\tau))}{\tau \text{Var}(z(t)-z(t-1))}$$

is equal to 1. There is another ready-made MATLAB Econometrics Toolbox function *vratiotest* for this, whose usage I demonstrate in Example 2.3.

## **Example 2.3: Using the Variance Ratio Test for Stationarity**

The *vratiotest* from MATALB Econometric Toolbox is applied to the same USD.CAD price series *y* that have been used in the previous examples in this chapter. The outputs are *h* and *pValue*: *h* = 1 means rejection of the random walk hypothesis at the 90 percent confi dence level, *h* = 0 means it may be a random walk. *pValue* gives the probability that the null (random walk) hypothesis is true.

[h,pValue]=vratiotest(log(y));

We fi nd that *h* = 0 and *pValue* = 0.367281 for USD.CAD, indicating that there is a 37 percent chance that it is a random walk, so we cannot reject this hypothesis.

# **Half-Life of Mean Reversion**

The statistical tests I described for mean reversion or stationarity are very demanding, with their requirements of at least 90 percent certainty. But in practical trading, we can often be profi table with much less certainty. In this section, we shall fi nd another way to interpret the λ coeffi cient in Equation 2.1 so that we know whether it is negative enough to make a trading strategy practical, even if we cannot reject the null hypothesis that its actual value is zero with 90 percent certainty in an ADF test. We shall fi nd that λ is a measure of how long it takes for a price to mean revert.

To reveal this new interpretation, it is only necessary to transform the discrete time series Equation 2.1 to a diff erential form so that the changes in prices become infi nitesimal quantities. Furthermore, if we ignore the drift (β*t*) and the lagged diff erences (Δ*y*(*t* − 1), …, Δ*y*(*t* − *k*)) in Equation 2.1, then it becomes recognizable in stochastic calculus as the Ornstein-Uhlenbeck formula for mean-reverting process:

$$dy(t) = (\lambda y(t-1) + \mu)dt + d\varepsilon \qquad (2.5)$$

where *d*ε is some Gaussian noise. In the discrete form of 2.1, linear regression of Δ*y*(*t*) against *y*(*t* − 1) gave us λ, and once determined, this value of λ carries over to the diff erential form of 2.5. But the advantage of writing the equation in the diff erential form is that it allows for an analytical solution for the expected value of *y*(*t*):

$$E(y(t)) = y_0 \exp(\lambda t) - \frac{\mu}{\lambda} (1 - \exp(\lambda t)) \qquad (2.6)$$

Remembering that $\lambda$ is negative for a mean-reverting process, this tells us that the expected value of the price decays exponentially to the value $-\mu/\lambda$ with the half-life of decay equals to $-\text{log}(2)/\lambda$. This connection between a regression coeffi cient $\lambda$ and the half-life of mean reversion is very useful to traders. First, if we fi nd that $\lambda$ is positive, this means the price series is not at all mean reverting, and we shouldn't even attempt to write a meanreverting strategy to trade it. Second, if $\lambda$ is very close to zero, this means the half-life will be very long, and a mean-reverting trading strategy will not be very profi table because we won't be able to complete many round-trip trades in a given time period. Third, this $\lambda$ also determines a natural time scale for many parameters in our strategy. For example, if the half life is 20 days, we shouldn't use a look-back of 5 days to compute a moving average or standard deviation for a mean-reversion strategy. Often, setting the lookback to equal a small multiple of the half-life is close to optimal, and doing so will allow us to avoid brute-force optimization of a free parameter based on the performance of a trading strategy. We will demonstrate how to compute half-life in Example 2.4.

#### **Example 2.4: Computing Half-Life for Mean Reversion**

We concluded in the previous example that the price series USD.CAD is not stationary with at least 90 percent probability. But that doesn't necessarily mean we should give up trading this price series using a mean reversion model because most profi table trading strategies do not require such a high level of certainty. To determine whether USD.CAD is a good candidate for mean reversion trading, we will now determine its half-life of mean reversion.

To determine λ in Equations 2.1 and 2.5, we can run a regression fi t with *y*(*t*) − *y*(*t* − 1) as the dependent variable and *y*(*t* − 1) as the independent variable. The regression function *ols* as well as the function *lag* are both part of the jplv7 package. (You can also use the

#### **Example 2.4 (***Continued***)**

MATLAB Statistics Toolbox *regress* function for this as well.) This code fragment is part of *stationaryTests.m*.

```
ylag=lag(y, 1); % lag is a function in the jplv7 
 % (spatial-econometrics.com) package.
deltaY=y-ylag;
deltaY(1)=[]; % Regression functions cannot handle the NaN 
 in the first bar of the time series.
ylag(1)=[];
regress_results=ols(deltaY, [ylag ones(size(ylag))]); 
halflife=-log(2)/regress_results.beta(1);
```

The result is about 115 days. Depending on your trading horizon, this may or may not be too long. But at least we know what look-back to use and what holding period to expect.

# **A Linear Mean-Reverting Trading Strategy**

Once we determine that a price series is mean reverting, and that the halflife of mean reversion for a price series short enough for our trading horizon, we can easily trade this price series profi tably using a simple linear strategy: determine the normalized deviation of the price (moving standard deviation divided by the moving standard deviation of the price) from its moving average, and maintain the number of units in this asset negatively proportional to this normalized deviation. The look-back for the moving average and standard deviation can be set to equal the half-life. We see in Example 2.5 how this linear mean reversion works for USD.CAD.

You might wonder why it is necessary to use a moving average or standard deviation for a mean-reverting strategy at all. If a price series is stationary, shouldn't its mean and standard deviation be fi xed forever? Though we usually assume the mean of a price series to be fi xed, in practice it may change slowly due to changes in the economy or corporate management. As for the standard deviation, recall that Equation 2.4 implies even a "stationary" price series with 0 < *H* < 0.5 has a variance that increases with time, though not as rapidly as a geometric random walk. So it is appropriate to use moving average and standard deviation to allow ourselves to adapt to an ever-evolving mean and standard deviation, and also to capture profi t more quickly. This point will be explored more thoroughly in Chapter 3, particularly in the context of "scaling-in."

## **Example 2.5: Backtesting a Linear Mean-Reverting Trading Strategy**

In this simple strategy, we seek to own a number of units of USD.CAD equal to the negative normalized deviation from its moving average. The market value (in USD) of one unit of a currency pair USD.X is nothing but the quote USD.X, so in this case the linear mean reversion is equivalent to setting the market value of the portfolio to be the negative of the Z-Score of USD.CAD. The functions *movingAvg* and *movingStd* can be downloaded from my website. (This code fragment is part of *stationaryTests.m*.)

```
lookback=round(halflife); % setting lookback to the halflife 
 % found above
mktVal=-(y-movingAvg(y, lookback))./movingStd(y, lookback); 
pnl=lag(mktVal, 1).*(y-lag(y, 1))./lag(y, 1); % daily P&L of 
 % the strategy
```

The cumulative P&L of this strategy is plotted in Figure 2.3.

Despite the long half-life, the total profi t and loss (P&L) manages to be positive, albeit with a large drawdown. As with most example strategies in this book, we do not include transaction costs. Also, there is a look-ahead bias involved in this particular example due to

![](_page_10_Figure_7.jpeg)

**FIGURE 2.3** Equity Curve of Linear Trading Strategy on AUDCAD.

## **Example 2.5 (***Continued***)**

the use of in-sample data to fi nd the half-life and therefore the lookback. Furthermore, an unlimited amount of capital may be needed to generate the P&L because there was no maximum imposed on the market value of the portfolio. So I certainly don't recommend it as a practical trading strategy. (There is a more practical version of this mean-reverting strategy in Chapter 5.) But it does illustrate that a nonstationary price series need not discourage us from trading a mean reversion strategy, and that we don't need very complicated strategies or technical indicators to extract profi ts from a meanreverting series.

Since the goal for traders is ultimately to determine whether the expected return or Sharpe ratio of a mean-reverting trading strategy is good enough, why do we bother to go through the stationarity tests (ADF or Variance Ratio) and the calculation of half-life at all? Can't we just run a backtest on the trading strategy directly and be done with it? The reason why we went through all these preliminary tests is that their statistical signifi cance is usually higher than a direct backtest of a trading strategy. These preliminary tests make use of every day's (or, more generally, every bar's) price data for the test, while a backtest usually generates a signifi cantly smaller number of round trip trades for us to collect performance statistics. Furthermore, the outcome of a backtest is dependent on the specifi cs of a trading strategy, with a specifi c set of trading parameters. However, given a price series that passed the stationarity statistical tests, or at least one with a short enough half-life, we can be assured that we can eventually fi nd a profi table trading strategy, maybe just not the one that we have backtested.

# ■ **Cointegration**

As we stated in the introduction of this chapter, most fi nancial price series are not stationary or mean reverting. But, fortunately, we are not confi ned to trading those "prefabricated" fi nancial price series: We can proactively create a portfolio of individual price series so that the market value (or price) series of this portfolio is stationary. This is the notion of cointegration: If we can fi nd a stationary linear combination of several nonstationary price series, then these price series are called *cointegrated.* The most common combination is that of two price series: We long one asset and simultaneously short another asset, with an appropriate allocation of capital to each asset. This is the familiar "pairs trading" strategy. But the concept of cointegration easily extends to three or more assets. And in this section, we will look at two common cointegration tests: the CADF and the Johansen test. The former is suitable only for a pair of price series, while the latter is applicable to any number of series.

# **Cointegrated Augmented Dickey-Fuller Test**

An inquisitive reader may ask: Why do we need any new tests for the stationarity of the portfolio price series, when we already have the trusty ADF and Variance Ratio tests for stationarity? The answer is that given a number of price series, we do not know *a priori* what hedge ratios we should use to combine them to form a stationary portfolio. (The hedge ratio of a particular asset is the number of units of that asset we should be long or short in a portfolio. If the asset is a stock, then the number of units corresponds to the number of shares. A negative hedge ratio indicates we should be short that asset.) Just because a set of price series is cointegrating does not mean that *any* random linear combination of them will form a stationary portfolio. But pursuing this line of thought further, what if we fi rst determine the optimal hedge ratio by running a linear regression fi t between two price series, use this hedge ratio to form a portfolio, and then fi nally run a stationarity test on this portfolio price series? This is essentially what Engle and Granger (1987) did. For our convenience, the spatial-econometrics.com jplv7 package has provided a *cadf* function that performs all these steps. Example 2.6 demonstrates how to use this function by applying it to the two exchange-traded funds (ETFs) EWA and EWC.

## **Example 2.6: Using the CADF Test for Cointegration**

ETFs provide a fertile ground for fi nding cointegrating price series—and thus good candidates for pair trading. For example, both Canadian and Australian economies are commodity based, so they seem likely to cointegrate. The program *cointegrationTest.m* can be downloaded from my website. We assume the price series of EWA is (*Continued* )

## **Example 2.6 (***Continued***)**

![](_page_13_Figure_2.jpeg)

**FIGURE 2.4** Share Prices of EWA versus EWC

contained in the array *x*, and that of EWC is contained in the array *y*. From Figure 2.4, we can see that they do look quite cointegrating.

A scatter plot of EWA versus EWC in Figure 2.5 is even more convincing, as the price pairs fall on a straight line.

We can use the *ols* function found in the jplv7 package to fi nd the optimal hedge ratio.

![](_page_13_Figure_7.jpeg)

**FIGURE 2.5** Scatter Plot of EWA versus EWC

## **Example 2.6 (***Continued***)**

```
regression_result=ols(y, [x ones(size(x))]);
hedgeRatio=regression_result.beta(1);
```

As expected, the plot of the residual EWC-hedgeRatio\*EWA in Figure 2.6 does look very stationary.

We use the *cadf* function of the jplv7 package for our test. Other than an extra input for the second price series, the inputs are the same as the *adf* function. We again assume that there can be a nonzero off set of the pair portfolio's price series, but the drift is zero. Note that in both the regression and the CADF test we have chosen EWA to be the independent variable *x*, and EWC to be the dependent variable *y*. If we switch the roles of EWA and EWC, will the result for the CADF test diff er? Unfortunately, the answer is "yes." The hedge ratio derived from picking EWC as the independent variable will not be the exact reciprocal of the one derived from picking EWA as the independent variable. In many cases (though not for EWA-EWC, as we shall confi rm later with Johansen test), only one of those hedge ratios is "correct," in the sense that only one hedge ratio will lead to a stationary portfolio. If you use the CADF test, you would have to try each variable as independent and see which order gives the best (most negative) *t*-statistic, and use that order to obtain the

![](_page_14_Figure_6.jpeg)

**FIGURE 2.6** Stationarity of Residuals of Linear Regression between EWA versus EWC

## **Example 2.6 (***Continued***)**

hedge ratio. For brevity, we will just assume EWA to be independent, and run the CADF test.

```
results=cadf(y, x, 0, 1); 
% Print out results
prt(results);
% Output:
% Augmented DF test for co-integration variables:
  % variable 1,variable 2 
% CADF t-statistic # of lags AR(1) estimate 
% -3.64346635 1 -0.020411 
% 
% 1% Crit Value 5% Crit Value 10% Crit Value 
% -3.880 -3.359 -3.038 
% -3.880 -3.359 -3.038
```

We fi nd that the ADF test statistic is about –3.64, certainly more negative than the critical value at the 95 percent level of –3.359. So we can reject the null hypothesis that λ is zero. In other words, EWA and EWC are cointegrating with 95 percent certainty.

# **Johansen Test**

In order to test for cointegration of more than two variables, we need to use the Johansen test. To understand this test, let's generalize Equation 2.1 to the case where the price variable *y*(*t*) are actually vectors representing multiple price series, and the coeffi cients λ and α are actually matrices. (Because I do not think it is practical to allow for a constant drift in the price of a stationary portfolio, we will assume β*t* = 0 for simplicity.) Using English and Greek capital letters to represent vectors and matrices respectively, we can rewrite Equation 2.1 as

$$\Delta Y(t) = \Lambda Y(t-1) + M + A_1 \Delta Y(t-1) + \dots + A_k \Delta Y(t-k) + \epsilon_t \qquad (2.7)$$

Just as in the univariate case, if Λ = 0, we do not have cointegration. (Recall that if the next move of *Y* doesn't depend on the current price level, there can be no mean reversion.) Let's denote the rank (remember this quaint linear algebraic term?) of Λ as *r,* and the number of price series *n.* The number of independent portfolios that can be formed by various linear combinations of the cointegrating price series is equal to *r.* The Johansen test will calculate *r* for us in two diff erent ways, both based on eigenvector decomposition of Λ. One test produces the so-called trace statistic, and other produces the eigen statistic. (A good exposition can be found in Sorensen, 2005.) We need not worry what they are exactly, since the jplv7 package will provide critical values for each statistic to allow us to test whether we can reject the null hypotheses that *r* = 0 (no cointegrating relationship), *r* ≤ 1, …, up to *r* ≤ *n* – 1. If all these hypotheses are rejected, then clearly we have *r* = *n.* As a useful by-product, the eigenvectors found can be used as our hedge ratios for the individual price series to form a stationary portfolio. We show how to run this test on the EWA-EWC pair in Example 2.7, where we fi nd that the Johansen test confi rms the CADF test's conclusion that this pair is cointegrating. But, more interestingly, we add another ETF to the mix: IGE, an ETF consisting of natural resource stocks. We will see how many cointegrating relations can be found from these three price series. We also use the eigenvectors to form a stationary portfolio, and fi nd out its half-life for mean reversion.

#### **Example 2.7: Using the Johansen Test for Cointegration**

We take the EWA and EWC price series that we used in Example 2.6 and apply the Johansen test to them. There are three inputs to the *johansen* function of the jplv7 package: *y*, *p*, and *k*. *y* is the input matrix, with each column vector representing one price series. As in the ADF and CADF tests, we set *p* = 0 to allow the Equation 2.7 to have a constant off set (*M* ≠ 0), but not a constant drift term (β = 0). The input *k* is the number of lags, which we again set to 1. (This code fragment is part of *cointegrationTests.m*.)

```
% Combine the two time series into a matrix y2 for input 
 % into Johansen test
y2=[y, x];
results=johansen(y2, 0, 1); 
% Print out results
prt(results);
```

#### **Example 2.7 (***Continued***)**

## % Output: Johansen MLE estimates NULL: Trace Statistic Crit 90% Crit 95% Crit 99% r <= 0 variable 1 19.983 13.429 15.494 19.935 r <= 1 variable 2 3.983 2.705 3.841 6.635 NULL: Eigen Statistic Crit 90% Crit 95% Crit 99% r <= 0 variable 1 16.000 12.297 14.264 18.520 r <= 1 variable 2 3.983 2.705 3.841 6.635

We see that for the Trace Statistic test, the hypothesis *r* = 0 is rejected at the 99% level, and *r* ≤ 1 is rejected at the 95 percent level. The Eigen Statistic test concludes that hypothesis *r* = 0 is rejected at the 95 percent level, and *r* ≤ 1 is rejected at the 95 percent as well. This means that from both tests, we conclude that there are two cointegrating relationships between EWA and EWC.

What does it mean to have two cointegrating relations when we have only two price series? Isn't there just one hedge ratio that will allocate capital between EWA and EWC to form a stationary portfolio? Actually, no. Remember when we discussed the CADF test, we pointed out that it is order dependent. If we switched the role of the EWA from the independent to dependent variable, we may get a diff erent conclusion. Similarly, when we use EWA as the dependent variable in a regression against EWC, we will get a diff erent hedge ratio than when we use EWA as the independent variable. These two diff erent hedge ratios, which are not necessarily reciprocal of each other, allow us to form two independent stationary portfolios. With the Johansen test, we do not need to run the regression two times to get those portfolios: Running it once will generate all the independent cointegrating relations that exist. The Johansen test, in other words, is independent of the order of the price series.

Now let us introduce another ETF to the portfolio: IGE, which consists of natural resource stocks. Assuming that its price series is contained in an array *z*, we will run the Johansen test on all three price series to fi nd out how many cointegrating relationships we can get out of this trio.

#### **Example 2.7 (***Continued***)**

```
y3=[y2, z];
results=johansen(y3, 0, 1); 
% Print out results
prt(results);
% Output:
% Johansen MLE estimates 
% NULL: Trace Statistic Crit 90% Crit 95% Crit 99% 
% r <= 0 variable 1 34.429 27.067 29.796 35.463 
% r <= 1 variable 2 17.532 13.429 15.494 19.935 
% r <= 2 variable 3 4.471 2.705 3.841 6.635 
% 
% NULL: Eigen Statistic Crit 90% Crit 95% Crit 99% 
% r <= 0 variable 1 16.897 18.893 21.131 25.865 
% r <= 1 variable 2 13.061 12.297 14.264 18.520 
% r <= 2 variable 3 4.471 2.705 3.841 6.635
```

Both Trace statistic and Eigen statistic tests conclude that we should have three cointegrating relations with 95 percent certainty.

The eigenvalues and eigenvectors are contained in the arrays *results.eig* and *results.evec,* respectively.

```
results.eig % Display the eigenvalues
```

```
% ans =
% 
% 0.0112
% 0.0087
% 0.0030
results.evec % Display the eigenvectors
% ans =
% 
% -1.0460 -0.5797 -0.2647
% 0.7600 -0.1120 -0.0790
% 0.2233 0.5316 0.0952
```

Notice that the eigenvectors (represented as column vectors in *results.evec*) are ordered in decreasing order of their corresponding eigenvalues. So we should expect the fi rst cointegrating relation to be (*Continued* )

## **Example 2.7 (***Continued***)**

the "strongest"; that is, have the shortest half-life for mean reversion. Naturally, we pick this eigenvector to form our stationary portfolio (the eigenvector determines the shares of each ETF), and we can fi nd its half-life by the same method as before when we were dealing with a stationary price series. The only diff erence is that we now have to compute the T × 1 array *yport,* which represents the net market value (price) of the portfolio, which is equal to the number of shares of each ETF multiplied by the share price of each ETF, then summed over all ETFs. *yport* takes the role of *y* in Example 2.4.

```
yport=smartsum(repmat(results.evec(:, 1)', [size(y3, 1) ...
 1]).*y3, 2); 
% Find value of lambda and thus the half-life of mean 
 % reversion by linear regression fit
ylag=lag(yport, 1); % lag is a function in the jplv7 
 % (spatial-econometrics.com) package.
deltaY=yport-ylag;
deltaY(1)=[]; % Regression functions cannot handle the NaN 
 % in the first bar of the time series.
ylag(1)=[];
regress_results=ols(deltaY, [ylag ones(size(ylag))]); 
halflife=-log(2)/regress_results.beta(1);
```

The half-life of 23 days is considerably shorter than the 115 days for USD.CAD, so we expect a mean reversion trading strategy to work better for this triplet.

# **Linear Mean-Reverting Trading on a Portfolio**

In Example 2.7 we determined that the EWA-EWC-IGE portfolio formed with the "best" eigenvector from the Johansen test has a short half-life. We can now confi dently proceed to backtest our simple linear mean-reverting strategy on this portfolio. The idea is the same as before when we own a number of units in USD.CAD proportional to their negative normalized deviation from its moving average (i.e., its Z-Score). Here, we also accumulate units of the portfolio proportional to the negative Z-Score of the "unit" portfolio's price. A unit portfolio is one with shares determined by the Johansen eigenvector. The share price of a unit portfolio is like the share price of a mutual fund or ETF: it is the same as its market value. When a unit portfolio has only a long and a short position in two instruments, it is usually called a *spread*. (We express this in more mathematical form in Chapter 3.)

Note that by a "linear" strategy we mean only that the number of units invested is proportional to the Z-Score, not that the market value of our investment is proportional.

This linear mean-reverting strategy is obviously not a practical strategy, at least in its simplest version, as we do not know the maximum capital required

## **Example 2.8: Backtesting a Linear Mean-Reverting Strategy on a Portfolio**

The *yport* is a Tx1 array representing the net market value of the "unit" portfolio calculated in the preceding code fragment. *numUnits* is a Tx1 array representing the multiples of this unit portfolio we wish to purchase. (The multiple is a negative number if we wish to short the unit portfolio.) All other variables are as previously calculated. The *positions* is a Tx3 array representing the position (market value) of each ETF in the portfolio we have invested in. (This code fragment is part of *cointegrationTests.m*.)

```
% Apply a simple linear mean reversion strategy to EWA-EWC-
 % IGE
lookback=round(halflife); % setting lookback to the halflife 
 % found above
numUnits =-(yport-movingAvg(yport, lookback))...
 ./movingStd(yport, lookback); % multiples of unit 
 % portfolio . movingAvg and movingStd are functions from 
 % epchan.com/book2
positions=repmat(numUnits, [1 size(y3, 2)]).*repmat(results. ...
 evec(:, 1)', [size(y3, 1) 1]).*y3;
 % results.evec(:, 1)' is the shares allocation, while 
 % positions is the capital (dollar) 
 % allocation in each ETF.
pnl=sum(lag(positions, 1).*(y3-lag(y3, 1))./lag(y3, 1), 2); 
 % daily P&L of the strategy
ret=pnl./sum(abs(lag(positions, 1)), 2); % return is P&L 
 % divided by gross market value of portfolio
```

 Figure 2.7 displays the cumulative returns curve of this linear meanreverting strategy for a stationary portfolio of EWA, EWC, and IGE. (*Continued* )

#### **Example 2.8 (***Continued***)**

![](_page_21_Figure_3.jpeg)

**FIGURE 2.7** Cumulative Returns of a Linear Trading Strategy on EWA-EWC-IGE Stationary Portfolio

We fi nd that APR = 12.6 percent with a Sharpe ratio of 1.4 for the strategy.

at the outset and we cannot really enter and exit an infi nitesimal number of shares whenever the price moves by an infi nitesimal amount. Despite such impracticalities, the importance of backtesting a mean-reverting price series with this simple linear strategy is that it shows we can extract profi ts without any data-snooping bias, as the strategy has no parameters to optimize. (Remember that even the look-back is set equal to the half-life, a quantity that depends on the properties of the price series itself, not our specifi c trading strategy.) Also, as the strategy continuously enters and exits positions, it is likely to have more statistical signifi cance than any other trading strategies that have more complicated and selective entry and exit rules.

# ■ **Pros and Cons of Mean-Reverting Strategies**

It is often fairly easy to construct mean-reverting strategies because we are not limited to trading instruments that are intrinsically stationary. We can pick and choose from a great variety of cointegrating stocks and ETFs to create our own stationary, mean-reverting portofolio. The fact that every year there are new ETFs created that may be just marginally diff erent from existing ones certainly helps our cause, too.

Besides the plethora of choices, there is often a good fundamental story behind a mean-reverting pair. Why does EWA cointegrate with EWC? That's because both the Canadian and the Australian economies are dominated by commodities. Why does GDX cointegrate with GLD? That's because the value of gold-mining companies is very much based on the value of gold. Even when a cointegrating pair falls apart (stops cointegrating), we can often still understand the reason. For example, as we explain in Chapter 4, the reason GDX and GLD fell apart around the early part of 2008 was high energy prices, which caused mining gold to be abnormally expensive. We hope that with understanding comes remedy. This availability of fundamental reasoning is in contrast to many momentum strategies whose only justifi cation is that there are investors who are slower than we are in reacting to the news. More bluntly, we must believe there are greater fools out there. But those fools do eventually catch up to us, and the momentum strategy in question may just stop working without explanation one day.

Another advantage of mean-reverting strategies is that they span a great variety of time scales. At one extreme, market-making strategies rely on prices that mean-revert in a matter of seconds. At the other extreme, fundamental investors invest in undervalued stocks for years and patiently wait for their prices to revert to their "fair" value. The short end of the time scale is particularly benefi cial to traders like ourselves, since a short time scale means a higher number of trades per year, which in turn translates to higher statistical confi dence and higher Sharpe ratio for our backtest and live trading, and ultimately higher compounded return of our strategy.

Unfortunately, it is because of the seemingly high consistency of meanreverting strategy that may lead to its eventual downfall. As Michael Dever pointed out, this high consistency often lulls traders into overconfi dence and overleverage as a result (Dever, 2011). (Think Long Term Capital Management.) When a mean-reverting strategy suddenly breaks down, perhaps because of a fundamental reason that is discernible only in hindsight, it often occurs when we are trading it at maximum leverage after an unbroken string of successes. So the rare loss is often very painful and sometimes catastrophic. Hence, risk management for mean reverting is particularly important, and particularly diffi cult since the usual stop losses cannot be logically deployed. In Chapter 8, I discuss why this is the case, as well as techniques for risk management that are suitable for meanreverting strategies.

#### **KEY POINTS**

- Mean reversion means that the change in price is proportional to the difference between the mean price and the current price.
- Stationarity means that prices diffuse slower than a geometric random walk.
- The ADF test is designed to test for mean reversion.
- The Hurst exponent and Variance Ratio tests are designed to test for stationarity.
- Half-life of mean reversion measures how quickly a price series reverts to its mean, and is a good predictor of the profi tability or Sharpe ratio of a meanreverting trading strategy when applied to this price series.
- A linear trading strategy here means the number of units or shares of a unit portfolio we own is proportional to the negative Z-Score of the price series of that portfolio.
- If we can combine two or more nonstationary price series to form a stationary portfolio, these price series are called cointegrating.
- Cointegration can be measured by either CADF test or Johansen test.
- The eigenvectors generated from the Johansen test can be used as hedge ratios to form a stationary portfolio out of the input price series, and the one with the largest eigenvalue is the one with the shortest half-life.