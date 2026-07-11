# CHAPTER 6 Money and Risk Management

All trading strategies suffer occasional losses, technically known as drawdowns. The drawdowns may last a few minutes or a few years. To profit from a quantitative trading business, it is essential to manage your risks in a way that limits your drawdowns to a tolerable level and yet be positioned to use optimal leverage of your equity to achieve maximum possible growth of your wealth. Furthermore, if you have more than one strategy, you will also need to find a way to optimally allocate capital among them so as to maximize overall riskadjusted return.

The optimal allocation of capital and the optimal leverage to use so as to strike the right balance between risk management and maximum growth is the focus of this chapter, and the central tool we use is called the Kelly formula.

### OPTIMAL CAPITAL ALLOCATION AND LEVERAGE

Suppose you plan to trade several strategies, each with their own expected returns and standard deviations. How should you allocate capital among them in an optimal way? Furthermore, what should be the overall leverage (ratio of the size of your portfolio to your account equity)? Dr. Edward Thorp, whom I mentioned in the preface, has written an excellent expository article on this subject in one of his papers (Thorp, 1997), and I shall follow his discussion closely in this chapter. (Dr. Thorp's discussion is centered on a portfolio of securities, and mine is constructed around a portfolio of strategies. However, the mathematics are almost identical.)

Every optimization problem begins with an objective. Our objective here is to maximize our long-term wealth—an objective that I believe is not controversial for the individual investor. Maximizing longterm wealth is equivalent to maximizing the long-term compounded growth rate $g$ of your portfolio. Note that this objective implicitly means that ruin (i.e., equity's going to zero or less because of a loss) must be avoided. This is because if ruin can be reached with nonzero probability at some point, the long-term wealth is surely zero, as is the long-term growth rate.

(In all of the discussions, I assume that we reinvest all trading profits, and therefore it is the levered, compounded growth rate that is of ultimate importance.)

One approximation that I will make is that the probability distribution of the returns of each of the trading strategy i is Gaussian, with a fixed mean $m_{i}$ and standard deviation $s_{i \cdot}$ (The returns should be net of all financing costs; that is, they should be excess returns.) This is a common approximation in finance, but it can be quite inaccurate. Certain big losses in the financial markets occur with far higher frequencies (or viewed alternatively, at far higher magnitudes) than Gaussian probability distributions will allow. However, every scientific or engineering endeavor starts with the simplest model with the crudest approximation, and finance is no exception. I will discuss the remedies to such inaccuracies later in this chapter.

Let's denote the optimal fractions of your equity that you should allocate to each of your $n$ strategies by a column vector $\boldsymbol { F }^{\ast} = \big ( \boldsymbol { f }_{\mathrm { ~ 1 ~} }^{\ast} \boldsymbol { f }_{\mathrm { ~ 2 ~} }^{\ast}$ $. . . , f_{n} ^ { * } ) ^ { T } .$ . Here, T means transpose.

Given our optimization objective and the Gaussian assumption, Dr. Thorp has shown that the optimal allocation is given by

$$
F^{*} = C^{- 1} M
$$

Here, $C$ is the covariance matrix such that matrix element $C_{i j}$ is the covariance of the returns of the $i^{\mathrm { { t h} } }$ and $j^{\mathrm{th} }$ strategies, $\mathbf { - 1 }$ indicates matrix inverse, and $M = ( m_{1} , m_{2} , . . . , m_{\mathrm{n} } ) ^ { T }$ is the column vector of mean returns of the strategies. Note that these returns are oneperiod, simple (uncompounded), unlevered returns. For example, if the strategy is long $\$ 1$ of stock A and short $\$ 1$ of stock B and made $\$ 0.10$ profit in a period, $m$ is 0.05, no matter what the equity in the account is.

If we assume that the strategies are all statistically independent, the covariance matrix becomes a diagonal matrix, with the diagonal elements equal to the variance of the individual strategies. This leads to an especially simple formula:

$$
f_{i} = m_{i} / s_{i} ^ { 2 }
$$

This is the famous Kelly formula (for the many interesting stories surrounding this formula, see, for example, Poundstone, 2005) as applied to continuous finance as opposed to gambling with discrete outcomes, and it gives the optimal leverage one should employ for a particular trading strategy.

Interested readers can look up a simple derivation of the Kelly formula at the end of this chapter in the simple one-strategy case.

### Example 6.1: An Interesting Puzzle (or Why Risk Is Bad for You)1

Here is a little puzzle that may stymie many a professional trader. Suppose a certain stock exhibits a true (geometric) random walk, by which I mean there is a 50–50 chance that the stock is going up 1 percent or down 1 percent every minute. If you buy this stock, are you most likely—in the long run and ignoring financing costs—to make money, lose money, or be flat?

Most traders will blurt out the answer “Flat!” and that is wrong. The correct answer is that you will lose money, at the rate of 0.005 percent (or 0.5 basis point) every minute! This is because for a geometric random walk, the average compounded rate of return is not the short-term (or one-period) return m (0 here), but is $g = m - s^{2} / 2$ . This follows from the general formula for compounded growth $g ( f )$ given in the appendix to this chapter, with the leverage $f$ set to 1 and risk-free rate $r$ set to 0. This is also consistent with the fact that the geometric mean of a set of numbers is always smaller than the arithmetic mean (unless the numbers are identical, in which case the two means are the same). When we assume, as I did, that the arithmetic mean of the returns is zero, the geometric mean, which gives the average compounded rate of return, must be negative.

The take-away lesson here is that risk always decreases long-term growth rate—hence the importance of risk management! (See also Box 6.1 on “Loss Aversion Is Not a Behavioral Bias.”)

Often, because of uncertainties in parameter estimations, and also because return distributions are not really Gaussian, traders prefer to cut this recommended leverage in half for safety. This is called half-Kelly betting.

If you have a retail trading account, your maximum overall leverage l will be restricted to either 2 or 4, depending on whether you hold the positions overnight or just intraday. In this situation, you would have to reduce each $f_{i}$ by the same factor $l / ( | f_{1} | + | f_{2} | + . . . + | f_{n} | )$ , where $| f_{1} | + | f_{2} | + . . . + | f_{n} |$ is the total unrestricted leverage of the portfolio. Here, we ignore the possibility that some of your individual strategies may hold positions that offset each other (such as a long and a short position each balanced with short and long T-bills, respectively), which may allow you to hold a higher leverage than this formula suggests.

I stated that adopting this capital allocation and leverage will allow us to maximize the long-term compounded growth rate of your equity. So what is this maximum compounded growth rate? It turns out to be

$$
g = r + S^{2} / 2
$$

where the $\boldsymbol { s }$ is none other than the Sharpe ratio of your portfolio! As I mentioned in Chapter 2, the higher the Sharpe ratio of your portfolio (or strategy), the higher the maximum growth rate of your equity (or wealth), provided you use the optimal leverage recommended by the Kelly formula. Here is the simple mathematical embodiment of this fact.

### Example 6.2: Calculating the Optimal Leverage Based on the Kelly Formula

Let's see an example of the Kelly formula at work. Suppose our portfolio consists of just a long position in SPY, the exchangetraded fund (ETF) tracking the S&P 500 index. Let's suppose that the mean annual return of SPY is 11.23 percent, with an annualized standard deviation of 16.91 percent, and that the riskfree rate is 4 percent. Hence, the portfolio has an annual mean excess return of 7.231 percent and an annual standard deviation of 16.91 percent, giving it a Sharpe ratio of 0.4275. The optimal leverage according to the Kelly formula is $f = 0 . 0 7 2 3 1 / 0 . 1 6 9 1^{2} =$ 2.528. (Notice one interesting tidbit: The Kelly $f$ is independent of time scale, so it actually does not matter whether you annualize your return and standard deviation, as opposed to the Sharpe ratio, which is time scale dependent.) Finally, the annualized compounded, levered growth rate is 13.14 percent, which includes the financing costs.

You can verify these numbers yourselves by downloading the SPY daily prices from Yahoo! Finance and computing the various quantities on a spreadsheet. I did that on December 29, 2007, and my spreadsheet is available at   
epchan.com/book/example6_2.xls. In column H, I have   
computed the daily returns of the (adjusted) closing prices of SPY, while in row 3760 starting at column H, I have computed the (annualized) mean return of SPY, the standard deviation of SPY, the mean excess return of the portfolio, the Sharpe ratio of the portfolio, the Kelly leverage, and, finally, the compounded growth rate.

The Kelly leverage of 2.528 that we computed is saying that, for this strategy, if you have $\$ 100$ ,000 in cash to invest, and if you really believe the expected values of your returns and standard deviations, you should borrow money to buy $\$ 252$ ,800 worth of SPY. Furthermore, expect an annual compounded return on your $\$ 1$ 100,000 investment to be 13.14 percent.

For comparison, let's see what compounded growth rate we will get if we did not leverage our investment (see the formula in the appendix to this chapter): $g = r + m - s^{2} / 2 = 0 . 1 1 2 3 -$ $( 0 . 1 6 9 1 ) ^ { 2 } / 2 = 9 . 8$ percent, where m is the annualized mean return and s is the annualized standard deviation of returns. This, and not mean annual return of 11.23 percent, is the long-term growth rate of buying SPY with cash only.

### Example 6.3: Calculating the Optimal Allocation Using the Kelly Formula

We pick three sector-specific ETFs and see how we should allocate capital among them to achieve the maximum growth rate for the portfolio. The three ETFs are: OIH (oil service), RKH (regional bank), and RTH (retail). The daily prices are downloaded from Yahoo! Finance and saved in epchan.com/book as OIH.xls, RKH.xls, and RTH.xls.

### Using MATLAB

Here is the MATLAB program   
(epchan.com/book/example6_3.m) to retrieve these files and calculate $M , C ,$ and $F^{^ { \ast} }$ . % make sure previously defined variables are erased.   
clear;   
% read a spreadsheet named "OIH.xls" into MATLAB.   
[num1, txt1] $=$ xlsread('OIH');   
% the first column (starting from the second row) is   
% the trading days in format mm/dd/yyyy.   
tday1 $=$ txt1(2:end, 1);   
tday1 $=$ datestr(datenum(tday1, 'mm/dd/yyyy'), 'yyyymmdd'); % convert the format into yyyymmdd. $\%$ convert the date strings first into cell arrays and then into numeric format.   
tday1 $=$ str2double(cellstr(tday1));   
% the last column contains the adjusted close prices.   
adjcls1 $=$ num1(:, end);   
% read a spreadsheet named "RKH.xls" into MATLAB.   
[num2, txt2] $=$ xlsread('RKH');   
% the first column (starting from the second row) is % the trading days in format mm/dd/yyyy.   
tday2 $=$ txt2(2:end, 1);   
% convert the format into yyyymmdd.   
tday2 $=$ ..   
datestr(datenum(tday2, 'mm/dd/yyyy'), 'yyyymmdd');

% convert the date strings first into cell arrays and

% then into numeric format.   
tday2 $=$ str2double(cellstr(tday2));   
adjcls2 $=$ num2(:, end); % read a spreadsheet named "RTH.xls" into MATLAB.   
[num3, txt3] $=$ xlsread('RTH');   
% the first column (starting from the second row) is % the trading days in format mm/dd/yyyy.   
tday3 $=$ txt3(2:end, 1);   
% convert the format into yyyymmdd.   
tday $3 =$ ..   
datestr(datenum(tday3, 'mm/dd/yyyy'), 'yyyymmdd'); % convert the date strings first into cell arrays and % then into numeric format.   
tday3 $=$ str2double(cellstr(tday3));   
adjcls3 $=$ num3(:, end);

% merge these data tday $=$ union(tday1, tday2); tday $=$ union(tday, tday3); adjcls $=$ NaN(length(tday), 3);

[foo idx1 idx] $=$ intersect(tday1, tday);   
adjcls(idx, 1) $=$ adjcls1(idx1);   
[foo idx2 idx] $=$ intersect(tday2, tday);   
adjcls(idx, 2) $=$ adjcls2(idx2);   
[foo idx3 idx] $=$ intersect(tday3, tday);   
adjcls(idx, 3) $=$ adjcls3(idx3);

ret $=$ (adjcls-lag1(adjcls))./lag1(adjcls); % returns

% days where any one return is missing   
baddata $\underline { { \underline { { \mathbf { \Pi } } } } } =$ find(any(\~isfinite(ret), 2));   
% eliminate days where any one return is missing   
ret(baddata,:) $= [ ~ ]$ ;   
% excess returns: assume annualized risk free rate is 4% excessRet $=$ ret-repmat(0.04/252, size(ret));   
$\%$ annualized mean excess returns   
$_ { \mathrm{M} = 2 5 2 } \star$ mean(excessRet, 1)'% M $=$   
%   
% 0.1396 % 0.0294 % -0.0073

C=252\*cov(excessRet) % annualized covariance matrix $\begin{array}{r l} { \frac { 0 } { 0 } } & { { } \mathsf { C } } & { = } \end{array}$

% 0.1109 0.0200 0.0183   
% 0.0200 0.0372 0.0269   
% 0.0183 0.0269 0.0420   
F=inv(C)\*M % Kelly optimal leverages   
% F $=$   
%   
% 1.2919   
% 1.1723   
% -1.4882

Notice that the mean excess return of RTH is negative. Given this, it is not surprising that the Kelly formula recommends we short RTH.

You might wonder what the Sharpe ratio and the maximum compounded growth rate generated using this optimal allocation are. It turns out that the maximum growth rate of a multistrategy Gaussian process is

$$
g \left( F^{*} \right) = r + F^{*} { }^{T} C F^{*} / 2
$$

and the Sharpe ratio is given by

$$
S = \sqrt { F^{*} " C F^{*} } .
$$

Here is the MATLAB code snippet that calculates these two quantities:

$\%$ Maximum annualized compounded growth rate   
$\mathtt { g = 0 \_ 0 4 + F^{\prime} \times C^{\star} F / 2 \_ 9^{\circ} } \mathtt{g} =$   
$\%$   
% 0.1529   
S=sqrt(F'\*C\*F) % Sharpe ratio of portfolio   
% ${ \mathrm { ~ \bf ~ S ~ } } =$   
$\%$   
$\%$ 0.4751

Notice that the compounded growth rate of the portfolio is 15.29 percent, higher than that of the maximum growth rate achievable by any of the individual stocks. (As an exercise, you can verify

that the compounded growth rate of OIH, which has the highest one-period return among the three stocks, is 12.78 percent.)

### Using Python

Here is the equivalent Python Jupyter Notebook code, downloadable as example6_3.ipynb.

Calculating the Optimal Allocation Using Kelly formula   
import numpy as np   
import pandas as pd   
from numpy.linalg import inv   
df1 $=$ pd.read_excel('OIH.xls')   
df2 $=$ pd.read_excel('RKH.xls')   
df=pd.merge(df1, df2, on $=$ 'Date', suffixes $=$ ('_OIH', '_RKH'))   
df.set_index('Date', inplace $=$ True)   
df3 $=$ pd.read_excel('RTH.xls')   
df=pd.merge(df, df3, on $=$ 'Date')   
df.rename(columns $=$ {"Adj Close": "Adj Close_RTH"},   
inplace $=$ True)   
df.set_index('Date', inplace $=$ True)   
df.sort_index(inplace $=$ True)   
dailyret $=$ df.loc[:, ('Adj Close_OIH', 'Adj Close_RKH', 'Adj   
Close_RTH')].pct_change()   
dailyret.rename(columns $=$ {"Adj Close_OIH": "OIH", "Adj   
Close_RKH": "RKH", "Adj Close_RTH": "RTH"}, inplace $=$ True)   
excessRet $=$ dailyret-0.04/252   
$_ { \mathrm{M} = 2 5 2 } \star$ excessRet.mean()   
M   
OIH 0.139568 RKH 0.029400 RTH -0.007346 dtype: float64   
${ \mathrm{C} } { = } 2 5 2^{\star}$ excessRet.cov()   
C   
OIH   
RKH   
RTH   
OIH   
0.110901   
0.020014   
0.018255   
RKH   
0.020014   
0.037165   
0.026893   
RTH   
0.018255   
0.026893   
0.041967   
$\mathtt{F} =$ np.dot(inv(C), M)   
F   
array([ 1.2919082, 1.17226473, -1.48821285])   
$\mathtt { g = } 0 \mathtt { . } 0 4 \mathtt { + } \mathtt{np}$ .dot(F.T, np.dot(C, F))/2   
g   
0.1528535789840623   
${ \sf S } =$ np.sqrt(np.dot(F.T, np.dot(C, F)))   
S   
0.47508647420035505

### Using R

Here is the R code, downloadable as example6_3.R.

library('zoo') source('calculateReturns.R') source('calculateMaxDD.R') source('backshift.R')

data1 <- read.delim("OIH.txt") # Tab-delimited   
data_sort1 <- data1[order(as.Date(data1[,1], '%m/%d/%Y')),]   
### sort in ascending order of dates (1st column of data)   
tday1 <- as.integer(format(as.Date(data_sort1[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls1 <- data_sort1[,ncol(data_sort1)]   
data2 <- read.delim("RKH.txt") # Tab-delimited   
data_sort2 <- data2[order(as.Date(data2[,1], '%m/%d/%Y')),] # sort in ascending order of dates (1st   
column of data)   
tday2 <- as.integer(format(as.Date(data_sort2[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls2 <- data_sort2[,ncol(data_sort2)]   
data3 <- read.delim("RTH.txt") # Tab-delimited   
data_sort3 <- data3[order(as.Date(data3[,1], '%m/%d/%Y')),] # sort in ascending order of dates (1st   
column of data)   
tday3 <- as.integer(format(as.Date(data_sort3[,1], '%m/%d/%Y'), '%Y%m%d'))   
adjcls3 <- data_sort3[,ncol(data_sort3)]

### merge these data tday <- union(tday1, tday2) tday <- union(tday, tday2) tday <- tday[order(tday)] adjcls <- matrix(NaN, length(tday), 3)

adjcls[tday %in% tday1, 1] <- adjcls1   
adjcls[tday %in% tday2, 2] <- adjcls2   
adjcls[tday %in% tday3, 3] <- adjcls3

ret <- calculateReturns(adjcls, 1) # daily returns excessRet <- ret - 0.04/252 # excess returns: assume annualized risk free rate is 4%

### annualized mean excess returns   
M <- 252\*colMeans(excessRet, na.rm $=$ TRUE) #   
c(0.143479780597111, 0.0560305170502084, -0.0073464585163155)   
### annualized covariance matrix   
C<- 252\*cov(excessRet, use $=$ "pairwise.complete.obs")   
### 0.11305722 0.01986547 0.01825486   
### 0.01986547 0.04263365 0.02689284   
### 0.01825486 0.02689284 0.04196684   
### Kelly optimal leverage   
F <- solve(C) %\*% M   
### 1.240554   
### 1.992328   
### -1.991381

### Maximum annualized compounded growth rate g <- 0.04+t(F) %\*% C %\*% F/2 # 0.1921276

### Sharpe ratio of portfolio S <- sqrt(t(F) %\*% C %\*% F) # 0.5515933

Note that following the Kelly formula requires you to continuously adjust your capital allocation as your equity changes so that it remains optimal. Based on the SPY example (Example 6.2), let's say you followed the Kelly formula and bought a portfolio worth $\$ 252$ ,800. The next day, disaster struck, and you lost 10 percent on SPY. So now your portfolio is worth only $\$ 227,520$ , and your equity is now only $\$ 1$ 74,720. What should you do now? Kelly's criterion will dictate that you immediately reduce the size of your portfolio to $\$ 188$ ,892. Why? Because the optimal leverage of 2.528 times the current equity of $\$ 74,720$ is $\$ 123,892$ .

As a practical procedure, this continuous updating of the capital allocation should occur at least once at the end of each trading day. In addition to updating the capital allocation, one should also periodically update $F^{*}$ itself by recalculating the most recent trailing mean return and standard deviation. What should the lookback period be, and how often do you need to update these inputs to the Kelly formula? These depend on the average holding period of your strategy. If you hold your positions for only one day or so, then as a rule of thumb, I would advise using a lookback period of six months. Using a relatively short lookback period has the advantage of allowing you to gradually reduce your exposure to strategies that have been losing their performance. As for the frequency of update, it should not be a burden to update $F^{*}$ daily once you have written a program to do so.

One last point: Some strategies generate a variable number of trading signals each day, which may result in a variable number of positions and thus total capital each day. How should the Kelly formula be used to determine the capital in this case when we don't know what it will be beforehand? One can still use the Kelly formula to determine the maximum number of positions and thus the maximum capital allowed. It is always safer to have a leverage below what the Kelly formula recommends.

### RISK MANAGEMENT

We saw in the previous section that the Kelly formula is not only useful for the optimal allocation of capital and for the determination of the optimal leverage, but also for risk management. In fact, the SPY example (Example 6.2) illustrated that the Kelly formula would advise you to reduce the portfolio size in the face of trading losses. This selling at a loss is the frequent result of risk management, whether or not the risk management scheme is based on the Kelly formula.

Risk management always dictates that you should reduce your position size whenever there is a loss, even when it means realizing those losses. (The other face of the coin is that optimal leverage dictates that you should increase your position size when your strategy generates profits.) This kind of selling is believed by some analysts to be the cause of “financial contagion” affecting many large hedge funds simultaneously when one faces a large loss.

An example of this is the summer 2007 meltdown, described in the previously cited article “What Happened to the Quants in August 2007?” by Amir Khandani and Andrew Lo (Khandani and Lo, 2007). During August 2007, under the ominous cloud of a housing and mortgage default crisis, a number of well-known hedge funds experienced unprecedented losses, with Goldman Sachs's Global Alpha fund falling 22.5 percent. Several billion dollars evaporated within one week. Even Renaissance Technologies Corporation, arguably the most successful quantitative hedge fund of all time, lost 8.7 percent in the first half of August, though it later recovered most of it. Not only is the magnitude of the loss astounding, but the widespread nature of it was causing great concern in the financial community. Strangest of all, few of these funds hold any mortgagebacked securities at all, ostensibly the root cause of the panic. It therefore became a classic study of financial contagion as propagated by hedge funds.

Another example is the January 2021 GameStop short squeeze Wang (2021). As this stock was surging due to traders’ promotion on Reddit's r/wallstreetbets forum and the subsequent coordinated buying of the stock and especially its call options, large hedge funds suffered enormous losses as they bought cover for their short positions. Renaissance Institutional Equities Fund fell 9.5 percent, while Melvin Capital lost a whopping 53 percent.

This kind of contagion occurs because a large loss by one hedge fund causes it to sell off some large positions that it holds (whether or not these are the positions that cause the loss in the first place). This selling causes the prices of the securities to drop (or rise in the case of short positions). If other hedge funds are holding similar positions, they will then suffer large losses also, causing their own risk management system to sell off their own positions, and on and on. For example, in the summer of 2007, one large hedge fund might have been holding subprime mortgage-backed securities and suffered a large loss in that sector. Risk management then required that it sell off liquid stock positions in their portfolio that might, up to that point, be unaffected by the subprime debacle. Because of the selling of such stock positions, other statistical arbitrage hedge funds that hold no mortgage-backed securities might now have suffered big losses, and have proceeded to sell their stocks as well. Hence, a selloff in the mortgage-backed securities market suddenly became a selloff in the stock market—a nice demonstration of the meaning of contagion. Similarly, in January 2021, the losses in short GameStop positions in some long–short funds caused them to buy cover other unrelated short positions while simultaneously selling unrelated long positions due to overall portfolio deleveraging, leading to losses in other funds that hold similar short and long positions. They were also forced to buy and sell the same stocks due to their own deleveraging, leading to a contagion.

Given the necessity of realizing losses as well as the scale and frequency of trading required to constantly rebalance the portfolio in order to closely follow the Kelly formula, it is understandable that most traders prefer to trade at half-Kelly leverage. A lower leverage implies a smaller size of the selling required for risk management.

Sometimes, even taking the conservative half-Kelly formula may be too aggressive, and traders may want to limit their portfolio size further by additional constraints. This is because, as I pointed out previously, the application of the Kelly formula to continuous finance is premised on the assumption that return distribution is Gaussian. (Finance is continuous in the sense that the outcomes of making bets in the financial market fall on a continuum of profits or losses, as opposed to a game of cards where the outcomes fall into discrete cases.) But, of course, the returns are not really Gaussian: large losses occur at far higher frequencies than would be predicted by a nice bell-shaped curve. Some people refer to the true distributions of returns as having fat tails. What this means is that the probability of an event far, far away from the mean is much higher than allowed by the Gaussian bell curve. These highly improbable events have been called black swan events by the author Nassim Taleb (see Taleb, 2007).

To handle extreme events that fall outside the Gaussian distribution, we can use our simple backtest technique to roughly estimate what the maximum one-period loss was historically. (The period may be one week, one day, or one hour. The only criterion to use is that you should be ready to rebalance your portfolio according to the Kelly formula at the end of every period.) You should also have in mind what is the maximum one-period drawdown on your equity that you are willing to suffer. Dividing the maximum tolerable one-period drawdown on equity by the maximum historical loss will tell you whether even half-Kelly leverage is too large for your comfort. The leverage to use is always the smaller of the half-Kelly leverage and the maximum leverage obtained using the worst historical loss. In the S&P 500 index example in the previous section, the maximum historical one-day loss is about 20.47 percent, which occurred on October 19, 1987—“Black Monday.” If you can tolerate only a 20 percent one-day drawdown on equity, then the maximum leverage you can apply is about 1. Meanwhile, the leverage recommended by half-Kelly is 1.26. Hence, in this case, even half-Kelly leverage would not be conservative enough to survive Black Monday.

The truly scary scenario in risk management is the one that has not occurred in history before. Echoing the philosopher Ludwig Wittgenstein, “Whereof one cannot speak, thereof one must be silent”—on such unknowables, theoretical models are appropriately silent.

### IS THE USE OF STOP LOSS A GOOD RISK MANAGEMENT PRACTICE?

Some traders believe that good risk management means imposing stop loss on every trade; that is, if a position incurs a certain percent loss, the trader will exit the position. It is a common fallacy to believe that imposing stop loss will prevent the portfolio from suffering catastrophic losses. When a catastrophic event occurs, securities prices will drop discontinuously, so the stop-loss orders to exit the positions will only be filled at prices much worse than those before the event. So, by exiting the positions, we are actually realizing the catastrophic loss and not avoiding it. For stop loss to be beneficial, we must believe that we are in a momentum, or trending, regime. In other words, we must believe that the prices will get worse within the expected lifetime of our trade. Otherwise, if the market is mean reverting within that lifetime, we will eventually recoup our losses if we didn't exit the position too quickly.

Of course, it is not easy to tell whether one is in a momentum regime (when stop loss is beneficial) or in a mean-reverting regime (when stop loss is harmful). My own observation is that when the movement of prices is due to news or other fundamental reasons (such as a company's deteriorating revenue), one is likely to be in a momentum regime, and one should not “stand in front of a freight train,” in traders' vernacular. For example, if a fundamental analysis of a company reveals that it is currently overvalued, its stock price will likely gradually decrease (at least in relation to the market index) in order to reach a new, lower equilibrium price. This movement to the lower equilibrium price is irreversible as long as the fundamental economics of the company does not change. However, when securities prices move drastically without any apparent news or reasons, it is likely that the move is the result of a liquidity event—for example, major holders of the securities suddenly need to liquidate large positions for their own idiosyncratic reasons, or major speculators suddenly decide to

cover their short positions. These liquidity events are of relatively short durations and mean reversion to the previous price levels is likely.

I will discuss in some more detail the appropriate exit strategies for mean-reverting versus momentum strategies in Chapter 7.

Beyond position risk (which is comprised of both market risk and specific risk), there are other forms of risks to consider: model risk, software risk, and natural disaster risk, in decreasing order of likelihood.

### Model Risk

Model risk simply refers to the possibility that trading losses are not due to the statistical vagaries of the market but to the fact that the trading model is wrong. It could be wrong for a large number of reasons, some of which were detailed in Chapter 3: data-snooping bias, survivorship bias, and so on. To eliminate all these different biases and errors in the backtest programs, it is extremely helpful to have a collaborator or consultant to duplicate your backtest results independently to ensure their validity. This need to duplicate results is routinely done in scientific research and is no less essential in financial research.

Model risk can also come not from any bias or error in your model or backtesting procedure, but from increased competition from other institutional traders all running the same strategy as you; or it could be a result of some fundamental change in market structure that eliminated the edge of your trading model. This is the regime shift that I talked about in Chapter 3.

There is not much you can do to alleviate these sources of model risk, except to gradually lower the leverage of the model as it racks up losses, up to the point where the leverage is zero. This can be accomplished in a systematic way if you constantly update the leverage according to the Kelly formula based on the trailing mean return and standard deviation. (As the mean return decreases to zero in the lookback period, your Kelly leverage will be driven to zero.) This is preferable to abruptly shutting down a model because of a

large drawdown (see my discussion of the psychological pressure to shut down models prematurely in the following section on psychological preparedness).

### Software Risk

Software risk refers to the case where the automated trading system that generates trades every day actually does not faithfully reflect your backtest model. This happens because of the omnipresent software bugs. I discussed the way to eliminate such software errors in Chapter 5: you should compare the trades generated by your automated trading system with the theoretical trades generated by your backtest system to ensure that they are the same.

### Natural Disaster Risk

Finally, physical or natural disasters can happen, which can cause big losses, and they don't have to be anything dramatic like earthquakes or tsunami. What if your internet connection went down before you could enter a hedging position? What if your power went down in the middle of transmitting a trade? The different methods of preventing physical disasters from causing major disruptions to your trading can be found in the section on physical infrastructure in Chapter 5.

### PSYCHOLOGICAL PREPAREDNESS

It may seem strange that a book on quantitative trading would include a section on psychological preparedness. After all, isn't quantitative trading supposed to liberate us from our emotions and let the computer make all the trading decisions in a disciplined manner? If only it were this easy: human traders who are not psychologically prepared will often override their automated trading systems' decisions, especially when there is a position or day with abnormal profit or loss. Hence, it is critical even if we trade using quantitative strategies to understand some of our own psychological weaknesses.

Fortunately, there is a field of financial research called behavioral finance (Thaler, 1994) that studies irrational financial decisionmaking. I will try to highlight a few of the common irrational behaviors that affect trading.

The first behavioral bias is known variously as the endowment effect, status quo bias, or loss aversion. The first two effects cause some traders to hold on to a losing position for too long, because traders (and people in general) give too much preference to the status quo (the status quo bias), or because they demand much more to give up the stock than what they would pay to acquire it (the endowment effect). As I argued in the risk management section, there are rational reasons to hold on to a losing position (e.g., when you expect mean-reverting behavior); however, these behavioral biases cause traders to hold on to losing positions even when there is no rational reason (e.g., when you expect trending behavior, and the trend is such that your positions will lose even more). At the same time, the loss aversion bias causes some traders to exit their profitable positions too soon, even if holding longer will lead to a larger profit on average. Why do they exit the profitable positions so soon? Because the pain from possibly losing some of the current profits outweighs the pleasure from gaining higher profits.

This behavioral bias manifests itself most clearly and most disastrously when one has entered a position by mistake (because of either a software bug, an operational error, or a data problem) and has incurred a big loss. The rational step to take is to exit the position immediately upon discovery of the error. However, traders are often tempted to wait for mean reversion such that the loss is smaller before they exit. Unless you have a model for mean reversion that suggests now is a good time to enter into this position, this wait for mean reversion may very well lead to bigger losses instead.

While loss aversion leads to suboptimal trading in such a situation, there are other times when loss aversion is wise and is not a behavioral bias. Most economic arguments against loss aversion assumes that we have a large number of “gamblers” (read: traders) playing a risky game, and as long as the average return is positive, the economist suggests that this risky game is worthwhile, even if some of these gamblers may be ruined. However, when you are the trader, it is rational to avoid ruin at all cost, no matter what return the “average” trader enjoys. This contrast between the ensemble average (across different traders) and the time series average (over a long time horizon for a single trader) is a profound mathematical observation by the physicists Ole-Peters and Nobel laureate Murray Gell-Mann. See Box 6.1 for more details.

### BOX 6.1 LOSS AVERSION IS NOT A BEHAVIORAL BIAS

In his famous book Thinking, Fast and Slow, the Nobel laureate Daniel Kahneman (2011) described one common example of a behavioral finance bias:

“You are offered a gamble on the toss of a [fair] coin.

If the coin shows tails, you lose $\$ 100$ .

If the coin shows heads, you win $\$ 110$ .

Is this gamble attractive? Would you accept it?”

(I have modified the numbers to be more realistic in a financial market setting, but otherwise it is a direct quote.)

Experiments show that most people would not accept this gamble, even though the expected gain is $\$ 5$ . This is the so-called “loss aversion” behavioral bias, and is considered irrational. Kahneman went on to write that “professional risk takers” (read “traders”) are more willing to act rationally and accept this gamble.

It turns out that the loss-averse “layman” is the one acting rationally here.

It is true that if we have infinite capital, and can play infinitely many rounds of this game simultaneously, we should expect a $\$ 5$ gain per round. But trading isn't like that. We are dealt one coin at a time, and if we suffer a string of losses, our capital will be depleted and we will be in debtor prison if we keep playing. The proper way to evaluate whether this game is attractive is to evaluate the expected compound rate of growth of our capital.

Let's say we are starting with a capital of $\$ 1,000$ . The expected return of playing this game once is initially 0.005. The standard deviation of the return is 0.105. To simplify matters, let's say we are allowed to adjust the payoff of each round so we have the same expected return and standard deviation of return each

round. For example, if at some point we earned so much that we doubled our capital to $\$ 2$ ,000, we are allowed to win $\$ 220$ or lose $\$ 1$ 200 per round. What is the expected growth rate of our capital? As Example 6.1 shows, in the continuous approximation it is – 0.0005125 per round – we are losing, not gaining! The layman is right to refuse this gamble.

Loss aversion, in the context of a risky game played repeatedly, is rational, and not a behavioral bias. Our primitive, primate instinct grasped a truth that behavioral economists cannot. It only seems like a behavioral bias if we take an “ensemble view” (i.e., allowed infinite capital to play many rounds of this game simultaneously), instead of a “time series view” (i.e. allowed only finite capital to play many rounds of this game in sequence, provided we don't go broke at some point). The time series view is the one relevant to all traders. In other words, take time average, not ensemble average, when evaluating real-world risks.

The important difference between ensemble average and time average has been raised in a paper by physicist Ole Peters and Nobel laureate Murray Gell-Mann (Peters, et al., 2016). It deserves to be much more widely read in the behavioral economics community. But beyond academic interest, there is a practical importance in emphasizing that loss aversion is rational. As traders, we should not only focus on average returns: risks can depress compound returns severely.

Another common bias that I have personally experienced is the representativeness bias—people tend to put too much weight on recent experience and underweight long-term average (Ritter, 2003). (This reference has a good introduction to various biases studied by behavioral finance.) After a big loss, traders—even quantitative traders—tend to immediately modify certain parameters of their strategies so that they would have avoided the big loss if they were to trade this modified system. But, of course, this is unwise because this modification may invite some other big loss that is yet to happen, or it may have eliminated many profit opportunities that existed. We must remember that we are operating in a probabilistic regime: No system can avoid all the market vagaries that can result in losses.

If you feel that your system really is deficient and want to tweak it, you should always backtest the modified version to make sure that it does outperform the old system over a sufficiently long backtest period, not just over the last few weeks.

There are two major psychological weaknesses that are more well known to the traders than to economists: despair and greed.

Despair occurs when a trading model is in a major, prolonged drawdown. Many traders (and their managers, investors, etc.) will be under great pressure under this circumstance to shut down the model completely. Other overly self-confident traders with a reckless bent will do the opposite: They will double their bets on their losing models, hoping to recoup their losses eventually, if and when the models rebound. Neither behavior is rational: if you have been managing your capital allocation and leverage by the Kelly formula, you would lower the capital allocation for the losing model gradually.

Greed is the more usual emotion when the model is having a good run and is generating a lot of profits. The temptation now is to increase its leverage quickly in order to get rich quickly. Once again, a well-disciplined quantitative trader will keep the leverage below the dictates of the Kelly formula as well as the caution imposed by the possibility of fat-tail events.

Both despair and greed can lead to overleveraging (i.e., trading an overly large portfolio): In despair, one tries to recoup the losses by adding fresh capital; in greed, one adds capital too quickly after initial successes with a strategy. Therefore, the one golden rule in risk management is to keep the size of your portfolio under control at all times. This is, however, easier said than done. Large, well-known funds have succumbed to the temptation to overleverage and failed: Long-Term Capital Management in 2000 (Lowenstein, 2000) and Amaranth Advisors in 2006 (Chan, 2006a). In the Amaranth Advisors case, the leverage employed on one single strategy (natural gas calendar spread trade) due to one single trader (Brian Hunter) is so large that a $\$ 6$ billion loss was incurred, comfortably wiping out the fund's equity—a textbook case of risk mismanagement.

I have experienced this pressure myself both in an institutional setting and in a personal setting, and the unfortunate result both times was to succumb prematurely. When I was with a money management firm, I lost over $\$ 1$ million for the fund's investors because, in a fit of greed, I added over $\$ 100$ million to a portfolio based on a strategy that had been traded for barely six months. (That was before I learned of the Kelly criterion and other stress testing methodologies.) As if this is not enough lesson, I repeated the same mistake again when I started trading independently. It concerns a mean-reverting spread strategy involving XLE, an energy exchangetraded fund (ETF) and the crude oil future (CL). When the spread refused to mean revert over time, I stubbornly increased the size of the spread to almost $\$ 500$ ,000. Finally, despair set in, and I exited the spread with close to a six-figure loss. Naturally, the spread started to revert afterward when I wasn't around to benefit.

(Fortunately, several of my other strategies performed well in that first year of my independent trading, so the fiscal year ended with only a small overall loss.)

How should we train ourselves to overcome these psychological weaknesses and learn not to override the models manually and to remedy trading errors correctly and expeditiously? As with most human endeavors, the way to do this is to start with a small portfolio and gradually gain psychological preparedness, discipline, and confidence in your models. As you become emotionally more able to handle the daily swings in profit and loss (P&L) and rein in the primordial urges of the psyche, your portfolio's actual performance will hew to the theoretically expected performance of your strategy.

I have certainly found that to be the case after getting over those aforementioned disastrous trades. My newfound discipline and faith in the Kelly formula has so far prevented similar disasters from happening again.

### SUMMARY

Risk management is a crucial discipline in trading. The trading world is littered with numerous examples of giant hedge funds and investment banks laid low by enormous losses due to a single trade or in a very short period of time. Most of these losses are due to overleveraging positions and not to an inherently erroneous model. Typically, traders will not overleverage a model that has not worked very well. It is a hitherto superbly performing model that is at the greatest risk of huge loss due to overconfidence and overleverage. This chapter therefore provides an important tool for risk management: the determination of the optimal leverage using the Kelly formula.

Besides the determination of the optimal leverage, the Kelly formula has a very useful side benefit: It also determines the optimal allocation of capital among different strategies, based on the covariance of their returns.

But no risk management formula or system will prevent disasters if you are not psychologically prepared for the ups and downs of trading and thus deviating from the prescriptions of rational decision making (i.e., your models). The ultimate risk management mind-set is very simple: Do not succumb to either despair or greed. To gain practice in this psychological discipline, one must proceed slowly with small position size, and thoroughly test various aspects of the trading business (model, software, operational procedure, money and risk management) before scaling up according to the Kelly formula.

I have found that in order to proceed slowly and cautiously, it is helpful to have other sources of income or other businesses to help sustain yourself either financially or emotionally (to avoid the boredom associated with slow progress). It is indeed possible that finding a diversion, whether income producing or not, may actually help improve the long-term growth of your wealth.

## APPENDIX: A SIMPLE DERIVATION OF THE KELLY FORMULA WHEN RETURN DISTRIBUTION IS GAUSSIAN

If we assume that the return distribution of a strategy (or security) is Gaussian, then the Kelly formula can be derived very easily. We start with the formula for a compounded, levered growth rate applicable to a Gaussian process:

$$
g \big ( f \big ) = r + f m - s^{2} f^{2} / 2
$$

where $f$ is the leverage; r is the risk-free rate; m is the average simple, uncompounded one-period excess return; and s is the standard deviation of those uncompounded returns. This formula for compounded growth rate can itself be derived quite simply, but not as simply as the Kelly formula, so I leave its derivation for the reader to look up in the Thorp article referenced earlier.

To find the optimal $f ,$ which maximizes $g$ , simply take its first derivative with respect to $f$ and set the derivative to zero:

$$
d g / d f = m - s^{2} f = 0
$$

Solving this equation for $f$ gives us $f = m / s^{2}$ , the Kelly formula for one strategy or security under the Gaussian assumption\*.

## REFERENCES

Chan, Ernest. 2006a. “A ‘Highly Improbable’ Event? A Historical Analysis of the Natural Gas Spread Trade That Bought Down Amaranth.” Quantitative Trading blog, October 2, http://epchan.blogspot.com/2006/10/highly-improbableevent.html.

Kahneman, Daniel. 2011. Thinking, Fast and Slow. Farrar, Straus and Giroux.

Khandani, Amir E., and Andrew Lo. 2007. “What Happened to the Quants in August 2007?” MIT. https://web.mit.edu/Alo/www/Papers/august07.pdf.

Lowenstein, Roger. 2000. When Genius Failed: The Rise and Fall of Long-Term Capital Management. Random House.

Peters, O., and M. Gell-Mann. 2016. “Evaluating Gambles Using Dyanmics.” Chaos 26, 023103. https://doi.org/10.1063/1.4940236.

Poundstone, William. 2005. Fortune's Formula. New York: Hill and Wang.

Ritter, Jay. 2003. “Behavioral Finance.” Pacific-Basin Finance Journal 11(4, September): 429–437.

Taleb, Nassim. 2007. The Black Swan: The Impact of the Highly Improbable. Random House.

Thaler, Richard. 1994. The Winner's Curse. Princeton, NJ: Princeton University Press.

Thorp, Edward. 1997. “The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market.” Handbook of Asset and Liability Management, Volume I, Zenios and Ziemba (eds.). Elsevier 2006. www.EdwardOThorp.com.

### NOTES

1 This example was reproduced with corrections from my blog article “Maximizing Compounded Rate of Return,” which you can find at epchan.blogspot.com/2006/10/maximizing-compounded-rate-ofreturn.html.

\* This commentary was reproduced from my blog article of the same title at predictnow.ai/blog/loss-aversion-is-not-a-behavioral-bias/ \* This commentary was reproduced from my blog article of the same title at predictnow.ai/blog/loss-aversion-is-not-a-behavioral-bias/