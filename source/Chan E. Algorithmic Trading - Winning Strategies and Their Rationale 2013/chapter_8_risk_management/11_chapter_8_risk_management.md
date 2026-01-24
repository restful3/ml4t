# Risk Management

Risk management means different things to different people. To novice traders, risk management is driven by "loss aversion": we simply don't like the feeling of losing money. In fact, research has suggested that the average human being needs to have the potential for making \$2 to compensate for the risk of losing \$1, which may explain why a Sharpe ratio of 2 is so emotionally appealing (Kahneman, 2011). However, this dislike of risk in itself is not rational. Our goal should be the maximization of long-term equity growth, and we avoid risk only insofar as it interferes with this goal. Risk management in this chapter is based on this objective.

The key concept in risk management is the prudent use of leverage, which we can optimize via the Kelly formula or some numerical methods that maximize compounded growth rate. But sometimes reality forces us to limit the maximum drawdown of an account. One obvious way of accomplishing this is the use of stop loss, but it is often problematic. The other way is constant proportion portfolio insurance, which tries to maximize the upside of the account in addition to preventing large drawdowns. Both will be discussed here. Finally, it may be wise to avoid trading altogether during times when the risk of loss is high. We will investigate whether the use of certain leading indicators of risk is an effective loss-avoidance technique.

# ■ **Optimal Leverage**

It is easy to say that we need to be prudent when using leverage, but much harder to decide what constitutes a prudent, or optimal, leverage for a particular strategy or portfolio because, obviously, if we set leverage to zero, we will suff er no risks but will generate no returns, either.

To some portfolio managers, especially those who are managing their own money and answerable to no one but themselves, the sole goal of trading is the maximization of net worth over the long term. They pay no mind to drawdowns and volatilities of returns. So the optimal leverage to them means one that can maximize the net worth or, equivalently, the compounded growth rate.

We'll discuss here three methods of computing the optimal leverage that maximizes the compounded growth rate. Each method has its own assumptions and drawbacks, and we try to be agnostic as to which method you should adopt. But, in all cases, we have to make the assumption that the future probability distribution of returns of the *market* is the same as in the past. This is usually an incorrect assumption, but this is the best that quantitative models can do. Even more restrictive, many risk management techniques assume further that the probability distribution of returns of the *strategy* itself is the same as in the past. And fi nally, the most restrictive of all assumes that the probability distribution of returns of the strategy is Gaussian. As is often the case in mathematical modeling, the most restrictive assumptions give rise to the most elegant and simple solution, so I will start this survey with the Kelly formula under the Gaussian assumption.

If the maximum drawdown of an account with a certain leverage is −100 percent, this leverage cannot be optimal because the compounded growth rate will also be −100 percent. So an optimal leverage implies that we must not be ruined (equity reaching zero) at any point in history, rather selfevidently! But sometimes our risk managers (perhaps it is a spouse for independent traders) tell us that we are allowed to have a much smaller magnitude of drawdown than 1. In this case, the maximum drawdown allowed forms an additional constraint in the leverage optimization problem.

No matter how the optimal leverage is determined, the one central theme is that the leverage should be kept constant. This is necessary to optimize the growth rate whether or not we have the maximum drawdown constraint. Keeping a constant leverage may sound rather mundane, but can be counterintuitive when put into action. For example, if you have a long stock portfolio, and your profi t and loss (P&L) was positive in the last trading period, the constant leverage requirement forces you to buy more stocks for

### **Example 8.1: The Implications of the Constant Leverage Requirement**

The central requirement for all ways of optimizing leverage described in this chapter is that the leverage be kept constant at all times. This can have some counterintuitive consequences.

If you started with \$100K equity in your account, and your strategy's optimal leverage was determined to be 5, then you should have a portfolio with market value of \$500K.

If, however, you lost \$10K in one day and your equity was reduced to \$90K, with a portfolio market value of \$490K, then you need to liquidate a *further* \$40K of your portfolio so that its updated market value became 5 × \$90K = \$450K. This selling into the loss may make some people uncomfortable, but it is a necessary part of many risk management schemes.

Suppose you then gained \$20K the next day. What should your portfolio market value be? And what should you do to achieve that market value?

The new portfolio market value should be 5 × (\$90K + \$20K) = \$550K. Since your current portfolio market value was just \$450K + \$20K = \$470K, this means you need to add \$80K worth of (long or short) securities to the portfolio. Hopefully, your broker will lend you the cash to buy all these extra securities!

this period. However, if your P&L was negative in the last period, it forces you to sell stocks into the loss. Example 8.1 illustrates this.

Many analysts believe that this "selling into losses" feature of the risk management techniques causes contagion in fi nancial crises. (In particular, this was cited as a cause of the August 2007 meltdown of quant funds; see Khandani and Lo, 2007). This is because often many funds are holding similar positions in their portfolios. If one fund suff ers losses, perhaps due to some unrelated strategies, it is prone to liquidate positions across all its portfolios due to the constant leverage requirement, causing losses for all other funds that hold those positions. The losses force all these other funds to also liquidate their positions and thus exacerbate the losses for everyone: a vicious cycle. One might think of this as a tragedy of the commons: self-preservation ("risk management") for one fund can lead to catastrophe for all.

# **Kelly Formula**

If one assumes that the probability distribution of returns is Gaussian, the Kelly formula gives us a very simple answer for optimal leverage *f:* 

$$f = m / s^2 \qquad (8.1)$$

where *m* is the mean excess return, and *s* 2 is the variance of the excess returns.

One of the best expositions of this formula can be found in Edward Thorp's (1997) paper, and I also devoted an entire chapter in *Quantitative Trading* (Chan, 2009) to it. It can be proven that if the Gaussian assumption is a good approximation, then the Kelly leverage *f* will generate the highest compounded growth rate of equity, assuming that all profi ts are reinvested. However, even if the Gaussian assumption is really valid, we will inevitably suff er estimation errors when we try to estimate what the "true" mean and variance of the excess return are. And no matter how good one's estimation method is, there is no guarantee that the future mean and variance will be the same as the historical ones. The consequence of using an overestimated mean or an underestimated variance is dire: Either case will lead to an overestimated optimal leverage, and if this overestimated leverage is high enough, it will eventually lead to ruin: equity going to zero. However, the consequence of using an underestimated leverage is merely a submaximal compounded growth rate. Many traders justifi ably prefer the later scenario, and they routinely deploy a leverage equal to half of what the Kelly formula recommends: the so-called half-Kelly leverage.

My actual experience using Kelly's optimal leverage is that it is best viewed as an upper bound rather than as the leverage that must be used. Often, the Kelly leverage given by the backtest (or a short period of walkforward test) is so high that it far exceeds the maximum leverage allowed by our brokers. At other times, the Kelly leverage would have bankrupted us even in backtest, due to the non-Gaussian distributions of returns. In other words, the maximum drawdown in backtest is −1 using the Kelly leverage, which implies setting the leverage by numerically optimizing the growth rate using a more realistic non-Gaussian distribution might be more practical. Alternatively, we may just optimize on the empirical, historical returns. These two methods will be discussed in the next sections.

But just using Kelly optimal leverage as an upper bound can sometimes provide interesting insights. For example, I once calculated that both the Russell 1000 and 2000 indices have Kelly leverage at about 1.8. But exchange-traded fund (ETF) sponsor Direxion has been marketing triple leveraged ETFs BGU and TNA tracking these indices. By design, they have a leverage of 3. Clearly, there is a real danger that the net asset value (NAV) of these ETFs will go to zero. Equally clearly, no investors should buy and hold these ETFs, as the sponsor itself readily agrees.

There is another usage of the Kelly formula besides setting the optimal leverage: it also tells us how to optimally allocate our buying power to different portfolios or strategies. Let's denote F as a column vector of optimal leverages that we should apply to the different portfolios based on a common pool of equity. (For example, if we have \$1 equity, then  $F = [3.2 \ 1.5]^T$  means the first portfolio should have a market value of \$3.2 while the second portfolio should have a market value of \$1.5. The T signifies matrix transpose.) The Kelly formula says

$$F = C^{-1}M \qquad (8.2)$$

where C is the covariance matrix of the returns of the portfolios and M is the mean excess returns of these portfolios.

There is an extensive example on how to use this formula in *Quantitative Trading*. But what should we do if our broker has set a maximum leverage  $F_{max}$  that is smaller than the total gross leverage  $\sum_{i=1}^{n} |F_{i}|$ ? (We are concerned with the gross leverage, which is equal to the absolute sum of the long and short market values divided by our equity, not the net leverage, which is the net of the long and short market values divided by our equity.) The usual recommendation is to multiply all  $F_{i}$  by the factor  $F_{max}/\sum_{i=1}^{n} |F_{i}|$  so that the total gross leverage is equal to  $F_{max}$ . The problem with this approach is that the compounded growth rate will no longer be optimal under this maximum leverage constraint. I have constructed Example 8.2 to demonstrate this. The upshot of that example is that when  $F_{max}$  is much smaller than  $\sum_{i=1}^{n} |F_{i}|$ , it is often optimal (with respect to maximizing the growth rate) to just invest most or all our buying power into the portfolio or strategy with the highest mean excess return.

# Optimization of Expected Growth Rate Using Simulated Returns

If one relaxes the Gaussian assumption and substitutes another analytic form (e.g., Student's t) for the returns distribution to take into account the fat tails, we can still follow the derivations of the Kelly formula in Thorp's

#### Example 8.2: Optimal Capital Allocation Under a Maximum Leverage Constraint

When we have multiple portfolios or strategies, the Kelly formula says that we should invest in each portfolio i with leverage  $F_i$  determined by Equation 8.2. But often, the total gross leverage  $\sum_{i}^{n} |F_i|$  computed this way exceeds the maximum leverage  $F_{max}$  imposed on us by our brokerage or our risk manager. With this constraint, it is often not optimal to just multiply all these  $F_i$  by the factor  $F_{max}/\sum_{i}^{n} |F_i|$ , as I will demonstrate here.

Suppose we have two strategies, 1 and 2. Strategy 1 has annualized mean excess return and volatility of 30 percent and 26 percent, respectively. Strategy 2 has annualized mean excess return and volatility of 60 percent and 35 percent, respectively. Suppose further that their returns distributions are Gaussian, and that there is zero correlation between the returns of 1 and 2. So the Kelly leverages for them are 4.4 and 4.9, respectively, with a total gross leverage of 9.3. The annualized compounded growth rate is (Thorp, 1997)

$$g = F^{T}CF/2 = 2.1 \qquad (8.3)$$

where we have also assumed that the risk-free rate is 0. Now, let's say our brokerage tells us that we are allowed a maximum leverage of 2.

![](_page_5_Figure_6.jpeg)

**FIGURE 8.1** Constrained Growth Rate g as Function of  $F_2$ 

So the leverages for the strategies have to be reduced to 0.95 and 1.05, respectively. The growth rate is now reduced to

$$g = \sum_{i=1}^{2} (F_i M_i - F_i^2 s_i^2 / 2) = 0.82 \qquad (8.4)$$

(Equation 8.3 for *g* applies only when the leverages used are optimal.)

But do these leverages really generate the maximum *g* under our maximum leverage constraint? We can fi nd out by setting *F*<sup>1</sup> to *Fmax* − *F*2, and plot *g* as a function of *F*2 over the allowed range 0 to *Fmax* = *F*2.

It is obvious that the growth rate is optimized when *F*<sup>2</sup> = *Fmax* = 2. The optimized *g* is 0.96, which is higher than the 0.82 given in Equation 8.4. This shows that when we have two or more strategies with very diff erent independent growth rates, and when we have a maximum leverage constraint that is much lower than the Kelly leverage, it is often optimal to just apply all of our buying power on the strategy that has the highest growth rate.

paper and arrive at another optimal leverage, though the formula won't be as simple as Equation 8.1. (This is true as long as the distribution has a fi nite number of moments, unlike, for example, the Pareto Levy distribution.) For some distributions, it may not even be possible to arrive at an analytic answer. This is where Monte Carlo simulations can help.

The expected value of the compounded growth rate as a function of the leverage *f* is (assuming for simplicity that the risk-free rate is zero)

$$g(f) = \langle \log(1 + fR) \rangle \qquad (8.5)$$

where 〈…〉 indicates an average over some random sampling of the unlevered return-per-bar *R*(*t*) of the strategy (not of the market prices) based on some probability distribution of *R*. (We typically use daily bars for *R*(*t*), but the bar can be as long or short as we please.) If this probability distribution is Gaussian, then *g*( *f* ) can be analytically reduced to *g*( *f* ) = *fm* − *f* <sup>2</sup> *m*2 /2, which is the same as Equation 8.4 in the single strategy case. Furthermore, the maxima of *g*( *f* ) can of course be analytically determined by taking the derivative of*g*( *f* ) with respect to*f* and setting it to zero. This will reproduce the Kelly formula in Equation 8.1 and will reproduce the maximum growth rate indicated by Equation 8.3 in the single strategy case. But this is not our interest here. We would like to compute Equation 8.5 using a non-Gaussian distribution of *R*.

Even though we do not know the true distribution of *R*, we can use the so-called Pearson system (see [www.mathworks.com/help/toolbox/stats](http://www.mathworks.com/help/toolbox/stats/br5k833-1.html) [/br5k833-1.html](http://www.mathworks.com/help/toolbox/stats/br5k833-1.html) or [mathworld.wolfram.com/PearsonSystem.html\)](http://mathworld.wolfram.com/PearsonSystem.html) to model it. The Pearson system takes as input the mean, standard deviation, skewness, and kurtosis of the empirical distribution of *R*, and models it as one of seven probability distributions expressible analytically encompassing Gaussian, beta, gamma, Student's *t,* and so on. Of course, these are not the most general distributions possible. The empirical distribution might have nonzero higher moments that are not captured by the Pearson system and might, in fact, have infi nite higher moments, as in the case of the Pareto Levy distribution. But to capture all the higher moments invites data-snooping bias due to the limited amount of empirical data usually available. So, for all practical purposes, we use the Pearson system for our Monte Carlo sampling.

We illustrate this Monte Carlo technique by using the mean reversion strategy described in Example 5.1. But fi rst, we can use the daily returns in the test set to easily calculate that the Kelly leverage is 18.4. We should keep this number in mind when comparing with the Monte Carlo results. Next, we use the fi rst four moments of these daily returns to construct a Pearson system and generate 100,000 random returns from this system. We can use the *pearsrnd* function from the MATLAB Statistics Toolbox for this. (The complete code is in *monteCarloOptimLeverage.m.*)

**BOX 8.1** We assume that the strategy daily returns are contained in the Nx1 array ret. We will use the fi rst four moments of ret to generate a Pearson system distribution, from which any number of simulated returns ret\_sim can be generated.

```
moments={mean(ret), std(ret), skewness(ret), kurtosis(ret)};
[ret_sim, type]=pearsrnd(moments{:}, 100000, 1);
```

In the code, *ret* contains the daily returns from the backtest of the strategy, whereas *ret\_sim* are 100,000 randomly generated daily returns with the same four moments as *ret*. The *pearsrnd* function also returns *type*, which indicates which type of distribution fi ts our data best. In this example, *type* is 4, indicating that the distribution is not one of the standard ones such as Student's *t.* (But we aren't at all concerned whether it has a name.) Now we can use *ret\_sim* to compute the average of*g*( *f* ). In our code,*g*( *f* ) is an inline function with leverage *f* and a return series *R* as inputs.

**BOX 8.2**

An inline function for calculating the compounded growth rate based on leverage f and return per bar of R.

```
g=inline('sum(log(1+f*R))/length(R)', 'f', 'R');
```

Plotting *g*( *f* ) for*f* = 0 to *f* = 23 reveals that *g*( *f* ) does in fact have a maximum somewhere near 19 (see Figure 8.2), and a numerical optimization using the *fminbnd* function of the MATLAB Optimization Toolbox yields an optimal *f* of 19, strikingly close to the Kelly's optimal *f* of 18.4!

**BOX 8.3** Finding the minimum of the negative of the growth rate based on leverage f and the simulated returns ret\_sim (same as fi nding the maximum of the positive growth rate).

```
minusGsim=@(f)-g(f, ret_sim);
optimalF=fminbnd(minusGsim, 0, 24);
```

Of course, if you run this program with a diff erent random seed and therefore diff erent series of simulated returns, you will fi nd a somewhat diff erent value for the optimal *f,* but ideally it won't be too diff erent from my value. (As a side note, the only reason we minimized −*g* instead of maximized *g* is that MATLAB does not have a *fmaxbnd* function.)

There is another interesting result from running this Monte Carlo optimization. If we try *f* of 31, we shall fi nd that the growth rate is −1; that is, ruin. This is because the most negative return per period is −0.0331, so any leverage higher than 1 / 0.0331 = 30.2 will result in total loss during that period.

# **Optimization of Historical Growth Rate**

Instead of optimizing the expected value of the growth rate using our analytical probability distribution of returns as we did in the previous section,

![](_page_9_Figure_2.jpeg)

**FIGURE 8.2** Expected Growth Rate *g* as Function of*f*.

one can of course just optimize the historical growth rate in the backtest with respect to the leverage. We just need one particular realized set of returns: that which actually occurred in the backtest. This method suff ers the usual drawback of parameter optimization in backtest: data-snooping bias. In general, the optimal leverage for this particular historical realization of the strategy returns won't be optimal for a diff erent realization that will occur in the future. Unlike Monte Carlo optimization, the historical returns off er insuffi cient data to determine an optimal leverage that works well for many realizations.

Despite these caveats, brute force optimization over the backtest returns sometimes does give a very similar answer to both the Kelly leverage and Monte Carlo optimization. Using the same strategy as in the previous section, and altering the optimization program slightly to feed in the historical returns *ret* instead of the simulated returns *ret\_sim*.

**BOX 8. 4**

Finding the minimum of the negative of the growth rate based on leverage f and the historical returns ret.

```
minusG=@(f)-g(f, ret);
optimalF=fminbnd(minusG, 0, 21);
```

we obtain the optimal f of 18.4, which is again the same as the Kelly optimal f.

# **Maximum Drawdown**

For those portfolio managers who manage other people's assets, maximizing the long-term growth rate is not the only objective. Often, their clients (or employers) will insist that the absolute value of the drawdown (return calculated from the historic high watermark) should never exceed a certain maximum. That is to say, they dictate what the maximum drawdown can be. This requirement translates into an additional constraint into our leverage optimization problem.

Unfortunately, this translation is not as simple as multiplying the unconstrained optimal leverage by the ratio of the maximum drawdown allowed and the original unconstrained maximum drawdown. Using the example in the section on optimization of expected growth rate with simulated returns *ret\_sim*, the maximum drawdown is a frightening –0.999. This is with an unconstrained optimal *f* of 19.2. Suppose our risk manager allows a maximum drawdown of only half this amount. Using half the optimal *f* of 9.6 still generates a maximum drawdown of –0.963. By trial and error, we fi nd that we have to lower the leverage by a factor of 7, to 2.7 or so, in order to reduce the magnitude of the maximum drawdown to about 0.5. (Again, all these numbers depend on the exact series of simulated returns, and so are not exactly reproducible.)

**BOX 8.5**

Using my function calculateMaxDD (available on [http://epchan.com/book2\)](http://epchan.com/book2) to compute maximum drawdowns with different leverages on the same simulated returns series ret\_sim.

maxDD=calculateMaxDD(cumprod(1+optimalF/7\*ret\_sim)-1);

Of course, setting the leverage equal to this upper bound will only prevent the simulated drawdown from exceeding the maximum allowed, but it will not prevent our future drawdown from doing so. The only way to guarantee that the future drawdown will not exceed this maximum is to either use constant proportion insurance or to impose a stop loss. We will discuss these techniques in the next two sections.

It is worth noting that this method of estimating the maximum drawdown is based on a simulated series of strategy returns, not the historical strategy returns generated in a backtest. We can, of course, use the historical strategy returns to calculate the maximum drawdown and use that to determine the optimal leverage instead. In this case, we will fi nd that we just need to decrease the unconstrained optimal *f* by a factor of 1.5 (to 13) in order to reduce the maximum drawdown to below −0.49.

Which method should we use? The advantage of using simulated returns is that they have much better statistical signifi cance. They are akin to the value-at-risk (VaR) methodology used by major banks or hedge funds to determine the likelihood that they will lose a certain amount of money over a certain period. The disadvantage is the maximum drawdown that occurs in the simulation may be so rare that it really won't happen more than once in a million years (a favorite excuse for fund managers when they come to grief). Furthermore, the simulated returns inevitably miss some crucial serial correlations that may be present in the historical returns and that may persist into the future. These correlations may be reducing the maximum drawdown in the real world. The advantage of using the historical strategy returns is that they fully capture these correlations, and furthermore the drawdown would cover a realistic life span of a strategy, not a million years. The disadvantage is, of course, that the data are far too limited for capturing a worst-case scenario. A good compromise may be a leverage somewhere in between those generated by the two methods.

# ■ **Constant Proportion Portfolio Insurance**

The often confl icting goals of wishing to maximize compounded growth rate while limiting the maximum drawdown have been discussed already. There is one method that allows us to fulfi ll both wishes: constant proportion portfolio insurance (CPPI).

Suppose the optimal Kelly leverage of our strategy is determined to be *f*. And suppose we are allowed a maximum drawdown of −D. We can simply set aside D of our initial total account equity for trading, and apply a leverage of*f* to this subaccount to determine our portfolio market value. The other 1 − D of the account will be sitting in cash. We can then be assured that we won't lose all of the equity of this subaccount, or, equivalently, we won't suff er a drawdown of more than −D in our total account. If our trading strategy is profi table and the total account equity reaches a new high water mark, then we can reset our subaccount equity so that it is again D of the total equity, moving some cash back to the "cash" account. However, if the strategy suff ers losses, we will not transfer any cash between the cash and the trading subaccount. Of course, if the losses continue and we lose all the equity in the trading subaccount, we have to abandon the strategy because it has reached our maximum allowed drawdown of −D. Therefore, in addition to limiting our drawdown, this scheme serves as a graceful, principled way to wind down a losing strategy. (The more common, less optimal, way to wind down a strategy is driven by the emotional breakdown of the portfolio manager.)

Notice tha t because of this separation of accounts, this scheme is *not* equivalent to just using a leverage of *L* = *f*D in our total account equity. There is no guarantee that the maximum drawdown will not exceed −D even with a lowered leverage of*f*D. Even if we were to further impose a stop loss of −D, or if the drawdown never went below −D, applying the leverage of*f*D to the full account still won't generate the exact same compounded return as CPPI, unless every period's returns are positive (i.e., maximum drawdown is zero). As long as we have a drawdown, CPPI will decrease order size much faster than the alternative, thus making it almost impossible (due to the use of Kelly leverage on the subaccount) that the account would approach the maximum drawdown –D.

I don't know if there is a mathematical proof that CPPI will be the same as using a leverage of*f*D in terms of the long-run growth rate, but we can use the same simulated returns in the previous sections to demonstrate that after 100,000 days, the growth rate of CPPI is very similar to the alternative scheme: 0.002484 versus 0.002525 per day in one simulation with D = 0.5. The main advantage of CPPI is apparent only when we look at the maximum drawdown. By design, the magnitude of the drawdown in CPPI is less than 0.5, while that of the alternative scheme without using stop loss is a painful 0.9 even with just half of the optimal leverage. The code for computing the growth rate using CPPI is shown in Box 8.6.

#### **Computing Growth Rate Using CPPI**

**B**

**OX 8.6**

Assume the return series is ret\_sim and the optimal leverage is optimalF, both from previous calculations. Also assume the maximum drawdown allowed is –D = –0.5.

```
g_cppi=0;
drawdown=0;
D=0.5;
for t=1:length(ret_sim)
 g_cppi=g_cppi+log(1+ ret_sim (t)*D*optimalF*(1+drawdown));
 drawdown=min(0, (1+drawdown)*(1+ ret_sim (t))-1);
end
g_cppi=g_cppi/length(ret_sim);
```

Note that this scheme should only be applied to an account with one strategy only. If it is a multistrategy account, it is quite possible that the profi table strategies are "subsidizing" the nonprofi table ones such that the drawdown is never large enough to shut down the complete slate of strategies. This is obviously not an ideal situation unless you think that the losing strategy will somehow return to health at some point.

There is one problem with using CPPI, a problem that it shares with the use of stop loss: It can't prevent a big drawdown from occurring during the overnight gap or whenever trading in a market has been suspended. The purchases of out-of-the-money options prior to an expected market close can eliminate some of this risk.

# ■ **Stop Loss**

There are two ways to use stop losses. The common usage is to use stop loss to exit an existing position whenever its unrealized P&L drops below a threshold. But after we exit this position, we are free to reenter into a new position, perhaps even one of the same sign, sometime later. In other words, we are not concerned about the cumulative P&L or the drawdown of the strategy.

 The less common usage is to use stop loss to exit the strategy completely when our drawdown drops below a threshold. This usage of stop loss is awkward—it can happen only once during the lifetime of a strategy, and ideally we would never have to use it. That is the reason why CPPI is preferred over using stop loss for the same protection. The rest of this section is concerned with the fi rst, more common usage of stop loss.

Stop loss can only prevent the unrealized P&L from exceeding our selfimposed limit if the market is always open whenever we are holding a position. For example, it is eff ective if we do not hold positions after the market closes or if we are trading in currencies or some futures where the electronic market is always open except for weekends and holidays. Otherwise, if the prices "gap" down or up when the market reopens, the stop loss may be executed at a price much worse than what our maximum allowable loss dictates. As we said earlier, the purchases of options will be necessary to eliminate this risk, but that may be expensive to implement and is valuable only for expected market downtime.

In some extreme circumstances, stop loss is useless even if the market is open but when all liquidity providers decide to withdraw their liquidity

simultaneously. This happened during the fl ash crash of May 6, 2010, since modern-day market makers merely need to maintain a bid of \$0.01 (the infamous "stub quote") in times of market stress (Arnuk and Saluzzi, 2012). This is why an unfortunate sell stop order on Accenture, a company with multibillion-dollar revenue, was executed at \$0.01 per share that day.

But even if the market is open and there is normal liquidity, it is a matter of controversy whether we should impose stop loss for mean-reverting strategies. At fi rst blush, stop loss seems to contradict the central assumption of mean reversion. For example, if prices drop and we enter into a long position, and prices drop some more and thus induce a loss, we should expect the prices to rise eventually if we believe in mean reversion of this price series. So it is not sensible to "stop loss" and exit this position when the price is so low. Indeed, I have never backtested any mean-reverting strategy whose APR or Sharpe ratio is increased by imposing a stop loss.

There is just one problem with this argument: What happens if the mean reversion model has permanently stopped working while we are in a position? In fi nance, unlike in physics, laws are not immutable. As I have been repeating, what was true of a price series before may not be true in the future. So a mean-reverting price series can undergo a regime change and become a trending price series for an extended period of time, maybe forever. In this case, a stop loss will be very eff ective in preventing catastrophic losses, and it will allow us time to consider the possibility that we should just shut down the strategy before incurring a 100 percent loss. Furthermore, these kinds of "turncoat" price series that regime-change from mean reversion to momentum would never show up in our catalog of profi table mean reversion strategies because our catalog would not have included meanreverting strategies that failed in their backtests. Survivorship bias was in action when I claimed earlier that stop loss always lowers the performance of mean-reverting strategies. It is more accurate to say that stop loss always lowers the performance of mean-reverting strategies when the prices *remain mean reverting*, but it certainly improves the performance of those strategies when the prices suff er a regime change and start to trend!

Given this consideration of regime change and survivorship bias, how should we impose a stop loss on a mean-reverting strategy, since any successfully backtested mean-reverting strategy suff ers survivorship bias and will always show lowered performance if we impose a stop loss? Clearly, we should impose a stop loss that is greater than the backtest maximum intraday drawdown. In this case, the stop loss would never have been triggered in the backtest period and could not have aff ected the backtest performance,

yet it can still eff ectively prevent a black swan event in the future from leading to ruin.

In contrast to mean-reverting strategies, momentum strategies benefit from stop loss in a very logical and straightforward way. If a momentum strategy is losing, it means that momentum has reversed, so logically we should be exiting the position and maybe even reversing the position. Thus, a continuously updated momentum trading signal serves as a de facto stop loss. This is the reason momentum models do not present the same kind of tail risk that mean-reverting models do.

# ■ **Risk Indicators**

Many of the risk management measures we discussed above are reactive: We lower the order size when we incur a loss, or we stop trading altogether when a maximum drawdown has been reached. But it would be much more advantageous if we could proactively avoid those periods of time when the strategy is likely to incur loss. This is the role of leading risk indicators.

The obvious distinction between leading risk indicators and the more general notion of risk indicators is that leading risk indicators let us predict whether the next period will be risky for our investment, while general risk indicators are just contemporaneous with a risky period.

There is no one risk indicator that is applicable to all strategies: What is a risky period to one strategy may be a highly profi table period for another. For example, we might try using the VIX, the implied volatility index, as the leading risk indicator to predict the risk of the next-day return of the buy-on-gap stock strategy described in Chapter 4. That strategy had an annualized average return of around 8.7 percent and a Sharpe ratio of 1.5 from May 11, 2006, to April 24, 2012. But if the preceding day's VIX is over 35, a common threshold for highly risky periods, then the day's annualized average return will be 17.2 percent with a Sharpe ratio of 1.4. Clearly, this strategy benefi ts from the socalled risk! However, VIX > 35 is a very good leading risk indicator for the FSTX opening gap strategy depicted in Chapter 7. That strategy had an annualized average return of around 13 percent and a Sharpe ratio of 1.4 from July 16, 2004, to May 17, 2012. If the preceding day's VIX is over 35, then the day's annualized average return drops to 2.6 percent and the Sharpe ratio to 0.16. Clearly, VIX tells us to avoid trading on the following day.

Besides VIX, another commonly used leading risk indicator is the TED spread. It is the diff erence between the three-month London Interbank Off ered Rate (LIBOR) and the three-month T-bill interest rate, and it measures the risk of bank defaults. In the credit crisis of 2008, TED spread rose to a record 457 basis points. Since the credit market is dominated by large institutional players, presumably they are more informed than those indicators based on the stock market where the herd-like instinct of retail investors contributes to its valuation. (The TED spread is useful notwithstanding the fraudulent manipulation of LIBOR rates by the banks to make them appear lower, as discovered by Snider and Youle, 2010. What matters is the relative value of the TED spread over time, not its absolute value.)

There are other risky assets that at diff erent times have served as risk indicators, though we would have to test them carefully to see if they are leading indicators. These assets include high yield bonds (as represented, for example, by the ETF HYG) and emerging market currencies such as the Mexican peso (MXN). During the European debt crisis of 2011, the MXN became particularly sensitive to bad news, even though the Mexican economy remained healthy throughout. Commentators attributed this sensitivity to the fact that traders are using the MXN as a proxy for all risky assets in general.

More recently, traders can also watch the ETF's ONN and OFF. ONN goes up when the market is in a "risk-on" mood; that is, when the prices of risky assets are bid up. ONN basically holds a basket of risky assets. OFF is just the mirror image of ONN. So a high value of OFF may be a good leading risk indicator. At the time of this writing, these ETFs have only about seven months of history, so there is not enough evidence to confi rm that they have predictive value.

As we mentioned in the section on high-frequency trading in Chapter 7, at short time scales, those who have access to order fl ow information can detect a sudden and large change in order fl ow, which often indicates that important information has come into the possession of institutional traders. This large change in order fl ow is negative if the asset in question is risky, such as stocks, commodities, or risky currencies; it is positive if the asset is low risk, such as U.S. treasuries or USD, JPY, or CHF. As we learned before, order fl ow is a predictor of future price change (Lyons, 2001). Thus, order fl ow can be used as a short-term leading indicator of risk before that information becomes more widely dispersed in the market and causes the price to change more.

There are also risk indicators that are very specifi c to a strategy. We mentioned in Chapter 4 that oil price is a good leading risk indicator for the pair trading of GLD versus GDX. Other commodity prices such as that of gold may also be good leading risk indicators for pair trading of ETFs for countries or companies that produce them. Similarly, the Baltic Dry Index may be a good leading indicator for the ETFs or currencies of export-oriented countries.

I should conclude, though, with one problem with the backtesting of leading risk indicators. Since the occurrence of fi nancial panic or crises is relatively rare, it is very easy to fall victim to data-snooping bias when we try to decide whether an indicator is useful. And, of course, no fi nancial indicators can predict natural and other nonfi nancial disasters. As the order fl ow indicator works at higher frequency, it may turn out to be the most useful of them all.

#### **KEY POINTS**

- Maximization of long-term growth rate:
  - Is your goal the maximization of your net worth over the long term? If so, consider using the half-Kelly optimal leverage.
  - Are your strategy returns fat-tailed? You may want to use Monte Carlo simulations to optimize the growth rate instead of relying on Kelly's formula.
  - Keeping data-snooping bias in mind, sometimes you can just directly optimize the leverage based on your backtest returns' compounded growth rate.
  - Do you want to ensure that your drawdown will not exceed a preset maximum, yet enjoy the highest possible growth rate? Use constant proportion portfolio insurance.
- Stop loss:
  - Stop loss will usually lower the backtest performance of mean-reverting strategies because of survivorship bias, but it can prevent black swan events.
  - Stop loss for mean-reverting strategies should be set so that they are never triggered in backtests.
  - Stop loss for momentum strategies forms a natural and logical part of such strategies.
- Risk indicators:
  - Do you want to avoid risky periods? You can consider one of these possible leading indicators of risk: VIX, TED spread, HYG, ONN/OFF, MXN.
  - Be careful of data-snooping bias when testing the effi cacy of leading risk indicators.
  - Increasingly negative order fl ow of a risky asset can be a short-term leading risk indicator.