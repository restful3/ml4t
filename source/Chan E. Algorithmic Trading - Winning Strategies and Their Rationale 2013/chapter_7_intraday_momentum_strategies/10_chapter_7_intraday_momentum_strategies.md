# Intraday Momentum Strategies

In the preceding chapter we saw that most instruments, be they stocks or futures, exhibit cross-sectional momentum, and often time-series momentum as well. Unfortunately, the time horizon of this momentum behavior tends to be long—typically a month or longer. Long holding periods present two problems: They result in lower Sharpe ratios and backtest statistical significance because of the infrequent independent trading signals, and they suffer from underperformance in the aftermath of financial crises. In this chapter, we describe short-term, intraday momentum strategies that do not suffer these drawbacks.

We previously enumerated four main causes of momentum. We will see that all but one of them also operate at the intraday time frame. (The only exception is the persistence of roll return, since its magnitude and volatility are too small to be relevant intraday.)

There is an additional cause of momentum that is mainly applicable to the short time frame: the triggering of stops. Such triggers often lead to the so-called breakout strategies. We'll see one example that involves an entry at the market open, and another one that involves intraday entry at various support or resistance levels.

Intraday momentum can be triggered by specific events beyond just price actions. These events include corporate news such as earnings announcements or analyst recommendation changes, as well as macro-economic news. That

these events generate time series momentum has long been known, but I present some new research on the eff ects of each specifi c category of events.

Intraday momentum can also be triggered by the actions of large funds. I examine how the daily rebalancing of leveraged ETFs leads to short-term momentum.

Finally, at the shortest possible time scale, the imbalance of the bid and ask sizes, the changes in order fl ow, or the aforementioned nonuniform distribution of stop orders can all induce momentum in prices. Some of the common high-frequency trading tactics that take advantage of such momentum will be presented in this chapter.

## ■ **Opening Gap Strategy**

In Chapter 4, we discussed a mean-reverting buy-on-gap strategy for stocks. The opposite momentum strategy will sometimes work on futures and currencies: buying when the instrument gaps up, and shorting when it gaps down.

After being tested on a number of futures, this strategy proved to work best on the Dow Jones STOXX 50 index futures (FSTX) trading on Eurex, which generates an annual percentage rate (APR) of 13 percent and a Sharpe ratio of 1.4 from July 16, 2004, to May 17, 2012. Example 7.1 shows the gap momentum code (available for download as *gapFutures\_FSTX.m*).

#### **Example 7.1: Opening Gap Strategy for FSTX**

This code assumes the open, high, low, and close prices are contained in *T* × 1 arrays *op, hi, lo, cl.* It makes use of utilities function *smartMovingStd* and *backshift* available from epchan.com/book2.

```
entryZscore=0.1;
stdretC2C90d=backshift(1, smartMovingStd(calculateReturns ...
 (cl, 1), 90));
longs=op > backshift(1, hi).*(1+entryZscore*stdretC2C90d);
shorts=op < backshift(1, lo).*(1-entryZscore*stdretC2C90d);
positions=zeros(size(cl));
positions(longs)=1;
positions(shorts)=-1;
ret=positions.*(op-cl)./op;
```

![](_page_2_Figure_2.jpeg)

**FIGURE 7.1** Equity Curve of FSTX Opening Gap Strategy

The same strategy works on some currencies, too. However, the daily "open" and "close" need to be defi ned diff erently. If we defi ne the close to be 5:00 p.m. ET, and the open to be 5:00 a.m. ET (corresponding to the London open), then applying this strategy to GBPUSD yields an APR of 7.2 percent and a Sharpe ratio of 1.3 from July 23, 2007, to February 20, 2012. Naturally, you can experiment with diff erent defi nitions of opening and closing times for diff erent currencies. Most currency markets are closed from 5:00 p.m. on Friday to 5:00 p.m. on Sunday, so that's a natural "gap" for these strategies.

What's special about the overnight or weekend gap that sometimes triggers momentum? The extended period without any trading means that the opening price is often quite diff erent from the closing price. Hence, stop orders set at diff erent prices may get triggered all at once at the open. The execution of these stop orders often leads to momentum because a cascading eff ect may trigger stop orders placed further away from the open price as well. Alternatively, there may be signifi cant events that occurred overnight. As discussed in the next section, many types of news events generate momentum.

## ■ **News-Driven Momentum Strategy**

If, as many people believe, momentum is driven by the slow diff usion of news, surely we can benefi t from the fi rst few days, hours, or even seconds after a newsworthy event. This is the rationale behind traditional post–earnings announcement drift (PEAD) models, as well as other models based on various corporate or macroeconomic news.

# **Post–Earnings Announcement Drift**

There is no surprise that an earnings announcement will move stock price. It is, however, surprising that this move will persist for some time after the announcement, and in the same direction, allowing momentum traders to benefi t. Even more surprising is that though this fact has been known and studied since 1968 (Bernard and Thomas, 1989), the eff ect still has not been arbitraged away, though the duration of the drift may have shortened. What I will show in this section is that as recently as 2011 this strategy is still profitable if we enter at the market open after the earnings announcement was made after the previous close, buying the stock if the return is very positive and shorting if the return is very negative, and liquidate the position at the same day's close. Notice that this strategy does not require the trader to interpret whether the earnings announcement is "good" or "bad." It does not even require the trader to know whether the earnings are above or below analysts' expectations. We let the market tell us whether it thinks the earnings are good or bad.

Before we backtest this strategy, it is necessary to have historical data of the times of earnings annoucements. You can use the function *parseEarnings CalendarFromEarningsDotcom.m* displayed in the box to retrieve one year or so of such data from earnings.com given a certain stock universe specifi ed by the stock symbols array *allsyms*. The important feature of this program is that it carefully selects only earnings announcements occurring after the previous trading day's market close and before today's market open. Earnings announcements occurring at other times should not be triggers for our entry trades as they occur at today's market open.

# **Function for Retrieving Earnings Calendar from earnings.com**

This function takes an input 1xN stock symbols cell array allsyms and creates a 1 × N logical array earnann, which tells us whether (with values true or false) the corresponding stock has an earnings announcement after the previous day's 4:00 P.M. ET (U.S. market closing time) and before today's 9:30 A.M. ET (U.S. market opening time). The inputs prevDate and todayDate should be in yyyymmdd format.

```
function [earnann]= ...
 parseEarningsCalendarFromEarningsDotCom(prevDate, ...
 todayDate, allsyms)
```

**BOX 7.1**

```
% [earnann]==parseEaringsCalendarFromEarningsDotCom
 % (prevDate,todayDate, allsyms)
earnann=zeros(size(allsyms));
prevEarningsFile=urlread(['http://www.earnings.com/earning ...
 .asp?date=', num2str(prevDate), '&client=cb']);
todayEarningsFile=urlread(['http://www.earnings.com ...
 /earning.asp?date=', num2str(todayDate), '&client=cb']);
prevd=day(datenum(num2str(prevDate), 'yyyymmdd'));
todayd=day(datenum(num2str(todayDate), 'yyyymmdd'));
prevmmm=datestr(datenum(num2str(prevDate), 'yyyymmdd'), ...
 'mmm');
todaymmm=datestr(datenum(num2str(todayDate), 'yyyymmdd'), ...
 'mmm');
patternSym='<a\s+href="company.asp\?ticker=([%\*\w\._ ...
 /-]+)&coid';
% prevDate
patternPrevDateTime=['<td align="center"><nobr>', ...
 num2str(prevd), '-', num2str(prevmmm), '([ :\dABPMCO]*) ...
 </nobr>'];
symA=regexp(prevEarningsFile, patternSym , 'tokens');
timeA=regexp(prevEarningsFile, patternPrevDateTime, ...
 'tokens');
symsA=[symA{:}];
timeA=[timeA{:}];
assert(length(symsA)==length(timeA));
isAMC=~cellfun('isempty', regexp(timeA, 'AMC'));
patternPM='[ ]+\d:\d\d[ ]+PM'; % e.g. ' 6:00 PM'
isAMC2=~cellfun('isempty', regexp(timeA, patternPM));
symsA=symsA(isAMC | isAMC2);
[foo, idxA, idxALL]=intersect(symsA, allsyms);
earnann(idxALL)=1;
% today
patternTodayDateTime=['<td align="center"><nobr>', ...
 num2str(todayd), '-', num2str(todaymmm), ...
 '([ :\dABPMCO]*)</nobr>'];
```

(Continued)

```
symA=regexp(todayEarningsFile, patternSym , 'tokens');
timeA=regexp(todayEarningsFile, patternTodayDateTime, ...
 'tokens');
symsA=[symA{:}];
timeA=[timeA{:}];
symsA=symsA(1:length(timeA));
assert(length(symsA)==length(timeA));
isBMO=~cellfun('isempty', regexp(timeA, 'BMO'));
patternAM='[ ]+\d:\d\d[ ]+AM'; % e.g. ' 8:00 AM'
isBMO2=~cellfun('isempty', regexp(timeA, patternAM));
symsA=symsA(isBMO | isBMO2);
[foo, idxA, idxALL]=intersect(symsA, allsyms);
earnann(idxALL)=1;
end
```

We need to call this program for each day in the backtest for the PEAD strategy. We can then concatenate the resulting 1 × *N* earnann arrays into one big historical *T* × *N* earnann array for the *T* days in the backtest.

Assuming that we have compiled the historical earnings announcement logical array, whether using our function above or through other means, the actual backtest program for the PEAD strategy is very simple, as shown in Example 7.2. We just need to compute the 90-day moving standard deviation of previous-close-to-next day's-open return as the benchmark for deciding whether the announcement is "surprising" enough to generate the post announcement drift.

#### **Example 7.2: Backtest of Post-Earnings Annoucement Drift Strategy**

We assume the historical open and close prices are stored in the *T* × *N* arrays *op* and *cl*. The input *T* × *N* logical array *earnann* indicates whether there is an earnings announcement for a stock on a given day prior to that day's market open but after the previous trading day's market close. The utility functions backshift, smartMovingStd and

#### **Example 7.2 (***Continued***)**

smartsum are available for download from epchan.com/book2. The backtest program itself is named *pead.m*.

```
lookback=90;
retC2O=(op-backshift(1, cl))./backshift(1, cl);
stdC2O=smartMovingStd(retC2O, lookback);
positions=zeros(size(cl));
longs=retC2O >= 0.5*stdC2O & earnann;
shorts=retC2O <= -0.5*stdC2O & earnann;
positions(longs)=1;
positions(shorts)=-1;
ret=smartsum(positions.*(cl-op)./op, 2)/30;
```

For a universe of S&P 500 stocks, the APR from January 3, 2011, to April 24, 2012, is 6.7 percent, while the Sharpe ratio is a very respectable 1.5. The cumulative returns curve is displayed in Figure 7.2. Note that we have used 30 as the denominator in calculating returns, since there is a maximum of 30 positions in one day during that backtest period. Of course, there is a certain degree of look-ahead bias in using this number, since we don't know exactly what the maximum will be. But given that the maximum number of

![](_page_6_Figure_6.jpeg)

**FIGURE 7.2** Cumulative Returns Curve of PEAD Strategy

announcements per day is quite predictable, this is not a very grievous bias. Since this is an intraday strategy, it is possible to lever it up by at least four times, giving an annualized average return of close to 27 percent.

You might wonder whether holding these positions overnight will generate additional profi ts. The answer is no: the overnight returns are negative on average. On the contrary, many published results from 10 or 20 years ago have shown that PEAD lasted more than a day. This may be an example where the duration of momentum is shortened due to increased awareness of the existence of such momentum. It remains to be tested whether an even shorter holding period may generate better returns.

# **Drift Due to Other Events**

Besides earnings announcements, there are other corporate events that may exhibit post-announcement drift: An incomplete list includes earnings guidance, analyst ratings and recommendation changes, same store sales, and airline load factors. (A reasonable daily provider of such data is the Dow Jones newswire delivered by Newsware because it has the code specifi c to the type of event attached to each story and is machine readable.) In theory, any announcements that prompt a reevaluation of the fair market value of a company should induce a change in its share price toward a new equilibrium price. (For a recent comprehensive study of all these events and their impact on the stock's post-event returns, see Hafez, 2011.) Among these events, mergers and acquisitions, of course, draw the attention of specialized hedge funds that possess in-depth fundamental knowledge of the acquirer and acquiree corporations. Yet a purely technical model like the one described earlier for PEAD can still extract an APR of about 3 percent for mergers and acquisitions (M&As). (It is interesting to note that contrary to common beliefs, Hafez found that the acquiree's stock price falls more than the acquirer's after the initial announcement of the acquisition.)

In Chapter 6, we described how momentum in a stock's price is generated by large funds' forced buying or selling of the stock. For index funds (whether mutual or exchange traded), there is one type of forced buying and selling that is well known: index composition changes. When a stock is added to an index, expect buying pressure, and vice versa when a stock is deleted from an index. These index rebalancing trades also generate momentum immediately following the announced changes. Though some researchers have reported that such momentum used to last many days, my

own testing with more recent data suggests that the drift horizon has also been reduced to intraday (Shankar and Miller, 2006).

While we are on the subject of momentum due to scheduled announcements, what about the impact of macroeconomic events such as Federal Open Market Committee's rate decisions or the release of the latest consumer price index? I have tested their eff ects on EURUSD, but unfortunately have found no signifi cant momentum. However, Clare and Courtenay reported that U.K. macroeconomic data releases as well as Bank of England interest rate announcements induced momentum in GBPUSD for up to at least 10 minutes after the announcements (Clare and Courtnenay, 2001). These results were based on data up to 1999, so we should expect that the duration of this momentum to be shorter in recent years, if the momentum continues to exist at all.

## ■ **Leveraged ETF Strategy**

Imagine that you have a portfolio of stocks that is supposed to track the MSCI US REIT index (RMZ), except that you want to keep the leverage of the portfolio at 3, especially at the market close. As I demonstrate in Example 8.1, this constant leverage requirement has some counterintuitive and important consequences. Suppose the RMZ dropped precipitously one day. That would imply that you would need to substantially reduce the positions in your portfolio by selling stocks across the board in order to keep the leverage constant. Conversely, if the RMZ rose that day, you would need to increase the positions by buying stocks.

Now suppose you are actually the sponsor of an ETF, and that portfolio of yours is none other than a 3× leveraged ETF such as DRN (a real estate ETF), and its equity is over a hundred million dollars. If you think that this rebalancing procedure (selling the component stocks when the portfolio's return is negative, and vice versa) near the market close would generate momentum in the market value of the portfolio, you would be right.

(A large change in the market index generates momentum in the same direction for both leveraged long or short ETFs. If the change is positive, a short ETF would experience a decrease in equity, and its sponsor would need to reduce its short positions. Therefore, it would also need to buy stocks, just as the long ETF would.)

We can test this hypothesis by constructing a very simple momentum strategy: buy DRN if the return from previous day's close to 15 minutes before market close is greater than 2 percent, and sell if the return is smaller than −2 percent. Exit the position at the market close. Note that this momentum strategy is based on the momentum of the underlying stocks, so it should be aff ecting the near-market-close returns of the unlevered ETFs such as SPY as well. We use the leveraged ETFs as trading instruments simply to magnify the eff ect. The APR of trading DRN is 15 percent with a Sharpe ratio of 1.8 from October 12, 2011, to October 25, 2012.

Naturally, the return of this strategy should increase as the aggregate assets of all leveraged ETFs increase. It was reported that the total AUM of leveraged ETFs (including both long and short funds) at the end of January 2009 is \$19 billion (Cheng and Madhavan, 2009). These authors also estimated that a 1 percent move of SPX will necessitate a buying or selling of stocks constituting about 17 percent of the market-on-close volume. This is obviously going to have signifi cant market impact, which is momentum inducing. (A more updated analysis was published by Rodier, Haryanto, Shum, and Hejazi, 2012.)

There is of course another event that will aff ect the equity of an ETF, leveraged or not: the fl ow of investors' cash. A large infl ow into long leveraged ETFs will cause positive momentum on the underlying stocks' prices, while a large infl ow into short leveraged ("inverse") ETFs will cause negative momentum. So it is theoretically possible that on the same day when the market index had a large positive return many investors sold the long leveraged ETFs (perhaps as part of a mean-reverting strategy). This would have neutralized the momentum. But our backtests show that this did not happen often.

## ■ **High-Frequency Strategies**

Most high-frequency momentum strategies involve extracting information from the order book, and the basic idea is simple: If the bid size is much bigger than the ask size, expect the price to tick up and vice versa. This idea is backed by academic research. For example, an approximately linear relationship between the imbalance of bid versus ask sizes and short-term price changes in the Nasdaq market was found (Maslov and Mills, 2001). As expected, the eff ect is stronger for lower volume stocks. The eff ect is not limited to just the national best bid off er (NBBO) prices: an imbalance of the entire order book also induces price changes for a stock on the Stockholm stock market (Hellström and Simonsen, 2006).

There are a number of high-frequency momentum strategies based on this phenomenon. Many of those were described in books about market microstructure or high-frequency trading (Arnuk and Saluzzi, 2012; Durbin, 2010; Harris, 2003; and Sinclair, 2010). (In my descriptions that follow, I focus on making an initial long trade, but, of course, there is a symmetrical opportunity on the short side.)

In markets that fi ll orders on a pro-rata basis such as the Eurodollar futures trading on CME, the simplest way to benefi t from this expectation is just to "join the bid" immediately, so that whenever there is a fill on the bid side, we will get allocated part of that fill. To ensure that the bid and ask prices are more likely to move higher rather than lower after we are fi lled, we join the bid only when the original bid size is much larger than the ask size. This is called the *ratio trade*, because we expect the proportion of the original order to be filled is equal to the ratio between our own order size and the aggregate order size at the bid price. Once the buying pressure causes the bid price to move up one or more ticks, then we can sell at a profi t, or we can simply place a sell order at the best ask (if the bid-ask spread is larger than the round trip commission per share). If the bid price doesn't move up or our sell limit order doesn't get fi lled, we can probably still sell at the original best bid price because of the large bid size, with the loss of commissions only.

In markets where the bid-ask spread is bigger than two ticks, there is another simple trade to benefi t from the expectation of an uptick. Simply place the buy order at the best bid plus one tick. If this is fi lled, then we place a sell order at the best ask minus one tick and hope that it is fi lled. But if it is not, we can probably still sell it at the original best bid, with the loss of commissions plus one tick. This is called *ticking* or *quote matching*. For this trade to be profi table, we need to make sure that the round trip commission per share is less than the bid-ask spread minus two ticks. This strategy is illustrated in Figure 7.3.

![](_page_10_Figure_5.jpeg)

**FIGURE 7.3** Ticking Strategy. The original spread must be greater than two ticks. After the buy order is fi lled at B, we will try to sell it at S for a profi t of at least one tick. But if the sell order cannot be fi lled, then we will sell it at S′ at a loss of one tick.

(Ticking is not a foolproof strategy, of course. The original best bid before it was front-run may be cancelled if the trader knows that he has been front-run, leaving us with a lower bid price to unload our inventory. Or the whole situation could be set up as a trap for us: the trader who placed the original best bid actually wanted to sell us stocks at a price better than her own bid. So once we bought her stocks plus one tick, she would immediately cancel the bid.)

Even when there is no preexisting buying pressure or bid-ask size imbalance, we can create the illusion of one (often called *momentum ignition*). This works for markets with time priority for orders instead of using pro-rata fills. Let's assume we start with very similar best bid and ask sizes. We will place a large buy limit order at the best bid to create the impression of buying pressure, and simultaneously place a small sell limit order at the best ask. This would trick traders to buy at the ask price since they anticipate an uptick, fi lling our small sell order. At this point, we immediately cancel the large buy order. The best bid and ask sizes are now roughly equal again. Many of those traders who bought earlier expecting a large buying pressure may now sell back their holdings at a loss, and we can then buy them at the original best bid. This is called *fl ipping*.

There is a danger to creating the illusion of buying pressure—somebody just might call our bluff and actually fi ll our large buy order. In this case, we might have to sell it at a loss. Conversely, if we suspect a large buy order is due to fl ippers, then we can sell to the fl ippers and drive down the bid price. We hope that the fl ippers will capitulate and sell their new inventory, driving the ask price down as well, so that we can then cover our short position below the original bid price. How do we know that the large buy order is due to fl ippers in the fi rst place? We may have to record how often a large bid gets canceled instead of getting fi lled. If you subscribe to the private data feeds from the exchanges such as ITCH from Nasdaq, EDGX Book Feed from Direct Edge, or the PITCH feed from BATS, you will receive the detailed life history of an order including any modifi cations or partial fi lls (Arnuk and Saluzzi, 2012). Such information may help you detect fl ippers as well.

All these strategies and their defenses, bluff s, and counterbluff s illustrate the general point that high-frequency traders can profi t only from slower traders. If only high-frequency traders are left in the market, the net average profi t for everyone will be zero. Indeed, because of the prevalence of these types of high-frequency strategies that "front-run" large bid or ask orders, many traditional market makers no longer quote large sizes. This has led to a general decrease of the NBBO sizes across many markets. For example, even in highly liquid stocks such as AAPL, the NBBO sizes are often just a few hundred shares. And even for the most liquid ETFs such as SPY on ARCA, the NBBO sizes are often fewer than 10,000 shares. Only after these small orders are filled will the market maker go back to requote at the same prices to avoid being taken advantage of by the high-frequency traders. (Of course, there are other reasons for avoiding displaying large quotes: market makers do not like to keep large inventories that can result from having their large quotes filled.) Similarly, large institutional orders that were formerly executed as block trades are now broken up into tiny child orders to be scattered around the diff erent market venues and executed throughout the day.

*Stop hunting* is another favorite high-frequency momentum strategy. Research in the currencies markets indicated that once support (resistance) levels are breached, prices will go further down (up) for a while (Osler, 2000, 2001). These support and resistance levels can be those reported daily by banks or brokerages, or they can just be round numbers in the proximity of the current price levels. This short-term price momentum occurs because of the large number of stop orders placed at or near the support and resistance levels.

To understand this further, let's just look at the support levels, as the situation with resistance levels is symmetrical. Once the price drops enough to breach a support level, those sell stop orders are triggered and thereby drive the prices down further. Given this knowledge, high-frequency traders can, of course, create artifi cial selling pressure by submitting large sell orders when the price is close enough to a support level, hoping to drive the next tick down. Once the stop orders are triggered and a downward momentum is in force, these high-frequency traders can cover their short positions for a quick profi t.

If we have access to the *order fl ow* information of a market, then we have a highly valuable information stream that goes beyond the usual bid/ask/last price stream. As Lyons discussed in the context of currencies trading, "order fl ow" is signed transaction volume (Lyons, 2001). If a trader buys 100 units from a dealer/market maker/order book, the order fl ow is 100, and it is −100 if the trader sells 100 units instead. What "buying" from an order book means is that a trader buys at the ask price, or, equivalently, the trader submits a market order to buy. Empirical research indicates that order fl ow information is a good predictor of price movements. This is because market makers can distill important fundamental information from order fl ow information, and set the bid-ask prices accordingly. For example, if a major hedge fund just learns about a major piece of breaking news, their algorithms will submit large market orders of the same sign in a split second. A market maker monitoring

the order fl ow will deduce, quite correctly, that such large one-directional demands indicate the presence of informed traders, and they will immediately adjust their bid-ask prices to protect themselves. The urgency of using market orders indicates that the information is new and not widely known.

Since most of us are not large market makers or operators of an exchange, how can we access such order fl ow information? For stocks and futures markets, we can monitor and record every tick (i.e., changes in best bid, ask, and transaction price and size), and thus determine whether a transaction took place at the bid (negative order fl ow) or at the ask (positive order fl ow). For the currencies market, this is diffi cult because most dealers do not report transaction prices. We may have to resort to trading currency futures for this strategy. Once the order fl ow per transaction is computed, we can easily compute the cumulative or average order fl ow over some look-back period and use that to predict whether the price will move up or down.

#### **KEY POINTS**

- Intraday momentum strategies do not suffer from many of the disadvantages of interday momentum strategies, but they retain some key advantages.
- "Breakout" momentum strategies involve a price exceeding a trading range.
- The opening gap strategy is a breakout strategy that works for some futures and currencies.
- Breakout momentum may be caused by the triggering of stop orders.
- Many kinds of corporate and macroeconomic news induce short-term price momentum.
- Index composition changes induce momentum in stocks that are added to or deleted from the index.
- Rebalancing of leveraged ETFs near the market close causes momentum in the underlying index in the same direction as the market return from the previous close.
- Many high-frequency momentum strategies involve imbalance between bid and ask sizes, an imbalance that is sometimes artifi cially created by the high-frequency traders themselves.
- Stop hunting is a high-frequency trading strategy that relies on triggering stop orders that typically populate round numbers near the current market price.
- Order fl ow can predict short-term price movement in the same direction.