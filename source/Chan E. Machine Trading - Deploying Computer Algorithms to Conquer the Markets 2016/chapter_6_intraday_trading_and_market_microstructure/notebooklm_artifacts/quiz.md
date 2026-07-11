# Microstructure Quiz

## Question 1
According to the text, why does an intraday strategy typically possess a higher Sharpe ratio than a monthly strategy, assuming they share the same annualized unlevered returns?

- [x] The law of large numbers reduces the variance between realized and expected returns through a higher frequency of independent bets.
- [ ] Intraday strategies are naturally immune to 'black swan' events due to their short holding periods.
- [ ] The ability to apply higher leverage via Kelly's formula directly increases the unlevered Sharpe ratio.
- [ ] Intraday strategies benefit from lower transaction costs due to the higher volume of trades.

**Hint:** Consider how the frequency of 'bets' affects the statistical confidence of the mean return.

## Question 2
If a trader submits a market order for $\$1$ billion in the S&P 500 ES futures market, where the 'at-the-touch' quote size is $\$30$ million, what is the most likely immediate consequence?

- [x] The order will 'walk the book', executing against multiple price levels and incurring significant market impact.
- [ ] The exchange will automatically route the excess order size to dark pools to prevent price slippage.
- [ ] The order will be rejected by the exchange's matching engine due to insufficient quote size at the NBBO.
- [ ] Arbitrageurs will immediately provide the necessary liquidity at the current midprice to capture the order flow.

**Hint:** Think about what happens when a single order consumes all available shares at the best price and continues to fill.

## Question 3
In the context of US stock market microstructure, how does a 'Hide Not Slide' or 'Post No Preference Blind' order function to benefit a liquidity provider?

- [x] It keeps the order hidden at a non-displayed price to avoid routing under Reg NMS, then 'lights up' and gains time priority at the original timestamp once the NBBO shifts.
- [ ] It automatically executes as a market order at the best available price across all 50+ market centers to capture the highest possible rebate.
- [ ] It allows a trader to bypass the SIP feed and execute directly against dark pool liquidity at the midprice.
- [ ] It ensures the order is only executed if the entire size can be filled at the midpoint of the NBBO.

**Hint:** Focus on how an order can stay on a local exchange even if its price currently matches a better offer elsewhere.

## Question 4
What is the primary regulatory justification for a trader to use an Intermarket Sweep Order (ISO) when executing a large trade?

- [x] The trader assumes responsibility for satisfying protected quotes on other exchanges, allowing them to 'walk the book' on a specific exchange without delay.
- [ ] ISOs are the only order type permitted to interact with non-displayed liquidity in dark pools under Reg NMS.
- [ ] Using an ISO guarantees that the order will be executed at the consolidated NBBO midprice regardless of exchange fragmentation.
- [ ] ISO orders provide 'last look' rights to the trader, allowing them to cancel if the market moves adversely within $1$ millisecond.

**Hint:** Consider the relationship between the Order Protection Rule (Rule 611) and the trader's ability to execute at multiple price levels simultaneously.

## Question 5
How does the 'last look' feature in spot currency markets induce adverse selection for liquidity takers?

- [x] Liquidity providers can reject orders after seeing them, typically filling unprofitable trades while rejecting those where the market moves in the taker's favor.
- [ ] It requires the taker to display their order size to the entire market $5$ milliseconds before execution.
- [ ] It forces all market orders to be routed to the primary exchange where spreads are wider.
- [ ] It creates a pro-rata allocation system where large orders are favored over high-speed small orders.

**Hint:** Think about the advantage gained by a party who can see a trade request and then decide whether to honor it based on the latest price movement.

## Question 6
Which of the following describes a common pitfall when backtesting a mean-reverting strategy using consolidated daily data instead of primary exchange auction data?

- [x] The consolidated close price may be an 'off-market' small trade, creating fictitious profits that cannot be replicated using MOC orders at the primary exchange.
- [ ] Daily data typically underestimates the bid-ask spread, leading to overly optimistic results for high-frequency strategies.
- [ ] Primary exchange data is time-stamped in nanoseconds, which makes it incompatible with standard backtesting platforms.
- [ ] Consolidated data excludes dark pool transactions, which are essential for determining the true NBBO.

**Hint:** Recall why a closing price of $\$262.04$ on a small exchange might differ from the $\$262.26$ auction price at Nasdaq.

## Question 7
In the Bulk Volume Classification (BVC) method, if the price change $\Delta P$ is significantly positive relative to its standard deviation $\sigma_{\Delta P}$, what is the estimated net order flow for a bar with volume $V$?

- [x] The net order flow approaches $+V$, as the Gaussian CDF value $Z(\frac{\Delta P}{\sigma_{\Delta P}})$ approaches $1$.
- [ ] The net order flow is $0$, because BVC assumes all price moves within a bar are perfectly balanced.
- [ ] The net order flow is determined by the number of individual trade ticks within the bar, regardless of $\Delta P$.
- [ ] The net order flow becomes negative as the increased price attracts more sellers into the order book.

**Hint:** Evaluate the behavior of the Gaussian CDF as its input becomes a large positive number.

## Question 8
Why might a trader running a mean-reversion strategy prefer to send limit orders to BATS' BZX exchange, which charges a fee for adding liquidity, rather than the BYX exchange, which pays a rebate?

- [x] The 'inverted' fee structure discourages high-frequency rebate seekers, potentially allowing the trader's order to reach the head of the queue more easily.
- [ ] BZX is a dark pool, meaning orders are less likely to be front-run by aggressive momentum traders.
- [ ] The BZX exchange automatically applies the ISO modifier to all limit orders to ensure they stay local.
- [ ] Orders on BZX are executed on a pro-rata basis, which benefits smaller retail traders over large institutions.

**Hint:** Consider the incentive for other HFT firms to place orders on an exchange that costs money to use versus one that pays them.

## Question 9
Which specific technical optimization allows scripting languages like Python or MATLAB to approach the execution speed of compiled languages like $C++$ for intraday trading?

- [x] Replacing computationally intensive functions with compiled code using tools like Numba, Cython, or Mex files.
- [ ] Using cloud-based virtual private servers (VPS) to host the scripting environment near exchange data centers.
- [ ] Increasing the interval of the market data snapshots to $250$ ms to reduce the CPU load on the interpreter.
- [ ] Transitioning from parallel processing to sequential processing to simplify the memory management of the script.

**Hint:** Identify the method used to bypass the 'interpreter' overhead of dynamic languages for heavy math.

## Question 10
What is the primary risk associated with 'spoofing' in the context of dark pool midprice manipulation?

- [x] A trader places small limit orders on lit exchanges to artificially move the NBBO midprice, triggering a large execution in a dark pool at the manipulated price.
- [ ] The spoofer uses direct sponsored access to 'walk the book' of a dark pool before the SIP feed can update.
- [ ] The trader sends massive market orders to dark pools to force them to become lit exchanges under Reg NMS rules.
- [ ] The spoofer identifies hidden orders in dark pools and front-runs them on the primary exchange to earn a rebate.

**Hint:** Focus on how non-executed orders on one venue can affect the execution price on another hidden venue.

## Question 11
Order book imbalance $\rho$ is defined as $\frac{V_B - V_S}{V_B + V_S}$. If $\rho$ is consistently large and positive, what behavior is often observed in market participants according to the source?

- [x] It attracts further buy market orders, and market makers may adjust their quote prices upward in anticipation of the imbalance.
- [ ] It signals an overbought condition, leading mean-reversion traders to immediately enter short positions.
- [ ] The imbalance is ignored by institutional traders as it only reflects displayed (and thus 'uninformed') liquidity.
- [ ] Market makers will increase the depth of their offers to earn higher rebates from the incoming sell flow.

**Hint:** Consider how a visible 'wall' of buy orders affects the expectations of other buyers and the pricing of sellers.

## Question 12
Why is it difficult for a billionaire to execute an intraday strategy in a highly liquid stock like AAPL, despite its $\$4$ billion average daily volume?

- [x] The 'at-the-touch' quote size is typically very small, meaning a large order would cause massive market impact that wipes out potential profits.
- [ ] Reg NMS forbids any single trader from accounting for more than $1\%$ of the daily volume in a single session.
- [ ] High-frequency traders use microwave transmissions to 'front-run' every large order before it reaches the SIP feed.
- [ ] AAPL's status as a 'Primary Exchange' stock requires all large orders to be executed through the end-of-day auction only.

**Hint:** Distinguish between total daily aggregate volume and the instantaneous liquidity available on the order book.
