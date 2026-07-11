# Intraday Flashcards

## Card 1

**Front:** Why do intraday investment strategies typically have higher Sharpe ratios than longer-term strategies with the same average returns?

**Back:** Intraday strategies make more independent bets, allowing the law of large numbers to reduce the variance between realized and expected returns.

---

## Card 2

**Front:** According to Kelly’s formula, how does a higher Sharpe ratio affect a strategy's leverage?

**Back:** A higher Sharpe ratio allows for the application of higher leverage to the strategy.

---

## Card 3

**Front:** What is the primary risk associated with holding a position for longer periods compared to intraday trading?

**Back:** Longer holding periods increase vulnerability to black swan risks.

---

## Card 4

**Front:** What is the main 'capacity' issue facing billion-dollar funds attempting day trading strategies?

**Back:** Order sizes are often much larger than the available 'at-the-touch' quote size, leading to significant market impact.

---

## Card 5

**Front:** Term: Walking the Book

**Back:** Definition: Executing a large market order by taking out multiple price layers of the limit order book.

---

## Card 6

**Front:** What are the three distinct types of latencies in intraday trading?

**Back:** Order submission latency, order status latency, and market data latency.

---

## Card 7

**Front:** Where is Equinix’s NY4 data center located, frequently used for retail colocation?

**Back:** Secaucus, New Jersey.

---

## Card 8

**Front:** How does 'sponsored' direct access differ from standard broker-routed order submission?

**Back:** Orders go directly from the user's program to the market center, bypassing the broker's server.

---

## Card 9

**Front:** What is the typical snapshot interval for stock price data provided by Interactive Brokers (IB)?

**Back:** 250 milliseconds.

---

## Card 10

**Front:** Why should intraday traders prioritize compiled languages like $C++$ over scripting languages like $R$ or Python?

**Back:** Compiled languages generally run at least 10 times faster than scripting languages.

---

## Card 11

**Front:** What tool can be used to bring the speed of MATLAB functions within a factor of 2 of $C++$?

**Back:** Mex files written in $C++$.

---

## Card 12

**Front:** How does latency lead to 'opportunity cost' in mean-reversion strategies?

**Back:** A late limit order placement may miss the fill because most order books use time priority for execution.

---

## Card 13

**Front:** Concept: Adverse Selection

**Back:** Definition: The tendency for a limit order to be filled only when the market moves against the trader, usually due to more informed counterparties.

---

## Card 14

**Front:** In momentum trading, how does latency manifest as 'slippage'?

**Back:** Other informed traders may take the best quotes milliseconds before the trader's market order arrives.

---

## Card 15

**Front:** What is the purpose of the SEC's 'Order Protection Rule' (Rule 611)?

**Back:** It requires marketable orders to be routed to the exchange displaying the best available quote (NBBO).

---

## Card 16

**Front:** What does the acronym 'NBBO' stand for?

**Back:** National Best Bid Offer.

---

## Card 17

**Front:** How does a 'hide-and-light' order modifier help a limit order stay on a local exchange book?

**Back:** It lowers the displayed price to avoid locking the market while maintaining a hidden working price at the better level.

---

## Card 18

**Front:** What is the time priority advantage of a 'hide-and-light' order when the national offer is raised?

**Back:** It is time-stamped at the original arrival time rather than the repricing time, granting it higher queue priority.

---

## Card 19

**Front:** By approximately how much time does a direct exchange data feed typically lead the consolidated SIP feed?

**Back:** 0.5 milliseconds.

---

## Card 20

**Front:** Term: Mil

**Back:** Definition: A unit of measurement for exchange rebates equal to 0.1 cents per share.

---

## Card 21

**Front:** What does the acronym 'ISO' stand for in market microstructure?

**Back:** Intermarket Sweep Order.

---

## Card 22

**Front:** Why is an ISO exempt from the standard routing requirements of the Order Protection Rule?

**Back:** The sender assumes the obligation to simultaneously route orders to execute against all protected superior quotes.

---

## Card 23

**Front:** How does using ISOs for large orders facilitate 'parallel processing'?

**Back:** It allows multiple orders to be sent to different exchanges simultaneously to sweep liquidity before HFTs can react.

---

## Card 24

**Front:** Which order modifier combination ensures a large order 'walks the book' at a local exchange without rerouting?

**Back:** ISO (Intermarket Sweep Order) combined with a limit price set to sweep several layers.

---

## Card 25

**Front:** According to research, what percentage of flash crashes are caused by ISOs?

**Back:** 71.49 percent.

---

## Card 26

**Front:** How do dark pools differ from lit exchanges regarding quote visibility?

**Back:** Dark pools do not display their quotes publicly and only report trades after execution.

---

## Card 27

**Front:** What is the standard execution price for most orders in a dark pool?

**Back:** The NBBO midprice.

---

## Card 28

**Front:** Unlike time-priority exchanges, how do most dark pools allocate trade fills?

**Back:** On a pro rata basis.

---

## Card 29

**Front:** What is the primary motive for using IEX’s 350-microsecond delay?

**Back:** To update to the latest NBBO midprice before execution, preventing latency arbitrage by HFTs.

---

## Card 30

**Front:** Concept: Spoofing

**Back:** Definition: An illegal activity where a trader places and cancels orders to manipulate the NBBO midprice for a better fill in a dark pool.

---

## Card 31

**Front:** Who are considered 'uninformed' traders in the context of adverse selection?

**Back:** Liquidity traders, such as market makers or mutual funds selling due to redemption requests.

---

## Card 32

**Front:** Why does the 'last look' feature in FX markets induce adverse selection for buy-side traders?

**Back:** Liquidity providers can reject orders after checking if the market moved or if the trader is likely informed.

---

## Card 33

**Front:** How does the impact of 'last look' in FX differ from adverse selection in stock markets?

**Back:** In FX, it affects market/IOC orders, whereas in stocks, it primarily affects resting limit orders.

---

## Card 34

**Front:** Why is using consolidated daily close prices inaccurate for backtesting mean-reversion strategies?

**Back:** The consolidated close is random noise from any market center and may differ significantly from the primary exchange auction price.

---

## Card 35

**Front:** In TAQ data, what flag indicates a trade participated in the opening or closing auction?

**Back:** The 'Cross' flag.

---

## Card 36

**Front:** Term: Implied-out Quotes

**Back:** Definition: Outright market quotes generated by limit orders on calendar spreads.

---

## Card 37

**Front:** How is 'Order Flow' defined in terms of transaction volume?

**Back:** It is signed transaction volume, where buy market orders are positive and sell market orders are negative.

---

## Card 38

**Front:** What exchange-provided data tag identifies the side that initiated a trade?

**Back:** The 'Aggressor' tag.

---

## Card 39

**Front:** Rule: Tick Rule

**Back:** A trade at a price higher than the previous trade is a 'buy' (+ flow), and a trade at a lower price is a 'sell' (- flow).

---

## Card 40

**Front:** Rule: Quote Rule

**Back:** A trade at a price higher than the midprice is a buy, and a trade lower than the midprice is a sell.

---

## Card 41

**Front:** The Lee-Ready algorithm uses the _____ for trades at the midprice.

**Back:** Tick Rule

---

## Card 42

**Front:** What is the primary advantage of Bulk Volume Classification (BVC) over the Lee-Ready algorithm?

**Back:** BVC uses bar data (price change and volume) instead of tick data, reducing computational intensity and including hidden orders.

---

## Card 43

**Front:** What is the formula for net order flow in Bulk Volume Classification using volume $V$ and CDF $Z$?

**Back:** $V \cdot [2Z(\frac{\Delta P}{\sigma_{\Delta P}}) - 1]$

---

## Card 44

**Front:** Formula: Order Book Imbalance ($\rho$)

**Back:** $\rho = \frac{VB - VS}{VB + VS}$, where $VB$ is bid size and $VS$ is offer size.

---

## Card 45

**Front:** How does high order book imbalance ($ho$) influence the behavior of market makers?

**Back:** They typically adjust their quote prices in the direction of the imbalance to manage risk and potential market orders.

---

## Card 46

**Front:** What is the typical duration for which the correlation between order book imbalance and price change persists?

**Back:** Up to 200 seconds.

---

## Card 47

**Front:** Why are market orders necessary when trading based on order flow signals?

**Back:** The strategy is trend-following, and there is urgency to fill before the predicted midprice move occurs.

---

## Card 48

**Front:** Which data structure is recommended for simulating an order book from ITCH messages?

**Back:** Binary search trees.

---

## Card 49

**Front:** What does a 'WORKING_CONFIRMED' status signify in an ITCH-style message?

**Back:** The addition of a new limit order to the book.

---

## Card 50

**Front:** In the BATS exchange, what is the 'maker/taker' fee difference between BYX and BZX?

**Back:** BYX pays a rebate for adding liquidity, while BZX charges a fee for adding liquidity.

---
