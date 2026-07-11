# Time-Series Quiz

## Question 1
In the $AR(1)$ process defined as $Y(t) - \mu = \phi(Y(t-1) - \mu) + \epsilon(t)$, what is the statistical implication for the time series if the absolute value of the autoregressive coefficient $|\phi|$ is exactly equal to $1$?

- [x] The time series becomes a random walk.
- [ ] The time series exhibits strong mean reversion.
- [ ] The time series will follow a deterministic trend.
- [ ] The variance of the series remains constant over time.

**Hint:** Consider what happens to the relationship between the current price and the previous price when the coefficient is unity.

## Question 2
When determining the optimal lag $p$ for an $AR(p)$ model using an exhaustive search, the Bayesian Information Criterion (BIC) is minimized. What is the primary purpose of the penalty term in the BIC formula?

- [x] To penalize model complexity and prevent overfitting.
- [ ] To increase the log likelihood of the training data.
- [ ] To ensure the autoregressive coefficients remain below $1$.
- [ ] To account for the bid-ask bounce in high-frequency data.

**Hint:** Think about the trade-off between the goodness of fit and the number of parameters used in the model.

## Question 3
Why does the author recommend using midprices instead of trade prices when applying time-series analysis to one-minute bars of currency pairs like $AUD.USD$?

- [x] Trade prices create 'phantom' mean-reversion due to bid-ask bounce that is not tradeable.
- [ ] Midprices are more stationary than trade prices.
- [ ] The $MATLAB$ Econometrics Toolbox only accepts midprice inputs.
- [ ] Trade prices are always a random walk, whereas midprices are trending.

**Hint:** Focus on the mechanical behavior of prices flipping between the bid and offer levels.

## Question 4
In an $ARIMA(p, d, q)$ model, what does the parameter $d=1$ signify regarding the relationship between prices and returns?

- [x] The first difference of the series is modeled as an $ARMA(p, q)$ process.
- [ ] The model assumes the series is already stationary without differencing.
- [ ] The model incorporates a deterministic linear trend into the price series.
- [ ] The model uses only moving average terms for the price prediction.

**Hint:** Recall how the 'I' in ARIMA (Integrated) relates to the 'change' in a variable over one time step.

## Question 5
A $VAR(1)$ model is applied to a portfolio of computer hardware stocks. If you transform this into a Vector Error Correction ($VEC$) model, what does the Error Correction Matrix $C$ help a trader understand?

- [x] The long-term equilibrium and mean-reverting relationships between the stocks.
- [ ] The optimal number of lags needed to minimize the BIC.
- [ ] The specific volatility of each individual stock in the group.
- [ ] The exact dollar allocation for a sector-neutral strategy.

**Hint:** Think about how this matrix relates the change in price to the previous price level.

## Question 6
The sector-neutral dollar allocation formula $w_i = (r_i - \langle r \rangle) / \sum |r_j - \langle r \rangle|$ is used for the $VAR(1)$ strategy. What does this formula ensure about the portfolio's market exposure?

- [x] The initial gross market value of the portfolio is always $1$.
- [ ] The portfolio is always long-only and ignores shorting opportunities.
- [ ] The portfolio is immune to idiosyncratic risk from individual stocks.
- [ ] The weights are inversely proportional to the stock's historical volatility.

**Hint:** Examine the denominator of the provided weight equation.

## Question 7
In a linear State Space Model ($SSM$), the measurement equation is defined as $y(t) = C(t)x(t) + D(t)\epsilon(t)$. If we are using the Kalman filter to estimate a time-varying hedge ratio between two ETFs ($EWA$ and $EWC$), what does the term $C(t)$ represent?

- [x] The price of one ETF (e.g., $EWA$) augmented with a constant.
- [ ] The hidden hedge ratio and the constant offset.
- [ ] The measurement noise covariance matrix.
- [ ] The log returns of the pair over the training period.

**Hint:** Recall how the Kalman filter is used like a dynamic linear regression where $y$ is the dependent variable.

## Question 8
The author notes that while $ARMA(2, 5)$ provided a better fit in-sample for $AUD.USD$, its out-of-sample performance was significantly worse than the simpler $AR(10)$ model. What risk does this highlight in algorithmic time-series modeling?

- [x] Increased model complexity can lead to overfitting of noise.
- [ ] Moving average models are inherently inferior to autoregressive models.
- [ ] BIC is an unreliable metric for currency pair prediction.
- [ ] Currency pairs do not exhibit mean-reverting behavior.

**Hint:** Consider why a model that looks 'better' on past data might fail on new data.

## Question 9
In the Kalman filter application to computer hardware stocks, the author assumes $B$ and $D$ are diagonal matrices. What is the practical motivation for this constraint during the 'estimate' phase?

- [x] To reduce the number of parameters and the risk of overfitting.
- [ ] Because stock prices in the same industry are never correlated.
- [ ] To ensure the state transition matrix $A$ remains the identity matrix.
- [ ] The filter function in $MATLAB$ requires diagonal matrices for noise.

**Hint:** Think about the consequences of having too many unknown parameters in a maximum likelihood estimation.

## Question 10
When trading the $EWA-EWC$ pair using the Kalman filter, a long position in the spread is entered when the forecast error $e$ is less than $-\sqrt{ymse}$. What does $ymse$ represent in this context?

- [x] The forecasted standard deviation or uncertainty of the observation.
- [ ] The mean squared error of the training set residuals.
- [ ] The historical volatility of the hedge ratio.
- [ ] The profit target for the mean-reversion trade.

**Hint:** Look at the variables used to determine entry signals in the provided code fragment.
