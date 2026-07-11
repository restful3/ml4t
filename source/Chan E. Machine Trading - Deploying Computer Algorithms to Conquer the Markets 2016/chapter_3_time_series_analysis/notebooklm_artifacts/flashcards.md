# Econometrics Flashcards

## Card 1

**Front:** What is the primary goal shared by economists, electrical engineers, and traders in the context of time-series analysis?

**Back:** Predicting the next signal in a series.

---

## Card 2

**Front:** In the $AR(1)$ model equation $Y(t) - \mu = \phi(Y(t-1) - \mu) + \epsilon(t)$, what does the term $\epsilon(t)$ represent?

**Back:** Gaussian noise with zero mean, also known as innovation.

---

## Card 3

**Front:** A time series where the mean and variance are constant over time is described as being _____.

**Back:** Weakly stationary

---

## Card 4

**Front:** What is the mathematical condition for an $AR(1)$ process to be considered weakly stationary?

**Back:** $|\phi| < 1$

---

## Card 5

**Front:** In an $AR(1)$ model, if the coefficient $\phi$ is exactly 1, the time series is characterized as a _____.

**Back:** Random walk

---

## Card 6

**Front:** In an $AR(1)$ model, if the coefficient $|\phi| > 1$, the time series will exhibit what behavior?

**Back:** It will trend.

---

## Card 7

**Front:** According to the text, a weakly stationary time series is also inherently _____.

**Back:** Mean reverting

---

## Card 8

**Front:** Which MATLAB Econometrics Toolbox function is used to specify an $AR(1)$ model?

**Back:** $arima(1, 0, 0)$

---

## Card 9

**Front:** What statistical method does the MATLAB function $estimate$ use to find parameters for an $ARIMA$ model?

**Back:** Maximum likelihood estimation

---

## Card 10

**Front:** Why does the author recommend testing on midprices instead of trade prices for currency time-series analysis?

**Back:** To reduce bid-ask bounce, which creates phantom mean-reversion.

---

## Card 11

**Front:** Term: $AR(p)$ model

**Back:** Definition: A multiple regression model relating the current price to past prices up to a lag of $p$ bars.

---

## Card 12

**Front:** Formula: The general form of an $AR(p)$ model.

**Back:** $Y(t) = \mu + \phi_1 Y(t-1) + \phi_2 Y(t-2) + \dots + \phi_p Y(t-p) + \epsilon(t)$

---

## Card 13

**Front:** What does the Bayesian information criterion (BIC) penalize to avoid overfitting?

**Back:** Model complexity (the number of parameters $p$).

---

## Card 14

**Front:** When using BIC to find the optimal lag $p$ for an $AR(p)$ model, should the objective be to maximize or minimize the BIC value?

**Back:** Minimize the BIC.

---

## Card 15

**Front:** In the provided MATLAB example for $AUD.USD$, what was the optimal lag $p$ determined by the brute-force BIC search?

**Back:** 10

---

## Card 16

**Front:** When using the $forecast$ function, how many recent data points are required to predict the next bar in an $AR(pMin)$ model?

**Back:** $pMin$ data points.

---

## Card 17

**Front:** If $yF(t)$ is the forecast made with data up to time $t$, for which time period is the price actually predicted?

**Back:** $t + 1$

---

## Card 18

**Front:** What is the simple trading rule used for the $AR(10)$ strategy on $AUD.USD$?

**Back:** Buy if the predicted price is higher than the current price; otherwise sell.

---

## Card 19

**Front:** What type of execution program is necessary to realize the high returns of the $AR(10)$ $AUD.USD$ strategy mentioned in the text?

**Back:** A low-latency execution program that manages limit orders at the midprice.

---

## Card 20

**Front:** Term: $ARMA(p, q)$ model

**Back:** Definition: An autoregressive moving average process that combines $p$ lagged price terms and $q$ lagged noise terms.

---

## Card 21

**Front:** In an $ARMA(p, q)$ model, the $q$ lagged noise terms are collectively referred to as the _____ part of the model.

**Back:** Moving average

---

## Card 22

**Front:** Formula: The mathematical representation of an $ARMA(p, q)$ model.

**Back:** $Y(t) = \mu + \sum_{i=1}^p \phi_i Y(t-i) + \epsilon(t) + \sum_{j=1}^q \theta_j \epsilon(t-j)$

---

## Card 23

**Front:** When performing an exhaustive search for $ARMA(p, q)$ parameters, how is the penalty term for BIC calculated in relation to $p$ and $q$?

**Back:** It is proportional to $p + q + 1$.

---

## Card 24

**Front:** For $AUD.USD$, how did the optimal lags for the $ARMA$ model compare to the $AR$ model?

**Back:** The $ARMA$ lags ($p=2, q=5$) were shorter than the $AR$ lag ($p=10$).

---

## Card 25

**Front:** In the $AUD.USD$ backtest, did adding the complexity of moving average ($ARMA$) improve the annualized return compared to the simpler $AR$ model?

**Back:** No, it decreased the return from 158% to 60%.

---

## Card 26

**Front:** What does the 'I' stand for in the $ARIMA(p, d, q)$ model?

**Back:** Integrated

---

## Card 27

**Front:** If $Y(t)$ follows an $ARIMA(p, 1, q)$ model, what model does the first difference $\Delta Y(t)$ follow?

**Back:** $ARMA(p, q)$

---

## Card 28

**Front:** Using $ARMA(p, q)$ to model log returns is equivalent to using $ARIMA(p, 1, q)$ to model _____.

**Back:** Log prices

---

## Card 29

**Front:** Why might modeling price $Y$ directly be more flexible than modeling the change in price $\Delta Y$ using $ARMA$?

**Back:** A model for $Y$ can use both $Y$ and $\Delta Y$ as independent variables, whereas $ARMA$ for $\Delta Y$ only uses $\Delta Y$.

---

## Card 30

**Front:** Term: $VAR(p)$ model

**Back:** Definition: A Vector Autoregressive model that generalizes the $AR(p)$ model to multivariate time series.

---

## Card 31

**Front:** In a $VAR(p)$ model, the autoregressive coefficients $\phi$ are interpreted as _____.

**Back:** $m \times m$ matrices

---

## Card 32

**Front:** What characteristic of the noise terms $\epsilon(t)$ distinguishes $VAR(p)$ from simple univariate models?

**Back:** They can have nonzero cross-sectional correlations but must have zero serial correlations.

---

## Card 33

**Front:** For which type of financial instruments is a $VAR$ model particularly suitable?

**Back:** Instruments with correlated returns, such as stocks within the same industry group.

---

## Card 34

**Front:** Which industry group from the S&P 500 did the author use to demonstrate the $VAR(1)$ model?

**Back:** Computer hardware

---

## Card 35

**Front:** What is the equivalent MATLAB function for $estimate$ when working with $VAR$ models?

**Back:** $vgxvarx$

---

## Card 36

**Front:** What was the optimal lag $p$ determined for the computer hardware stocks in the $VAR$ model training?

**Back:** 1

---

## Card 37

**Front:** Formula: Calculation of a sector-neutral target dollar allocation $w_i$ for stock $i$ with predicted return $r_i$.

**Back:** $w_i = (r_i - \langle r \rangle) / \sum_j |r_j - \langle r \rangle|$

---

## Card 38

**Front:** In the author's sector-neutral $VAR(1)$ strategy, the sum of absolute target dollar allocations is always equal to _____.

**Back:** $1

---

## Card 39

**Front:** Term: $VEC(q)$ model

**Back:** Definition: A Vector Error Correction model that transforms $VAR(p)$ to express changes in price $\Delta Y$ as the dependent variable.

---

## Card 40

**Front:** Formula: The $VEC(q)$ model equation.

**Back:** $\Delta Y(t) = M + CY(t-1) + A_1\Delta Y(t-1) + \dots + A_k\Delta Y(t-k) + \epsilon(t)$

---

## Card 41

**Front:** In the $VEC$ model, what is the name of the $m \times m$ matrix $C$?

**Back:** Error correction matrix

---

## Card 42

**Front:** What is the relationship between the lag order $p$ of a $VAR$ model and the lag order $q$ of its corresponding $VEC$ model?

**Back:** $q = p - 1$

---

## Card 43

**Front:** In a $VEC$ model, a negative diagonal element in the matrix $C$ typically indicates that the stock is _____.

**Back:** Serially mean reverting

---

## Card 44

**Front:** According to the author, if a portfolio is cointegrating, what should be observed in the Johansen test regarding matrix $C$?

**Back:** It would give rise to a significantly negative eigenvalue.

---

## Card 45

**Front:** What is the primary difference between AR/VAR models and State Space Models (SSM)?

**Back:** SSMs use hidden variables (states) to determine observed values, rather than just using observable lagged prices.

---

## Card 46

**Front:** Term: Linear State Space Model (SSM)

**Back:** Definition: A model consisting of a state transition equation and a measurement equation, often implemented via the Kalman filter.

---

## Card 47

**Front:** Formula: The state transition equation in a State Space Model.

**Back:** $x(t) = A(t)x(t-1) + B(t)u(t)$

---

## Card 48

**Front:** Formula: The measurement equation in a State Space Model.

**Back:** $y(t) = C(t)x(t) + D(t)\epsilon(t)$

---

## Card 49

**Front:** In an SSM, if the state transition matrix $A(t)$ is the identity matrix, how does the hidden state $x(t)$ evolve?

**Back:** As a random walk: $x(t) = x(t-1) + B u(t)$.

---

## Card 50

**Front:** How can the unknown matrices $B$ and $D$ in a State Space Model be determined if they are not directly observable?

**Back:** By applying maximum likelihood estimation (MLE) on training data.

---

## Card 51

**Front:** In the context of SSMs, what is a common trading hypothesis regarding the relationship between observed prices and a hidden moving average state?

**Back:** Prices are trending, so the best guess for tomorrow's price is the current estimated moving average.

---

## Card 52

**Front:** In the computer hardware SSM example, what did the $NaN$ inputs to the MATLAB function represent?

**Back:** Unknown parameters to be estimated by the model.

---

## Card 53

**Front:** Why did the author use diagonal matrices for $B$ and $D$ in the hardware stock SSM?

**Back:** To assume state and measurement noises are uncorrelated between stocks, reducing the danger of overfitting.

---

## Card 54

**Front:** What does the MATLAB $filter$ function provide in the context of an SSM?

**Back:** Estimates of both the hidden states and the forecasted observations.

---

## Card 55

**Front:** What is the 'filtered price' at time $t$ in a Kalman filter model?

**Back:** The estimated state variable (e.g., moving average) given observed prices up to time $t$.

---

## Card 56

**Front:** Besides the Kalman filter, name two other filters mentioned that are used for signal processing in finance.

**Back:** Hodrick-Prescott filter and wavelet filter.

---

## Card 57

**Front:** How was the Kalman filter used for the EWA/EWC ETF pair in the author's previous work?

**Back:** To find the best estimates of the time-varying hedge ratio and offset.

---

## Card 58

**Front:** When using a Kalman filter for a hedge ratio, which variable acts as the 'measurement' $y(t)$?

**Back:** One of the two price series (e.g., EWC).

---

## Card 59

**Front:** When modeling a hedge ratio with Kalman filter, why is the matrix $C(t)$ augmented with a column of ones?

**Back:** To allow for a constant offset in the linear regression relationship.

---

## Card 60

**Front:** In the EWA/EWC Kalman filter strategy, when does a trader enter a long position in EWC?

**Back:** When the observed price is smaller than the forecasted value by more than one standard deviation of the forecast error.

---

## Card 61

**Front:** What is a major risk when applying linear time-series models with many parameters to trading?

**Back:** Overfitting the training data.

---

## Card 62

**Front:** What are two common constraints used to reduce the number of parameters and combat overfitting in ARMA or VAR models?

**Back:** Limiting the number of lags to 1 and assuming zero cross-correlations for noises.

---

## Card 63

**Front:** Why are time-series techniques particularly promising for intraday trading?

**Back:** They can be trained on a large amount of data, which helps mitigate overfitting.

---

## Card 64

**Front:** What is the primary reason the EWA/EWC equity curve might flatten at the end of a backtest period?

**Back:** Regime change (falling out of cointegration) or overfitting the noise covariance matrix.

---

## Card 65

**Front:** In an $AR(1)$ process, if the variance of $Y(t)$ is to be constant (weakly stationary), what constraint must be on the regression coefficient?

**Back:** $|\phi| < 1$

---

## Card 66

**Front:** Does an $ARMA$ model for $\Delta Y$ always imply an $ARMA$ model for $Y$?

**Back:** Yes.

---

## Card 67

**Front:** Does an $ARMA$ model for $Y$ always imply an $ARMA$ model for $\Delta Y$?

**Back:** No, because a model for $Y$ is more flexible.

---

## Card 68

**Front:** In the sector-neutral weighting formula, what is the purpose of the denominator $\sum |r_j - \langle r \rangle|$?

**Back:** To normalize the portfolio so the initial gross market value is exactly $1.

---

## Card 69

**Front:** In the $VEC(q)$ model, what does the term $M$ represent?

**Back:** The constant offsets.

---

## Card 70

**Front:** Process: Identifying optimal $p$ and $q$ for an $ARMA$ model.

**Back:** Perform a brute-force search over a range of $p$ and $q$, calculate BIC for each, and select the pair that minimizes it.

---

## Card 71

**Front:** Which software packages are recommended for $R$ users to perform time-series analysis similar to the MATLAB Econometrics Toolbox?

**Back:** forecast, vars, and dlm.

---

## Card 72

**Front:** What is the significance of the $A(t)$ matrix in the state transition equation of an SSM?

**Back:** It specifies how the hidden state evolves from the previous time step.

---

## Card 73

**Front:** In the computer hardware $VAR(1)$ model, how were the autoregressive coefficients represented in Table 3.3?

**Back:** As a matrix where subscripts refer to individual stocks.

---

## Card 74

**Front:** The assumption that state innovation noises are uncorrelated corresponds to which matrix structure in an SSM?

**Back:** A diagonal $B$ matrix.

---

## Card 75

**Front:** In the Kalman filter strategy for EWA/EWC, the 'forecast error' is defined as the difference between _____ and _____.

**Back:** Observed price ($y$); Forecasted price ($yF$)

---

## Card 76

**Front:** What was the estimated standard error of $\phi$ for $AUD.USD$ in the text, and what did it imply about the market's efficiency?

**Back:** 0.00001; it implied the market is very close to a random walk.

---

## Card 77

**Front:** In Equation 3.2 ($AR(p)$), which variable is the dependent (response) variable?

**Back:** The price at time $t$ ($Y(t)$).

---

## Card 78

**Front:** How does the $aicbic$ function in MATLAB assist in model selection?

**Back:** It calculates the AIC and BIC values for a set of models based on their log likelihoods.

---

## Card 79

**Front:** Concept: Sector-neutral trading

**Back:** A strategy where the net market exposure to an industry group or sector is zero.

---

## Card 80

**Front:** What is the dimensionality of the noise vector $u(t)$ in a state transition equation involving $m$ state variables?

**Back:** $k$-dimensional (where $k$ is the number of noise sources).

---

## Card 81

**Front:** In the $VEC$ model, if $k=0$, the model is referred to as a _____.

**Back:** $VEC(0)$ model

---

## Card 82

**Front:** True or False: According to Professor Lyons, textbook models explain a high proportion of monthly exchange rate changes.

**Back:** False (he stated it is essentially zero).

---

## Card 83

**Front:** In the context of the Econometrics Toolbox, what parameters does $arima(p, d, q)$ require?

**Back:** $p$ (AR order), $d$ (degree of differencing), and $q$ (MA order).

---

## Card 84

**Front:** What does the $smartsum$ function likely do in the MATLAB code for sector-neutral positioning?

**Back:** Calculates the sum while handling potentially problematic data points (like NaNs).

---

## Card 85

**Front:** In the computer hardware $VAR(1)$ study, what was the data source for the midprices at market close?

**Back:** Center for Research of Security Prices (CRSP).

---
