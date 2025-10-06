# Calendar-based clustering of weekly extremes: Empirical failure of stochastic models

Im Hyeon Lee[^corresponding]  
Department of Finance, Dong-A University, Busan, Republic of Korea

#### Abstract

This study uncovers a significant deviation from randomness in the occurrence of weekly highs and lows in financial markets, a phenomenon we term the clustering of weekly extremes. Unlike the well-known day-of-the-week effect, which focuses on differences in mean returns and volatility, this clustering represents the concentration of highs and lows that cannot be explained by random variation. To address this gap, this study introduces a day-dependent Markov-switching GARCH model that incorporates weekday-specific transition probabilities. The proposed model more effectively captures the clustering of weekly extremes than conventional models and provides a novel analytical framework for understanding calendar-driven market dynamics more accurately.

**Keywords**: clustering of weekly extremes, calendar anomaly, day-of-the-week effect, market microstructure, financial time-series analysis  
**JEL**: C58, C22, C52, G14

## 1. Introduction

Since French (1980) formalized the day-of-the-week effect, a substantial body of empirical finance literature has documented various calendar anomalies, with particular attention on weekend-specific differences in mean returns or volatility (e.g., Chiah and Zhong, 2021; Dicle and Levendis, 2014; Qadan et al., 2022; Valadkhani and O'Mahony, 2024).

Bowles et al. (2024) argue that the conventional rebalancing practice following Fama and French (1992, hereafter FF92)—which sets the update date to 30 June, lagging information releases by several months—relies on stale information and therefore understates anomaly predictability. Their finding that abnormal returns are concentrated within a few weeks, or even days, of disclosure implies that ignoring calendar precision at the level of specific dates or weekdays can distort both the existence of anomalies and the assessment of risk, underscoring the importance of research on calendar effects.

Building on these insights, this study shifts the focus from average returns to a distinct phenomenon: the clustering of weekly price extremes on specific weekdays. Unlike the day-of-the-week effect, this phenomenon is driven not by differences in average returns or volatility, but by which days of the week the lowest and highest prices occur within a given week, suggesting a hidden calendar structure underlying market movements.

We examine the distribution of weekly highs and lows across major asset classes, including index futures, bond futures, commodities, and currency markets. By analyzing data in calendar weeks, we identify the specific weekday responsible for each week's extreme values. Under the efficient market hypothesis, these extremes should follow a random distribution (Campbell et al., 1997). However, G-tests reveal significant deviations from randomness, motivating the development of models that can capture this phenomenon.

## 2. Methodology

The primary focus of this study is the clustering of weekly extremes, defined as the phenomenon in which weekly high and low prices occur more frequently on specific weekdays than would be expected under random movement. To investigate this empirically, we compile daily time-series data from four representative markets:

1. Index futures: E-mini S\&P 500 futures (CME, 2001-10-09–2024-12-16)
2. Bond futures: U.S. 30-year Treasury bond futures (CBOT, 2001-07-19–2025-01-10)
3. Commodities: Goldman Sachs Commodity Index (S\&P, 2001-10-09–2025-01-10)
4. Currencies: EUR/USD exchange rate (ICE, 2001-10-09–2025-01-10)

For each asset, only weeks with exactly five trading days are retained, and we record the weekday associated with the highest and lowest prices, yielding an empirical distribution of weekly extremes.

To assess whether the observed distribution of weekly extremes deviates from behavior consistent with weak-form efficiency, three benchmark stochastic models are employed: geometric Brownian motion (GBM), the Heston model, and a jump-diffusion model.

The GBM model describes the evolution of an asset price \(S_t\) as

$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t,
$$

where \(\mu\) is the drift, \(\sigma\) is the volatility, and \(W_t\) is a standard Brownian motion. It follows from Itô's lemma that the log-price process satisfies

$$
d \ln S_t = \left(\mu - \frac{1}{2}\sigma^{2}\right) dt + \sigma\, dW_t.
$$

Assuming that the log returns

$$
r_t = \ln \left(\frac{S_t}{S_{t-1}}\right)
$$

are normally distributed, maximum likelihood estimation (MLE) is used to estimate \(\mu\) and \(\sigma\) from the observed data.

The Heston model extends GBM by introducing a stochastic process for volatility (Heston, 1993). It is defined by the system

$$
\begin{aligned}
dS_t &= \mu S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^{(1)}, \\
dv_t &= \kappa\left(\theta - v_t\right) dt + \xi \sqrt{v_t}\, dW_t^{(2)}, \\
dW_t^{(1)} dW_t^{(2)} &= \rho\, dt,
\end{aligned}
$$

where \(v_t\) denotes instantaneous variance, \(\kappa\) is the rate of mean reversion toward the long-run variance \(\theta\), \(\xi\) governs volatility of variance, and \(\rho\) captures the correlation between the Brownian motions \(W_t^{(1)}\) and \(W_t^{(2)}\).

The jump-diffusion model augments the diffusion process by introducing Poisson-driven price jumps:

$$
dS_t = \mu S_t\, dt + \sigma S_t\, dW_t + (J_t - 1) S_t\, dN_t,
$$

where \(N_t\) is a Poisson process with intensity \(\lambda\) and \(J_t\) denotes the jump size. The resulting likelihood adopts a mixture of normal components reflecting the number of jumps in each interval.

For each specification, we estimate model parameters via MLE or, in the case of the Heston model, particle filtering to integrate over latent variance states. Simulations are then generated to evaluate whether the implied weekday distribution of weekly extremes matches the empirical distribution. Specifically, latent states are drawn according to the day-specific transition probabilities, returns are simulated from their conditional distributions, and prices are updated via \(P_t = P_{t-1} \exp(r_t)\). Model fit is evaluated with Kullback–Leibler (KL) divergence and the G-test statistic comparing simulated and observed frequencies.

In addition to the continuous-time benchmarks, we consider a day-dependent Markov-switching GARCH (MSGARCH) model in which both transition probabilities and conditional variances vary with the weekday. Let \(S_t \in \{1,\dots,K\}\) denote the latent state at time \(t\), and let \(d(t)\) identify the weekday. Conditional returns follow

$$
r_t \mid (S_t = i, d(t) = d) \sim \mathcal{N}\left(\mu_{i,d}, \sigma_{i,d}^2\right),
$$

with the conditional variance evolving according to a GARCH(1,1) structure,

$$
\sigma_{i,d}^2 = \alpha_{i,d} + \beta_{i,d} \epsilon_{t-1}^2 + \gamma_{i,d} \sigma_{t-1}^2,
\quad \epsilon_t = r_t - \mu_{i,d},
$$

subject to \(\alpha_{i,d}>0\), \(\beta_{i,d} \geq 0\), \(\gamma_{i,d} \geq 0\), and \(\beta_{i,d} + \gamma_{i,d} < 1\). The parameter set comprises weekday-specific transition matrices \(p_{ij}(d)\), state- and day-specific means, GARCH coefficients, and the initial state distribution \(\pi_0\). Estimation proceeds via an expectation–maximization algorithm using the forward–backward (Baum–Welch) routine in the E-step and maximum-likelihood updates in the M-step.

## 3. Empirical Results

Simulation from the benchmark stochastic models yields weekday distributions of weekly highs and lows that differ materially from the empirical data (Table 1). The discrepancies emphasize that diffusion processes, stochastic volatility, and jump risk alone are insufficient to reproduce the clustering of weekly extremes observed in markets.

Table 1. KL divergence and G-test statistics for stochastic model simulations versus empirical data.

| Asset | Model | KL div (High) | KL div (Low) | G-stat (High) | G-stat (Low) | p-value (High) | p-value (Low) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ES | GBM | 0.015 | 0.006 | 30.98 | 12.79 | 0.000*** | 0.012** |
| ES | Heston | 0.011 | 0.006 | 23.22 | 11.43 | 0.000*** | 0.022** |
| ES | Jump-diffusion | 0.001 | 0.007 | 2.39 | 14.44 | 0.665 | 0.006*** |
| ZB | GBM | 0.002 | 0.006 | 4.37 | 11.67 | 0.358 | 0.020** |
| ZB | Heston | 0.001 | 0.008 | 2.71 | 17.42 | 0.608 | 0.002*** |
| ZB | Jump-diffusion | 0.001 | 0.014 | 2.32 | 28.92 | 0.677 | 0.000*** |
| GSCI | GBM | 0.009 | 0.008 | 17.73 | 16.52 | 0.001*** | 0.002*** |
| GSCI | Heston | 0.010 | 0.009 | 19.05 | 17.91 | 0.001*** | 0.001*** |
| GSCI | Jump-diffusion | 0.005 | 0.003 | 9.80 | 6.22 | 0.044** | 0.183 |
| EUR/USD | GBM | 0.006 | 0.002 | 14.66 | 5.67 | 0.005*** | 0.225 |
| EUR/USD | Heston | 0.003 | 0.003 | 7.15 | 7.30 | 0.128 | 0.121 |
| EUR/USD | Jump-diffusion | 0.006 | 0.005 | 13.21 | 12.34 | 0.010*** | 0.015** |

Note: Significance levels are denoted as *** (1%), ** (5%), and * (10%).

The baseline MSGARCH specification, which excludes weekday-dependent transition matrices and variances, improves upon the stochastic benchmarks but still fails to match the empirical weekday clustering (Table 2). By contrast, the day-dependent MSGARCH model that allows both transitions and variances to vary by weekday closely replicates the observed distributions, as illustrated in Figure 2 and summarized in Table 3.

Table 2. KL divergence and G-test statistics for the baseline MSGARCH model.

| Asset | KL div (High) | KL div (Low) | G-stat (High) | G-stat (Low) | p-value (High) | p-value (Low) |
| --- | --- | --- | --- | --- | --- | --- |
| ES | 0.004 | 0.001 | 10.79 | 3.37 | 0.029** | 0.498 |
| ZB | 0.004 | 0.004 | 9.80 | 9.97 | 0.044** | 0.041** |
| GSCI | 0.002 | 0.004 | 5.95 | 9.85 | 0.279 | 0.043** |
| EUR/USD | 0.001 | 0.002 | 3.28 | 4.04 | 0.516 | 0.401 |

Table 3. KL divergence and G-test statistics for the day-dependent MSGARCH model.

| Asset | KL div (High) | KL div (Low) | G-stat (High) | G-stat (Low) | p-value (High) | p-value (Low) |
| --- | --- | --- | --- | --- | --- | --- |
| ES | 0.002 | 0.001 | 3.84 | 3.25 | 0.428 | 0.516 |
| ZB | 0.003 | 0.002 | 6.76 | 4.90 | 0.151 | 0.298 |
| GSCI | 0.001 | 0.000 | 2.68 | 0.45 | 0.610 | 0.987 |
| EUR/USD | 0.001 | 0.002 | 3.05 | 5.57 | 0.550 | 0.234 |

Note: K equals 3 for ES and ZB, 4 for GSCI, and 5 for EUR/USD. Significance levels follow the notation in Table 1.

<img src="./image/Calendar-based clustering of weekly extremes_fig_01.png" width="800" alt="Weekday clustering of weekly extremes in empirical data versus stochastic model simulations" />

Figure 1. Day-of-week clustering of weekly extremes in empirical data and mean model-implied frequencies from stochastic simulations for four assets. Panels (a)–(d) show the distribution of weekly high counts by weekday; panels (e)–(h) show the corresponding distributions of weekly low counts.

<img src="./image/Calendar-based clustering of weekly extremes_fig_02.png" width="800" alt="Weekday clustering of weekly extremes in data versus day-dependent MSGARCH" />

Figure 2. Day-of-week clustering of weekly extremes in empirical data and the day-dependent MSGARCH model for four assets. Panels (a)–(d) show the distribution of weekly high counts by weekday; panels (e)–(h) show the corresponding distributions of weekly low counts.

### 3.1 Robustness Tests

To evaluate the day-dependent MSGARCH model's out-of-sample performance and temporal stability, we conducted two exercises: (1) an 80/20 split with KL divergence and G-test statistics computed on the held-out weeks, and (2) rolling windows comprising 750-day estimation and 375-day evaluation periods advanced in 375-day increments.

Full-sample calibration yields KL divergences below 0.003 and no G-test rejections, confirming an excellent in-sample fit. However, in the out-of-sample split, 5 of 8 G-tests (62.5%) reject the null at the 5% level. In the rolling-window analysis, the proportion of windows in which the null hypothesis is not rejected equals 77% for ES (20 of 26), 80.8% for GSCI (21 of 26), 57.7% for ZB (15 of 26), and 46.2% for EUR/USD (12 of 26). Overall, 64 out of 104 tests (65.4%) fail to reject the null (p ≥ 0.05). The detailed statistics are summarized in Tables 4 and 5.

Table 4. Out-of-sample KL divergence and G-test statistics for the day-dependent MSGARCH model.

| Asset | KL div (High) | KL div (Low) | G-stat (High) | G-stat (Low) | p-value (High) | p-value (Low) |
| --- | --- | --- | --- | --- | --- | --- |
| ES | 0.012 | 0.003 | 5.69 | 1.20 | 0.223 | 0.878 |
| ZB | 0.037 | 0.009 | 17.80 | 4.23 | 0.001*** | 0.375 |
| GSCI | 0.002 | 0.020 | 0.96 | 9.78 | 0.913 | 0.045** |
| EUR/USD | 0.042 | 0.009 | 20.54 | 4.55 | 0.000*** | 0.336 |

Table 5. Rolling-window KL divergence and G-test statistics for the day-dependent MSGARCH model.

| Asset | Window index | KL div (High) | KL div (Low) | G-stat (High) | G-stat (Low) | p-value (High) | p-value (Low) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ES | 1875 | 0.011 | 0.034 | 1.76 | 5.25 | 0.780 | 0.264 |
| ES | 3000 | 0.054 | 0.022 | 8.46 | 3.50 | 0.076* | 0.478 |
| ES | 4125 | 0.095 | 0.018 | 14.86 | 2.80 | 0.005*** | 0.592 |
| ZB | 1875 | 0.030 | 0.027 | 4.68 | 4.32 | 0.322 | 0.364 |
| ZB | 3000 | 0.045 | 0.087 | 6.96 | 13.51 | 0.138 | 0.009*** |
| ZB | 4125 | 0.056 | 0.107 | 8.77 | 16.92 | 0.067* | 0.002*** |
| GSCI | 1875 | 0.082 | 0.051 | 6.98 | 4.71 | 0.012 | 0.094* |
| GSCI | 3000 | 0.176 | 0.074 | 24.71 | 9.30 | 0.000*** | 0.019 |
| GSCI | 4125 | 0.008 | 0.049 | 3.54 | 2.47 | 0.883 | 0.103 |
| EUR/USD | 1875 | 0.030 | 0.034 | 4.57 | 5.14 | 0.334 | 0.273 |
| EUR/USD | 3000 | 0.135 | 0.086 | 32.63 | 13.52 | 0.000*** | 0.011 |
| EUR/USD | 4125 | 0.030 | 0.025 | 8.13 | 2.13 | 0.336 | 0.430 |
| ES (mean) | — | 0.048 | 0.046 | 7.47 | 7.10 | — | — |
| ZB (mean) | — | 0.063 | 0.061 | 9.88 | 9.44 | — | — |
| GSCI (mean) | — | 0.046 | 0.050 | 7.12 | 7.83 | — | — |
| EUR/USD (mean) | — | 0.091 | 0.063 | 13.66 | 9.50 | — | — |

Note: K equals 2 for ES, 3 for ZB and EUR/USD, and 5 for GSCI. Significance levels follow the notation in Table 1.

These findings imply that, although the weekday-conditional transition and volatility dynamics improve explanatory power over baseline stochastic models, the framework still lacks robustness under distributional shifts.

To address this limitation, further extensions are needed. More flexible inference procedures, such as Bayesian Markov chain Monte Carlo or sequential Monte Carlo, could accommodate time-varying parameter uncertainty and latent state dynamics. Additionally, allowing the transition matrix and volatility structure to incorporate macroeconomic or market-based covariates may improve responsiveness to structural breaks. Lastly, robustness may be enhanced by conducting window-sensitivity analyses, including variations in window lengths and step sizes, as well as by applying time-series cross-validation frameworks to mitigate sample dependency.

Taken together, the results validate the model's structural design but also underscore the need for methodological refinements to ensure robustness and generalization in dynamic market environments.

## 4. Discussion

Building on the studies reviewed in Section 1, this study empirically identifies the clustering of weekly extremes in financial markets, defined as the statistically significant concentration of weekly high and low prices on specific weekdays, contrary to what would be expected under random price movements. To replicate and predict this phenomenon, a day-dependent MSGARCH model is proposed, incorporating weekday-specific state transition probabilities and volatility dynamics. Empirical analysis shows that, unlike conventional stochastic models, the proposed model effectively captures the clustering of weekly extremes. However, the model does not achieve perfect performance in out-of-sample and rolling-window evaluations, revealing limitations in its ability to generalize beyond the estimation sample under dynamic market conditions.

A synthesis of prior microstructure research may help explain the asset-class heterogeneity in our findings. Hasbrouck (2003) shows that roughly 90% of price discovery in the S\&P 500 is initiated in E-mini futures (information share, IS ≈ 0.89–0.93) and then diffuses sequentially from futures to ETFs and finally to the cash index. Kilian and Murphy (2014), using a structural VAR, report that about 78% of crude-oil price fluctuations arise from physical demand-and-supply shocks, with speculative activity accounting for no more than 20%. By contrast, fixed-income and FX markets are dominated by concentrated public signals: Fleming and Remolona (1999) find that approximately 60–70% of intraday variance in U.S. Treasury yields and major exchange rates is realized within the first 30 minutes after key macro releases, while Kuttner (2001) estimates that a 1 bp monetary-policy surprise moves the 30-year Treasury yield by 0.19–0.33 bp almost instantaneously.

These discussions suggest that, while information in the ES and GSCI markets is dispersed, asynchronous, and incorporated sequentially, information in the ZB and EUR/USD markets is incorporated almost immediately into prices. Accordingly, the routine of information assimilation appears stronger in ES and GSCI than in ZB and EUR/USD, leading to the hypothesis that the economic and theoretical drivers documented in this study may be partly explained by market-specific information-incorporation routines.

Subsequent research could extend toward enhancing model robustness through the alternative estimation methodologies proposed in Section 3.1 and toward uncovering the fundamental drivers behind the observed phenomenon. Additionally, exploring practical and policy-oriented approaches to integrate these phenomena and models into risk-management frameworks, such as dynamic value-at-risk estimation or variable margin systems, remains an important avenue for future research.

## Appendix A. Detailed estimation procedures: MLE and particle filter

This appendix outlines the estimation procedures employed for the continuous-time models. The discussion covers MLE for the GBM and jump-diffusion models, as well as the particle-filtering approach used for the Heston model.

### Appendix A.1. MLE for the GBM and jump-diffusion models

The log returns introduced in Section 2 are assumed to follow a normal distribution characterized by mean \(\mu - \frac{1}{2}\sigma^{2}\) and volatility \(\sigma\). Parameter estimation proceeds by maximizing the log-likelihood function computed over the observed log returns.

Within the jump-diffusion framework, the diffusion process is augmented by a jump component with Poisson arrivals. The resulting likelihood function takes the form of a mixture of normal components, each representing the probability structure associated with zero or multiple jumps. To enhance numerical stability, a log-sum-exp formulation is implemented. The parameter set \(\{\mu, \sigma, \lambda, \mu_J, \sigma_J\}\) is inferred by maximizing the likelihood.

### Appendix A.2. Particle filter for the Heston model

The Heston model introduces a latent variance process \(v_t\), whose unobserved nature necessitates particle filtering for likelihood approximation. The procedure unfolds through the following steps:

1. **Initialization**: Generate a set of \(N\) particles \(\{v_0^{(i)}\}\) from an initial distribution over \(v_0\) and assign uniform weights \(w_0^{(i)} = 1/N\).
2. **Propagation**: At each time step, update particle variances via a discretized approximation of the Heston dynamics,
   $$
   v_t^{(i)} \approx v_{t-1}^{(i)} + \kappa(\theta - v_{t-1}^{(i)}) \Delta t + \xi \sqrt{\max\{v_{t-1}^{(i)}, \varepsilon\}}\, \epsilon_t^{(i)},
   $$
   where \(\epsilon_t^{(i)} \sim \mathcal{N}(0, \Delta t)\).
3. **Weight update**: Update each particle's weight using the likelihood of the observed log return \(r_t\) conditional on the particle-specific variance, then normalize the weights so they sum to one.
4. **Resampling**: Whenever the effective sample size falls below a threshold, resample to mitigate particle degeneracy.
5. **Likelihood approximation**: Approximate the overall likelihood via
   $$
   L(\theta) \approx \prod_{t=1}^{T-1} \left(\sum_{i=1}^{N} w_t^{(i)}\right).
   $$

## Appendix B. Implementation of the day-dependent MSGARCH model

This appendix details the implementation of the day-dependent MSGARCH model introduced in Section 2. Let \(S_t\) denote the latent state at time \(t\) with weekday-dependent transition probabilities \(p_{ij}(d(t+1))\), where \(d(t)\in\{0,\dots,4\}\). The return distribution conditional on state \(i\) and weekday \(d\) follows

$$
r_t \sim \mathcal{N}\left(\mu_{i,d}, \sigma_{i,d}^2\right),
$$

with \(\sigma_{i,d}^2\) governed by a GARCH(1,1) process. Estimation mirrors Haas et al. (2004) and proceeds via an expectation–maximization routine:

1. **E-step**: Compute posterior probabilities of latent states and joint probabilities of consecutive states given observed returns, using the forward–backward (Baum–Welch) algorithm.
2. **M-step**: Update transition probabilities \(p_{ij}(d)\), state- and day-specific means \(\mu_{i,d}\), and GARCH parameters \(\{\alpha_{i,d}, \beta_{i,d}, \gamma_{i,d}\}\) by maximizing the expected complete-data log-likelihood.

## References

Anderson, H.M., Nam, K., Vahid, F., 1999. Asymmetric nonlinear smooth transition GARCH models. In: Rothman, P. (Ed.), *Nonlinear Time Series Analysis of Economic and Financial Data*. Springer US, pp. 191–207.

Bates, D.S., 1996. Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options. *Review of Financial Studies* 9(1), 69–107.

Bowles, B., Reed, A.V., Ringgenberg, M.C., Thornock, J.R., 2024. Anomaly time. *Journal of Finance* 79(5), 3543–3579.

Campbell, J.Y., Lo, A.W., MacKinlay, A., 1997. *The Econometrics of Financial Markets*. Princeton University Press.

Chiah, M., Zhong, A., 2021. Tuesday blues and the day-of-the-week effect in stock returns. *Journal of Banking & Finance* 133, 106243.

Dicle, M.F., Levendis, J.D., 2014. The day-of-the-week effect revisited: international evidence. *Journal of Economics and Finance* 38(3), 407–437.

Fama, E.F., French, K.R., 1992. The cross-section of expected stock returns. *Journal of Finance* 47(2), 427–465.

Fleming, M.J., Remolona, E.M., 1999. Price formation and liquidity in the U.S. Treasury market: the response to public information. *Journal of Finance* 54(5), 1901–1915.

French, K.R., 1980. Stock returns and the weekend effect. *Journal of Financial Economics* 8(1), 55–69.

Gray, S.F., 1996. Modeling the conditional distribution of interest rates as a regime-switching process. *Journal of Financial Economics* 42(1), 27–62.

Haas, M., Mittnik, S., Paolella, M.S., 2004. A new approach to Markov-switching GARCH models. *Journal of Financial Econometrics* 2(4), 493–530.

Hasbrouck, J., 2003. Intraday price formation in U.S. equity index markets. *Journal of Finance* 58(6), 2375–2400.

Heston, S.L., 1993. A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies* 6(2), 327–343.

Kilian, L., Murphy, D.P., 2014. The role of inventories and speculative trading in the global market for crude oil. *Journal of Applied Econometrics* 29(3), 454–478.

Kuttner, K.N., 2001. Monetary policy surprises and interest rates: evidence from the Fed funds futures market. *Journal of Monetary Economics* 47(3), 523–544.

Qadan, M., Aharon, D.Y., Eichel, R., 2022. Seasonal and calendar effects and the price efficiency of cryptocurrencies. *Finance Research Letters* 46, 102354.

Valadkhani, A., O'Mahony, B., 2024. Sector-specific calendar anomalies in the U.S. equity market. *International Review of Financial Analysis* 95, 103347.

[^corresponding]: Corresponding author. Email: 24167688@donga.ac.kr.
