# Calendar-based clustering of weekly extremes: Empirical failure of stochastic models 

Im Hyeon Lee*<br>Department of Finance, Dong-A University, Busan, Republic of Korea


#### Abstract

This study uncovers a significant deviation from randomness in the occurrence of weekly highs and lows in financial markets, a phenomenon we term the clustering of weekly extremes. Unlike the well-known day-of-the-week effect, which focuses on differences in mean returns and volatility, this clustering represents the concentration of highs and lows that cannot be explained by random variation. To address this gap, this study introduces a day-dependent Markov-switching GARCH model that incorporates weekday-specific transition probabilities. The proposed model more effectively captures the clustering of weekly extremes than conventional models and provides a novel analytical framework for more accurately understanding calendar-driven market dynamics.


Keywords: Clustering of weekly extremes, calendar anomaly, day-of-the-week effect, market microstructure, financial time series analysis
JEL: C58, C22, C52, G14

## 1. Introduction

Since French $(1980)^{1}$, a substantial body of empirical finance literature has documented various calendar anomalies, with the day-of-the-week effect, which refers to weekend-specific differences in mean returns or volatility, receiving particular attention (e.g., Chiah and Zhong, 2021; Dicle and Levendis, 2014; Qadan et al., 2022; Valadkhani and O'Mahony, 2024).

Furthermore, Bowles et al. (2024) argue that the conventional FF92 ${ }^{2}$ practice of rebalancing on 30 June, which lags the actual information release date by an average of five to six months, relies on stale information and therefore understates anomaly predictability. Their empirical finding that abnormal returns are concentrated within only a few weeks, or even a few days, after disclosure implies that ignoring calendar precision at the level of specific dates or weekdays can distort both the existence of anomalies and the assessment of risk, which may underscore the importance of research on calendar effects.

Building on these insights, this study shifts the focus from average returns to a distinct phenomenon: the clustering of weekly price extremes on specific weekdays. Unlike the day-of-the-week effect, this phenomenon is driven not by differences in average returns or volatility, but by which days of the week the lowest and highest prices occur within a given week, suggesting a hidden calendar structure underlying market movements.

We examine the distribution of weekly highs and lows across major asset classes, including index futures, bond futures, commodities, and currency markets. By analyzing the data in calendar weeks, we identify the specific weekday responsible for each week's extreme values. Under the efficient market hypothesis, these extremes should follow a random distribution (Campbell et al., 1997). However, G-tests reveal significant deviations from randomness, motivating the development of models that can capture this phenomenon.

[^0]
[^0]:    *Corresponding author. Email: 24167688donga.ac.kr
    ${ }^{1}$ Pre-1980 evidence was scattered; French (1980) formalized the day-of-the-week effect.
    ${ }^{2}$ Fama and French (1992)

# 2. Methodology 

The primary focus of this study is the clustering of weekly extremes, which is defined as the phenomenon in which weekly high and low prices occur more frequently on specific weekdays than would be expected under random movement. To empirically investigate this phenomenon, we compile daily time-series data from four representative markets:

1. Index futures: E-mini S\&P 500 futures (CME, 2001-10-09-2024-12-16)
2. Bond futures: U.S. 30-year Treasury bond futures (CBOT, 2001-07-19-2025-01-10)
3. Commodities: Goldman Sachs commodity index (S\&P, 2001-10-09-2025-01-10)
4. Currencies: EUR/USD exchange rate (ICE, 2001-10-09-2025-01-10)

For each asset, only weeks with exactly five trading days were retained, and for each week, we recorded the weekday associated with the highest and lowest prices, yielding an empirical distribution of weekly extremes.

To assess whether this observed distribution of weekly extremes deviates from the behavior expected under the assumption of weak-form efficiency, three stochastic models are employed: geometric Brownian motion (GBM), the Heston model, and the jump-diffusion model.

The GBM model describes the evolution of an asset price $S_{t}$ as

$$
d S_{t}=\mu S_{t} d t+\sigma S_{t} d W_{t}
$$

where $\mu$ is the drift and $\sigma$ is the volatility, and $W_{t}$ is a standard Brownian motion. It follows from Itô's lemma that the log-price process satisfies

$$
d \ln S_{t}=\left(\mu-\frac{1}{2} \sigma^{2}\right) d t+\sigma d W_{t}
$$

Assuming the log returns

$$
r_{t}=\ln \left(\frac{S_{t}}{S_{t-1}}\right)
$$

are normally distributed, maximum likelihood estimation (MLE) is used to estimate $\mu$ and $\sigma$ from the observed data.

The Heston model extends GBM by introducing a stochastic process for volatility (Heston, 1993). It is defined by the system

$$
\begin{aligned}
d S_{t} & =\mu S_{t} d t+\sqrt{v_{t}} S_{t} d W_{t}^{S} \\
d v_{t} & =\kappa\left(\theta-v_{t}\right) d t+\xi \sqrt{v_{t}} d W_{t}^{v}
\end{aligned}
$$

where $v_{t}$ denotes the instantaneous variance, $\kappa$ is the speed of mean reversion, $\theta$ is the long-run average variance, and $\xi$ is the volatility of volatility. The Brownian motions $d W_{t}^{S}$ and $d W_{t}^{v}$ are correlated, with $\mathbb{E}\left[d W_{t}^{S} d W_{t}^{v}\right]=\rho d t$. Due to the latent nature of $v_{t}$, a particle filter (details are provided in Appendix A) is employed to construct the conditional likelihood and to estimate the parameters $v_{0}, \kappa, \theta, \xi, \rho, \mu$.

As demonstrated by Bates (1996), the jump-diffusion model augments the continuous diffusion process with a jump component to account for sudden, discontinuous movements due to rare events. The dynamics are given by

$$
d S_{t}=(\mu-\lambda k) S_{t} d t+\sigma S_{t} d W_{t}+S_{t} d J_{t}
$$

where $d J_{t}$ represents a jump process (often modeled via a Poisson process with intensity $\lambda$ ), and the jump size is typically drawn from a normal distribution. The correction term $k=\mathbb{E}\left[e^{J_{t}}-1\right]$ adjusts for the mean impact of the jumps. Parameter estimation involves MLE techniques with a log-sum-exp formulation to simultaneously estimate $\lambda, \mu_{J}$ (the mean jump size), and $\sigma_{J}$ (the jump volatility).

The parameters for each model were estimated using their respective procedures. The model then generated 2,000 weekly price paths, starting from an initial price. The frequencies of these extreme values were aggregated and rescaled to match the expected frequency based on the real data. Under the assumption that markets exhibit weak-form efficiency, it is hypothesized that the distribution of weekly extremes by weekday does not significantly differ from that implied by those stochastic models.

In this study, the existence of such weekday-dependent clustering in weekly extremes is primarily evaluated using the Kullback–Leibler divergence (KL divergence) and the G-statistic, by comparing distributions derived from simulated and actual data. KL divergence is defined as

$D_{KL}(O\|\left.E)=\sum_{d=0}^{4} O_{d}\log\frac{O_{d}}{E_{d}}\right.$,

and the G-statistic is given by

$G=2\sum_{d=0}^{4} O_{d}\ln\left(\frac{O_{d}}{E_{d}}\right),$

where $O_{d}$ and $E_{d}$ represent the observed and expected frequencies for day $d$, respectively.

# 2.1. Day-dependent Markov-switching GARCH model 

In this study, to replicate and predict the clustering of weekly extremes, we build on the regime-switching framework initially proposed by Gray (1996) and introduce an extended day-dependent Markov-switching GARCH model (day-dependent MSGARCH model; cf. Anderson et al., 1999) that incorporates weekdaydependent state transitions and volatility dynamics.

Formally, the market state at time $t$ is defined as

$$
S_{t} \in\{1,2, \ldots, K\}
$$

with an initial distribution $\pi_{0}=\left(\pi_{1}, \ldots, \pi_{K}\right)$, where $\pi_{i}=P\left(S_{1}=i\right)$. Each day of the week is represented as $d(t) \in\{0,1,2,3,4\}$ (Mon-Fri). The state transition probability from state $i$ to state $j$ depends on the day of the week

$$
p_{i j}(d(t+1))=P\left(S_{t+1}=j \mid S_{t}=i, d(t+1)\right)
$$

with $\sum_{j=1}^{K} p_{i j}(d)=1$ for each $i$.
For each asset, candidate models with $K=2$ to $K=5$ states were evaluated. The optimal $K$ was determined based on the minimization of the KL divergence between the empirical and simulated distributions of weekly extremes. (The full results can be found in the Supplementary Material.)

Returns at time $t$ conditional on the current state $S_{t}=i$ and day $d(t)$ follow

$$
r_{t} \mid\left(S_{t}=i, d(t)\right) \sim \mathcal{N}\left(\mu_{i, d(t)}, \sigma_{i, d(t)}^{2}\right)
$$

where the conditional variance is governed by a $\operatorname{GARCH}(1,1)$ process

$$
\sigma_{t}^{2}=\alpha_{i, d(t)}+\beta_{i, d(t)} \epsilon_{t-1}^{2}+\gamma_{i, d(t)} \sigma_{t-1}^{2}, \quad \epsilon_{t}=r_{t}-\mu_{i, d(t)}
$$

with constraints $\alpha_{i, d(t)}>0, \beta_{i, d(t)} \geq 0, \gamma_{i, d(t)} \geq 0$, and $\beta_{i, d(t)}+\gamma_{i, d(t)}<1$.
The full parameter set is thus defined as

$$
\theta=\left\{p_{i j}(d), \mu_{i, d}, \alpha_{i, d}, \beta_{i, d}, \gamma_{i, d}, \pi_{0}\right\}
$$

Given observed returns $r_{1: T}$, the likelihood function is computed by summing over all latent states

$$
L\left(\theta ; r_{1: T}\right)=\sum_{S_{1}=1}^{K} \cdots \sum_{S_{T}=1}^{K} \pi_{S_{1}} f\left(r_{1} \mid S_{1}, d(1)\right) \prod_{t=2}^{T} p_{S_{t-1}, S_{t}}(d(t)) f\left(r_{t} \mid S_{t}, d(t)\right)
$$

where $f(\cdot)$ denotes the normal probability density function. Parameter estimation is conducted through expectation-maximization (EM) methods (details are provided in Appendix B).

Simulation involves drawing states according to the day-specific transition probabilities and generating returns from their conditional distributions

$$
\begin{gathered}
S_{t+1} \sim \operatorname{Multinomial}\left(1 ; p_{S_{t}, 1}(d(t+1)), \ldots, p_{S_{t}, K}(d(t+1))\right) \\
r_{t} \mid\left(S_{t}, d(t)\right) \sim \mathcal{N}\left(\mu_{S_{t}, d(t)}, \sigma_{S_{t}, d(t)}^{2}\right)
\end{gathered}
$$

Prices are then updated according to $P_{t}=P_{t-1} \exp \left(r_{t}\right)$.
Likewise, the model's ability to replicate weekly extremes was assessed based on Equations (6) and (7).

Table 1: KL divergence and G-test for the day-of-week distribution of weekly extreme values, based on stochastic model simulations and corresponding empirical market data.

|  |  | KL div |  | G-stat |  | $p$-value |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |   |   |   |   |   |   |   |
|  Asset | Model | High | Low | High | Low | High | Low |
|  |   |   |   |   |   |   |   |
|  ES | GBM | 0.015 | 0.006 | 30.98 | 12.79 | $0.000^{* * *}$ | $0.012^{* *}$ |
|  |   |   |   |   |   |   |   |
|   | Heston | 0.011 | 0.006 | 23.22 | 11.43 | $0.000^{* * *}$ | $0.022^{* *}$ |
|  |   |   |   |   |   |   |   |
|   | Jump-diff | 0.001 | 0.007 | 2.39 | 14.44 | 0.665 | $0.006^{* * *}$ |
|  |   |   |   |   |   |   |   |
|  ZB | GBM | 0.002 | 0.006 | 4.37 | 11.67 | 0.358 | $0.020^{* *}$ |
|  |   |   |   |   |   |   |   |
|   | Heston | 0.001 | 0.008 | 2.71 | 17.42 | 0.608 | $0.002^{* * *}$ |
|  |   |   |   |   |   |   |   |
|   | Jump-diff | 0.001 | 0.014 | 2.32 | 28.92 | 0.677 | $0.000^{* * *}$ |
|  |   |   |   |   |   |   |   |
|  GSCI | GBM | 0.009 | 0.008 | 17.73 | 16.52 | $0.001^{* * *}$ | $0.002^{* * *}$ |
|  |   |   |   |   |   |   |   |
|   | Heston | 0.010 | 0.009 | 19.05 | 17.91 | $0.001^{* * *}$ | $0.001^{* * *}$ |
|  |   |   |   |   |   |   |   |
|   | Jump-diff | 0.005 | 0.003 | 9.80 | 6.22 | $0.044^{* *}$ | 0.183 |
|  |   |   |   |   |   |   |   |
|  EUR/USD | GBM | 0.006 | 0.002 | 14.66 | 5.67 | $0.005^{* * *}$ | 0.225 |
|  |   |   |   |   |   |   |   |
|   | Heston | 0.003 | 0.003 | 7.15 | 7.30 | 0.128 | 0.121 |
|  |   |   |   |   |   |   |   |
|   | Jump-diff | 0.006 | 0.005 | 13.21 | 12.34 | $0.010^{* * *}$ | $0.015^{* *}$ |

Note. $p$-values are reported in parentheses. ${ }^{* * *},{ }^{* *}$, and * indicate significance at the $1 \%, 5 \%$, and $10 \%$ levels, respectively.
when contrasted with actual market outcomes, deviate markedly. This finding highlights the insufficiency of diffusion processes, stochastic volatility, and jump risk alone to replicate salient clustering of weekly extremes.

The original MSGARCH specification, which excludes both the weekday-conditional transition matrix and the weekday state-specific variance structure, yields the results presented in Table 2 under the same analytical conditions as the day-dependent MSGARCH. Although this model exhibits improved performance compared to the previously analyzed stochastic models, it still fails to adequately capture the observed clustering of weekly extremes.

In contrast, the day-dependent MSGARCH model, by incorporating day-dependent transition matrices and state-contingent GARCH parameters, effectively replicates the observed market behavior. The model's ability to capture the clustering of weekly high and low frequencies is demonstrated in Figure 2. The alignment of day-dependent MSGARCH simulations with the observed distribution of weekly extremes, as summarized in Table 3, demonstrates that the model provides a more comprehensive explanation of the phenomenon, as evidenced by the G-test and KL divergence values.

Table 2: KL divergence and G-test for the day-of-week distribution of weekly extreme values, based on MSGARCH model simulations and corresponding empirical market data.

|  | KL div |  | G-stat |  | $p$-value |  |
| --- | --- | --- | --- | --- | --- | --- |
|  Asset | High | Low | High | Low | High | Low |
|  ES | 0.004 | 0.001 | 10.79 | 3.37 | $0.029^{* *}$ | 0.498 |
| ZB | 0.004 | 0.004 | 9.80 | 9.97 | $0.044^{* *}$ | $0.041^{* *}$ |
| GSCI | 0.002 | 0.004 | 5.95 | 9.85 | 0.279 | $0.043^{* *}$ |
| EUR/USD | 0.001 | 0.002 | 3.28 | 4.04 | 0.516 | 0.401 |

Note. $K=3$ for ES, and ZB, $K=4$ for GSCI, $K=5$ for EUR/USD. $p$-values are reported in parentheses. ${ }^{* * *},{ }^{* *}$, and * indicate significance at the $1 \%, 5 \%$, and $10 \%$ levels, respectively.

![img-0.jpeg](image/Calendar-based%20clustering%20of%20weekly%20extremes_fig_01.png)

Figure 1: Day-of-week clustering of weekly extremes in empirical data and mean model-implied frequencies from stochastic simulations for four assets. Panels (a)–(d) show the distribution of weekly high counts by weekday; panels (e)–(h) show the corresponding distributions of weekly low counts.

![img-1.jpeg](image/Calendar-based%20clustering%20of%20weekly%20extremes_fig_02.png)

Figure 2: Day-of-week clustering of weekly extremes in empirical data and day-dependent MSGARCH model for four assets. Panels (a)–(d) show the distribution of weekly high counts by weekday; panels (e)–(h) show the corresponding distributions of weekly low counts.

Table 3: KL divergence and G-test for the day-of-week distribution of weekly extreme values, based on day-dependent MSGARCH model simulations and corresponding empirical market data.

|  | KL div |  | G-stat |  | $p$-value |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |   |   |   |   |   |   |
|  Asset | High | Low | High | Low | High | Low |
|  |   |   |   |   |   |   |
|  ES | 0.002 | 0.001 | 3.84 | 3.25 | 0.428 | 0.516 |
| ZB | 0.003 | 0.002 | 6.76 | 4.90 | 0.151 | 0.298 |
| GSCI | 0.001 | 0.000 | 2.68 | 0.45 | 0.610 | 0.987 |
| EUR/USD | 0.001 | 0.002 | 3.05 | 5.57 | 0.550 | 0.234 |

Note. $K=3$ for ES, $K=4$ for GSCI, $K=5$ for ZB, and EUR/USD. $p$-values are reported in parentheses. ${ }^{* * *},{ }^{* *}$, and * indicate significance at the $1 \%, 5 \%$, and $10 \%$ levels, respectively.

# 3.1. Robustness test 

To evaluate the day-dependent MSGARCH model's out-of-sample performance and temporal stability, we conducted two exercises:

1. Out-of-sample: $80 / 20$ split; KL divergence and G-test on the held-out weeks
2. Rolling window: 750-day estimation and 375-day evaluation windows, advanced in 375-day steps

Full-sample calibration yields KL divergences below 0.003 and no G-test rejections, confirming an excellent in-sample fit. However, in the out-of-sample split, 5 of 8 G-tests ( $62.5 \%$ ) reject the null at the $5 \%$ level. In the rolling-window analysis, the proportion of windows in which the null hypothesis was not rejected was $77 \%$ for ES ( 20 of 26 ), $80.8 \%$ for GSCI ( 21 of 26 ), but lower for ZB ( 15 of $26 ; 57.7 \%$ ) and EUR/USD (12 of $26 ; 46.2 \%$ ). Overall, 64 out of 104 tests ( $65.4 \%$ ) failed to reject the null ( $p \geq 0.05$ ). These results are summarized in Tables 4 and 5 .

Table 4: KL divergence and G-test for the day-of-week distribution of weekly extreme values based on out-of-sample tests of the day-dependent MSGARCH model.

|  | KL div |  | G-stat |  | $p$-value |  |
| --- | --- | --- | --- | --- | --- | --- |
|  |   |   |   |   |   |   |
|  Asset | High | Low | High | Low | High | Low |
|  |   |   |   |   |   |   |
|  ES | 0.012 | 0.003 | 5.69 | 1.20 | 0.223 | 0.878 |
| ZB | 0.037 | 0.009 | 17.80 | 4.23 | $0.001^{* * *}$ | 0.375 |
| GSCI | 0.002 | 0.020 | 0.96 | 9.78 | 0.913 | $0.045^{* *}$ |
| EUR/USD | 0.042 | 0.009 | 20.54 | 4.55 | $0.000^{* * *}$ | 0.336 |

Note. $K=4$ for ES, $K=5$ for ZB, GSCI, and EUR/USD. $p$-values are reported in parentheses. ${ }^{* * *},{ }^{* *}$, and * indicate significance at the $1 \%, 5 \%$, and $10 \%$ levels, respectively.

These findings imply that, although the proposed weekday-conditional transition and volatility dynamics improve explanatory power over baseline stochastic models, the framework still lacks robustness under distributional shifts.

To address this limitation, further extensions are needed. More flexible inference procedures, such as Bayesian Markov chain Monte Carlo (MCMC) or sequential Monte Carlo, could accommodate time-varying parameter uncertainty and latent state dynamics. Additionally, allowing the transition matrix and volatility structure to incorporate macroeconomic or market-based covariates may improve responsiveness to structural breaks. Lastly, robustness may be enhanced by conducting window-sensitivity analyses, including variations in window lengths and step sizes, as well as by applying time-series cross-validation frameworks to mitigate sample dependency.

Taken together, the results validate the model's structural design but also underscore the need for methodological refinements to ensure robustness and generalization in dynamic market environments.

## 4. Discussion

Building on the studies reviewed in Section 1, this study empirically identifies the clustering of weekly extremes in financial markets, defined as the statistically significant concentration of weekly high and low

Table 5: KL divergence and G-test for the day-of-week distribution of weekly extreme values based on rolling window tests of the day-dependent MSGARCH model.

|  Asset | Index | KL div |  | G-stat |  | $p$-value |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
|   |  | High | Low | High | Low | High | Low  |
|  ES | 1875 | 0.011 | 0.034 | 1.76 | 5.25 | 0.780 | 0.264  |
|   | 3000 | 0.054 | 0.022 | 8.46 | 3.50 | $0.076^{*}$ | 0.478  |
|   | 4125 | 0.095 | 0.018 | 14.86 | 2.80 | $0.005^{ *** }$ | 0.592  |
|   | 1875 | 0.030 | 0.027 | 4.68 | 4.32 | 0.322 | 0.364  |
|  ZB | 3000 | 0.045 | 0.087 | 6.96 | 13.51 | 0.138 | $0.009^{ *** }$  |
|   | 4125 | 0.056 | 0.107 | 8.77 | 16.92 | $0.067^{*}$ | $0.002^{ *** }$  |
|   | 1875 | 0.082 | 0.051 | 6.98 | 4.71 | $0.012^{ }$ | $0.094^{*}$  |
|  GSCI | 3000 | 0.176 | 0.074 | 24.71 | 9.30 | $0.000^{ *** }$ | $0.019^{ }$  |
|   | 4125 | 0.008 | 0.049 | 3.54 | 2.47 | 0.883 | 0.103  |
|   | 1875 | 0.030 | 0.034 | 4.57 | 5.14 | 0.334 | 0.273  |
|  EUR/USD | 3000 | 0.135 | 0.086 | 32.63 | 13.52 | $0.000^{ *** }$ | $0.011^{ }$  |
|   | 4125 | 0.030 | 0.025 | 8.13 | 2.13 | 0.336 | 0.430  |
|  ES mean |  | 0.048 | 0.046 | 7.47 | 7.10 | - | -  |
|  ZB mean |  | 0.063 | 0.061 | 9.88 | 9.44 | - | -  |
|  GSCI mean |  | 0.046 | 0.050 | 7.12 | 7.83 | - | -  |
|  EUR/USD mean |  | 0.091 | 0.063 | 13.66 | 9.50 | - | -  |

Note. $K=2$ for ES, $K=3$ for ZB, and EUR/USD, $K=5$ for GSCI. $p$-values are reported in parentheses. ${ }^{ *** },{ }^{ }$, and * indicate significance at the $1 \%, 5 \%$, and $10 \%$ levels, respectively. prices on specific weekdays, contrary to what would be expected under random price movements. To replicate and predict this phenomenon, a day-dependent MSGARCH model is proposed, incorporating weekday-specific state transition probabilities and volatility dynamics. Empirical analysis shows that, unlike conventional stochastic models, the proposed model effectively captures the clustering of weekly extremes. However, the model does not achieve perfect performance in out-of-sample and rolling window evaluations, revealing limitations in its ability to generalize beyond the estimation sample under dynamic market conditions.

Here, a synthesis of prior microstructure research may help partly explain the asset-class heterogeneity in our findings. Hasbrouck (2003) shows that roughly $90 \%$ of price discovery in the S\&P 500 is initiated in E-mini futures (information share, IS $\approx 0.89-0.93$ ) and then diffuses sequentially from futures to ETFs and finally to the cash index. Kilian and Murphy (2014), using a structural VAR, report that about $78 \%$ of crude-oil price fluctuations arise from physical demand-and-supply shocks, with speculative activity accounting for no more than $20 \%$. By contrast, fixed-income and FX markets are dominated by concentrated public signals: Fleming and Remolona (1999) find that approximately $60 \%-70 \%$ of intraday variance in U.S. Treasury yields and major exchange rates is realized within the first 30 min after key macro releases, while Kuttner (2001) estimates that a 1 bp monetary-policy surprise moves the 30 -year Treasury yield by $0.19-0.33 \mathrm{bp}$ almost instantaneously.

These discussions may suggest that, while information in the ES and GSCI markets is dispersed, asynchronous, and incorporated sequentially, information in the ZB and EUR/USD markets is incorporated almost immediately into prices. Accordingly, the routine of information assimilation appears stronger in ES and GSCI than in ZB and EUR/USD, leading to the hypothesis that the economic and theoretical drivers documented in this study may be partly explained by market-specific information-incorporation routines.

Subsequent research could extend toward enhancing model robustness through the alternative estimation methodologies proposed in Section 3.1 and toward uncovering the fundamental drivers behind the observed phenomenon. Additionally, exploring practical and policy-oriented approaches to integrate these phenomena and models into risk management frameworks, such as dynamic VaR estimation or variable margin systems, remains an important avenue for future research.

# Appendix A. Detailed estimation procedures: MLE and particle filter 

This section delineates the estimation procedures employed for the continuous-time models under consideration. The methodological exposition encompasses MLE for the GBM and jump-diffusion models, as well as the particle filtering approach applied to the Heston model.

## Appendix A.1. MLE for the GBM and jump-diffusion models

The log returns, defined as Equation (3) are posited to follow a normal distribution characterized by a mean of $\mu-1 / 2 \sigma^{2}$ and volatility $\sigma$. Parameter estimation proceeds through the maximization of the log-likelihood function computed over the observed log returns.

Within the jump-diffusion framework, the diffusion process is augmented by a jump component, the occurrence of which adheres to a Poisson arrival process. The resultant likelihood function assumes the form of a mixture of normal components, each representing the probability structure associated with zero or multiple jumps. To enhance numerical stability, a log-sum-exp formulation is implemented. The parameter set $\left\{\mu, \sigma, \lambda, \mu_{J}, \sigma_{J}\right\}$ is inferred by maximizing the resultant likelihood function.

## Appendix A.2. Particle filter for the Heston model

The Heston model introduces a latent variance process, denoted $v_{t}$, whose unobserved nature necessitates the adoption of a particle filtering methodology for likelihood approximation. The procedure unfolds through the following sequential stages:

1. Initialization: A set of $N$ particles, $\left\{v_{0}^{(i)}\right\}$, is generated from an initial distribution over $v_{0}$. Uniform weights are assigned, such that $w_{0}^{(i)}=1 / N$.
2. Propagation: At each temporal increment, particle variances are updated via a discretized approximation of the Heston dynamics

$$
v_{t}^{(i)} \approx v_{t-1}^{(i)}+\kappa\left(\theta-v_{t-1}^{(i)}\right) \Delta t+\xi \sqrt{\max \left\{v_{t-1}^{(i)}, \epsilon\right\}} \epsilon_{t}^{(i)}
$$

where $\epsilon_{t}^{(i)} \sim N(0, \Delta t)$.
3. Weight update: Each particle's weight is updated in accordance with the likelihood of the observed log return $r_{t}$, conditional upon the particle-specific variance. The weights are subsequently normalized to ensure their sum equals unity.
4. Resampling: In instances where the effective sample size declines below a pre-specified threshold, resampling is conducted to mitigate particle degeneracy.
5. Likelihood approximation: The overall likelihood is approximated by the expression

$$
L(\theta) \approx \prod_{t=1}^{T-1}\left(\sum_{i=1}^{N} w_{t}^{(i)}\right)
$$

## Appendix B. Implementation: Day-dependent MSGARCH model

This section delineates the implementation of the day-dependent MSGARCH model, incorporating day-of-week-dependent transitions, as introduced in the primary text. Let Equation (8) denote the latent state at time $t$, with the transition probability defined as Equation (9) where $d(t+1)$ designates the weekday $(0-4)$. The return distribution conditional on state $i$ and day $d$ is expressed as

$$
r_{t} \sim \mathcal{N}\left(\mu_{i, d}, \sigma_{i, d}^{2}\right)
$$

with $\sigma_{i, d}^{2}$ governed by a $\operatorname{GARCH}(1,1)$ process. Following the estimation methodology articulated in Haas et al. (2004), the EM algorithm is employed.

1. E-step: Posterior probabilities of latent states and joint probabilities of consecutive states, conditional upon observed returns, are computed utilizing the forward-backward (Baum-Welch) algorithm.
2. M-step: Transition probabilities $p_{i j}(d)$, state- and day-specific means $\mu_{i, d}$, and GARCH parameters $\left\{\alpha_{i, d}, \beta_{i, d}, \gamma_{i, d}\right\}$ are updated by maximizing the expected complete-data log-likelihood.

# References 

Anderson, H.M., Nam, K., Vahid, F., 1999. Asymmetric nonlinear smooth transition GARCH models, in: Rothman, P. (Ed.), Nonlinear time series analysis of economic and financial data. Springer US, pp. 191-207.

Bates, D.S., 1996. Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options. Rev. Financ. Stud. 9 (1), 69-107.

Bowles, B., Reed, A.V., Ringgenberg, M.C., Thornock, J.R., 2024. Anomaly time. J. Finance 79 (5), $3543-3579$.

Campbell, J.Y., Lo, A.W., MacKinlay, A., 1997. The econometrics of financial markets. Princeton University Press.

Chiah, M., Zhong, A., 2021. Tuesday blues and the day-of-the-week effect in stock returns. J. Bank. Finance 133, 106243 .

Dicle, M.F., Levendis, J.D., 2014. The day-of-the-week effect revisited: international evidence. J. Econ. Finance 38 (3), 407-437.

Fama, E.F., French, K.R., 1992. The cross-section of expected stock returns. J. Finance 47 (2), 427-465.
Fleming, M.J., Remolona, E.M., 1999. Price formation and liquidity in the U.S. Treasury market: the response to public information. J. Finance 54 (5), 1901-1915.

French, K.R., 1980. Stock returns and the weekend effect. J. Financ. Econ. 8 (1), 55-69.
Gray, S.F., 1996. Modeling the conditional distribution of interest rates as a regime-switching process. J. Financ. Econ. 42 (1), 27-62.

Haas, M., Mittnik, S., Paolella, M.S., 2004. A new approach to Markov-switching GARCH models. J. Financ. Econom. 2 (4), 493-530.

Hasbrouck, J., 2003. Intraday price formation in U.S. equity index markets. J. Finance 58 (6), 2375-2400.
Heston, S.L., 1993. A closed-form solution for options with stochastic volatility with applications to bond and currency options. Rev. Financ. Stud. 6 (2), 327-343.

Kilian, L., Murphy, D.P., 2014. The role of inventories and speculative trading in the global market for crude oil. J. Appl. Econom. 29 (3), 454-478.

Kuttner, K.N., 2001. Monetary policy surprises and interest rates: evidence from the Fed funds futures market. J. Monet. Econ. 47 (3), 523-544.

Qadan, M., Aharon, D.Y., Eichel, R., 2022. Seasonal and calendar effects and the price efficiency of cryptocurrencies. Finance Res. Lett. 46, 102354.

Valadkhani, A., O'Mahony, B., 2024. Sector-specific calendar anomalies in the U.S. equity market. Int. Rev. Financ. Anal. 95, 103347.

