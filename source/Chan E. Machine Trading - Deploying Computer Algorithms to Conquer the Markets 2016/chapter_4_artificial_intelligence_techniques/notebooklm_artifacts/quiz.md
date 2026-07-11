# Trading Quiz

## Question 1
According to the source material, why do many finance practitioners derisively refer to the AI-driven approach in trading as 'data mining'?

- [x] Because financial data is often non-stationary and machine learning algorithms can easily identify rules that fail out-of-sample.
- [ ] Because the use of tick data provides an infinite amount of stationary samples that make patterns too obvious to be profitable.
- [ ] Because AI algorithms lack the mathematical rigor of the 'hunch-based' approach used by theoretical physicists.
- [ ] Because machine-learned rules are fundamentally more rational than those developed through human financial intuition.

**Hint:** Consider the statistical property of financial returns where probability distributions do not stay constant.

## Question 2
In the context of Stepwise Regression using MATLAB's `stepwiselm`, what is the primary difference between it and a standard multiple linear regression?

- [x] Stepwise regression automatically selects the most significant predictors by adding or removing them based on goodness of fit criteria.
- [ ] Stepwise regression uses non-linear products of independent variables by default to improve the model's Sharpe ratio.
- [ ] Multiple regression requires the use of a constant intercept term, whereas stepwise regression strictly ignores it.
- [ ] Stepwise regression is an unsupervised learning technique, while multiple regression is a supervised one.

**Hint:** Think about the automated process of 'feature selection' mentioned in the text.

## Question 3
What is the primary criterion used by a regression tree algorithm to determine the 'best' predictor and split point at a parent node?

- [x] Minimizing the variance of the response variable in the resulting child nodes.
- [ ] Maximizing the Gini Diversity Index to ensure the child nodes are as heterogeneous as possible.
- [ ] Ensuring that each child node contains exactly the same number of observations as the parent node.
- [ ] Maximizing the regression coefficient of the predictor within each leaf of the tree.

**Hint:** Focus on how the algorithm tries to reduce the error or spread of the dependent variable values.

## Question 4
Which ensemble technique specifically involves training models on replicas of the training set created by sampling $N$ observations with replacement?

- [x] Bagging
- [ ] Boosting
- [ ] Random Subspace
- [ ] Cross-Validation

**Hint:** This method is also referred to as 'bootstrapping' in the text.

## Question 5
In a Random Forest model, how is the 'Random Subspace' element implemented during the tree construction process?

- [x] By selecting the best predictor from a randomly chosen subset of available predictors at every node split.
- [ ] By averaging the predictions of trees that were each trained on a different randomly selected half of the observations.
- [ ] By discarding predictors that have a negative correlation with the response variable before training begins.
- [ ] By forcedly using all available predictors at every split to ensure the strongest possible learner is created.

**Hint:** Recall the MATLAB parameter `NumPredictorsToSample` and its role in the algorithm.

## Question 6
For a classification tree, if a node has a fraction of positive returns $p_+$ and a fraction of negative returns $p_-$, how is the Gini Diversity Index (GDI) calculated?

- [x] $1 - p^2_+ - p^2_-$
- [ ] $\frac{p_+}{p_-} \times 100$
- [ ] $p_+ \times p_-$
- [ ] $\sqrt{p^2_+ + p^2_-}$

**Hint:** The formula involves subtracting the squares of the class probabilities from one.

## Question 7
What is the primary objective of the Support Vector Machine (SVM) algorithm when dealing with discrete response categories?

- [x] To find a hyperplane that separates the classes with the largest possible margin.
- [ ] To iteratively minimize the residuals of a piecewise linear regression tree.
- [ ] To calculate the transition probabilities between hidden market regimes like 'bull' or 'bear'.
- [ ] To transform continuous returns into a normal distribution using a sigmoidal Kernel function.

**Hint:** Think about the geometric concept of a boundary in an $m$-dimensional space.

## Question 8
Which of the following is a characteristic of a Hidden Markov Model (HMM) as described in the text?

- [x] It is an unsupervised learning method because the underlying states (e.g., bull vs. bear) are not directly observable.
- [ ] It utilizes a Kernel function to transform linear data into a non-linear state space for better visualization.
- [ ] The Viterbi algorithm is used to determine the exact emission probabilities for the next time step.
- [ ] It assumes that market returns always follow a stationary probability distribution regardless of the current regime.

**Hint:** Distinguish between what we see (returns) and the 'regimes' that generate them.

## Question 9
In Neural Network architecture, what is the role of the sigmoid function $S(x) = \frac{1}{1+e^{-x}}$?

- [x] It acts as a non-linear activation function that transforms the weighted sum of inputs within a neuron.
- [ ] It is used to normalize the volatility of different stocks before aggregating them into a single training vector.
- [ ] It serves as the final linear layer that outputs the expected continuous return for a trading day.
- [ ] It is the primary optimization algorithm used to find the global minimum of the network's prediction error.

**Hint:** Consider why we don't just use an identity function for the neurons if we want to model more than simple linear relationships.

## Question 10
Why does the author emphasize the need for data normalization when aggregating stocks from the SPX index for a single machine learning model?

- [x] To ensure that stocks with vastly different volatilities (like WMT vs. WTW) have predictors and responses of similar magnitudes.
- [ ] To convert the returns into a discrete binary format required by regression tree algorithms.
- [ ] To remove survivorship bias from the CRSP database prior to the training phase.
- [ ] To ensure that fundamental factors like Earnings Per Share are always greater than the stock's P/E ratio.

**Hint:** Recall the example comparing the market cap and price movements of Walmart (WMT) and Weight Watchers (WTW).
