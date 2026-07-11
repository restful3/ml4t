# Trading Flashcards

## Card 1

**Front:** How does the methodology of a quantitative trader using AI differ from the approach of a theoretical physicist?

**Back:** AI exploration lacks a preconceived theory and uses algorithms to test as many factors and rules as possible.

---

## Card 2

**Front:** Why is the term 'data mining' often used derisively by finance practitioners regarding AI techniques?

**Back:** Financial data is limited and non-stationary, making it easy for algorithms to find rules that fail in the future.

---

## Card 3

**Front:** What statistical property of financial returns describes the fact that their probability distribution does not stay constant over time?

**Back:** Non-stationarity

---

## Card 4

**Front:** What is the primary advantage of machine-learned trading rules compared to handcrafted ones in terms of arbitrage?

**Back:** They may lack rational financial justification, which prevents them from being easily understood and arbitraged away.

---

## Card 5

**Front:** AI algorithms perform _____ to automatically select the most important independent variables for a prediction model.

**Back:** feature selection

---

## Card 6

**Front:** Which technique combines feature selection with linear regression to find the 'best' predictors?

**Back:** Stepwise regression

---

## Card 7

**Front:** What are the three common criteria used by stepwise regression to determine goodness of fit?

**Back:** Sum of squared error (SSE), Akaike information criterion (AIC), and Bayesian information criterion (BIC).

---

## Card 8

**Front:** How does stepwise regression add or remove variables during the modeling process?

**Back:** It adds the 'best' predictors one at a time and then tries to remove them until the goodness of fit stops improving.

---

## Card 9

**Front:** In a regression tree, what is the 'parent' node divided into based on an inequality condition?

**Back:** Child nodes

---

## Card 10

**Front:** What is the mathematical objective when picking the best predictor to split a node in a regression tree?

**Back:** Minimizing the variance of the response variable in each child node.

---

## Card 11

**Front:** In regression trees, minimizing the variance of the response variable is equivalent to minimizing which error metric?

**Back:** Mean squared error (MSE)

---

## Card 12

**Front:** What does a 'leaf' represent in a regression tree model?

**Back:** A child node without children that provides a set of inequalities and an average predicted response.

---

## Card 13

**Front:** The regression tree parameter 'MinLeafSize' is used to prevent _____ by ensuring nodes have enough observations.

**Back:** overfitting

---

## Card 14

**Front:** Which technique randomly divides the training set into $K$ subsets to test out-of-sample performance during model building?

**Back:** Cross validation

---

## Card 15

**Front:** In $K$-fold cross validation, how is the final model typically selected?

**Back:** The model with the highest cross-validation accuracy (or minimum loss) is chosen.

---

## Card 16

**Front:** Why might a high value of $K$ (e.g., 10) be problematic for cross validation with a small financial training set?

**Back:** The out-of-sample subsets become too small, leading to large statistical errors in accuracy measurements.

---

## Card 17

**Front:** Which ensemble method creates replicas of a training set by sampling $N$ observations with replacement?

**Back:** Bagging (or Bootstrapping)

---

## Card 18

**Front:** In bagging, what are the observations omitted from a specific replica called?

**Back:** Out-of-bag observations

---

## Card 19

**Front:** How is the final prediction determined in a bagging ensemble of models?

**Back:** By averaging the predicted responses from all the individual models built from the replicas.

---

## Card 20

**Front:** Which ensemble technique involves randomly sampling the predictors (features) rather than the data points?

**Back:** Random subspace

---

## Card 21

**Front:** What is the definition of a 'Random Forest' model?

**Back:** A hybrid ensemble method using both bagging (data sampling) and random subspace (predictor sampling) for tree-based models.

---

## Card 22

**Front:** Which learning method iteratively applies an algorithm to the prediction errors of the previous model?

**Back:** Boosting

---

## Card 23

**Front:** What specific boosting algorithm is used for gradient descent in regression or regression tree models?

**Back:** LSBoost

---

## Card 24

**Front:** Unlike regression trees, classification trees are designed to predict _____ response variables.

**Back:** discrete (or categorical)

---

## Card 25

**Front:** What is the formula for Gini's Diversity Index (GDI) in a binary classification problem?

**Back:** $1 - p_{+}^2 - p_{-}^2$, where $p_{+}$ and $p_{-}$ are the fractions of each class.

---

## Card 26

**Front:** What GDI value represents a 'perfect classification' in a classification tree node?

**Back:** Zero

---

## Card 27

**Front:** Which classification technique attempts to find a hyperplane in $m$-dimensional space that separates classes with the largest margin?

**Back:** Support Vector Machine (SVM)

---

## Card 28

**Front:** In an SVM model, what are the data points that lie closest to the separating hyperplane called?

**Back:** Support vectors

---

## Card 29

**Front:** Why might a basic linear SVM sometimes generalize better out-of-sample than a classification tree?

**Back:** A linear hyperplane is less flexible than piecewise linear models, which helps avoid overfitting.

---

## Card 30

**Front:** What function is used in SVMs to transform predictors nonlinearly into a higher dimensional space?

**Back:** Kernel function

---

## Card 31

**Front:** Which machine learning model is categorized as 'unsupervised' because the target states are not directly observable?

**Back:** Hidden Markov Model (HMM)

---

## Card 32

**Front:** In an HMM, what matrix describes the probability of moving from one hidden state to another?

**Back:** Transition probability matrix

---

## Card 33

**Front:** In an HMM, what does the 'emission' probability matrix represent?

**Back:** The probability that a specific hidden state will produce a certain observable symbol or value.

---

## Card 34

**Front:** Which algorithm is commonly used to estimate the parameters (transition and emission matrices) of an HMM?

**Back:** EM algorithm

---

## Card 35

**Front:** What is the purpose of the Viterbi algorithm (hmmviterbi) in the context of HMMs?

**Back:** To determine the most probable sequence of hidden states given a set of observed emissions.

---

## Card 36

**Front:** A neural network approximates functions using a linear combination of _____ functions, such as $S(x) = \frac{1}{1 + e^{-x}}$.

**Back:** sigmoid

---

## Card 37

**Front:** What are the three components of a standard feed-forward neural network architecture?

**Back:** Input layer, hidden layers, and an output layer.

---

## Card 38

**Front:** In a neural network, what is the purpose of a 'validation set' during the training process?

**Back:** To monitor prediction error and trigger 'early stopping' if error begins to increase, preventing overfitting.

---

## Card 39

**Front:** Why are neural networks in finance often sensitive to the 'initial guess' of weights?

**Back:** The optimization process can settle into different local minima depending on the starting values.

---

## Card 40

**Front:** According to the text, why might 'deep learning' be less effective for standard financial time-series prediction than for pattern recognition?

**Back:** Financial data lacks the high dimensionality and massive sample size required for deep architectures to generalize well.

---

## Card 41

**Front:** What process is necessary when aggregating returns from multiple stocks with different volatilities to train a single AI model?

**Back:** Data normalization

---

## Card 42

**Front:** How should fundamental variables like earnings per share (EPS) be normalized before cross-sectional aggregation?

**Back:** By total revenue or market capitalization.

---

## Card 43

**Front:** What is the 'black-box' reputation of AI trading strategies referring to?

**Back:** The difficulty in intuiting why an algorithm generates a specific signal, even if the underlying math is understood.

---

## Card 44

**Front:** Which ensemble technique is specifically designed to force a learning algorithm to improve on its past prediction errors?

**Back:** Boosting

---

## Card 45

**Front:** In a classification tree, how is the predicted response for a node determined?

**Back:** It is the class (observation category) that constitutes the highest fraction of the node's data.

---

## Card 46

**Front:** In MATLAB's 'fitrtree', which parameter limits the total number of splits allowed in the tree?

**Back:** MaxNumSplits

---

## Card 47

**Front:** What logic governs the transition from a 'weak learner' to a 'strong learner' in ensemble methods?

**Back:** Averaging the results of many weak learners reduces variance and the risk of overfitting to a single dataset.

---

## Card 48

**Front:** In the context of HMMs, bull and bear markets are treated as _____, as they cannot be directly seen.

**Back:** hidden states

---

## Card 49

**Front:** What is the primary risk of building a trading model on 'tick data' versus daily data in terms of stationarity?

**Back:** Tick data provides more samples but is often even less stationary, making long-term rules hard to find.

---

## Card 50

**Front:** If a neural network has one hidden layer with two neurons and a 4-dimensional input, how many weights (excluding constants) are calculated for the first layer?

**Back:** 8 (4 inputs $\times$ 2 neurons).

---

## Card 51

**Front:** What does the 'Upper' parameter value 'linear' signify in MATLAB's 'stepwiselm' function?

**Back:** The model should only include linear functions of variables, not products of independent variables.

---

## Card 52

**Front:** In the SPY example, which specific predictor was selected by the stepwise regression algorithm?

**Back:** ret2 (previous 2-day return).

---

## Card 53

**Front:** What did the negative regression coefficient for 'ret2' in the SPY model imply about market behavior?

**Back:** The model predicted mean-reversion from the past two-day return.

---

## Card 54

**Front:** What condition must be met for the probabilities in each row of an HMM transition matrix to be valid?

**Back:** They must sum to 1.

---

## Card 55

**Front:** Why does the text recommend starting with the simplest AI techniques like stepwise regression?

**Back:** In trading, complexity often leads to overfitting and doesn't necessarily pay off with better performance.

---

## Card 56

**Front:** Which technique is described as an 'online' algorithm in the text for updating estimates as new data arrives?

**Back:** Kalman filter

---

## Card 57

**Front:** How does 'Retraining' (as used for Neural Networks) resemble cross-validation?

**Back:** It trains multiple models with different initial weights and data subsets to find the one with the lowest validation error.

---

## Card 58

**Front:** Which database provided the survivorship-bias-free data used in the SPX component stocks example?

**Back:** CRSP (Center for Research in Security Prices).

---

## Card 59

**Front:** What is the main 'input feature' advice given by Domingos (2012) regarding machine learning efficacy?

**Back:** The efficacy of an algorithm relies heavily on the quality of the input features provided.

---

## Card 60

**Front:** In a regression tree, if the predicted response for a node is none other than the average of its response variables, what is being minimized?

**Back:** Mean squared error (MSE)

---

## Card 61

**Front:** Why is it important to use 'midprice' at market close for prices when dealing with consolidated closing price issues?

**Back:** To avoid inaccuracies and noise associated with the consolidated close which might skew return calculations.

---

## Card 62

**Front:** According to the text, what is the 'monopoly' AI challenges regarding our own minds?

**Back:** Our own minds and intuition do not have a monopoly on trading ideas; AI can surprise us with justification-free rules.

---

## Card 63

**Front:** What is the relationship between the number of boosting iterations ($M$) and the Sharpe ratio on the training set?

**Back:** The Sharpe ratio on the training set typically increases rapidly as the number of iterations increases.

---

## Card 64

**Front:** In the SPX fundamental factor example, what did the algorithm do with rows in the predictor matrix that contained NaN values?

**Back:** It automatically ignored those rows.

---

## Card 65

**Front:** True or False: An HMM can only handle discrete emissions like 'up' or 'down' days.

**Back:** False (they can also model continuous variables using distributions like Gaussian or Student-t).

---

## Card 66

**Front:** Which toolbox contains the 'learn_params_dbn_em' function used for training HMMs in the source material?

**Back:** Bayes Net Toolbox (BNT).

---

## Card 67

**Front:** What does a Sharpe ratio of 1 typically represent in financial research models?

**Back:** The threshold for achieving statistical significance.

---

## Card 68

**Front:** How does the 'Random Subspace' method choose predictors for each model in the ensemble?

**Back:** It samples predictors with replacement, often with decreasing probability for previously chosen ones.

---

## Card 69

**Front:** What is the default value for 'NumPredictorsToSample' in MATLAB's Random Forest implementation?

**Back:** One-third of the total number of predictors.

---

## Card 70

**Front:** Which method is specifically cited as a way to gain 'human understanding' of complex indicators?

**Back:** Turning AI techniques loose on them to see how they are used by the algorithm.

---

## Card 71

**Front:** In neural network 'averaging', how is the final predicted return for a test sample calculated?

**Back:** By averaging the predictions of a large number (e.g., 100) of independently trained networks.

---

## Card 72

**Front:** Which famous quant fund manager remarked that one of their most profitable strategies had no rational financial justification?

**Back:** Renaissance Technologies

---

## Card 73

**Front:** What was the 'revitalization' year for neural networks following a breakthrough in architecture?

**Back:** 2006

---

## Card 74

**Front:** Why is the use of 'cross validation' particularly critical in building machine-learning models compared to handcrafted ones?

**Back:** Machine-learning models have a higher propensity for overfitting to training data.

---

## Card 75

**Front:** In the classification tree example for SPY, what served as the response variable?

**Back:** A binary indicator of whether the next-day return was positive (True) or negative (False).

---

## Card 76

**Front:** In SVM, what does 'Kernel scale' set to 'auto' allow the algorithm to do?

**Back:** The algorithm selects an optimal scaling factor for the Kernel transformation.

---

## Card 77

**Front:** What benefit does an HMM provide regarding market 'regimes' beyond return prediction?

**Back:** It can identify the most probable sequence of hidden market states (e.g., bull or bear).

---

## Card 78

**Front:** In feed-forward networks, the final output layer typically represents a _____ function.

**Back:** linear

---

## Card 79

**Front:** What is the 'stopping condition' for splitting nodes in a regression tree when child nodes would have too few observations?

**Back:** MinLeafSize

---

## Card 80

**Front:** How did the SVM model on SPY perform relative to the classification tree in the test set?

**Back:** It was superior, achieving a higher CAGR and Sharpe ratio.

---
