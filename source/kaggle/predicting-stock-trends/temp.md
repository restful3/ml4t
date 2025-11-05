Overview
This competition challenges participants to predict whether a stock's closing price will be above or below its current value after 30 trading days. Using OHLC (Open, High, Low, Close) data from 5,000 stocks, participants must build accurate predictive models to identify future stock price trends. The objective is to create robust models that demonstrate strong forecasting capabilities.

Start

7 months ago
Close
5 months to go
Description
Develop a classification model to predict whether a stock's closing price will be above or below its current value after 30 trading days.

Account for seasonal trends, sector-specific patterns, and market-wide movements that could influence stock prices.

Handle missing data, outliers, and potential feature correlations to improve model accuracy.

Provide insights into which indicators or features most significantly impact stock price movements.

Evaluate the model based on Precision

Evaluation
The performance of the model will be evaluated based on Accuracy, a metric that measures the proportion of correctly predicted outcomes among all predictions. Accuracy is calculated as follows:

Accuracy = (True Positives + True Negatives) / (Total Predictions)

Explanation
True Positive (TP): Correctly predicted positive outcomes.
True Negative (TN): Correctly predicted negative outcomes.
False Positive (FP): Incorrectly predicted positive outcomes.
False Negative (FN): Incorrectly predicted negative outcomes.
Why Accuracy?
Accuracy is ideal for this competition since it provides a comprehensive view of the model's overall performance. A higher accuracy score indicates that the model is consistently making correct predictions across both positive and negative outcomes.

Submission File
For each ID in the test set, you must predict a binary prediction for the Pred variable. This prediction should be whether the Close price will be higher or lower than the last observed training Close. The file should contain a header and have the following format:

ID,Pred
ticker_1,1
ticker_2,0
ticker_3,0
ticker_4,1
Citation
jake wright. Predicting Stock Trends: Rise or Fall?. https://kaggle.com/competitions/predicting-stock-trends-rise-or-fall, 2025. Kaggle.


Dataset Description
train.csv (Training Data)
Purpose: Contains historical stock data for training your model.
Tickers: 5,000
Columns: 9
Columns Details:
Date — The date corresponding to the stock data entry.
Open — Opening price of the stock on that date.
High — Highest price of the stock on that date.
Low — Lowest price of the stock on that date.
Close — Closing price of the stock on that date.
Volume — Number of shares traded on that date.
Dividends — Dividend payout value for that date.
Stock Splits — Any stock splits that occurred on that date.
Ticker — The unique identifier for the stock.
test.csv (Test Data)
Purpose: Contains stock data for which predictions are required.
Rows: 5,000
Columns: 2
Columns Details:
ID — A unique identifier for each sample (in ticker_x format).
Date — The date for which predictions are required.
sample_submission.csv (Submission Format)
Purpose: Provides the required submission format for the competition.
Rows: 5,000
Columns: 2
Columns Details:
ID — Corresponds to entries in the test.csv file.
Pred — The predicted outcome:
1 — Closing price is predicted to be greater than its current value after 30 trading days.
0 — Closing price is predicted to be less than its current value after 30 trading days.
Notes
The training data should be used to train your model to predict whether a stock's closing price will be above or below its current value after 30 trading days.
The test data does not provide target labels — your model's output should follow the structure outlined in sample_submission.csv.
Ensure your submission matches the required format to be considered valid.
Files
3 files

Size
2.3 GB

Type
csv

License
Subject to Competition Rules

sample_submission.csv(68.9 kB)
Competition Rules


To see this data you need to agree to the competition rules.
Join the competition to view the data.


Join the competition
Data Explorer
2.3 GB

sample_submission.csv

test.csv

train.csv

Summary
3 files

13 columns


Download All
kaggle competitions download -c predicting-stock-trends-rise-or-fall
Download data

Metadata
License
Subject to Competition Rules
