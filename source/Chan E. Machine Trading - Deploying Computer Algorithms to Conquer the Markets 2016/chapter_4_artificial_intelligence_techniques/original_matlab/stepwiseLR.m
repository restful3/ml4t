% stepwiseLR.m
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build stepwise LR model on training data
trainset=1:floor(length(tday)/2);

% Select best linear model based on default criteria
model=stepwiselm([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'Upper', 'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.

% 1. Adding x2, FStat = 35.0074, pValue = 4.32227e-09
% 
% model = 
% 
% 
% Linear regression model:
%     y ~ 1 + x2
% 
% Estimated Coefficients:
%                     Estimate         SE         tStat       pValue  
%                    __________    __________    _______    __________
% 
%     (Intercept)    4.1525e-05    0.00043715    0.09499       0.92434
%     x2               -0.13006      0.021982    -5.9167    4.3223e-09
% 
% 
% Number of observations: 1158, Error degrees of freedom: 1156
% Root Mean Squared Error: 0.0149
% R-squared: 0.0294,  Adjusted R-Squared 0.0286
% F-statistic vs. constant model: 35, p-value = 4.32e-09

% Make "predictions" on training set (in-sample)
retPred1=predict(model, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)]);

% Backtest trading model based on "prediction" on training set
positions=zeros(length(trainset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(trainset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% In-sample: CAGR=0.436383 Sharpe ratio=1.649068 maxDD=-0.187332 maxDDD=74 Calmar ratio=2.329469

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

retPred1=predict(model, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Stepwise Regression on SPY');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample: CAGR=0.105685 Sharpe ratio=0.695228 maxDD=-0.197311 maxDDD=435 Calmar ratio=0.535627

