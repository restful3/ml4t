% lr.m
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

trainset=1:floor(length(tday)/2);

model=fitlm([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.
% 
% model = 
% 
% 
% Linear regression model:
%     y ~ 1 + x1 + x2 + x3 + x4
% 
% Estimated Coefficients:
%                          Estimate                   SE                   tStat                  pValue       
%                    ____________________    ____________________    __________________    ____________________
% 
%     (Intercept)    4.55250560811905e-05    0.000437616529777923     0.104029562375747       0.917163980386169
%     x1              -0.0249732825555171      0.0385466940773141    -0.647870930394901       0.517197417584629
%     x2               -0.130976974703952       0.033258363500598      -3.9381665517486    8.70296369153575e-05
%     x3               0.0139640149617444      0.0210811892963964     0.662392181267086       0.507852289631726
%     x4              0.00173116222712215     0.00926474088032084      0.18685489961185       0.851807273485198
% 
% 
% Number of observations: 1158, Error degrees of freedom: 1153
% Root Mean Squared Error: 0.0149
% R-squared: 0.0303,  Adjusted R-Squared 0.0269
% F-statistic vs. constant model: 9, p-value = 3.69e-07

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

% In-sample: CAGR=0.343339 Sharpe ratio=1.365304 maxDD=-0.216440 maxDDD=78 Calmar ratio=1.586302

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

retPred1=predict(model, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample: CAGR=0.003582 Sharpe ratio=0.103952 maxDD=-0.295969 maxDDD=778 Calmar ratio=0.012104

