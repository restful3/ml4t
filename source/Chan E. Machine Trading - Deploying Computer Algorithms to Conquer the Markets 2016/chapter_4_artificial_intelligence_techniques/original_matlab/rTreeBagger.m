% rTreeBagger.m
% Ensemble of regression trees (bagged trees)
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

K=5;

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

% Select best regression tree model based on default criteria
rng(1); % set fixed random seed for sampling training set, in order to get reproducible results
model=TreeBagger(K, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'Method', 'regression', 'MinLeaf', 100);
% model=TreeBagger(K, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'Method', 'regression', 'MinLeaf', 100, 'NumPredictorsToSample', 'all');

% Make "predictions" on training set (in-sample)
retPred1=predict(model, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)]);
% retPred1=oobPredict(model);

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
% In-sample: K=100. CAGR=0.957251 Sharpe ratio=2.993050 maxDD=-0.187332 maxDDD=146 Calmar ratio=5.109928
% In-sample: K=5. CAGR=0.856603 Sharpe ratio=2.760209 maxDD=-0.206181 maxDDD=70 Calmar ratio=4.154613
% In-sample: K=5, NVarToSample='all'. CAGR=0.723045 Sharpe ratio=2.433606 maxDD=-0.206181 maxDDD=65 Calmar ratio=3.506841

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

% retPred1=predict(model, [ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)]);
retPred1=predict(model, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

% plot(cumret);
plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on SPY with bagging. K=5.');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: K=100. CAGR=-0.017283 Sharpe ratio=-0.024335 maxDD=-0.232286 maxDDD=562 Calmar ratio=-0.074406
% Out-of-sample: K=5. CAGR=0.071967 Sharpe ratio=0.505925 maxDD=-0.170329 maxDDD=556 Calmar ratio=0.422520
% Out-of-sample: K=5. NVarSample=all: CAGR=0.015108 Sharpe ratio=0.173252 maxDD=-0.257029 maxDDD=537 Calmar ratio=0.058780
% Out-of-sample: K=10. CAGR=-0.029207 Sharpe ratio=-0.098758 maxDD=-0.265293 maxDDD=1023 Calmar ratio=-0.110092

