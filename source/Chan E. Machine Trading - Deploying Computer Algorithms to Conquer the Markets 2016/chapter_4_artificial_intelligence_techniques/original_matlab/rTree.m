% rTree.m
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

% Select best regression tree model based on default criteria
model=fitrtree([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset), 'MinLeafSize', 100);

view(model); % print the tree nodes as text
view(model, 'mode', 'graph'); % see the tree visually

% Make "predictions" on training set (in-sample)
% 1) Just the rules based on two leaves with the most extreme expected
% returns (positive vs negative)

positions=zeros(length(trainset), 1);

positions(ret2(trainset) >= 0.01531)=-1;
positions(ret2(trainset) < 0.01531 & ret1(trainset) < -0.01392)=1;

dailyRet=backshift(1, positions).*ret1(trainset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% In-sample: CAGR=0.287519 Sharpe ratio=1.529458 maxDD=-0.118432 maxDDD=255 Calmar ratio=2.427712

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

positions=zeros(length(testset), 1);

positions(ret2(testset) >= 0.015314)=-1;
positions(ret2(testset) < 0.015314 & ret1(testset) < -0.0139236)=1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on SPY');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample: CAGR=0.038665 Sharpe ratio=0.505706 maxDD=-0.086725 maxDDD=625 Calmar ratio=0.445838

% Make "predictions" on training set (in-sample)
% 2) Use entire tree to predict returns

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
% In-sample: CAGR=0.734833 Sharpe ratio=2.463130 maxDD=-0.153416 maxDDD=96 Calmar ratio=4.789795

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
% Out-of-sample: CAGR=-0.072422 Sharpe ratio=-0.376600 maxDD=-0.441188 maxDDD=1155 Calmar ratio=-0.164152

% Build new tree with cross-validation =============
rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

model_cv=fitrtree([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset),  'MinLeafSize', 100, 'CrossVal', 'On', 'KFold', 5);
L= kfoldLoss(model_cv,'mode','individual'); % Find the loss (mean squared error) between the predicted responses and true responses in a fold when compared against predictions made with a tree trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the tree with the minimum loss, i.e. with least overfitting error.

bestTree=model_cv.Trained{minLidx};
% view(bestTree, 'mode', 'graph'); % see the tree visually

retPred1=predict(bestTree, [ret1(testset) ret2(testset) ret5(testset) ret20(testset)]);

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on SPY: Cross-validated. K=5.');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=0.005605 Sharpe ratio=0.115906 maxDD=-0.257338 maxDDD=633 Calmar ratio=0.021781

% Use only extreme nodes from bestTree for prediction

positions=zeros(length(testset), 1);

positions(ret2(testset) >= 0.015314)=-1;
positions(ret2(testset) < 0.015314 & ret1(testset) < -0.0172668)=1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on SPY');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=0.047740 Sharpe ratio=0.366381 maxDD=-0.272897 maxDDD=457 Calmar ratio=0.174938
