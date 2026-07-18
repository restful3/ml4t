% rTree_SPX.m
% Predicting SPX stocks returns using only technical factors
clear;

load('C:/Users/Ernest/Dropbox/AI_WS/fundamentalData', 'tday', 'syms', 'mid');

cl=mid; % Use mid quote at close

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

vol1=smartMovingAvg(abs(ret1), 252); % daily returns volatility

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Normalization of all variables

ret1N=ret1./vol1;
ret2N=ret2./vol1;
ret5N=ret5./vol1;
ret20N=ret20./vol1;

retFut1N=retFut1./vol1;
% ret1N=ret1;
% ret2N=ret2;
% ret5N=ret5;
% ret20N=ret20;

% retFut1N=retFut1;

% Build stepwise LR model on training data
trainset=1:floor(length(tday)/2);

% Combine different independent variables into one matrix X for training
X=NaN(length(trainset)*length(syms), 4);
X(:, 1)=reshape(ret1N(trainset, :), [length(trainset)*length(syms) 1]);
X(:, 2)=reshape(ret2N(trainset, :), [length(trainset)*length(syms) 1]);
X(:, 3)=reshape(ret5N(trainset, :), [length(trainset)*length(syms) 1]);
X(:, 4)=reshape(ret20N(trainset, :), [length(trainset)*length(syms) 1]);

Y=reshape(retFut1N(trainset, :), [length(trainset)*length(syms) 1]); % dependent variable

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);
% Select best regression tree model based on default criteria
model_cv=fitrtree(X, Y, 'MinLeafSize', 100, 'CrossVal', 'On', 'KFold', 5);
L= kfoldLoss(model_cv,'mode','individual'); % Find the loss (mean squared error) between the predicted responses and true responses in a fold when compared against predictions made with a tree trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the tree with the minimum loss, i.e. with least overfitting error.

bestTree=model_cv.Trained{minLidx};

retPred1=reshape(predict(bestTree, X), [length(trainset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% Backtest trading model based on "prediction" on training set
positions=zeros(size(retPred1));

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=0.789787 Sharpe ratio=6.321720 maxDD=-0.081501 maxDDD=255 Calmar ratio=9.690483

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X=NaN(length(testset)*length(syms), 4);
X(:, 1)=reshape(ret1N(testset, :), [length(testset)*length(syms) 1]);
X(:, 2)=reshape(ret2N(testset, :), [length(testset)*length(syms) 1]);
X(:, 3)=reshape(ret5N(testset, :), [length(testset)*length(syms) 1]);
X(:, 4)=reshape(ret20N(testset, :), [length(testset)*length(syms) 1]);

Y=reshape(retFut1N(testset, :), [length(testset)*length(syms) 1]); % dependent variable


retPred1=reshape(predict(bestTree, X), [length(testset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

positions=zeros(size(retPred1));

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

positions=backshift(1, positions); % Actually enter positions 1 day later
positions(1, :)=0;

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on SPX components: Cross-validated. K=5.');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=0.023153 Sharpe ratio=0.872566 maxDD=-0.048680 maxDDD=540 Calmar ratio=0.475607

% Without normalization
% Out-of-sample: CAGR=-0.007218 Sharpe ratio=-0.374746 maxDD=-0.048693 maxDDD=715 Calmar ratio=-0.148238

