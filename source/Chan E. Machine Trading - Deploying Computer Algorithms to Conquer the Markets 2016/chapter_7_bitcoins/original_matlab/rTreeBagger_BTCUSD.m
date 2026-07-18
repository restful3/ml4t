% rTreeBagger.m
% Ensemble of regression trees (bagged trees)
clear;

load('Jonathan_BTCUSD_BBO_1minute', 'tday', 'HHMM', 'bid', 'ask');

cl=(bid+ask)/2;

idx=find(isfinite(cl));
tday(1:idx-1)=[];
HHMM(1:idx-1)=[];
cl(1:idx-1)=[];

ret1=calculateReturns(cl, 1);
ret5=calculateReturns(cl, 5);
ret10=calculateReturns(cl, 10);
ret30=calculateReturns(cl, 30);
ret60=calculateReturns(cl, 60);

K=5;

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

% Select best regression tree model based on default criteria
rng(1); % set fixed random seed for sampling training set, in order to get reproducible results
model=TreeBagger(K, [ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)], retFut1(trainset), 'Method', 'regression', 'MinLeaf', 100);

% Make "predictions" on training set (in-sample)
retPred1=predict(model, [ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)]);
% retPred1=oobPredict(model);

% Backtest trading model based on "prediction" on training set
positions=zeros(length(trainset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

minuteRet=backshift(1, positions).*ret1(trainset);
minuteRet(~isfinite(minuteRet))=0;

cumret=cumprod(1+minuteRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(minuteRet)/std(minuteRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=24831842721782886000000000000000000000000000000000000.000000 Sharpe ratio=91.657255 maxDD=-0.162480 maxDDD=308 Calmar ratio=152829737531608230000000000000000000000000000000000000.000000


% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

retPred1=predict(model, [ret1(testset) ret5(testset) ret10(testset) ret30(testset) ret60(testset)]);

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

minuteRet=backshift(1, positions).*ret1(testset);
minuteRet(~isfinite(minuteRet))=0;

cumret=cumprod(1+minuteRet)-1;

% plot(cumret);
plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Regression tree on BTC.USD with bagging. K=5.');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(minuteRet)/std(minuteRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=75340814462.263458 Sharpe ratio=31.810122 maxDD=-0.294237 maxDDD=3750 Calmar ratio=256055141558.025790

