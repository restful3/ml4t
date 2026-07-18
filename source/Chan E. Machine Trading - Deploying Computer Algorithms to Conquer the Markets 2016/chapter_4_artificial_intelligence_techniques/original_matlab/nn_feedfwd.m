% nn_feedfwd.m Feedforward neural network 
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

X=num2cell([ret1(trainset) ret2(trainset), ret5(trainset), ret20(trainset)]', [1 length(trainset)]); 
T=num2cell(retFut1(trainset)', [1 length(trainset)]);

rng('default'); % Fix random number generator seed to get repeatable results
rng(2);

hiddenSizes=[1];
net = feedforwardnet(hiddenSizes); % Use hid 

net.divideParam.trainRatio=0.6; % 0.6 (default is 0.7) Pick 4/5 of trainset randomly to serve as train data
net.divideParam.valRatio=0.4; % 0.4 (default is 0.15) Pick 1/5 of remaining trainset to serve as validation data for early stopping
net.divideParam.testRatio=0;

net = train(net,X, T); 
% view(net) 
y = net(X); 

% Now use trained network for prediction.

% To align the prediction with the input array, turn y back to column vector and add a NaN to beginning 
retPred1=cell2mat(y)';
retPred1=[repmat(NaN, [length(T)-length(retPred1) 1]); retPred1];


% Make "predictions" on training set (in-sample) similar to lr.m
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

% In-sample: CAGR=0.188458 Sharpe ratio=0.845639 maxDD=-0.224498 maxDDD=473 Calmar ratio=0.839464

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X=num2cell([ret1(testset), ret2(testset), ret5(testset), ret20(testset)]', [1 length(testset)]); 

y = net(X);
retPred1=cell2mat(y)';
retPred1=[repmat(NaN, [length(T)-length(retPred1) 1]); retPred1];

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

% Out-of-sample: CAGR=-0.039297 Sharpe ratio=-0.162372 maxDD=-0.369031 maxDDD=804 Calmar ratio=-0.106488
