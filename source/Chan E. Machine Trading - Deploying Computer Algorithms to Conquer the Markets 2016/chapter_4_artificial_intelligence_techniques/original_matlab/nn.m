% nn.m Neural Network
clear;

load('inputData_SPY');

ret1=calculateReturns(cl, 1);
ret2=calculateReturns(cl, 2);
ret5=calculateReturns(cl, 5);
ret20=calculateReturns(cl, 20);

% Note: no need for extra target variable, since ret1 is what we want to
% predict. By convention of Neural Network Toolbox, input upto t will be
% use to predict target at time t+1, unlike our old convention
% retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

% Prepare input data as cell array with T columns, each cell is a column
% vector with 3 rows. We do not need to include ret1 since it is used as
% target variable, and by default it will be fed back
% as input during training
X=num2cell([ret2(trainset), ret5(trainset), ret20(trainset)]', [1 length(trainset)]); 
T=num2cell(ret1(trainset)', [1 length(trainset)]);

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);


% "open loop" network: no explicit feedback, but target time series is used
% as input
% By default, X(t-2), X(t-1) (but not X(t)!) and T(t-2), T(t-1) are used as input
% Hence output Y has 2 fewer columns, and Y(t-2) corresponds to T(t) (i.e.
% Y(1) is prediction for T(3), but Y(end) is prediction for T(end)
net = narxnet; 
[Xs,Xi,Ai,Ts] = preparets(net,X,{},T); 

net.divideParam.trainRatio=0.6; % 0.6 (default is 0.7) Pick 4/5 of trainset randomly to serve as train data
net.divideParam.valRatio=0.4; % 0.4 (default is 0.15) Pick 1/5 of remaining trainset to serve as validation data for early stopping
net.divideParam.testRatio=0;

net = train(net,Xs,Ts,Xi,Ai); 
% view(net) 
Y = net(Xs,Xi,Ai); 
% plotresponse(Ts,Y) 

% Now use trained network for prediction.
% Output y is the same as Y, since input is the same. But y
% has 1 extra output at the end, since predictive network netp can use
% X(end-1), X(end) and T(end-1), T(end) to predict T(end+1) (not observable
% in training data of course.)
netp = removedelay(net);
% view(netp)
[Xs,Xi,Ai,Ts] = preparets(netp,X,{},T);
y = netp(Xs,Xi,Ai);

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

% In-sample, trainRatio=0.7 valRatio=0.15: CAGR=0.552864 Sharpe ratio=1.985561 maxDD=-0.212750 maxDDD=194 Calmar ratio=2.598661
% In-sample, trainRatio=0.8 valRatio=0.2: CAGR=0.602797 Sharpe ratio=2.122199 maxDD=-0.212750 maxDDD=176 Calmar ratio=2.833364
% In-sample, trainRatio=0.7 valRatio=0.3: CAGR=0.552213 Sharpe ratio=1.983753 maxDD=-0.212750 maxDDD=194 Calmar ratio=2.595601
% In-sample, trainRatio=0.6 valRatio=0.4: CAGR=0.196943 Sharpe ratio=0.876671 maxDD=-0.211257 maxDDD=290 Calmar ratio=0.932245
% In-sample, trainRatio=0.5 valRatio=0.5: CAGR=0.358382 Sharpe ratio=1.413631 maxDD=-0.223841 maxDDD=135 Calmar ratio=1.601058

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X=num2cell([ret2(testset), ret5(testset), ret20(testset)]', [1 length(testset)]); 
T=num2cell(ret1(testset)', [1 length(testset)]);

[Xs,Xi,Ai,Ts] = preparets(netp,X,{},T);
y = netp(Xs,Xi,Ai);
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

% Out-of-sample, trainRatio=0.7 valRatio=0.15: CAGR=0.043669 Sharpe ratio=0.342624 maxDD=-0.259253 maxDDD=594 Calmar ratio=0.168443
% Out-of-sample, trainRatio=0.8 valRatio=0.2: CAGR=0.009660 Sharpe ratio=0.140571 maxDD=-0.283227 maxDDD=701 Calmar ratio=0.034108
% Out-of-sample, trainRatio=0.7 valRatio=0.3: CAGR=0.043669 Sharpe ratio=0.342624 maxDD=-0.259253 maxDDD=594 Calmar ratio=0.168443
% Out-of-sample, trainRatio=0.6 valRatio=0.4: CAGR=0.116512 Sharpe ratio=0.754619 maxDD=-0.262600 maxDDD=439 Calmar ratio=0.443689
% Out-of-sample: trainRatio=0.5 valRatio=0.5: CAGR=0.046413 Sharpe ratio=0.358525 maxDD=-0.318260 maxDDD=701 Calmar ratio=0.145835
