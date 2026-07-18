% nn_feedfwd.m Feedforward neural network 
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

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable

% Build model on training data
trainset=1:floor(length(tday)/2);

trainset1=trainset(1:floor(length(trainset)*4/5));
trainset2=trainset(floor(length(trainset)*1/5)+1:end);

X1=num2cell([ret1(trainset1) ret5(trainset1) ret10(trainset1) ret30(trainset1) ret60(trainset1)]', [1 length(trainset1)]); 
T1=num2cell(retFut1(trainset1)', [1 length(trainset1)]);
X2=num2cell([ret1(trainset2) ret5(trainset2) ret10(trainset2) ret30(trainset2) ret60(trainset2)]', [1 length(trainset2)]); 
T2=num2cell(retFut1(trainset2)', [1 length(trainset2)]);

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

hiddenSizes=[1];
net = feedforwardnet(hiddenSizes); % Use hid 

net.divideParam.trainRatio=0.6; 
net.divideParam.valRatio=0.4; 
net.divideParam.testRatio=0;

numNN=100;
NN = cell(1,numNN);
% errors = zeros(1,numNN);
for i=1:numNN
  disp(['Training ' num2str(i) '/' num2str(numNN)])
  NN{i}  = train(net,X1, T1);
  %   y = NN{i}(X2);
  %   errors(i) = mse(NN{i},T2,y);
end

% To align the prediction with the input array, turn y back to column vector and add a NaN to beginning 
X=num2cell([ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)]', [1 length(trainset)]); 
y=zeros(size(retFut1(trainset)))';
for i=1:numNN
  disp(['In-sample testing ' num2str(i) '/' num2str(numNN)])
  y = y+cell2mat(NN{i}(X));
end
y = y/numNN;
retPred1=y;
retPred1=[repmat(NaN, [length(trainset)-length(retPred1) 1]); retPred1];


% Make "predictions" on training set (in-sample) similar to lr.m
positions=zeros(length(trainset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(trainset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);



% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X=num2cell([ret1(testset) ret5(testset) ret10(testset) ret30(testset) ret60(testset)]', [1 length(testset)]); 
y=zeros(size(retFut1(testset)))';
for i=1:numNN
  disp(['Out-of-sample testing ' num2str(i) '/' num2str(numNN)])
  y = y+cell2mat(NN{i}(X));
end
y = y/numNN;

retPred1=y;
retPred1=[repmat(NaN, [length(testset)-length(retPred1) 1]); retPred1];

positions=zeros(length(testset), 1);

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample: CAGR=1115340053.572500 Sharpe ratio=22.171526 maxDD=-0.703360 maxDDD=8951 Calmar ratio=1585730842.299765

