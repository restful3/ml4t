% svm.m
% SVM with cross validation
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

retFut1=fwdshift(1, ret1); % shifted next day's return to today's row to use as response variable.

% Build model on training data
trainset=1:floor(length(tday)/2);

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

% Select best regression tree model based on default criteria with 5-fold
% cross validation
model=fitcsvm([ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)], retFut1(trainset) >= 0, 'CrossVal', 'On', 'KFold', 5); % Response: True if >=0, False if < 0.
L= kfoldLoss(model,'mode','individual'); % Find the loss (mean squared error) between the observations in a fold when compared against predictions made with a SVM trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the SVM with the minimum loss, i.e. with least overfitting error.

bestSVM=model.Trained{minLidx};


% Make "predictions" on training set (in-sample)
% Buy long if predicted value is True, otherwise short
isRetPositiveOrZero=predict(bestSVM, [ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)]);

positions=zeros(length(trainset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(trainset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated SVM on BTC.USD: train set');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% % In-sample: CAGR=-0.368098 Sharpe ratio=0.328017 maxDD=-0.615623 maxDDD=230885 Calmar ratio=-0.597928


% Test set
testset=floor(length(tday)/2)+1:length(tday);

isRetPositiveOrZero=predict(bestSVM,[ret1(testset) ret5(testset) ret10(testset) ret30(testset) ret60(testset)]);

positions=zeros(length(testset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated SVM on SPY: test set');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample (with CrossVal=on): CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample (with CrossVal=on): CAGR=-0.856935 Sharpe ratio=-2.034599 maxDD=-0.759027 maxDDD=236742 Calmar ratio=-1.128992



% Tuning SVM by optimizing "kernel scale"
% rng(1); % Fixing random number seed for reproducibility, since procedure uses subsampling like cross validation
% model=fitcsvm([ret1(trainset) ret2(trainset) ret5(trainset) ret20(trainset)], retFut1(trainset) >= 0, 'KernelScale','auto'); % Response: True if >=0, False if < 0.

% Tuning SVM by using cross validation
model=fitcsvm([ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)], retFut1(trainset) >= 0, 'KernelFunction', 'polynomial', 'KernelScale','auto', 'CrossVal', 'on', 'KFold', 5); % Response: True if >=0, False if < 0.

L= kfoldLoss(model,'mode','individual'); % Find the loss (mean squared error) between the observations in a fold when compared against predictions made with a SVM trained on the out-of-fold data.
[~, minLidx]=min(L); % pick the SVM with the minimum loss, i.e. with least overfitting error.

bestSVM=model.Trained{minLidx};


% Make "predictions" on training set (in-sample)
% Buy long if predicted value is True, otherwise short
isRetPositiveOrZero=predict(bestSVM, [ret1(trainset) ret5(trainset) ret10(trainset) ret30(trainset) ret60(trainset)]);

positions=zeros(length(trainset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(trainset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated SVM on BTC.USD with KernelFunction=polynomial, KernelScale=auto: train set');
xlabel('Date');
ylabel('Cumulative Returns');


cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample (with KernelFunction=polynomial, KernelScale=auto, CrossVal=on): CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);



% Test set
testset=floor(length(tday)/2)+1:length(tday);

isRetPositiveOrZero=predict(bestSVM, [ret1(testset) ret5(testset) ret10(testset) ret30(testset) ret60(testset)]);

positions=zeros(length(testset), 1);

positions(isRetPositiveOrZero)=1;
positions(~isRetPositiveOrZero)=-1;

dailyRet=backshift(1, positions).*ret1(testset);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Cross-validated SVM on BTC.USD with KernelFunction=polynomial, KernelScale=auto: test set');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(365*24*60/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample (with KernelFunction=polynomial, KernelScale=auto, CrossVal=on): CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(365*24*60)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

