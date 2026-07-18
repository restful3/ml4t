% % computerFundamentalFactors2.m
% Use data from Sharadar instead
clear;

p=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms', 'bid', 'ask');
load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/fundamentals', 'tday', 'syms', 'ARQ*', 'ART*');
% load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/indicators', 'indQ', 'indT');

assert(all(strcmp(syms, p.syms)));
assert(all(tday==p.tday));
mid=(p.bid+p.ask)/2;


holdingDays=21; % hold for 21 days (1 month)

retM=calculateReturns(mid, holdingDays); % monthly return

retFut=fwdshift(holdingDays+1, retM); % shifted next quarter's return to today's row to use as response variable. Can enter only at next day's close.

trainset=1:floor(length(tday)/2);

% Combine different independent variables into one matrix X for training
X=NaN(length(trainset)*length(syms), 1);

Y=reshape(retFut(trainset, :), [length(trainset)*length(syms) 1]); % dependent variable

earningsInc=ARQ_EPS; % Earnings (net income) per Basic Share

bvpershr=ARQ_BVPS;
bvpershr_lag=backshift(1, fillMissingData(bvpershr));

ROE=1+earningsInc./bvpershr_lag;  
ROE(ROE <= 0)=NaN;

% BM=1./ARQ_PB;
% BM(BM <= 0)=NaN;

% X(:, 1)=reshape(log(BM(trainset, :)), [length(trainset)*length(syms) 1]);
X(:, 1)=reshape(log(ROE(trainset, :)), [length(trainset)*length(syms) 1]);


% Linear regression
model_train=fitlm(X, Y,  'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.
% model_train = 
% 
% 
% Linear regression model:
%     y ~ 1 + x1
% 
% Estimated Coefficients:
%                         Estimate                  SE                  tStat                 pValue       
%                    ___________________    ___________________    ________________    ____________________
% 
%     (Intercept)    0.00849309804147072    0.00156979541845461    5.41032158816706    6.53996435064286e-08
%     x1             -0.0527684550470039    0.00913169802481018    -5.7786027202866    7.91902769869272e-09
% 
% 
% Number of observations: 5878, Error degrees of freedom: 5876
% Root Mean Squared Error: 0.118
% R-squared: 0.00565,  Adjusted R-Squared 0.00548
% F-statistic vs. constant model: 33.4, p-value = 7.92e-09

retPred=reshape(predict(model_train, X), [length(trainset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions


longs=backshift(1, retPred>0); %1 day later
shorts=backshift(1, retPred<0);

longs(1, :)=false;
shorts(1, :)=false;

positions=zeros(size(retPred));

for h=0:holdingDays-1
    long_lag=backshift(h, longs);
    long_lag(isnan(long_lag))=false;
    long_lag=logical(long_lag);
    
    short_lag=backshift(h, shorts);
    short_lag(isnan(short_lag))=false;
    short_lag=logical(short_lag);
    
    positions(long_lag)=positions(long_lag)+1;
    positions(short_lag)=positions(short_lag)-1;
end

ret1=calculateReturns(mid, 1);

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=0.014004 Sharpe ratio=0.177303 maxDD=-0.426541 maxDDD=619 Calmar ratio=0.032832



% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

% X(:, 1)=reshape(log(BM(testset, :)), [length(testset)*length(syms) 1]);
X(:, 1)=reshape(log(ROE(testset, :)), [length(testset)*length(syms) 1]);


retPred=reshape(predict(model_train, X), [length(testset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

longs=backshift(1, retPred>0); %1 day later
shorts=backshift(1, retPred<0);

longs(1, :)=false;
shorts(1, :)=false;

positions=zeros(size(retPred));

for h=0:holdingDays-1
    long_lag=backshift(h, longs);
    long_lag(isnan(long_lag))=false;
    long_lag=logical(long_lag);
    
    short_lag=backshift(h, shorts);
    short_lag(isnan(short_lag))=false;
    short_lag=logical(short_lag);
    
    positions(long_lag)=positions(long_lag)+1;
    positions(short_lag)=positions(short_lag)-1;
end

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Linear regression on SPX log(ROE)');
xlabel('Date');
ylabel('Cumulative Returns');
% savefig('c05f003 (Linear regression on log ROE ).fig')

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% holdingDays=21
% Out-of-sample: Out-of-sample: CAGR=0.201661 Sharpe ratio=1.305036 maxDD=-0.182143 maxDDD=149 Calmar ratio=1.107156

