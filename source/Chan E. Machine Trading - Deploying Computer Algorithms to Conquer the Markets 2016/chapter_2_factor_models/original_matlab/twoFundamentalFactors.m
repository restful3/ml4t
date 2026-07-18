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
X=NaN(length(trainset)*length(syms), 2);

Y=reshape(retFut(trainset, :), [length(trainset)*length(syms) 1]); % dependent variable

earningsInc=ARQ_EPS; % Earnings (net income) per Basic Share

bvpershr=ARQ_BVPS;
bvpershr_lag=backshift(1, fillMissingData(bvpershr));

ROE=1+earningsInc./bvpershr_lag;  
ROE(ROE <= 0)=NaN;

BM=1./ARQ_PB;
BM(BM <= 0)=NaN;

X(:, 1)=reshape(log(BM(trainset, :)), [length(trainset)*length(syms) 1]);
X(:, 2)=reshape(log(ROE(trainset, :)), [length(trainset)*length(syms) 1]);


% Linear regression
model_train=fitlm(X, Y,  'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.

% model_train = 
% 
% 
% Linear regression model:
%     y ~ 1 + x1 + x2
% 
% Estimated Coefficients:
%                         Estimate                 SE                   tStat                 pValue       
%                    __________________    ___________________    _________________    ____________________
% 
%     (Intercept)      0.01726811103609     0.0022719264606344     7.60064699949314    3.42083178989552e-14
%     x1             0.0107086508799573    0.00203364533711983      5.2657416140828     1.4470349996317e-07
%     x2             -0.049573284642887     0.0113200791085867    -4.37923482401142    1.21212357864216e-05
% 
% 
% Number of observations: 5747, Error degrees of freedom: 5744
% Root Mean Squared Error: 0.116
% R-squared: 0.0119,  Adjusted R-Squared 0.0115
% F-statistic vs. constant model: 34.6, p-value = 1.2e-15

retPred=reshape(predict(model_train, X), [length(trainset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% Backtest trading model based on "prediction" on training set
% 
% 
% positions(retPred > 0)=1;
% positions(retPred < 0)=-1;
% 
% positions=backshift(1, positions); % Actually enter positions 1 day later
% positions(1, :)=0;
% 
% % Hold for a month
% pos=zeros(size(positions));
% for h=1:holdingDays-1
%    pos=backshift(h, positions);
%    pos(~isfinite(pos))=0;
%    %    positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
% end

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
% In-sample: CAGR=0.050343 Sharpe ratio=0.370628 maxDD=-0.259442 maxDDD=395 Calmar ratio=0.194043


% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X(:, 1)=reshape(log(BM(testset, :)), [length(testset)*length(syms) 1]);
X(:, 2)=reshape(log(ROE(testset, :)), [length(testset)*length(syms) 1]);


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
title('Linear regression on SPX log(ROE) and log(BM)');
xlabel('Date');
ylabel('Cumulative Returns');
% savefig('c05f002 (Linear regression on log ROE and BM).fig')

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% holdingDays=21
% ***Out-of-sample: CAGR=0.093199 Sharpe ratio=1.010660 maxDD=-0.120718 maxDDD=349 Calmar ratio=0.772039
% holdingDays=252/4
% Out-of-sample: CAGR=0.069957 Sharpe ratio=1.054519 maxDD=-0.087289 maxDDD=192 Calmar ratio=0.801444

