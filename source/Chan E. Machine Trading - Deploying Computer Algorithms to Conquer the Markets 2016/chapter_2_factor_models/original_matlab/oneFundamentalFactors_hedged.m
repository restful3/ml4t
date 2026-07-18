% Use SPY to make portfolio dollar neutral at all times
% twoFundamentalFactors_hedged.m
% Use data from Sharadar instead
clear;

p=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms', 'bid', 'ask');
s=load('C:/Projects/prod_data/inputDataOHLCDaily_ETF_20151023', 'tday', 'stocks', 'cl');
spy=s.cl(:, strcmp(s.stocks, 'SPY'));

load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/fundamentals', 'tday', 'syms', 'ARQ*', 'ART*');
% load('C:/Users/Ernest/Dropbox/Hedge fund/QTS/Backtests/Quandl/SF1/indicators', 'indQ', 'indT');



assert(all(strcmp(syms, p.syms)));
assert(all(tday==p.tday));
mid=(p.bid+p.ask)/2;

[~, idx1, idx2]=intersect(tday, s.tday);
assert(all(tday==s.tday(idx2)));

spy=spy(idx2);


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



% X(:, 1)=reshape(log(BM(trainset, :)), [length(trainset)*length(syms) 1]);
X(:, 1)=reshape(log(ROE(trainset, :)), [length(trainset)*length(syms) 1]);


% Linear regression
model_train=fitlm(X, Y,  'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.


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

% Hedge with SPY
positions=[positions -smartsum(positions, 2)];

ret1=[calculateReturns(mid, 1) calculateReturns(spy, 1)];

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=0.039345 Sharpe ratio=0.873104 maxDD=-0.056378 maxDDD=384 Calmar ratio=0.697878



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

% Hedge with SPY
positions=[positions -smartsum(positions, 2)];

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Linear regression on SPX log(ROE) and log(BM)');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=0.002676 Sharpe ratio=0.143295 maxDD=-0.041643 maxDDD=540 Calmar ratio=0.064261


