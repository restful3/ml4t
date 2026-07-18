% stepwiseLR_SPX.m
clear;

load('C:/Users/Ernest/Dropbox/AI_WS/fundamentalData', 'tday', 'syms', 'mid', 'indQ', 'indT', 'ARQ*', 'ART*');

holdingDays=252/4; % hold a quarter

ret1=calculateReturns(mid, 1);
vol1=smartMovingAvg(abs(ret1), 252); % daily returns svolatility
retQ=calculateReturns(mid, holdingDays); % quarterly return

retFut=fwdshift(holdingDays+1, retQ)./vol1; % shifted next quarter's return to today's row to use as response variable. Can enter only at next day's close. Normalized by volatility.

% Build stepwise LR model on training data
trainset=1:floor(length(tday)/2);

% Combine different independent variables into one matrix X for training
X=NaN(length(trainset)*length(syms), length(indQ)+length(indT));

Y=reshape(retFut(trainset, :), [length(trainset)*length(syms) 1]); % dependent variable

for iQ=1:length(indQ)
    eval(['ARQ_', indQ{iQ}, '=fillMissingData(ARQ_', indQ{iQ}, ');']);
    eval(['X(:, iQ)=reshape(ARQ_', indQ{iQ}, '(trainset, :), [length(trainset)*length(syms) 1]);']);
end

for iT=1:length(indT)
    eval(['ART_', indT{iT}, '=fillMissingData(ART_', indT{iT}, ');']);
    eval(['X(:, iQ+iT)=reshape(ART_', indT{iT}, '(trainset, :), [length(trainset)*length(syms) 1]);']);
end

% Select best linear model based on default criteria
model_train=stepwiselm(X, Y, 'Upper', 'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.

% 1. Adding x24, FStat = 21.7016, pValue = 3.34672e-06
% 2. Adding x21, FStat = 10.1019, pValue = 0.00149868
% 3. Adding x12, FStat = 13.3587, pValue = 0.000262354
% 4. Adding x1, FStat = 11.3753, pValue = 0.000755254
% 5. Adding x7, FStat = 8.7351, pValue = 0.0031497
% 6. Adding x5, FStat = 4.1443, pValue = 0.041877
% 7. Adding x15, FStat = 3.8475, pValue = 0.049929
% 
% model_train = 
% 
% 
% Linear regression model:
%     y ~ 1 + x1 + x5 + x7 + x12 + x15 + x21 + x24
% 
% Estimated Coefficients:
%                    Estimate        SE         tStat        pValue  
%                    _________    _________    ________    __________
% 
%     (Intercept)      -0.2492      0.65612    -0.37981       0.70412
%     x1               0.73582      0.19394      3.7941    0.00015157
%     x5             -0.013447    0.0067168      -2.002       0.04539
%     x7                5.4122       1.5858      3.4129    0.00065276
%     x12               3.5699       1.1307      3.1572     0.0016113
%     x15             0.043952     0.022408      1.9615      0.049929
%     x21              -1.1465      0.17551     -6.5324    7.7696e-11
%     x24              -16.146        2.978     -5.4216     6.455e-08
% 
% 
% Number of observations: 2585, Error degrees of freedom: 2577
% Root Mean Squared Error: 10.7
% R-squared: 0.0279,  Adjusted R-Squared 0.0253
% F-statistic vs. constant model: 10.6, p-value = 3.6e-13
% 
% indQ([1 5 ])
% 
% ans = 
% 
%     'CURRENTRATIO'    'TBVPS'
%     
% indT([7-5 12-5 15-5 21-5 24-5])
% 
% ans = 
% 
%     'EBITDAMARGIN'    'GROSSMARGIN'    'NCFOGROWTH1YR'    'PS'    'ROA'

retPred1=reshape(predict(model_train, X), [length(trainset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% Backtest trading model based on "prediction" on training set
positions=zeros(size(retPred1));

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

positions=backshift(1, positions); % Actually enter positions 1 day later
positions(1, :)=0;

% Hold for a quarter
pos=zeros(size(positions));
for h=1:holdingDays-1
   pos=backshift(h, positions);
   pos(~isfinite(pos))=0;
   positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
end

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% VERY BAD!!
% In-sample: CAGR=-0.007079 Sharpe ratio=0.075381 maxDD=-0.432976 maxDDD=692 Calmar ratio=-0.016351

assert(0, 'TODO');


% Try dollar-neutral strategy: long 50, short 50 stocks
topN=50;
retPred1=fillMissingData(retPred1);
positions=zeros(size(retPred1));
for t=1:size(retPred1, 1)
    goodData=find(isfinite(retPred1(t, :)));
    if (length(goodData) >= 2*topN)
        [~, idx]=sort(retPred1(t, goodData), 'ascend');
        positions(t, goodData(idx(1:topN)))=-1;
        positions(t, goodData(idx(end-topN+1:end)))=1;
    end
end

positions=backshift(1, positions); % Actually enter positions 1 day later
positions(1, :)=0;

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample (dollar-neutral): CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample (dollar-neutral): CAGR=0.035897 Sharpe ratio=0.622946
% maxDD=-0.136584 maxDDD=453 Calmar ratio=0.262820. 
% Not any better!

% Make real predictions on test (out-of-sample) set
testset=floor(length(tday)/2)+1:length(tday);

X=NaN(length(testset)*length(syms), length(indQ)+length(indT));

Y=reshape(retFut(testset, :), [length(testset)*length(syms) 1]); % dependent variable

for iQ=1:length(indQ)
    eval(['X(:, iQ)=reshape(ARQ_', indQ{iQ}, '(testset, :), [length(testset)*length(syms) 1]);']);
end

for iT=1:length(indT)
    eval(['X(:, iQ+iT)=reshape(ART_', indT{iT}, '(testset, :), [length(testset)*length(syms) 1]);']);
end

retPred1=reshape(predict(model_train, X), [length(testset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

positions=zeros(size(retPred1));

positions(retPred1 > 0)=1;
positions(retPred1 < 0)=-1;

positions=backshift(1, positions); % Actually enter positions 1 day later
positions(1, :)=0;

% Hold for a quarter
pos=zeros(size(positions));
for h=1:holdingDays-1
   pos=backshift(h, positions);
   pos(~isfinite(pos))=0;
   positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
end

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Stepwise regression on SPX fundamental facotrs');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);

% Out-of-sample: CAGR=0.038624 Sharpe ratio=1.112467 maxDD=-0.027653 maxDDD=240 Calmar ratio=1.396763

