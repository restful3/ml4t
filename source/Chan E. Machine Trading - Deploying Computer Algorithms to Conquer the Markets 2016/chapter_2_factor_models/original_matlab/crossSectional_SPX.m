% crossSectional_SPX.m
clear;

load('fundamentalData', 'tday', 'syms', 'mid', 'indQ', 'indT', 'ARQ*', 'ART*');

holdingDays=252/4; % hold a quarter

ret1=calculateReturns(mid, 1);
% vol1=smartMovingAvg(abs(ret1), 252); % daily returns svolatility
retQ=calculateReturns(mid, holdingDays); % quarterly return

% retFut=fwdshift(holdingDays+1, retQ)./vol1; % shifted next quarter's return to today's row to use as response variable. Can enter only at next day's close. Normalized by volatility.
retFut=fwdshift(holdingDays+1, retQ); % shifted next quarter's return to today's row to use as response variable. Can enter only at next day's close.

% Build  LR model on training data
trainset=1:floor(length(tday)/2);

% Combine different independent variables into one matrix X for training
X=NaN(length(trainset)*length(syms), length(indQ)+length(indT));

Y=reshape(retFut(trainset, :), [length(trainset)*length(syms) 1]); % dependent variable

for iQ=1:length(indQ)
%     eval(['ARQ_', indQ{iQ}, '=fillMissingData(ARQ_', indQ{iQ}, ');']);
    eval(['X(:, iQ)=reshape(ARQ_', indQ{iQ}, '(trainset, :), [length(trainset)*length(syms) 1]);']);
end

for iT=1:length(indT)
%     eval(['ART_', indT{iT}, '=fillMissingData(ART_', indT{iT}, ');']);
    eval(['X(:, iQ+iT)=reshape(ART_', indT{iT}, '(trainset, :), [length(trainset)*length(syms) 1]);']);
end

% Linear regression
model_train=fitlm(X, Y,  'linear')  % By default, there is a constant term in model, so do not include column of 1's in predictors.
% model_train = 
% 
% 
% Linear regression model:
%     y ~ [Linear formula with 28 terms in 27 predictors]
% 
% Estimated Coefficients:
%                     Estimate          SE          tStat        pValue  
%                    ___________    __________    _________    __________
% 
%     (Intercept)              0             0          NaN           NaN
%     x1                0.015319    0.00034896       43.899             0
%     x2             -0.00039995    4.0103e-05       -9.973     2.023e-23
%     x3                       0             0          NaN           NaN
%     x4              0.00021074     2.834e-05       7.4362    1.0397e-13
%     x5              1.1903e-05    1.1816e-05       1.0073       0.31377
%     x6              -0.0010164    0.00046848      -2.1695      0.030046
%     x7                       0             0          NaN           NaN
%     x8             -2.9817e-05    9.4152e-05     -0.31669       0.75148
%     x9              1.5217e-13    3.8232e-14       3.9802    6.8864e-05
%     x10             3.3417e-07    8.2406e-07      0.40552        0.6851
%     x11             2.3913e-05    7.5889e-06        3.151     0.0016274
%     x12                      0             0          NaN           NaN
%     x13             -3.813e-06    5.6534e-05    -0.067446       0.94623
%     x14             0.00090827    4.8986e-05       18.541    1.0885e-76
%     x15              0.0009909    5.5897e-05       17.727    2.8806e-70
%     x16             4.0453e-05    1.3379e-05       3.0237     0.0024975
%     x17                      0             0          NaN           NaN
%     x18             0.00072159     0.0003482       2.0723      0.038235
%     x19            -8.6964e-07    3.8115e-07      -2.2816      0.022513
%     x20            -6.2076e-06    6.5958e-06     -0.94114       0.34663
%     x21              -0.011034    0.00024537       -44.97             0
%     x22                      0             0          NaN           NaN
%     x23                      0             0          NaN           NaN
%     x24                      0             0          NaN           NaN
%     x25             -0.0071657     0.0003957      -18.109    3.0512e-73
%     x26                      0             0          NaN           NaN
%     x27            -0.00023315    0.00010027      -2.3251      0.020066
% 
% 
% Number of observations: 228564, Error degrees of freedom: 228545
% Root Mean Squared Error: 0.214
% R-squared: 0.0147,  Adjusted R-Squared 0.0146
% F-statistic vs. constant model: 189, p-value = 0

retPred=reshape(predict(model_train, X), [length(trainset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% Backtest trading model based on "prediction" on training set
% positions=zeros(size(retPred));

% for t=1:length(trainset)
%     goodData=find(isfinite(retPred(t, :)));
%     if (length(goodData) >= 2*topN)
%         
%         [~, I]=sort(retPred(t, goodData)); % ascending sort
%         positions(t, goodData(I(1:topN)))=-1;
%         positions(t, goodData(I(end-topN+1)))=1;
%     end
% end

% positions(retPred > 0)=1;
% positions(retPred < 0)=-1;
% 
% positions=backshift(1, positions); % Actually enter positions 1 day later
% positions(1, :)=0;
% 
% % Hold for a quarter
% pos=zeros(size(positions));
% for h=1:holdingDays-1
%    pos=backshift(h, positions);
%    pos(~isfinite(pos))=0;
%    positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
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

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=0.043255 Sharpe ratio=0.391330 maxDD=-0.231885 maxDDD=604 Calmar ratio=0.186536


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

retPred=reshape(predict(model_train, X), [length(testset) length(syms)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% positions=zeros(size(retPred));

% for t=1:length(testset)
%     goodData=find(isfinite(retPred(t, :)));
%     if (length(goodData) >= 2*topN)
%         
%         [~, I]=sort(retPred(t, goodData)); % ascending sort
%         positions(t, goodData(I(1:topN)))=-1;
%         positions(t, goodData(I(end-topN+1)))=1;
%     end
% end

% positions(retPred > 0)=1;
% positions(retPred < 0)=-1;
% 
% positions=backshift(1, positions); % Actually enter positions 1 day later
% positions(1, :)=0;
% 
% % Hold for a quarter
% pos=zeros(size(positions));
% for h=1:holdingDays-1
%    pos=backshift(h, positions);
%    pos(~isfinite(pos))=0;
%    positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
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

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Linear regression on SPX fundamental facotrs');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=0.123549 Sharpe ratio=1.675774 maxDD=-0.076124 maxDDD=174 Calmar ratio=1.623003


