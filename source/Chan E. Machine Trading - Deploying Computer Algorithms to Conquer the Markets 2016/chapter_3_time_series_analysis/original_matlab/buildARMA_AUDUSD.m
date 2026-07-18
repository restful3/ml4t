% Find best ARMA(p, q) model (i.e. best p, q) 
clear;
load('inputData_AUDUSD_20150807', 'tday', 'hhmm', 'mid');

idx=find(isfinite(mid));
tday(1:idx-1)=[];
hhmm(1:idx-1)=[];
mid(1:idx-1)=[];


LOGL=-Inf(10, 9); % log likelihood for up to 10 p and 9 q (10 minutes)
PQ=zeros(size(LOGL)); % p values

trainset=1:(length(mid)-252*(24*60-15)); % Use all but 1 year of bars for in-sample fitting

% p, q from buildARMA_findPQ_AUDUSD.m               
model=arima(2, 0, 5) % assumes an AR(pMin) with unknown parameters
fit=estimate(model, mid(trainset));


testset=trainset(end)+1:length(mid);

yF=NaN(size(mid));
for t=testset(1):size(mid, 1)
    [y, ~]=forecast(fit, 1, 'Y0', mid(t-5+1:t)); % Need only most recent pMin data points for prediction
    yF(t)=y(end);
end

deltaYF=yF-mid;

% Trading strategy
pos=zeros(size(mid));
pos(deltaYF > 0)=1;
pos(deltaYF < 0)=-1;

ret=backshift(1, pos).*(mid-backshift(1, mid))./backshift(1, mid);
ret(isnan(ret))=0;
cumret=cumprod(1+ret)-1;


plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset));
title('ARMA(2, 5) model on AUDUSD');
xlabel('Date');
ylabel('Cumulative Returns');

% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1

% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*(24*60-15)/length(testset))-1


%    ARIMA(2,0,5) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant    2.80383e-06   4.58975e-06       0.610891
%         AR{1}       0.649011   0.000249771        2598.42
%         AR{2}       0.350986   0.000249775        1405.21
%         MA{1}       0.345806   0.000499929        691.711
%         MA{2}    -0.00906282   0.000874713       -10.3609
%         MA{3}     -0.0106082   0.000896239       -11.8363
%         MA{4}     -0.0102606     0.0010664       -9.62167
%         MA{5}    -0.00251154   0.000910359       -2.75884
%      Variance          2e-07   1.10884e-09        180.368
% 
% ans =
% 
%    0.602164545675666
% 
% 
% ans =
% 
%    0.602164545675666

