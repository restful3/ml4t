% Find best AR(p) model (i.e. best p) 
clear;
load('inputData_AUDUSD_20150807', 'tday', 'hhmm', 'mid');

idx=find(isfinite(mid));
tday(1:idx-1)=[];
hhmm(1:idx-1)=[];
mid(1:idx-1)=[];

LOGL=zeros(60, 1); % log likelihood for up to 60 lags (1 hour)
P=zeros(size(LOGL)); % p values

trainset=1:(length(mid)-252*(24*60-15)); % Use all but 1 year of bars for in-sample fitting



for p=1:length(P)
    model=arima(p, 0, 0);
    [~,~,logL] = estimate(model, mid(trainset),'print',false); 
    LOGL(p) = logL;
    P(p) = p;
    
end

% Has P+1 parameters, including constant
[~, bic]=aicbic(LOGL, P+1, length(mid(trainset)));

[~, pMin]=min(bic)
% pMin =
% 
%     10

model=arima(pMin, 0, 0) % assumes an AR(pMin) with unknown parameters
fit=estimate(model, mid);

%     ARIMA(10,0,0) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant    1.37196e-06   8.65314e-07        1.58551
%         AR{1}       0.993434   0.000187164        5307.82
%         AR{2}    -0.00121205   0.000293356       -4.13166
%         AR{3}   -0.000352717   0.000305831       -1.15331
%         AR{4}    0.000753222   0.000354121        2.12702
%         AR{5}     0.00662641   0.000358673        18.4748
%         AR{6}    -0.00224118   0.000330092       -6.78956
%         AR{7}    -0.00305157   0.000365348       -8.35252
%         AR{8}     0.00351317   0.000394538        8.90452
%         AR{9}    -0.00154844   0.000398956       -3.88124
%        AR{10}     0.00407798   0.000281821        14.4701
%      Variance    5.03463e-08   3.43847e-10        146.421

testset=trainset(end)+1:length(mid);

yF=NaN(size(mid));
for t=testset(1):size(mid, 1)
    [y, ~]=forecast(fit, 1, 'Y0', mid(t-pMin+1:t)); % Need only most recent pMin data points for prediction
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
title('AR(10) model on AUDUSD');
xlabel('Date');
ylabel('Cumulative Returns');


% cumret(end)=0.731300107386533

% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1
% 1.584085074291988

% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*(24*60-15)/length(testset))-1
%    1.584085074291988
