% Find best ARMA(p, q) model (i.e. best p, q) 
clear;
load('inputData_AUDUSD_20150807', 'tday', 'hhmm', 'mid');

idx=find(isfinite(mid));
tday(1:idx-1)=[];
hhmm(1:idx-1)=[];
mid(1:idx-1)=[];

mid=log(mid);

trainset=1:(length(mid)-252*(24*60-15)); % Use all but 1 year of bars for in-sample fitting

% p, r, q from buildARIMA_findPQ_AUDUSD.m               
model=arima(1, 1, 9) % assumes an AR(pMin) with unknown parameters
fit=estimate(model, mid(trainset));


testset=trainset(end)+1:length(mid);

yF=NaN(size(mid));
for t=testset(1):size(mid, 1)

    [y, ~]=forecast(fit, 1, 'Y0', mid(t-9+1:t)); % Need only most recent pMin data points for prediction
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
title('ARIMA(1, 1, 9) model on AUDUSD');
xlabel('Date');
ylabel('Cumulative Returns');

% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1

% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*(24*60-15)/length(testset))-1
