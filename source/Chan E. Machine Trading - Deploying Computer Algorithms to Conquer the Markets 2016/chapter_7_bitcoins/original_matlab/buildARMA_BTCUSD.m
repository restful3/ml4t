% Find best ARMA(p, q) model (i.e. best p, q) 
clear;
load('Jonathan_BTCUSD_BBO_1minute', 'tday', 'HHMM', 'bid', 'ask');

mid=(bid+ask)/2;

idx=find(isfinite(mid));
tday(1:idx-1)=[];
HHMM(1:idx-1)=[];
mid(1:idx-1)=[];

trainset=1:(length(mid)-126*24*60); % Use all but 0.5 year of bars for in-sample fitting

% p, q from buildARMA_findPQ_BTCUSD.m   
p=3;
q=7;
model=arima(3, 0, 7)
fit=estimate(model, mid(trainset));

testset=trainset(end)+1:length(mid);

yF=NaN(size(mid));
for t=testset(1):size(mid, 1)
    [y, ~]=forecast(fit, 1, 'Y0', mid(t-max(p, q)+1:t)); % Need only most recent pMin data points for prediction
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
title(['ARMA(', num2str(p), ', ', num2str(q), ') model on BTCUSD']);
xlabel('Date');
ylabel('Cumulative Returns');

% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1

% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*24*60/length(testset))-1

