% Find best AR(p) model (i.e. best p) 
clear;
load('Jonathan_BTCUSD_BBO_1minute', 'tday', 'HHMM', 'bid', 'ask');
mid=(bid+ask)/2;

idx=find(isfinite(mid));
tday(1:idx-1)=[];
HHMM(1:idx-1)=[];
mid(1:idx-1)=[];

LOGL=zeros(60, 1); % log likelihood for up to 60 lags (1 hour)
P=zeros(size(LOGL)); % p values

trainset=1:(length(mid)-126*24*60); % Use all but 0.5 year of bars for in-sample fitting


% 
% for p=1:length(P)
%     model=arima(p, 0, 0);
%     [~,~,logL] = estimate(model, mid(trainset),'print',false); 
%     LOGL(p) = logL;
%     P(p) = p;
%     
% end
% 
% % Has P+1 parameters, including constant
% [~, bic]=aicbic(LOGL, P+1, length(mid(trainset)));
% 
% [~, pMin]=min(bic)
% % pMin =
% % 
% %     16

pMin=3; % Fix p=3 despite MLE 

model=arima(pMin, 0, 0) 
fit=estimate(model, mid);

%     ARIMA(3,0,0) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant     0.00757631    0.00625365         1.2115
%         AR{1}       0.686097   0.000101462        6762.11
%         AR{2}       0.257477   0.000192505        1337.51
%         AR{3}      0.0564074   0.000201178        280.386
%      Variance       0.879034   0.000105175         8357.8

testset=trainset(end)+1:length(mid);

yF=NaN(size(mid));
for t=testset(1)-1:size(mid, 1)-1
    [y, ~]=forecast(fit, 1, 'Y0', mid(t-pMin+1:t)); % Need only most recent pMin data points for prediction
    yF(t+1)=y(end);
end

deltaYF=yF-backshift(1, mid);
% deltaYF_test=deltaYF(testset);

% Trading strategy
pos=zeros(size(mid));
pos(deltaYF > 0)=1;
pos(deltaYF < 0)=-1;

ret=backshift(1, pos).*(mid-backshift(1, mid))./backshift(1, mid);
ret(isnan(ret))=0;
cumret=cumprod(1+ret)-1;


plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset));
title('AR(10) model on BTCUSD');
xlabel('Date');
ylabel('Cumulative Returns');


% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1
%  -0.999847474079019

% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*(24*60-15)/length(testset))-1
%  -0.999999972061635
