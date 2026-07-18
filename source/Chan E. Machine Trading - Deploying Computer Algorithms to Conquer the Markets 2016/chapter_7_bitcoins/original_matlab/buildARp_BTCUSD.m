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
%     16

model=arima(pMin, 0, 0) % assumes an AR(pMin) with unknown parameters
fit=estimate(model, mid);

%     ARIMA(16,0,0) Model:
%     --------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant     0.00670585    0.00627906        1.06797
%         AR{1}       0.685261   0.000103369        6629.28
%         AR{2}       0.257702   0.000208905        1233.58
%         AR{3}      0.0580414   0.000374436         155.01
%         AR{4}     0.00443159   0.000603402        7.34434
%         AR{5}    -0.00339982   0.000708867       -4.79614
%         AR{6}    -0.00496624   0.000805397        -6.1662
%         AR{7}     -0.0106182   0.000828498       -12.8162
%         AR{8}    -0.00189899   0.000583569        -3.2541
%         AR{9}     0.00326403   0.000667348        4.89105
%        AR{10}      0.0045775   0.000507884        9.01289
%        AR{11}    -0.00946253    0.00052602       -17.9889
%        AR{12}    0.000956515   0.000587355        1.62851
%        AR{13}      0.0015719   0.000605645        2.59541
%        AR{14}     -0.0017755   0.000583957       -3.04046
%        AR{15}     0.00752874   0.000751633        10.0165
%        AR{16}     0.00876893   0.000599384        14.6299
%      Variance       0.878134   0.000107755        8149.36


results=adf(mid(trainset), 0, 1);
prt(results);

% Augmented DF test for unit root variable:                   variable   1 
%  ADF t-statistic       # of lags   AR(1) estimate 
%        -3.001533               1         0.999937 
% 
%    1% Crit Value    5% Crit Value   10% Crit Value 
%           -3.458           -2.871           -2.594 
testset=trainset(end)+1:length(mid);

yF=NaN(size(mid));
for t=testset(1):size(mid, 1)
    [y, ~]=forecast(fit, 1, 'Y0', mid(t-pMin+1:t)); % Need only most recent pMin data points for prediction
    yF(t)=y(end);
end

deltaYF=yF-mid;
deltaYF_test=deltaYF(testset);

% Trading strategy
pos=zeros(size(mid));
pos(deltaYF > 0)=1;
pos(deltaYF < 0)=-1;

ret=backshift(1, pos).*(mid-backshift(1, mid))./backshift(1, mid);
ret(isnan(ret))=0;
cumret=cumprod(1+ret)-1;


plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset));
title('AR(16) model on BTCUSD');
xlabel('Date');
ylabel('Cumulative Returns');


% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1
%      2.019081357722237e+02


% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*24*60/length(testset))-1
%   4.117071156255915e+04
