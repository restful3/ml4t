clear;
%% Select data from computer hardware industry group
load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'syms', 'tday', 'bid', 'ask');

[num txt]=xlsread('C:/Projects/Compustat/AnnualFundamentalsSPX/Annual2.csv');
tickers=txt(2:end, strcmp('tic', txt(1, :)));
gind=num(:, strcmp('gind', txt(1, :))); % 65 GIC industry groups
fiscalYr=num(:, strcmp('fyear', txt(1, :)));

% Pick a an  industry group as example, and pick those stocks whos
% industry group has not changed
gind_uniq=unique(gind);

g=find(gind_uniq==452020); % Computer hardware
%         'AAPL'    'EMC'    'HPQ'    'NTAP'    'SNDK'    'STX'    'WDC'

myTickers=unique(tickers(gind==gind_uniq(g)));
% Make sure this stock does not change industry group
for s=1:length(myTickers)
    assert(length(unique(gind(strcmp(tickers, myTickers(1)))))==1);
end

[~, idx1, idx2]=intersect(syms, myTickers);


bid=fillMissingData(bid);
ask=fillMissingData(ask);
mid=(bid(:, idx1)+ask(:, idx1))/2;

incompleteDataIdx=find(sum(isnan(mid), 1) > 0);
mid(:, incompleteDataIdx)=[];


trainset=1:floor(length(tday)/2); % Use all but 1 year of bars for in-sample fitting


% Using Matlab ssm notation: 
% x(t)=A*x(t-1) + B*u(t), state transition equation. A=I, since this is moving average
% y(t)=C*x(t)+D*e(t), measurement equation
% where u and e are zero mean and unit variance Gaussian noise.

y=mid; % Observed prices is measurement (observation)

A=eye(size(mid, 2)); % State transition matrix
B=NaN(size(mid, 2)); % state-disturbance-loading matrix. Time invariant, undetermined values.
C=NaN(size(mid, 2)); % Time-invariant measurement matrix
D=NaN(size(mid, 2)); % measurement-innovation matrix, time-invariant variance, undetermined values.

model=ssm(A, B, C, D);

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

param0=randn(3*size(B, 1)^2, 1); % 5 unknown parameters per bar.
model=estimate(model, y(trainset, :), param0);

disp(model)

% State vector length: 5
% Observation vector length: 5
% State disturbance vector length: 5
% Observation innovation vector length: 5
% Sample size supported by model: Unlimited
% 
% State variables: x1, x2,...
% State disturbances: u1, u2,...
% Observation series: y1, y2,...
% Observation innovations: e1, e2,...
% 
% State equations:
% x1(t) = x1(t-1) - (0.16)u1(t) - (0.28)u2(t) - (0.19)u3(t) + (0.49)u4(t)
%         - (0.30)u5(t)
% x2(t) = x2(t-1) + (0.52)u1(t) - (0.48)u2(t) + (0.78)u3(t) - (1.44)u4(t)
%         + (0.55)u5(t)
% x3(t) = x3(t-1) - (0.11)u1(t) + (0.07)u2(t) - (0.63)u3(t) - (0.97)u4(t)
%         + (1.04)u5(t)
% x4(t) = x4(t-1) - (0.42)u1(t) - (0.31)u2(t) + (0.61)u3(t) + (0.36)u4(t)
%         + (1.02)u5(t)
% x5(t) = x5(t-1) - (0.51)u1(t) + (0.20)u2(t) - (0.05)u3(t) + (1.13)u4(t)
%         - (1.46)u5(t)
% 
% Observation equations:
% y1(t) = -(0.96)x1(t) + (0.97)x2(t) + (0.69)x3(t) + (0.60)x4(t) - (0.17)x5(t)
%         - (0.59)e1(t) + (0.05)e2(t) - (0.18)e3(t) + (0.18)e4(t) + (0.32)e5(t)
% y2(t) = -(0.22)x1(t) - (0.26)x2(t) - (0.13)x3(t) - (0.10)x4(t) - (0.37)x5(t)
%         - (0.07)e1(t) - (0.03)e2(t) + (7.94e-03)e3(t) - (0.03)e4(t) + (0.03)e5(t)
% y3(t) = (0.79)x1(t) + (0.06)x2(t) + (0.45)x3(t) - (0.18)x4(t) - (0.15)x5(t)
%         - (0.19)e1(t) - (0.01)e2(t) - (0.05)e3(t) + (0.05)e4(t) + (0.14)e5(t)
% y4(t) = -(0.98)x1(t) + (0.03)x2(t) + (0.54)x3(t) + (2.24e-05)x4(t) + (0.51)x5(t)
%         - (0.18)e1(t) - (0.01)e2(t) - (0.05)e3(t) + (0.08)e4(t) - (0.02)e5(t)
% y5(t) = (0.15)x1(t) + (0.13)x2(t) - (0.26)x3(t) - (0.68)x4(t) - (0.55)x5(t)
%         - (0.12)e1(t) - (0.03)e2(t) + (0.02)e3(t) - (0.01)e4(t) - (0.16)e5(t)
% 
% Initial state distribution:
% 
% Initial state means
%  x1  x2  x3  x4  x5 
%   0   0   0   0   0 
% 
% Initial state covariance matrix
%      x1     x2     x3     x4     x5    
%  x1  1e+07  0      0      0      0     
%  x2  0      1e+07  0      0      0     
%  x3  0      0      1e+07  0      0     
%  x4  0      0      0      1e+07  0     
%  x5  0      0      0      0      1e+07 
% 
% State types
%     x1       x2       x3       x4       x5   
%  Diffuse  Diffuse  Diffuse  Diffuse  Diffuse


[x, logL, output]=filter(model, y);

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), x);
title('Kalman Filter Estimate of moving average of computer hardware stocks');
xlabel('Date');
ylabel('x(t)');

figure;

yF=NaN(size(y));
for t=1:length(output)
    yF(t, :)=output(t).ForecastedObs';
end

retF=(fwdshift(1, yF)-y)./y;
sectorRetF=mean(retF, 2);

pos=zeros(size(retF));
pos=(retF-repmat(sectorRetF, [1 size(retF, 2)]))./repmat(smartsum(abs(retF-repmat(sectorRetF, [1 size(retF, 2)])), 2), [1, size(retF, 2)]);

ret=smartsum(backshift(1, pos).*(mid-backshift(1, mid))./backshift(1, mid), 2);
ret(isnan(ret))=0;
cumret=cumprod(1+ret)-1;


plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret(trainset));
title('Trainset: Kalman Filter model on computer hardware SPX stocks');
xlabel('Date');
ylabel('Cumulative Returns');

figure;

testset=trainset(end)+1:size(mid, 1);


plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), (1+cumret(testset))/(1+cumret(trainset(end)))-1);
title('Testset: Kalman Filter model on computer hardware SPX stocks');
xlabel('Date');
ylabel('Cumulative Returns');


% Annualized compound returns on testset
CAGR=((1+cumret(end))/(1+cumret(trainset(end))))^(252/length(testset))-1;
sharpe=sqrt(252)*mean(ret)/std(ret);

fprintf(1, 'CAGR=%f Sharpe=%f\n', CAGR, sharpe);
% CAGR=-0.102901 Sharpe=0.927018
