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

incompleteDataIdx=sum(isnan(mid), 1) > 0;
mid(:, incompleteDataIdx)=[];

syms=syms(idx1(~incompleteDataIdx));
%   'AAPL'    'EMC'    'HPQ'    'NTAP'    'SNDK'

trainset=1:floor(length(tday)/2); % Use all but 1 year of bars for in-sample fitting


% Using Matlab ssm notation: 
% x(t)=A*x(t-1) + B*u(t), state transition equation. A=I, since this is moving average
% y(t)=C*x(t)+D*e(t), measurement equation
% where u and e are zero mean and unit variance Gaussian noise.

y=mid; % Observed log prices is measurement (observation)

A=eye(size(y, 2)); % State transition matrix
% B=NaN(size(y, 2)); % state-disturbance-loading matrix. Time invariant, undetermined values.
B=diag(NaN(size(y, 2), 1))
C=eye(size(y, 2)); % Time-invariant measurement matrix
% D=NaN(size(y, 2)); % measurement-innovation matrix, time-invariant variance, undetermined values.
D=diag(NaN(size(y, 2), 1))

model=ssm(A, B, C, D);

disp(model);

rng('default'); % Fix random number generator seed to get repeatable results
rng(2);

% param0=randn(2*size(B, 1)^2, 1); 
param0=randn(2*size(B, 1), 1);
Options=optimoptions(@FMINUNC, 'MaxFunEvals', 10000);
model=estimate(model, y(trainset, :), param0, 'Options', Options);

disp(model)

% State equations:
% x1(t) = x1(t-1) - (3.74)u1(t)
% x2(t) = x2(t-1) + (0.34)u2(t)
% x3(t) = x3(t-1) - (0.73)u3(t)
% x4(t) = x4(t-1) - (0.67)u4(t)
% x5(t) = x5(t-1) - (1.00)u5(t)
% 
% Observation equations:
% y1(t) = x1(t) - (4.54e-05)e1(t)
% y2(t) = x2(t) - (0.08)e2(t)
% y3(t) = x3(t) + (0.22)e3(t)
% y4(t) = x4(t) + (0.19)e4(t)
% y5(t) = x5(t) - (0.15)e5(t)

[x, logL, output]=filter(model, y);

disp(model);

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), x);
title('Kalman Filter Estimate of moving average of computer hardware stocks');
xlabel('Date');
ylabel('x(t)');
legend(syms);
figure;

yF=NaN(size(y));
for t=2:length(output)-1
    yF(t-1, :)=output(t).ForecastedObs';
end

% retF=(fwdshift(1, yF)-y)./y;
retF=(yF-y)./y;
sectorRetF=mean(retF, 2);

pos=zeros(size(retF));
pos=(retF-repmat(sectorRetF, [1 size(retF, 2)]))./repmat(smartsum(abs(retF-repmat(sectorRetF, [1 size(retF, 2)])), 2), [1, size(retF, 2)]);

ret=smartsum(backshift(1, pos).*(y-backshift(1, y))./backshift(1, y), 2);
% ret=smartsum(backshift(1, pos).*(y-backshift(1, y)), 2);
ret(isnan(ret))=0;
cumret=cumprod(1+ret)-1;
% cumret=cumsum(ret);


plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret(trainset));
title('Trainset: Kalman Filter model on computer hardware SPX stocks');
xlabel('Date');
ylabel('Cumulative Returns');


% Annualized compound returns on testset
CAGR=(1+cumret(trainset(end)))^(252/length(trainset))-1;
sharpe=sqrt(252)*mean(ret(trainset))/std(ret(trainset));
fprintf(1, 'Train set: CAGR=%f Sharpe=%f\n', CAGR, sharpe);
% Train set: CAGR=0.407758 Sharpe=1.846934

figure;

testset=trainset(end)+1:size(mid, 1);


plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), (1+cumret(testset))/(1+cumret(trainset(end)))-1);
% plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset)-cumret(trainset(end)));
title('Testset: Kalman Filter model on computer hardware SPX stocks');
xlabel('Date');
ylabel('Cumulative Returns');


% Annualized compound returns on testset
CAGR=((1+cumret(end))/(1+cumret(trainset(end))))^(252/length(testset))-1;
sharpe=sqrt(252)*mean(ret(testset))/std(ret(testset));

fprintf(1, 'Test set: CAGR=%f Sharpe=%f\n', CAGR, sharpe);
% Test set: CAGR=-0.003842 Sharpe=0.046716