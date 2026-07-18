clear;
% Daily data on EWA-EWC
load('inputData_ETF', 'tday', 'syms', 'cl');
idxA=find(strcmp('EWA', syms));
idxC=find(strcmp('EWC', syms));

trainset=1:1250;


% Using Matlab ssm notation: 
% x(t)=A(t)*x(t-1) + B(t)*u(t), state transition equation
% y(t)=C(t)*x(t)+D(t)*e(t), measurement equation
% where u and e are zero mean and unit variance Gaussian noise.

y=cl(:, idxC); % EWC price is measurement (observation)

C=[cl(:, idxA) ones(size(cl, 1), 1)]; % EWA prices augmented with constant offset as time-varying measurement 1x2 matrix.

A=eye(2); % State transition matrix
% B=num2cell(NaN(2, 2, size(cl, 1)), [1 2]); % state-disturbance-loading matrix. Cell array of 2x2 matrices, undetermined values.
B=diag(repmat(NaN, [2 1])); % state-disturbance-loading matrix. Time invariant, undetermined values.
C=mat2cell(C, ones(size(cl, 1), 1)); % Time-varying measurement matrix
% D=mat2cell(NaN(size(cl, 1), 1), ones(size(cl, 1), 1)); % measurement-innovation matrix, time varying variance. Cell array of 1x1 scalar, undetermined values.
D=NaN; % measurement-innovation matrix, time-invariant variance, undetermined values.

model=ssm(A, B, C(trainset, :), D);

rng('default'); % Fix random number generator seed to get repeatable results
rng(1);

param0=randn(3, 1); % 5 unknown parameters per bar.
model=estimate(model, y(trainset), param0);

% Method: Maximum likelihood (fminunc)
% Sample size: 1250
% Logarithmic  likelihood:     -180.733
% Akaike   info criterion:      371.466
% Bayesian info criterion:      397.121
%       |     Coeff       Std Err   t Stat     Prob  
% ---------------------------------------------------
%  c(1) | -0.01015       0.00601   -1.68813  0.09139 
%  c(2) |  0.40606       0.09422    4.30955  0.00002 
%  c(3) |  0.02114       0.00323    6.53626   0      
%  c(4) | -0.32381       0.11004   -2.94268  0.00325 
%  c(5) | -0.07687       0.01242   -6.19126   0      
%       |                                            
%       |   Final State   Std Dev    t Stat    Prob  
%  x(1) |  0.66148       0.11789    5.61080   0      
%  x(2) | 15.68948       3.12574    5.01945   0    

% disp(model)
% 
% State vector length: 2
% Observation vector length: 1
% State disturbance vector length: 2
% Observation innovation vector length: 1
% Sample size supported by model: 1250
% 
% State variables: x1, x2,...
% State disturbances: u1, u2,...
% Observation series: y1, y2,...
% Observation innovations: e1, e2,...
% 
% State equations:
% x1(t) = x1(t-1) - (0.01)u1(t) + (0.02)u2(t)
% x2(t) = x2(t-1) + (0.41)u1(t) - (0.32)u2(t)
% 
% Observation equation of period 1  3:
% y1(t) = (16.10)x1(t) + x2(t) - (0.08)e1(t)

% Observation equation of period 2:
% y1(t) = (15.98)x1(t) + x2(t) + (0.07)e1(t)

% ...

% Observation equation of period 1250:
% y1(t) = (26.53)x1(t) + x2(t) - (0.08)e1(t)


% Initial state distribution:
% 
% Initial state means
%  x1  x2 
%   0   0 
% 
% Initial state covariance matrix
%      x1     x2    
%  x1  1e+07  0     
%  x2  0      1e+07 
% 
% State types
%     x1       x2   
%  Diffuse  Diffuse 

% B=[-0.01015 0.02114; 0.40606 -0.32381];
% D=-0.07687;

B=model.B;
D=model.D;

model=ssm(A, B, C, D); % Now fully specified

[beta, logL, output]=filter(model, y);

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), beta(:, 1));
title('Kalman Filter Estimate of Slope between EWC vs EWA');
xlabel('Date');
ylabel('Slope=x(t, 1)');

figure;

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), beta(:, 2));
title('Kalman Filter Estimate of Offset between EWC vs EWA');
xlabel('Date');
ylabel('Offset=x(t, 2)');
figure;


% yhat=NaN(size(y));
% ymse=NaN(size(y));
% for t=2:length(tday)
%     [yhat(t), ymse(t)]=forecast(model, 1, y(1:t-1), 'C', C(t));
% end

yF=NaN(size(y));
ymse=NaN(size(y));
for t=1:length(output)
    yF(t, :)=output(t).ForecastedObs';
    ymse(t, :)=output(t).ForecastedObsCov';
end
e=y-yF; % forecast error

% plot(e(3:end), 'r');
plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), e, 'r');
title('Measurement forecast error e(t) and standard deviation of e(t)');
xlabel('Date');
hold on;

ymse(1:3)=NaN; % Early error estimates are too high and they distort our plot
plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), sqrt(ymse));
% legend(  '$$\sqrt{|D|}$$' );
% set(legend, 'Interpreter', 'Latex');
legend('e(t)',  '$\sqrt{|D|}$' );
set(legend, 'Interpreter', 'Latex');

y2=[cl(:, idxA) y];

longsEntry=e < -sqrt(ymse); % a long position means we should buy EWC
longsExit=e > -sqrt(ymse);

shortsEntry=e > sqrt(ymse);
shortsExit=e < sqrt(ymse);

numUnitsLong=NaN(length(y2), 1);
numUnitsShort=NaN(length(y2), 1);

numUnitsLong(1)=0;
numUnitsLong(longsEntry)=1; 
numUnitsLong(longsExit)=0;
numUnitsLong=fillMissingData(numUnitsLong); % fillMissingData can be downloaded from epchan.com/book2. It simply carry forward an existing position from previous day if today's positio is an indeterminate NaN.

numUnitsShort(1)=0;
numUnitsShort(shortsEntry)=-1; 
numUnitsShort(shortsExit)=0;
numUnitsShort=fillMissingData(numUnitsShort);

numUnits=numUnitsLong+numUnitsShort;
positions=repmat(numUnits, [1 size(y2, 2)]).*[-beta(:, 1) ones(size(beta(:, 1)))].*y2; % [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
pnl=sum(lag(positions, 1).*(y2-lag(y2, 1))./lag(y2, 1), 2); % daily P&L of the strategy
ret=pnl./sum(abs(lag(positions, 1)), 2); % return is P&L divided by gross market value of portfolio
ret(isnan(ret))=0;

figure;

cumret=cumprod(1+ret(trainset))-1;
plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Trainset: Cumulative Returns of Kalman Filter Strategy on EWA-EWC');
xlabel('Date');
ylabel('Cumulative Returns');

figure;

testset=trainset(end)+1:length(tday);

cumret=cumprod(1+ret(testset))-1;

% plot(cumprod(1+ret(testset))-1); % Cumulative compounded return

% fprintf(1, 'APR=%f Sharpe=%f\n', prod(1+ret).^(252/length(ret))-1, sqrt(252)*mean(ret)/std(ret));


plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Testset: Cumulative Returns of Kalman Filter Strategy on EWA-EWC');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(ret(testset))/std(ret(testset)), maxDD, maxDDD, -cagr/maxDD);
