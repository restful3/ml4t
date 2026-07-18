clear;
% Daily data on EWA-EWC
load('C:/Projects/prod_data/inputDataOHLCDaily_ETF_20150828', 'tday', 'stocks', 'cl');
idxV=find(strcmp('XIV', stocks));
idxS=find(strcmp('SPY', stocks));

x=cl(:, idxV)/100; % Original data multiplied by 100, not that this matters here
y=cl(:, idxS)/100;

goodData=find(all(isfinite([x y]), 2));
x=x(goodData);
y=y(goodData);
tday=tday(goodData);

% Augment x with ones to  accomodate possible offset in the regression
% between y vs x.

x=[x ones(size(x))];

delta=0.0001; % delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).

yhat=NaN(size(y)); % measurement prediction
e=NaN(size(y)); % measurement prediction error
Q=NaN(size(y)); % measurement prediction error variance

% For clarity, we denote R(t|t) by P(t).
% initialize R, P and beta.
R=zeros(2);
P=zeros(2);
beta=NaN(2, size(x, 1));
Vw=delta/(1-delta)*eye(2);
Ve=0.001;


% Initialize beta(:, 1) to zero
beta(:, 1)=0;

% Given initial beta and R (and P)
for t=1:length(y)
    if (t > 1)
        beta(:, t)=beta(:, t-1); % state prediction. Equation 3.7
        R=P+Vw; % state covariance prediction. Equation 3.8
    end
    
    yhat(t)=x(t, :)*beta(:, t); % measurement prediction. Equation 3.9

    Q(t)=x(t, :)*R*x(t, :)'+Ve; % measurement variance prediction. Equation 3.10
    
    
    % Observe y(t)
    e(t)=y(t)-yhat(t); % measurement prediction error
    
    K=R*x(t, :)'/Q(t); % Kalman gain
    
    beta(:, t)=beta(:, t)+K*e(t); % State update. Equation 3.11
    P=R-K*x(t, :)*R; % State covariance update. Euqation 3.12
    
end


plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), beta(1, :)');
title('Hedge ratio');
xlabel('Date')
ylabel('Number of XIV shares per SPY share');

figure;

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), beta(2, :)');
title( 'intercept');


save('hedgeRatio_XIV_SPY', 'tday', 'beta', 'x', 'y');