function [x, yF, PF, QF]=KF_filter(y, A, P0, C, Q0)
%  [x, yF, PF, QF]=KF_filter(y, A, P0, C, Q0)
% 
%  x(t)=A(t)*x(t-1) + u(t), state transition equation. 
%  y(t)=C(t)*x(t)+ e(t), measurement equation
%  
%  where u and e are zero mean Gaussian noise, with covariance matrices P
%    and Q respectively.
%
%  Inputs:
%  y is the observed time series: T x n
%  A is the state transition matrix: if time-invariant, m x m; if
%    time-varying, T x 1 cell array of m x m matrices.
%  C is the measurement matrix: if time-invariant, n x m; if time-varying,
%    T x 1 cell array of n x m matrices.
%  P0 is the initial state noise covariance estimate: m x m.
%  Q0 is the initial measurement noise covariance estimate: n x n
% 
%  Outputs:
%  x(t) is the predicted (filtered) state given y(1), ..., y(t), and
%     y(t). T x n.
%  yF(t) is the predicted observation given y(1), ..., y(t-1). T x n.
%  PF(t) is the predicted (filtered) state noise covariance given y(1), ...,
%  y(t). m x m.
%  QF(t) is the predicted observation noise covariance given y(1), ...,
%  y(t-1). n x n.

assert(0, 'INCOMPLETE!');

T=size(y, 1);

isATimeInvariant=~iscell(A);
isCTimeInvariant=~iscell(C);

if (isATimeInvariant)
    m=size(A, 1);
    A=mat2cell(repmat(A, [T 1]), repmat(m, [1 T]), m);
else
    m=size(A{1}, 1);
end

if (isBTimeInvariant)
    n=size(B, 1);
else
    n=size(B{1}, 1);
    B=mat2cell(repmat(B, [T 1]), repmat(n, [1 T]), n);
end

x=NaN(T, m); % state prediction for t
yF=NaN(T, n); % measurement prediction for t

PF=cell(T, 1); % state noise covariance

e=cell(T, 1); % measurement prediction error
e{1}=NaN(n);
QF=cell(T, 1); % measurement prediction error variance

% Given initial beta and R (and P)
for t=1:T
    if (t > 1)
        x(t, :)=(A{t}*x(t-1, :)')'; % state prediction. Equation 7.3
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


plot(beta(1, :)');

figure;

plot(beta(2, :)');

figure;

plot(e(3:end), 'r');

hold on;
plot(sqrt(Q(3:end)));

y2=[x(:, 1) y];

longsEntry=e < -sqrt(Q); % a long position means we should buy EWC
longsExit=e > -sqrt(Q);

shortsEntry=e > sqrt(Q);
shortsExit=e < sqrt(Q);

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
positions=repmat(numUnits, [1 size(y2, 2)]).*[-beta(1, :)' ones(size(beta(1, :)'))].*y2; % [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
pnl=sum(lag(positions, 1).*(y2-lag(y2, 1))./lag(y2, 1), 2); % daily P&L of the strategy
ret=pnl./sum(abs(lag(positions, 1)), 2); % return is P&L divided by gross market value of portfolio
ret(isnan(ret))=0;

figure;
plot(cumprod(1+ret)-1); % Cumulative compounded return

fprintf(1, 'APR=%f Sharpe=%f\n', prod(1+ret).^(252/length(ret))-1, sqrt(252)*mean(ret)/std(ret));
% APR=0.262252 Sharpe=2.361162


