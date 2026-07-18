% ef.m
% Find efficient frontier of a set of ETFs
% Note in this program mean and sd of returns are row vectors, 

clear;

load('inputDataOHLCDaily_ETF_20150417', 'stocks', 'tday', 'cl');

% Remove EWZ and FXI
stocks=stocks(~strcmp('EWZ', stocks) & ~strcmp('FXI', stocks));
cl=cl(:, ~strcmp('EWZ', stocks) & ~strcmp('FXI', stocks));

R=calculateReturns(cl, 1); % 1-period "net" returns
R(1, :)=[]; % Skip first row since returns there are NaN

mi=mean(R, 1); % average return of each stock i. 
C=cov(R); % covariance of returns

m=[min(mi):(max(mi)-min(mi))/20:max(mi)]; % prepare different target mean portfolio returns for efficient frontier
v=NaN(size(m));

% Variance of portfolio to be minimized
H=2*C;

% short sale constraint
A=-eye(length(mi));
b=zeros(length(mi), 1);

% No linear term in minimization
f=zeros(1, length(mi)); 

% Fixing portfolio mean return and the normalization constraint
Aeq=[mi; ones(1, length(mi))];


for i=1:length(m)
    beq=[m(i); 1];
    [F, v(i)]=quadprog(H, f, A, b, Aeq, beq);
end

sd=sqrt(v);
scatter(sd, m);
hold on;

% Find tangency portfolio
sharpeRatio=m./sd;
[~, idx]=max(sharpeRatio); 
scatter(sd(idx), m(idx), 'red');

beq=[m(idx); 1];
[F]=quadprog(H, f, A, b, Aeq, beq)

% F =
% 
%    0.451338068065785
%    0.263444604411169
%    0.000019328104256
%    0.000004913963978
%    0.000007497342085
%    0.285185588112728

% Find minimum variance portfolio
[~, idxMin]=min(sd);
scatter(sd(idxMin), m(idxMin), 'green', 'filled');

beq=[m(idxMin); 1];
[F]=quadprog(H, f, A, b, Aeq, beq)

% F =
% 
%    0.381518036613685
%    0.000033166607068
%    0.604011066007258
%    0.000133505575004
%    0.014299849485899
%    0.000004375711085


