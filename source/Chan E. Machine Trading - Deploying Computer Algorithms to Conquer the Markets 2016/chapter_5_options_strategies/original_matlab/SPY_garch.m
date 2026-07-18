% Find best GARCH(p, q) model for SPY
clear;
load('inputDataOHLCDaily_ETF_20151125', 'tday', 'stocks', 'cl');

idxS=find(strcmp('SPY', stocks));
idxV=find(strcmp('VXX', stocks));
vxx=cl(:, idxV);
spy=cl(:, idxS);

ret=[NaN; price2ret(spy)]; % log returns

% Find the best (p,q) using training data

LOGL=-Inf(10, 9); % log likelihood for up to 10 p and 9 q (10 minutes)
PQ=zeros(size(LOGL)); % p values

trainset=1:1500; 

for p=1:size(PQ, 1)
    for q=1:size(PQ, 2)
               
        model=garch(p, q);
        try
            [~,~,logL] = estimate(model, ret(trainset),'print',false);
            LOGL(p, q) = logL;
            PQ(p, q) = p+q;
        catch
        end
    end
end

% Has p+q+1 parameters, including constant
LOGL_vector = reshape(LOGL, size(LOGL, 1)*size(LOGL, 2), 1);
PQ_vector = reshape(PQ, size(LOGL, 1)*size(LOGL, 2), 1);
[~, bic]=aicbic(LOGL_vector, PQ_vector+1, length(ret(trainset)));
[bicMin, pMin]=min(bic)

% bicMin =
% 
%     -9.140133342361014e+03
% pMin =
% 11

% LOGL_vector(pMin)= 4.584693111954688e+03
% PQ_vector(pMin)+1 = 4

bic(:)=NaN;
bic(pMin)=bicMin;
bic=reshape(bic,size(LOGL))

% These are the best (P, Q), which turn out to be (1, 2)
P=find(any(isfinite(bic), 2));
Q=find(isfinite(bic(P, :)));

% Now fits GARCH(P, Q) with data again to find coefficients
model=garch(P, Q);
fit=estimate(model, ret(trainset));

%    GARCH(1,2) Conditional Variance Model:
%     ----------------------------------------
%     Conditional Probability Distribution: Gaussian
% 
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant    3.00818e-06   8.39855e-07        3.58179
%      GARCH{1}       0.854272      0.015727        54.3187
%       ARCH{1}     0.00173239     0.0132146       0.131097
%       ARCH{2}       0.129495     0.0186767        6.93349

% Apply GARCH model for forecasting variance

vF=NaN(size(ret)); % predicted variance 1 period ahead
assert(P < Q); % assumption
for t=1:max(P, Q)
    vF(t)=forecast(fit, 1); % Need only most recent max(P, Q) data points for prediction
end
for t=max(P, Q)+1:size(ret, 1)
    vF(t)=forecast(fit, 1, 'Y0', ret(t-max(P, Q)+1:t), 'V0', vF(t-max(P, Q):t-1)); % Need only most recent max(P, Q) data points for prediction
end

deltaV=fwdshift(1, ret.^2)-ret.^2; % actual change in "volatility"
deltaVF=vF-ret.^2; %  forecasted change in "volatility"


% deltaVF=vF-ret.^2;
% deltaVF=vF-backshift(1, vF);
% deltaVF=vF-smartMovingSum(ret.^2, 22);

testset=trainset(end)+1:length(ret);

deltaV_train=deltaV(trainset);
deltaV_test=deltaV(testset);

deltaVF_train=deltaVF(trainset);
deltaVF_test=deltaVF(testset);

sum(sign(deltaV_train)==sign(deltaVF_train))/length(find(isfinite(deltaVF_train)))
%    0.7151
sum(sign(deltaV_test)==sign(deltaVF_test))/length(find(isfinite(deltaVF_test)))
%    0.6880


% Trading strategy
pos=zeros(size(vxx));
pos(deltaVF > 0)=-1;
pos(deltaVF < 0)=1;


dailyret=backshift(1, pos).*(vxx-backshift(1, vxx))./backshift(1, vxx);
dailyret(isnan(dailyret))=0;
cumret=cumprod(1+dailyret)-1;

dailyret=dailyret(testset);
cumret=(1+cumret(testset))/(1+cumret(testset(1)))-1;
tday_test=tday(testset);

plot(datetime(tday_test, 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative compounded return
title('RV(t+1)-RV(t) strategy');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+dailyret).^(252/length(dailyret))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret));

[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);
% CAGR=0.806584 Sharpe=1.273297
% maxDD=-0.424835 maxDDD=198 Calmar ratio=1.898583
deltaVF(trainset)=NaN;

% save('variance_SPY', 'tday', 'deltaVF');
