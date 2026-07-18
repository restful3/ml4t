% Enter at open and exit at close.
% Trade VXX in opposite direction to GARCH prediction of volatility change
% Find best GARCH(p, q) model for SPY
clear;
onewaytcost=5/10000;

load('inputDataOHLCDaily_ETF_20151125', 'tday', 'stocks', 'op', 'cl');
idxS=find(strcmp('SPY', stocks));
idxV=find(strcmp('VXX', stocks));
vxx=cl(:, idxV);
spy=cl(:, idxS);
vxxOp=op(:, idxV);

ret=[NaN; price2ret(spy)]; % log returns
trainset=1:1500;

% Find the best (p,q) using training data
if (0)
    LOGL=-Inf(10, 9); % log likelihood for up to 10 p and 9 q (10 minutes)
    PQ=zeros(size(LOGL)); % p values
    
    
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
    %     -9.267388198516723e+03
    % pMin =
    % 11
    
    % LOGL_vector(pMin)=  4.648320540032542e+03
    % PQ_vector(pMin)+1 = 4
    
    bic(:)=NaN;
    bic(pMin)=bicMin;
    bic=reshape(bic,size(LOGL))
    
    % These are the best (P, Q), which turn out to be (1, 2)
    P=find(any(isfinite(bic), 2));
    Q=find(isfinite(bic(P, :)));
else
    P=1;
    Q=2;
end

% Now fits GARCH(P, Q) with data again to find coefficients
model=garch(P, Q);
fit=estimate(model, ret(trainset));

vF=NaN(size(ret)); % predicted variance 1 period ahead
for t=max(P, Q)+1:size(ret, 1)
    vF(t)=forecast(fit, 1, 'Y0', ret(t-max(P, Q)+1:t)); % Need only most recent max(P, Q) data points for prediction
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

% sum(sign(deltaV_train)==sign(deltaVF_train))/length(deltaVF_train)
% %    0.656000000000000
% sum(sign(deltaV_test)==sign(deltaVF_test))/length(deltaVF_test)
% %    0.596000000000000


% Trading strategy
pos=zeros(size(vxx));
pos(deltaVF > 0)=-1;
% pos(deltaVF < 0)=1;
% pos=-ones(size(vxx));

dailyret=backshift(1, pos).*(vxx-vxxOp)./vxxOp-2*onewaytcost;
dailyret(isnan(dailyret))=0;
cumret=cumprod(1+dailyret)-1;

dailyret=dailyret(testset);
cumret=(1+cumret(testset))/(1+cumret(testset(1)))-1;
tday_test=tday(testset);

plot(datetime(tday_test, 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative compounded return
title('Intraday VXX with anti-GARCH: out-of-sample');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+dailyret).^(252/length(dailyret))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret));

[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);
% 20111206-20151125
% onewaytcost=0
% CAGR=0.837937 Sharpe=1.302980
% maxDD=-0.461515 maxDDD=179 Calmar ratio=1.815622

% onewaytcost=5bps
% CAGR=0.765657 Sharpe=1.237057
% maxDD=-0.465551 maxDDD=316 Calmar ratio=1.644626

% Short VXX only
% CAGR=0.776538 Sharpe=1.313672
% maxDD=-0.338944 maxDDD=109 Calmar ratio=2.291052
% Should run short-only model




deltaVF(trainset)=NaN;

% save('variance_SPY', 'tday', 'deltaVF');
