% Compare GARCH predicted volatility and current VIX to determine position
% Find best GARCH(p, q) model for SPY
clear;
onewaytcost=0/10000;

load('C:/Projects/prod_data/inputDataOHLCDaily_ETF_20160311', 'stocks', 'tday',  'cl');
VX=load('C:/Projects/Futures_data/inputDataDaily_VX_20160321', 'contracts', 'tday', 'cl');
idxV=find(strcmp('0000$', VX.contracts));
vix=VX.cl(:, idxV);

idxS=find(strcmp('SPY', stocks));
% idxV=find(strcmp('VXX', stocks));
% vxx=cl(:, idxV);
spy=cl(:, idxS);

[tday, idxSt, idxVt]=intersect(tday, VX.tday);
vix=vix(idxVt, :);
spy=spy(idxSt, :);
% vxx=vxx(idxSt, :);

ret=[NaN; price2ret(spy)]; % log returns
trainset=find(tday < 20101130);


contracts=VX.contracts;
contracts(idxV)=[];
VX=VX.cl(idxVt, :);
VX(:, idxV)=[];


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

%     GARCH(1,2) Conditional Variance Model:
%     ----------------------------------------
%     Conditional Probability Distribution: Gaussian
% 
%                                   Standard          t     
%      Parameter       Value          Error       Statistic 
%     -----------   -----------   ------------   -----------
%      Constant    1.27051e-06   5.20109e-07        2.44278
%      GARCH{1}       0.915054     0.0085389        107.163
%       ARCH{1}     0.00930322     0.0161144       0.577324
%       ARCH{2}      0.0648126     0.0173535        3.73483

vF=NaN(size(ret)); % predicted variance 1 period ahead
for t=max(P, Q)+1:size(ret, 1)
    vF(t)=forecast(fit, 1, 'Y0', ret(t-max(P, Q)+1:t)); % Need only most recent max(P, Q) data points for prediction
end

vF=100*sqrt(252*vF); % annualized volatility, not variance

testset=trainset(end)+1:length(ret);
tday_test=tday(testset);

plot(datetime(tday_test, 'ConvertFrom', 'yyyyMMdd'), [vix(testset) vF(testset)] );
legend('VIX', 'GARCH');

figure;
delta=vF-vix;
plot(datetime(tday_test, 'ConvertFrom', 'yyyyMMdd'), delta(testset));
title('RV(t+1) - VIX(t)');
figure;

% Trading strategy
positions=zeros(size(VX));

% pos(delta > 0)=1;
% pos(delta < 0)=-1;

isExpireDate=false(size(VX));
isExpireDate=isfinite(VX) & ~isfinite(fwdshift(1, VX));

% Define front month as 40 days to 1 days before expiration
numDaysStart=30;
numDaysEnd=1;


for c=1:length(contracts)
    expireIdx=find(isExpireDate(:, c));
    
    if (c==1)
        startIdx=expireIdx-numDaysStart;
        endIdx=expireIdx-numDaysEnd;
    else % ensure next front month contract doesn't start until current one ends
        startIdx=max(endIdx+1, expireIdx-numDaysStart);
        endIdx=expireIdx-numDaysEnd;
    end
        
    if (~isempty(expireIdx))
        idx=startIdx:endIdx;
        positions(idx(delta(idx)>0), c)=1;
        positions(idx(delta(idx)<0), c)=-1;

    end
end
%%

dailyret=smartsum(backshift(1, positions).*(VX-backshift(1, VX))./backshift(1, VX), 2);
dailyret(isnan(dailyret))=0;
cumret=cumprod(1+dailyret)-1;

dailyret=dailyret(testset);
cumret=(1+cumret(testset))/(1+cumret(testset(1)))-1;

plot(datetime(tday_test, 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative compounded return
title('RV(t+1)-VIX(t) Strategy');
xlabel('Date');
ylabel('Cumulative Returns');

fprintf(1, '%i-%i\n', tday_test(1), tday_test(end));

cagr= prod(1+dailyret).^(252/length(dailyret))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret));

[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);


% CAGR=0.417015 Sharpe=0.838745
% maxDD=-0.771810 maxDDD=482 Calmar ratio=0.540308
%% New trading strategy
