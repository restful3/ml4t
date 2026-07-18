% StatisticalFactors_predictive.m
% Use principal components for prediction
clear;

topN=50;
numFactors=5;
lookback=252; % minimum number of observations required for PCA

load('fundamentalData', 'tday', 'syms', 'mid');

ret1=calculateReturns(mid, 1);
retFut1=fwdshift(1, ret1); % Actual future 1-day return
retPred1=NaN(1, size(mid, 2)); % Predicted future 1-day return
positions=zeros(size(mid));

% Find principal components
for t=lookback+1:length(tday)
    trainset=t-lookback+1:t;
    
    R=ret1(trainset, :);
    hasData=find(all(isfinite(R), 1)); % avoid any stocks with missing returns
    if (length(hasData) > 2*topN)
        % length(hasData)
        %
        % ans =
        %
        %    389
        R=R(:, hasData);
        
        [factorLoadings, factors, factorVariances] = pca(R, 'Algorithm', 'eig', 'NumComponents', numFactors); % R=factors*factorLoadings'
        
        % length(find(factorLoadings(:, 1) < 0))
        %
        % ans =
        %
        %      0
        
        for s=1:length(hasData)
            model=fitlm(factors(1:end-1, :), retFut1(trainset(1:end-1), hasData(s)), 'linear');  % By default, there is a constant term in model, so do not include column of 1's in predictors.
            retPred1(1, hasData(s))=predict(model, factors(end, :));
        end
        
        isGoodData=find(isfinite(retPred1));
        [~, I]=sort(retPred1(isGoodData)); % ascending sort
        positions(t, isGoodData(I(1:topN)))=-1;
        positions(t, isGoodData(I(end-topN+1:end)))=1;
        
    end
end

dailyRet=smartsum(backshift(1, positions).*ret1, 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

testset=min(find(any(positions~=0, 2))):length(tday);

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset)); % Cumulative compounded return
title('Statistical factor prediction: Out-of-sample');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+dailyRet(testset)).^(252/length(dailyRet(testset)))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyRet(testset))/std(dailyRet(testset)));
[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);
% CAGR=0.155643 Sharpe=1.378843
% maxDD=-0.159421 maxDDD=555 Calmar ratio=0.976298