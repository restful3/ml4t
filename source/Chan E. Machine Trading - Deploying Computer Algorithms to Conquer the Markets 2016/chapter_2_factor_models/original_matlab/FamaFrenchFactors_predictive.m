% FamaFrenchFactors_predictive.m
clear;

topN=50;

load('fundamentalData', 'tday', 'syms', 'mid');

ret1=calculateReturns(mid, 1);
retFut1=fwdshift(1, ret1);

[num]=xlsread('F-F_Research_Data_Factors_daily.csv'); % yyyymmdd, Mkt-RF, SMB, HML, RF

[tday, idx1, idx2]=intersect(tday, num(:, 1));

ret1=ret1(idx1, :);
retFut1=retFut1(idx1, :);

mktRF=num(idx2, 2);
smb=num(idx2, 3);
hml=num(idx2, 4);



% Build stepwise LR model on training data
trainset=1:floor(length(tday)/2);
testset=floor(length(tday)/2)+1:length(tday);

tday(trainset([1 end]))

% ans =
% 
%     20070103
%     20100701

252*mean(mktRF(trainset))
%   -3.9016
252*mean(smb(trainset))
%   3.1264
252*mean(hml(trainset))
%  -1.2242

retPred1=NaN(size(mid));

for s=1:length(syms)
    model=fitlm([mktRF(trainset) smb(trainset) hml(trainset)], retFut1(trainset, s), 'linear');  % By default, there is a constant term in model, so do not include column of 1's in predictors.
    retPred1(:, s)=predict(model, [mktRF smb hml]);
end
    
positions=zeros(size(mid));

for t=1:length(tday)
    isGoodData=find(isfinite(retPred1(t, :)));
    [~, I]=sort(retPred1(t, isGoodData)); % ascending sort
    positions(t, isGoodData(I(1:topN)))=-1;
    positions(t, isGoodData(I(end-topN+1)))=1;
    
end

dailyRet=smartsum(backshift(1, positions).*ret1, 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

% Trainset
plot(datetime(tday(trainset), 'ConvertFrom', 'yyyyMMdd'), cumret(trainset)); % Cumulative compounded return
title('Fama-French factor prediction: In-sample');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+dailyRet(trainset)).^(252/length(dailyRet(trainset)))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyRet(trainset))/std(dailyRet(trainset)));
[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);
% CAGR=1.035627 Sharpe=2.464065
% maxDD=-0.240659 maxDDD=564 Calmar ratio=4.303290

% Testset
plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset)); % Cumulative compounded return
title('Fama-French factor prediction: Out-of-sample');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+dailyRet(testset)).^(252/length(dailyRet(testset)))-1;
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(dailyRet(testset))/std(dailyRet(testset)));
[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);
% CAGR=-0.070378 Sharpe=-0.643982
% maxDD=-0.240659 maxDDD=564 Calmar ratio=-0.292438