% computerFundamentalFactors.m
clear;

numTrainQtr=30; % Use minimum of 30 quarters for training.
holdingDays=21; % hold for 21 days
topN=100; % Long and short portfolio size

load('C:/Projects/prod_data/inputDataOHLCDaily_SPX_20140114', 'tday', 'stocks', 'op', 'hi', 'lo', 'cl', 'vol');
reports=false(size(cl));
% niy=NaN(size(cl)); % YTD Net Income
% ni=NaN(size(cl)); % Net Income
% sheq=ni; % Shareholder equity
earningsIncExtra=NaN(size(cl)); % Earnings per share including extraordinary items [THIS IS USED BY PAPER]
earningsExcExtra=NaN(size(cl)); % Earnings per share excluding extraordinary items
mktVal=NaN(size(cl)); % Market value
fiscalQtr=cell(size(cl)); % Fiscal Quarter
bvpershr=NaN(size(cl)); % Book value per share (annual)
totShr=NaN(size(cl)); % Total shares outstanding (annual)

[num txt]=xlsread('C:/Projects/Compustat/QuarterlyFundamentalsSPX/Quarterly.csv');

reportDateIdx=find(strcmp('rdq', txt(1, :)));
ticIdx=find(strcmp('tic', txt(1, :)));
% niyIdx=find(strcmp('niy', txt(1, :)));
% sheqIdx=find(strcmp('teqq', txt(1, :)));
earnIncIdx=find(strcmp('epspiq', txt(1, :)));
earnExcIdx=find(strcmp('epspxq', txt(1, :)));

mktValIdx=find(strcmp('mkvaltq', txt(1, :)));
fiscalQtrIdx=find(strcmp('datafqtr', txt(1, :)));

for i=1:size(num, 1)
   idxS=find(strcmp(stocks, txt(i+1, ticIdx)));
   idxT=1+find(tday==num(i, reportDateIdx)); % update data on trading day AFTER reportDate
   
   reports(idxT, idxS)=true;
   %    niy(idxT, idxS)=num(i, niyIdx);
   %    sheq(idxT, idxS)=num(i, sheqIdx);
   earningsIncExtra(idxT, idxS)=num(i, earnIncIdx);
   earningsExcExtra(idxT, idxS)=num(i, earnExcIdx);
   mktVal(idxT, idxS)=num(i, mktValIdx);   

   fiscalQtr(idxT, idxS)=txt(i+1, fiscalQtrIdx);
end

% niy=fillMissingData(niy);
% sheq=fillMissingData(sheq);
% earningsIncExtra=fillMissingData(earningsIncExtra);
% earningsExcExtra=fillMissingData(earningsExcExtra);
% mktVal=fillMissingData(mktVal);

fiscalQtr=regexprep(fiscalQtr, 'Q1', '1');
fiscalQtr=regexprep(fiscalQtr, 'Q2', '2');
fiscalQtr=regexprep(fiscalQtr, 'Q3', '3');
fiscalQtr=regexprep(fiscalQtr, 'Q4', '4');
% 
% fiscalQtr=fillMissingData(str2double(fiscalQtr));

fiscalYr=floor(str2double(fiscalQtr)/10);

% prevdayFqtr=backshift(1, fiscalQtr);
% newFiscalQtr=fiscalQtr ~= prevdayFqtr;
% newFiscalYr=newFiscalQtr & mod(fiscalQtr, 10)==1;
% 
% for t=2:length(tday)
%    ni(t, newFiscalYr(t, :))=niy(t, newFiscalYr(t, :)); 
%    ni(t, newFiscalQtr(t, :) & ~newFiscalYr(t, :))=niy(t, newFiscalQtr(t, :) & ~newFiscalYr(t, :))-niy(t-1, newFiscalQtr(t, :) & ~newFiscalYr(t, :));
% end
% 
% ni=fillMissingData(ni);
% ROE=ni./sheq;

[num txt]=xlsread('C:/Projects/Compustat/AnnualFundamentalsSPX/Annual.csv');
ticIdx=find(strcmp('tic', txt(1, :)));
bvpsIdx=find(strcmp('bkvlps', txt(1, :)));
csoIdx=find(strcmp('csho', txt(1, :)));
fiscalYrIdx=find(strcmp('fyear', txt(1, :)));

for i=1:size(num, 1)
   idxS=find(strcmp(stocks, txt(i+1, ticIdx)));
   idxT=find(fiscalYr(:, idxS)==num(i, fiscalYrIdx));
        
   bvpershr(idxT, idxS)=num(i, bvpsIdx);
   totShr(idxT, idxS)=num(i, csoIdx);
end

bvpershr(bvpershr <= 0)=NaN;

bvpershr_lag=backshift(1, fillMissingData(bvpershr));

ROE=1+earningsIncExtra./bvpershr_lag;  
% ROE=1+earningsExcExtra./bvpershr;  
BM=bvpershr.*totShr./mktVal;

ROE(ROE < 0)=NaN; % How to deal with negative 1+earningsIncExtra./bvpershr?

retFut=log(fwdshift(holdingDays, cl))-log(cl); % 1 month ahead return
trainset=1:numTrainQtr*3*holdingDays;

Y=reshape(retFut(trainset, :), [length(trainset)*size(cl, 2) 1]); % dependent variable


X=NaN(length(trainset)*size(cl, 2), 2);

X(:, 1)=reshape(log(BM(trainset, :)), [length(trainset)*size(cl, 2) 1]);
X(:, 2)=reshape(log(ROE(trainset, :)), [length(trainset)*size(cl, 2) 1]);

model=fitlm(X, Y, 'linear')
% 
% model = 
% 
% 
% Linear regression model:
%     y ~ 1 + x1 + x2
% 
% Estimated Coefficients:
%                          Estimate                  SE                   tStat                 pValue       
%                    ____________________    ___________________    _________________    ____________________
% 
%     (Intercept)    -0.00719062973547129    0.00170281456133413    -4.22279084214405    2.43453488676084e-05
%     x1              0.00450040014855594     0.0015408195031053     2.92078347884748     0.00349940623324208
%     x2               0.0428622691344051     0.0104763484353306     4.09133672853564    4.32293569588828e-05
% 
% 
% Number of observations: 9903, Error degrees of freedom: 9900
% Root Mean Squared Error: 0.102
% R-squared: 0.00187,  Adjusted R-Squared 0.00167
% F-statistic vs. constant model: 9.26, p-value = 9.58e-05

retPred=reshape(predict(model, X), [length(trainset) size(cl, 2)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

% Backtest trading model based on "prediction" on training set
positions=zeros(size(retPred));

positions(retPred > 0)=1;
positions(retPred < 0)=-1;

positions=backshift(1, positions); % Actually enter positions 1 day later
positions(1, :)=0;

% Hold for a quarter
pos=zeros(size(positions));
for h=1:holdingDays-1
   pos=backshift(h, positions);
   pos(~isfinite(pos))=0;
   positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
end

ret1=calculateReturns(cl, 1);

dailyRet=smartsum(backshift(1, positions).*ret1(trainset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(cumret);

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'In-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% In-sample: CAGR=-0.113450 Sharpe ratio=-0.406251 maxDD=-0.692530 maxDDD=771 Calmar ratio=-0.163820

% Make real predictions on test (out-of-sample) set
testset=trainset(end)+1:length(tday);

Y=reshape(retFut(testset, :), [length(testset)*size(cl, 2) 1]); % dependent variable


X=NaN(length(testset)*size(cl, 2), 2);

X(:, 1)=reshape(log(BM(testset, :)), [length(testset)*size(cl, 2) 1]);
X(:, 2)=reshape(log(ROE(testset, :)), [length(testset)*size(cl, 2) 1]);

retPred=reshape(predict(model, X), [length(testset) size(cl, 2)]); % Make "predictions" based on model on training set, reshape back to original matrix dimensions

positions=zeros(size(retPred));

positions(retPred > 0)=1;
positions(retPred < 0)=-1;

positions=backshift(1, positions); % Actually enter positions 1 day later
positions(1, :)=0;

% Hold for a quarter
pos=zeros(size(positions));
for h=1:holdingDays-1
   pos=backshift(h, positions);
   pos(~isfinite(pos))=0;
   positions(positions==0)=positions(positions==0)+pos(positions==0); % exit old position if new one exists
end

dailyRet=smartsum(backshift(1, positions).*ret1(testset, :), 2)./smartsum(abs(backshift(1, positions)), 2);
dailyRet(~isfinite(dailyRet))=0;

cumret=cumprod(1+dailyRet)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Linear regression on SPX ROE and BM facotrs');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'Out-of-sample: CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyRet)/std(dailyRet), maxDD, maxDDD, -cagr/maxDD);
% Out-of-sample: CAGR=-0.224255 Sharpe ratio=-1.429949 maxDD=-0.499556 maxDDD=603 Calmar ratio=-0.448910

