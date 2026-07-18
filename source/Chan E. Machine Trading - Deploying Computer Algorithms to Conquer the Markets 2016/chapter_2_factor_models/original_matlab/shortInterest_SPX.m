% Buy stocks largest short interest as fraction of shares outstanding
clear;

topN=50; % Long and short portfolio size
holddays=1;
onewaytcost=0/10000;
load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms',  'gvkey',  'bid', 'ask');

datadate=NaN(size(bid));
totShr=NaN(size(bid)); % Total shares outstanding (annual)
shortInt=NaN(size(bid)); % Short interest (not adjusted for splits)

[num txt]=xlsread('C:/Projects/Compustat/AnnualFundamentalsSPX/Annual2.csv');
gvkeyIdx=find(strcmp('gvkey', txt(1, :)));
datadateIdx=find(strcmp('datadate', txt(1, :)));
csoIdx=find(strcmp('csho', txt(1, :)));


for i=1:size(num, 1)
   idxS=find(gvkey==num(i, gvkeyIdx));
   idxT=find(tday>=num(i, datadateIdx));
   if (~isempty(idxT))
       idxT=idxT(1);
       totShr(idxT, idxS)=num(i, csoIdx)*1000000; % Measured in millions
   end
end

totShr=fillMissingData(totShr);

[num txt]=xlsread('C:/Projects/Compustat/shortInterest.csv');
gvkeyIdx=find(strcmp('gvkey', txt(1, :)));
datadateIdx=find(strcmp('datadate', txt(1, :)));
shortInterestIdx=find(strcmp('shortint', txt(1, :)));

for i=1:size(num, 1)
   idxS=find(gvkey==num(i, gvkeyIdx));
   idxT=find(tday>=num(i, datadateIdx));
   if (~isempty(idxT))
       idxT=idxT(1);
       shortInt(idxT, idxS)=num(i, shortInterestIdx);
   end
end

shortInt=fillMissingData(shortInt);

shortFrac=shortInt./totShr;

pos0=zeros(size(bid));

for t=1:length(tday)
    goodData=find(isfinite(shortFrac(t, :)));
    if (length(goodData) > 2*topN)
        [~, idxSort]=sort(shortFrac(t, goodData), 'ascend');
        pos0(t, goodData(idxSort(1:topN)))=-1;
        pos0(t, goodData(idxSort(end-topN+1:end)))=1;
    end
end


pos=zeros(size(pos0));

for h=0:holddays-1
    pos_lag=backshift(h, pos0);
    pos_lag(isnan(pos_lag))=0;
    pos=pos+pos_lag;
end

%%
mid=(bid+ask)/2;

pnl=smartsum(backshift(1, pos).*(mid-backshift(1, mid))./backshift(1, mid)-onewaytcost*abs(pos-backshift(1, pos)), 2);
dailyret=pnl./smartsum(abs(pos), 2);
dailyret(~isfinite(dailyret))=0;

% Start with day with positions
idx=find(any(pos~=0, 2));
% idx=find(tday==20130102);

dailyret(1:(idx(1)-1))=[];
tday(1:(idx(1)-1))=[];


cumret=cumprod(1+dailyret)-1;

[maxDD maxDDD]=calculateMaxDD(cumret);
avgHoldDays=calculateAvgHoldday(pos);
% fprintf(1, '%i-%i: AvgAnnRet=%f Sharpe=%f maxDD=%f maxDDD=%i avgHoldDays=%i\n', tday(1), tday(end), 252*mean(dailyret), sqrt(252)*mean(dailyret)/std(dailyret), maxDD, maxDDD, round(avgHoldDays));
cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret), maxDD, maxDDD, -cagr/maxDD);


plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Short Interest as factor');
xlabel('Date');
ylabel('Cumulative Returns');


% TopN=50
%  tcost=0
% 20070112-20131231: AvgAnnRet=0.023765 Sharpe=0.378465 maxDD=-0.104838 maxDDD=554 avgHoldDays=59
% CAGR=0.022034 Sharpe ratio=0.378465 maxDD=-0.104838 maxDDD=554 Calmar ratio=0.210176

