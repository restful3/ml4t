% Use DTC (days to cover) instead of SIR
% SELL stocks with largest short interest as fraction of shares outstanding
clear;

topN=50; % Long and short portfolio size
onewaytcost=0/10000;
load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms', 'permco', 'gvkey', 'cusip', 'fromDate', 'thruDate', 'op', 'hi', 'lo', 'cl', 'bid', 'ask', 'vol', ...
    'op_unadj', 'hi_unadj', 'lo_unadj', 'cl_unadj', 'bid_unadj', 'ask_unadj', 'vol_unadj');

% datadate=NaN(size(cl));
% totShr=NaN(size(cl)); % Total shares outstanding (annual)
shortInt=NaN(size(cl)); % Short interest (not adjusted for splits)

ADV=smartMovingAvg(vol, 20);

% [num txt]=xlsread('C:/Projects/Compustat/AnnualFundamentalsSPX/Annual2.csv');
% gvkeyIdx=find(strcmp('gvkey', txt(1, :)));
% datadateIdx=find(strcmp('datadate', txt(1, :)));
% csoIdx=find(strcmp('csho', txt(1, :)));


% for i=1:size(num, 1)
%    idxS=find(gvkey==num(i, gvkeyIdx));
%    idxT=find(tday>=num(i, datadateIdx));
%    if (~isempty(idxT))
%        idxT=idxT(1);
%        totShr(idxT, idxS)=num(i, csoIdx)*1000000; % Measured in millions
%    end
% end
% 
% totShr=fillMissingData(totShr);

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

shortFrac=shortInt./ADV;

pos=zeros(size(cl));

for t=1:length(tday)
    goodData=find(isfinite(shortFrac(t, :)));
    if (length(goodData) > 2*topN)
        [foo, idxSort]=sort(shortFrac(t, goodData), 'ascend');
        pos(t, goodData(idxSort(1:topN)))=1;
        pos(t, goodData(idxSort(end-topN+1:end)))=-1;
    end
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
fprintf(1, '%i-%i: AvgAnnRet=%f Sharpe=%f maxDD=%f maxDDD=%i avgHoldDays=%i\n', tday(1), tday(end), 252*mean(dailyret), sqrt(252)*mean(dailyret)/std(dailyret), maxDD, maxDDD, round(avgHoldDays));

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret), maxDD, maxDDD, -cagr/maxDD);


plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Short Interest as factor');
xlabel('Date');
ylabel('Cumulative Returns');



% TopN=50
%  tcost=0
% 20070112-20131231: AvgAnnRet=0.059999 Sharpe=1.128597 maxDD=-0.085830 maxDDD=594 avgHoldDays=24
% CAGR=0.058981 Sharpe ratio=1.104392 maxDD=-0.086961 maxDDD=780 Calmar ratio=0.678252

% tcost=5 bps
% 20070131-20131231: AvgAnnRet=0.049501 Sharpe=0.930228 maxDD=-0.087737 maxDDD=1230 avgHoldDays=24
% 20130102-20131231: AvgAnnRet=-0.004679 Sharpe=-0.181356 maxDD=-0.026611 maxDDD=237 avgHoldDays=24
