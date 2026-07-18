% Use liquidity (monthly volume as fraction of total shares outstanding)
% BUY stocks with largest liquidity (opposite to Ibbotson 2014).
clear;

topN=50; % Long and short portfolio size
onewaytcost=0/10000;
load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms', 'gvkey',  'bid', 'ask', 'vol');

ADV=smartMovingAvg(vol, 21);

datadate=NaN(size(bid));
totShr=NaN(size(bid)); % Total shares outstanding (annual)

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

liquid=ADV./totShr;

% liquid=smartMovingAvg(liquid, 252);

pos=zeros(size(bid));

for t=1:length(tday)
    goodData=find(isfinite(liquid(t, :)));
    if (length(goodData) > 2*topN)
        [foo, idxSort]=sort(liquid(t, goodData), 'ascend');
        pos(t, goodData(idxSort(1:topN)))=-1;
        pos(t, goodData(idxSort(end-topN+1:end)))=1;
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

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(dailyret)/std(dailyret), maxDD, maxDDD, -cagr/maxDD);
% CAGR=0.084476 Sharpe ratio=0.797348 maxDD=-0.119586 maxDDD=721 Calmar ratio=0.706408


plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Liquidity as factor');
xlabel('Date');
ylabel('Cumulative Returns');

