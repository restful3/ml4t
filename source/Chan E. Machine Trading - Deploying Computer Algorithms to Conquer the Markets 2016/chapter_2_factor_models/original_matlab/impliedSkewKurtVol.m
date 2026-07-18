% impliedSkewKurtVol.m

% Use CRSP data

% First sort by implied skewness, take top 30%, then sort by implied
% kurtosis, take top 30%, finally sort by implied volatility, take top 30%.
% This is the long portfolio. Do the opposite for the short portfolio.
% Rebalance monthly.

% (see Option Implied Volatility, Skewness, and
% Kurtosis and the Cross-Section of Expected Stock Returns by Bali et al)

clear;
% onewaytcost=5/10000;
onewaytcost=0;

topPct=0.3;
holddays=21;

load('inputData_SPX_200401_201312_IVsurface', 'tday', 'syms', 'cl', 'impVolP_OTM', 'impVolP_ATM', 'impVolC_OTM', 'impVolC_ATM');

pos0=zeros(size(cl));

for t=1:length(tday)
   impVol=(impVolC_ATM(t, :)+impVolP_ATM(t, :))/2;
   impSkew=impVolC_OTM(t, :)-impVolP_OTM(t, :);
   impKurtosis=impVolC_OTM(t, :)+impVolP_OTM(t, :)-impVolC_ATM(t, :)-impVolP_ATM(t, :);
   
   goodData=find(isfinite(impSkew));
   if (length(goodData) >= 100)
       [~, idx1]=sort(impSkew(goodData), 'ascend');
             
       topN1=round(length(goodData)*topPct);
       shortCandidates=goodData(idx1(1:topN1)); % Short low Skew
       longCandidates=goodData(idx1(end-topN1+1:end)); % Buy high Skew
       
       assert(all(isfinite(impKurtosis(longCandidates))));
       assert(all(isfinite(impKurtosis(shortCandidates))));
       
       [~, idx2S]=sort(impKurtosis(shortCandidates), 'ascend');
       [~, idx2L]=sort(impKurtosis(longCandidates), 'descend');

       topN2S=round(length(shortCandidates)*topPct);
       topN2L=round(length(longCandidates)*topPct);
     
       shortCandidates2=shortCandidates(idx2S(1:topN2S)); % Short low kurtosis
       longCandidates2=longCandidates(idx2L(1:topN2L)); % Buy high kurtosis     
       
       assert(all(isfinite(impVol(longCandidates2))));
       assert(all(isfinite(impVol(shortCandidates2))));

       [~, idx3S]=sort(impVol(shortCandidates2), 'ascend');
       [~, idx3L]=sort(impVol(longCandidates2), 'descend');

       topN3S=round(length(shortCandidates2)*topPct);
       topN3L=round(length(longCandidates2)*topPct);

       shortCandidates3=shortCandidates2(idx3S(1:topN3S)); % Short low IV
       longCandidates3=longCandidates2(idx3L(1:topN3L)); % Buy high IV     

       pos0(t, shortCandidates3)=-1;
       pos0(t, longCandidates3)=1; 

   end
end

pos=zeros(size(pos0));

for h=0:holddays-1
    pos_lag=backshift(h, pos0);
    pos_lag(isnan(pos_lag))=0;
    pos=pos+pos_lag;
end



pnl=smartsum(backshift(1, pos).*(cl-backshift(1, cl))./backshift(1, cl)-onewaytcost*abs(pos-backshift(1, pos)), 2);
ret=pnl./smartsum(abs(backshift(1, pos)), 2);
ret(~isfinite(ret))=0;

idx=find(any(pos~=0, 2));
ret(1:(idx(1)-1))=[];
tday(1:(idx(1)-1))=[];

cumret=cumprod(1+ret)-1;

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), cumret);
title('Implied volatility/skewness/kurtosis as factor');
xlabel('Date');
ylabel('Cumulative Returns');

cagr=(1+cumret(end))^(252/length(cumret))-1;
[maxDD, maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'CAGR=%f Sharpe ratio=%f maxDD=%f maxDDD=%i Calmar ratio=%f\n', cagr, sqrt(252)*mean(ret)/std(ret), maxDD, maxDDD, -cagr/maxDD);
% CAGR=-0.009646 Sharpe ratio=-0.173205 maxDD=-0.134491 maxDDD=721 Calmar ratio=-0.071721

