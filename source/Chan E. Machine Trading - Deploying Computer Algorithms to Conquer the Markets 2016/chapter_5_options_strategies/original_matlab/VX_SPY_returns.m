clear;

load('C:/Projects/Futures_data/inputDataDaily_VX_20150828', 'tday', 'contracts', 'cl');

vixIdx=find(strcmp(contracts, '0000$'));
cl(:, vixIdx)=[];

isExpireDate=false(size(cl));
isExpireDate=isfinite(cl) & ~isfinite(fwdshift(1, cl));

% Define front month as 31 days to 7 days before expiration
numDaysStart=31;
numDaysEnd=7; 

ret=calculateReturns(cl, 1);
ret_VX=NaN(size(tday));

for c=1:length(contracts)-1
    
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
        
        ret_VX(idx)=-ret(idx, c); % assume short position
        
     end
end

ret_VX(~isfinite(ret_VX))=0;

% Compare with SPY
S=load('C:/Projects/prod_data/inputDataOHLCDaily_ETF_20150828', 'tday', 'stocks', 'cl');

spyIdx=find(strcmp(S.stocks, 'SPY'));
spy=S.cl(:, spyIdx);

ret_SPY=calculateReturns(spy, 1);
ret_SPY(~isfinite(ret_SPY))=0;

[tday, idxV, idxS]=intersect(tday, S.tday);

ret_VX=ret_VX(idxV);
ret_SPY=ret_SPY(idxS);

vx_cumret=cumprod(1+ret_VX)-1;
spy_cumret=cumprod(1+ret_SPY)-1;

% vx_cumret=(1+vx_cumret(idxV))./(1+vx_cumret(idxV(1)))-1;
% spy_cumret=(1+spy_cumret(idxS))./(1+spy_cumret(idxS(1)))-1;

plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), [spy_cumret vx_cumret]);

kelly_vx=mean(ret_VX)/var(ret_VX)
kelly_spy=mean(ret_SPY)/var(ret_SPY)

kelly_vx=1;
kelly_spy=1;

vx_kelly_cumret=cumprod(1+kelly_vx*ret_VX)-1; %  0.878754263173843
spy_kelly_cumret=cumprod(1+kelly_spy*ret_SPY)-1; %   2.152202064271893


plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), [spy_kelly_cumret vx_kelly_cumret]);
legend({'SPY x2.15', 'VX x-0.88'});
title('Long SPY vs Short VX');
xlabel('Date');
ylabel('Cumulative Returns');

cagr_SPY=(1+spy_cumret(end))^(252/length(spy_kelly_cumret))-1
%  0.072509105277968
cagr_VX=(1+vx_cumret(end))^(252/length(vx_kelly_cumret))-1
%    0.177794122502132

[maxDD_SPY, maxDDD_SPY]=calculateMaxDD(spy_kelly_cumret)
%   -0.862501548306116
%  1462

[maxDD_VX, maxDDD_VX]=calculateMaxDD(vx_kelly_cumret)
%   -0.917810513629244
% 1544

Calmar_SPY=-cagr_SPY/maxDD_SPY
%    0.084068376944216

Calmar_VX=-cagr_VX/maxDD_VX
%    0.193715499944635


