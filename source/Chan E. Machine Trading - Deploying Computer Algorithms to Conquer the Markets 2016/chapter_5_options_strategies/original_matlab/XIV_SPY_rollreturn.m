% Trade XIV vs SPY, but use same signals from VX. Hedge ratio from Kalman
% filter.

clear;
entryThreshold=0.1;

vx=load('C:/Projects/Futures_data/inputDataDaily_VX_20150828', 'tday', 'contracts', 'cl'); % TODO: all input files need to be in local folder
VX=vx.cl/1000; % prices were multiplied by 1000 
contracts=vx.contracts;
vixIdx=find(strcmp(contracts, '0000$'));
VIX=VX(:, vixIdx);
VX(:, vixIdx)=[];
contracts(vixIdx)=[];

load('hedgeRatio_XIV_SPY', 'tday', 'beta', 'x', 'y'); % x=XIV, y=SPY

[tday idx1 idx2]=intersect(vx.tday, tday);
VIX=VIX(idx1, :);
VX=VX(idx1, :);
x=x(idx2);
y=y(idx2);
beta=beta(:, idx2);
isExpireDate=false(size(VX));
isExpireDate=isfinite(VX) & ~isfinite(fwdshift(1, VX));

% Define front month as 40 days to 10 days before expiration
numDaysStart=30;
numDaysEnd=1;
% numDaysStart=40;
% numDaysEnd=10;

positions=zeros(size(x, 1), 2);

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
        dailyRoll=-(VX(idx, c)-VIX(idx))./[expireIdx-startIdx+1:-1:expireIdx-endIdx+1]'; % roll return
        positions(idx(dailyRoll > entryThreshold), 1)=-beta(1, idx(dailyRoll > entryThreshold));
        positions(idx(dailyRoll > entryThreshold), 2)=1;
               
        positions(idx(dailyRoll < -entryThreshold), 1)=beta(1, idx(dailyRoll < -entryThreshold));
        positions(idx(dailyRoll < -entryThreshold), 2)=-1;
        
    end
end

cl=[x y];

ret=smartsum(backshift(1, positions).*(cl-backshift(1, cl)), 2)./smartsum(abs(backshift(1, positions.*cl)), 2);
ret(isnan(ret))=0;

idx=find(tday >= 20100629);
ret=ret(idx);
tday=tday(idx);

cumret=cumprod(1+ret)-1;
plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative compounded return
title('XIV vs SPY using Kalman Filter');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+ret).^(252/length(ret))-1;
% fprintf(1, 'APR=%f Sharpe=%f\n', prod(1+ret(idx(501:end))).^(252/length(ret(idx(501:end))))-1, sqrt(252)*mean(ret(idx(501:end)))/std(ret(idx(501:end))));
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(ret)/std(ret));

[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);
% CAGR=0.160755 Sharpe=1.325540
% maxDD=-0.169324 maxDDD=173 Calmar ratio=0.949392


