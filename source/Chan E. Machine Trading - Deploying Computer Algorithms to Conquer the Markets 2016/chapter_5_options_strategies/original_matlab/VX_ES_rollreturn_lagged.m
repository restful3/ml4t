% Use 1-day lagged VX return as signal
clear;
entryThreshold=0.1;
onewaytcost=0/10000;

load('C:/Projects/Futures_data/inputDataDaily_VX_20150828', 'tday', 'contracts', 'cl'); % TODO: all input files need to be in local folder
VX=cl/1000; % prices were multiplied by 1000 
vixIdx=find(strcmp(contracts, '0000$'));
VIX=VX(:, vixIdx);
VX(:, vixIdx)=[];
contracts(vixIdx)=[];

% es=load('C:/Projects/Futures_data/inputDataDaily_ES_20150828', 'tday', 'contracts', 'cl');
es=load('C:/Projects/Futures_data/inputData_ES_continuous_20170111', 'tday',  'cl');
ES=es.cl;
tday_ES=es.tday;

[tday idx1 idx2]=intersect(tday, tday_ES);
VIX=VIX(idx1, :);
VX=VX(idx1, :);
ES=ES(idx2);

% OOS=find(tday>20120507); % Out of sample period
% tday=tday(OOS);
% VIX=VIX(OOS);
% VX=VX(OOS, :);
% ES=ES(OOS, :);



isExpireDate=false(size(VX));
isExpireDate=isfinite(VX) & ~isfinite(fwdshift(1, VX));

% Define front month as 40 days to 10 days before expiration
numDaysStart=30;
numDaysEnd=1;
% numDaysStart=40;
% numDaysEnd=10;

positions=[zeros(size(VX)) zeros(size(ES))];

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
        dailyRoll=(VX(idx-1, c)-VIX(idx-1))./[expireIdx-startIdx+2:-1:expireIdx-endIdx+2]'; % Use previous day's VX to compute roll retunr
        positions(idx(dailyRoll > entryThreshold), c)=-1*0.3906;
        positions(idx(dailyRoll > entryThreshold), end)=-1;
               
        positions(idx(dailyRoll < -entryThreshold), c)=1*0.3906;
        positions(idx(dailyRoll < -entryThreshold), end)=1;
        
    end
end

y=[VX*1000 ES*50];

% ret=smartsum(lag(positions).*(y-lag(y, 1)), 2)./smartsum(abs(lag(positions.*y)), 2)-...
%     onewaytcost*smartsum(abs(positions.*y-lag(positions.*y)), 2)./smartsum(abs(lag(positions.*y)), 2);
ret=smartsum(backshift(1, positions).*(y-backshift(1, y)), 2)./smartsum(abs(backshift(1, positions.*y)), 2)-...
    onewaytcost*smartsum(abs(positions.*y-backshift(1, positions.*y)), 2)./smartsum(abs(backshift(1, positions.*y)), 2);
ret(isnan(ret))=0;

% idx=find(tday >= 20100629);
idx=find(tday >= 20040405 & tday <= 20150819);
% idx=find(tday >= 20101130 & tday <= 20150819);
ret=ret(idx);
tday=tday(idx);

cumret=cumprod(1+ret)-1;
plot(datetime(tday, 'ConvertFrom', 'yyyyMMdd'), cumret); % Cumulative compounded return
title('VX vs ES');
xlabel('Date');
ylabel('Cumulative Returns');

cagr= prod(1+ret).^(252/length(ret))-1;
% fprintf(1, 'APR=%f Sharpe=%f\n', prod(1+ret(idx(501:end))).^(252/length(ret(idx(501:end))))-1, sqrt(252)*mean(ret(idx(501:end)))/std(ret(idx(501:end))));
fprintf(1, 'CAGR=%f Sharpe=%f\n', cagr, sqrt(252)*mean(ret)/std(ret));

[maxDD maxDDD]=calculateMaxDD(cumret);
fprintf(1, 'maxDD=%f maxDDD=%i Calmar ratio=%f\n', maxDD, maxDDD, -cagr/maxDD);

% tday([1 end])
% 
% ans =
% 
%     20040405
%     20150819
% CAGR=0.049482 Sharpe=0.709558
% maxDD=-0.090426 maxDDD=599 Calmar ratio=0.547208

% tday([1 end])
% 
% ans =
% 
%     20101130
%     20150819

% CAGR=0.036923 Sharpe=0.703974
% maxDD=-0.090426 maxDDD=599 Calmar ratio=0.408325

