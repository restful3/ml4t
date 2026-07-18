% bollinger_5min.m
clear;

load('Jonathan_BTCUSD_BBO_1minute', 'tday', 'HHMM', 'bid', 'ask');

cl=(bid+ask)/2;

idx=find(isfinite(cl));
tday(1:idx-1)=[];
HHMM(1:idx-1)=[];
cl(1:idx-1)=[];

lookback=60;
entryZscore=2;
exitZscore=0;

% buyEntry=cl <= smartMovingAvg(cl, lookback)-entryZscore*smartMovingStd(cl, lookback);
% sellEntry=cl >= smartMovingAvg(cl, lookback)+entryZscore*smartMovingStd(cl, lookback);
buyEntry=cl <= movingAvg(cl, lookback)-entryZscore*movingStd(cl, lookback);
sellEntry=cl >= movingAvg(cl, lookback)+entryZscore*movingStd(cl, lookback);

positionL=NaN(size(cl));
positionL(1)=0;
positionL(buyEntry)=1;
positionS=NaN(size(cl));
positionS(1)=0;
positionS(sellEntry)=-1;

% buyExit= cl >= smartMovingAvg(cl, lookback)-exitZscore*smartMovingStd(cl, lookback);
% sellExit= cl <= smartMovingAvg(cl, lookback)+exitZscore*smartMovingStd(cl, lookback);
buyExit= cl >= movingAvg(cl, lookback)-exitZscore*movingStd(cl, lookback);
sellExit= cl <= movingAvg(cl, lookback)+exitZscore*movingStd(cl, lookback);

positionL(buyExit)=0;
positionS(sellExit)=0;

positionL=fillMissingData(positionL);
positionS=fillMissingData(positionS);
position=positionL+positionS;

ret=(backshift(1, position).*(cl-backshift(1, cl)))./backshift(1, cl);
ret(isnan(ret))=0;

trainset=1:(length(cl)-126*24*60); % Use all but 0.5 year of bars for in-sample fitting
testset=trainset(end)+1:length(cl);

cumret=cumprod(1+ret)-1;

plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), (1+cumret(testset))/(1+cumret(trainset(end)))-1);
title('Bollinger band model on BTC.USD');
xlabel('Date');
ylabel('Cumulative Returns');

% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1
%       0.201024204778726


% Annualized compound returns on testset
((1+cumret(end))/(1+cumret(trainset(end))))^(252*24*60/length(testset))-1
%     0.442459140464370


% With transaction cost of 10 bps per trade
tcost=10/10000;
ret=(backshift(1, position).*(cl-backshift(1, cl)))./backshift(1, cl)-...
    tcost*abs(position-backshift(1, position));

ret(isnan(ret))=0;
cumret=cumprod(1+ret)-1;

% Test set cumulative return
(1+cumret(end))/(1+cumret(trainset(end)))-1
% -0.994913611902576