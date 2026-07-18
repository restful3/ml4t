clear;

load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'syms', 'tday', 'bid', 'ask');

[num txt]=xlsread('C:/Projects/Compustat/AnnualFundamentalsSPX/Annual2.csv');
tickers=txt(2:end, strcmp('tic', txt(1, :)));
gind=num(:, strcmp('gind', txt(1, :))); % 65 GIC industry groups
fiscalYr=num(:, strcmp('fyear', txt(1, :)));

% Pick a an  industry group as example, and pick those stocks whos
% industry group has not changed
gind_uniq=unique(gind);

g=find(gind_uniq==452020); % Computer hardware
%         'AAPL'    'EMC'    'HPQ'    'NTAP'    'SNDK'    'STX'    'WDC'
myTickers=unique(tickers(gind==gind(1)));

% Make sure this stock does not change industry group
for s=1:length(myTickers)
    assert(length(unique(gind(strcmp(tickers, myTickers(1)))))==1);
end

[~, idx1, idx2]=intersect(syms, myTickers);

bid=fillMissingData(bid);
ask=fillMissingData(ask);
mid=(bid(:, idx1)+ask(:, idx1))/2;

incompleteDataIdx=find(sum(isnan(mid), 1) > 0);
mid(:, incompleteDataIdx)=[]; % Eliminate stocks that were not present at beginning of trainset

trainset=1:(length(mid)-252); % Use all but 1 year of bars for in-sample fitting

% LOGL=zeros(25, 1); % log likelihood for up to 25 days
LOGL=zeros(5, 1); % log likelihood for up to 5 days
P=zeros(size(LOGL)); % p values

for p=1:length(P)
    model=vgxset('n', size(mid, 2), 'nAR', p, 'Constant', true); % with additive offset
    [model,~,logL,~] = vgxvarx(model,mid(trainset, :));
    [NumParam,~] = vgxcount(model);

    LOGL(p) = logL;
    P(p) = NumParam;
    
end

[~, bic]=aicbic(LOGL, P, length(trainset));

[~, pMin]=min(bic)
% pMin =
% 
%      1
