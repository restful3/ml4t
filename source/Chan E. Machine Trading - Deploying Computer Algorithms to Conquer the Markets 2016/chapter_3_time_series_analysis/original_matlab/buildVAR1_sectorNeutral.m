clear;

load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'syms', 'tday', 'bid', 'ask');

[num txt]=xlsread('C:/Projects/Compustat/AnnualFundamentalsSPX/Annual2.csv');
tickers=txt(2:end, strcmp('tic', txt(1, :)));
gind=num(:, strcmp('gind', txt(1, :))); % 65 GIC industry groups
fiscalYr=num(:, strcmp('fyear', txt(1, :)));

% Pick a random industry group as example, and pick those stocks whos
% industry group has not changed
gind_uniq=unique(gind);

for g=1:length(gind_uniq)
    
    myTickers=unique(tickers(gind==gind_uniq(g)));
    % Make sure this stock does not change industry group
    for s=1:length(myTickers)
        assert(length(unique(gind(strcmp(tickers, myTickers(1)))))==1);
    end
    
    [~, idx1, idx2]=intersect(syms, myTickers);
    
    
    bid=fillMissingData(bid);
    ask=fillMissingData(ask);
    mid=(bid(:, idx1)+ask(:, idx1))/2;
    
    incompleteDataIdx=find(sum(isnan(mid), 1) > 0);
    mid(:, incompleteDataIdx)=[];
    
    if (~isempty(mid))
        
        trainset=1:(length(mid)-252); % Use all but 1 year of bars for in-sample fitting
        
        pMin=1;
        model=vgxset('n', size(mid, 2), 'nAR', pMin, 'Constant', true); % with additive offset
        
        [model, estStdErrors]=vgxvarx(model,mid(trainset, :));
        
        %     vgxdisp(model, estStdErrors);
        
        
        testset=trainset(end)+1:size(mid, 1);
        
        yF=NaN(size(mid));
        
        for t=testset(1):size(mid, 1)
            FY = vgxpred(model,1, [], mid(t-pMin+1:t, :));
            yF(t, :)=FY;
        end
        
        retF=(yF-mid)./mid;
        sectorRetF=mean(retF, 2);
        
        pos=zeros(size(retF));
        pos=(retF-repmat(sectorRetF, [1 size(retF, 2)]))./repmat(smartsum(abs(retF-repmat(sectorRetF, [1 size(retF, 2)])), 2), [1, size(retF, 2)]);
        
        ret=smartsum(backshift(1, pos).*(mid-backshift(1, mid))./backshift(1, mid), 2);
        ret(isnan(ret))=0;
        cumret=cumprod(1+ret)-1;
        
        % plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset));
        % title('VAR(1) model on electric utilities SPX stocks');
        % xlabel('Date');
        % ylabel('Cumulative Returns');
        
        
        % Annualized compound returns on testset
        CAGR=((1+cumret(end))/(1+cumret(trainset(end))))^(252/length(testset))-1;
        sharpe=sqrt(252)*mean(ret)/std(ret);
        
        if (CAGR >=0.05 ) 
            fprintf(1, 'gind=%i CAGR=%f Sharpe=%f\n', gind_uniq(g), CAGR, sharpe);
        end
    end
end

% gind=201060 CAGR=0.100771 Sharpe=0.527903
% gind=252030 CAGR=0.329203 Sharpe=0.734381
% gind=253010 CAGR=0.059262 Sharpe=0.222797
% gind=254010 CAGR=0.185317 Sharpe=0.431981
% gind=351020 CAGR=0.104741 Sharpe=0.615125
% gind=352010 CAGR=0.070572 Sharpe=0.251663
% gind=352020 CAGR=0.056426 Sharpe=0.245557
% gind=402010 CAGR=0.070578 Sharpe=0.307767
% gind=452020 CAGR=0.483658 Sharpe=0.923363 (Computer hardware)
% gind=452030 CAGR=0.281896 Sharpe=0.546229
