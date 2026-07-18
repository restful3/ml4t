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
        
        pos=zeros(size(retF));
        pos=retF./repmat(smartsum(abs(retF), 2), [1, size(retF, 2)]);
        
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

% Best industry groups
% gind=151010 CAGR=0.053266 Sharpe=0.188995
% gind=151020 CAGR=0.196442 Sharpe=0.284792
% gind=151040 CAGR=0.539128 Sharpe=0.712724
% gind=253010 CAGR=0.277122 Sharpe=0.611753
% gind=254010 CAGR=0.539260 Sharpe=0.856102
% gind=452020 CAGR=0.652175 Sharpe=1.018252
% gind=452030 CAGR=0.286707 Sharpe=0.460048
% gind=452040 CAGR=0.617929 Sharpe=0.749629
% gind=501020 CAGR=0.225108 Sharpe=0.448142
% gind=551010 CAGR=0.304256 Sharpe=0.853411
% gind=551030 CAGR=0.231232 Sharpe=0.756425
% gind=551050 CAGR=0.418403 Sharpe=0.634163


% gind=101010 CAGR=-0.197189
% gind=101020 CAGR=0.003674
% gind=151010 CAGR=0.053266
% gind=151020 CAGR=0.196442
% gind=151030 CAGR=-0.323492
% gind=151040 CAGR=0.539128
% gind=151050 CAGR=-0.246556
% gind=201010 CAGR=-0.349346
% gind=201020 CAGR=-0.341114
% gind=201030 CAGR=-0.189163
% gind=201040 CAGR=-0.283593
% gind=201050 CAGR=-0.300020
% gind=201060 CAGR=-0.185858
% gind=201070 CAGR=-0.224995
% gind=202010 CAGR=-0.351369
% gind=202020 CAGR=-0.266231
% gind=203010 CAGR=-0.348381
% gind=203020 CAGR=-0.292393
% gind=203040 CAGR=-0.230579
% gind=251010 CAGR=-0.454637
% gind=251020 CAGR=-0.271298
% gind=252010 CAGR=-0.286290
% gind=252020 CAGR=-0.348694
% gind=252030 CAGR=-0.280057
% gind=253010 CAGR=0.277122
% gind=253020 CAGR=-0.228235
% gind=254010 CAGR=0.539260
% gind=255010 CAGR=-0.259778
% gind=255020 CAGR=-0.397261
% gind=255030 CAGR=-0.223521
% gind=255040 CAGR=-0.478064
% gind=301010 CAGR=-0.336146
% gind=302010 CAGR=-0.243121
% gind=302020 CAGR=-0.245052
% gind=302030 CAGR=-0.192001
% gind=303010 CAGR=-0.216881
% gind=303020 CAGR=-0.110149
% gind=351010 CAGR=-0.378863
% gind=351020 CAGR=0.016259
% gind=352010 CAGR=-0.466290
% gind=352020 CAGR=-0.299635
% gind=352030 CAGR=-0.242628
% gind=401010 CAGR=-0.278772
% gind=402010 CAGR=-0.282898
% gind=402020 CAGR=-0.345134
% gind=402030 CAGR=-0.350012
% gind=403010 CAGR=-0.345455
% gind=404020 CAGR=-0.064529
% gind=404030 CAGR=-0.270442
% gind=451010 CAGR=-0.302501
% gind=451020 CAGR=-0.284280
% gind=451030 CAGR=-0.331971
% gind=452010 CAGR=-0.138040
% gind=452020 CAGR=0.652175
% gind=452030 CAGR=0.286707
% gind=452040 CAGR=0.617929
% gind=453010 CAGR=-0.371760
% gind=501010 CAGR=-0.116527
% gind=501020 CAGR=0.225108
% gind=551010 CAGR=0.304256
% gind=551030 CAGR=0.231232
% gind=551050 CAGR=0.418403
