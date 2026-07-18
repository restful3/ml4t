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
    
    vgxdisp(model, estStdErrors);
    
%       Model  : 5-D VAR(1) with Additive Constant
%            Conditional mean is not AR-stable and is MA-invertible
%            Standard errors without DoF adjustment (maximum likelihood)
%        Parameter          Value     Std. Error    t-Statistic
%   -------------- -------------- -------------- --------------
%             a(1)        3.88363        1.15299        3.36832 
%             a(2)       0.669367      0.0970334        6.89832 
%             a(3)        1.75474       0.227636        7.70853 
%             a(4)          1.701       0.249767        6.81035 
%             a(5)         1.8752       0.282581        6.63596 
%       AR(1)(1,1)       0.991815     0.00320597        309.365 
%            (1,2)      0.0735881       0.100607       0.731441 
%            (1,3)      -0.105676      0.0289858       -3.64581 
%            (1,4)      0.0359698      0.0243106         1.4796 
%            (1,5)    -0.00619303      0.0204095      -0.303438 
%            (2,1)   -7.15594e-05    0.000269809      -0.265222 
%            (2,2)       0.970934     0.00846691        114.674 
%            (2,3)     -0.0103416     0.00243939        -4.2394 
%            (2,4)     0.00524778     0.00204594        2.56498 
%            (2,5)     0.00354032     0.00171763        2.06117 
%            (3,1)    -0.00158962    0.000632961       -2.51141 
%            (3,2)      -0.024093       0.019863       -1.21296 
%            (3,3)       0.965626     0.00572271        168.736 
%            (3,4)     0.00898799     0.00479969        1.87262 
%            (3,5)     0.00190162     0.00402949       0.471927 
%            (4,1)   -0.000771673    0.000694498       -1.11112 
%            (4,2)     -0.0409408      0.0217941       -1.87852 
%            (4,3)     -0.0284176     0.00627908       -4.52576 
%            (4,4)        1.00662     0.00526631        191.144 
%            (4,5)     0.00308001     0.00442123        0.69664 
%            (5,1)   -0.000526824    0.000785739      -0.670482 
%            (5,2)     -0.0579403      0.0246573       -2.34982 
%            (5,3)     -0.0309631       0.007104       -4.35854 
%            (5,4)        0.01704     0.00595818        2.85993 
%            (5,5)       0.998657     0.00500208        199.648 
%           Q(1,1)        36.2559                               
%           Q(2,1)        1.67571                               
%           Q(2,2)       0.256786                               
%           Q(3,1)        3.37592                               
%           Q(3,2)       0.449846                               
%           Q(3,3)        1.41323                               
%           Q(4,1)        3.78265                               
%           Q(4,2)       0.513747                               
%           Q(4,3)        1.20474                               
%           Q(4,4)        1.70138                               
%           Q(5,1)        4.39542                               
%           Q(5,2)       0.522437                               
%           Q(5,3)        1.26443                               
%           Q(5,4)        1.41357                               
%           Q(5,5)        2.17779        
    
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
    
    plot(datetime(tday(testset), 'ConvertFrom', 'yyyyMMdd'), cumret(testset));
    title('VAR(1) model on computer hardware SPX stocks');
    xlabel('Date');
    ylabel('Cumulative Returns');
    
    
    % Annualized compound returns on testset
    CAGR=((1+cumret(end))/(1+cumret(trainset(end))))^(252/length(testset))-1;
    sharpe=sqrt(252)*mean(ret)/std(ret);
    
    %         if (CAGR >=0.05 )
    fprintf(1, 'gind=%i CAGR=%f Sharpe=%f\n', gind_uniq(g), CAGR, sharpe);
    %         end
    
    [model_vec, C]=vartovec(model);
    C
    
    %     C =
    %
    %    -0.0082    0.0736   -0.1057    0.0360   -0.0062
    %    -0.0001   -0.0291   -0.0103    0.0052    0.0035
    %    -0.0016   -0.0241   -0.0344    0.0090    0.0019
    %    -0.0008   -0.0409   -0.0284    0.0066    0.0031
    %    -0.0005   -0.0579   -0.0310    0.0170   -0.0013
    
    result=johansen(mid(trainset, :), 0, 1);

end

