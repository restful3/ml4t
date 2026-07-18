% Output separate file for each stock, but capture ALL options of all
% strikes and expirations.

% Use survivorship-bias-free SPX components for stock and option prices.
% Time to expiration >= 14 days & <= 62 days
clear;

% minTimeToExpiration=14;

stk=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday', 'syms', 'cusip', 'permco', 'gvkey', 'fromDate', 'thruDate', 'op', 'hi', 'lo', 'cl', 'bid', 'ask', 'vol', ...
    'op_unadj', 'hi_unadj', 'lo_unadj', 'cl_unadj', 'bid_unadj', 'ask_unadj', 'vol_unadj', 'earningsAnn');

% stk_intra=load('C:/Projects/reversal_data/inputData_eSignal_adj_20130910', 'dn', 'syms', 'trade');

tday=stk.tday;

cl_unadj=stk.cl_unadj;
cusip8=char(stk.cusip');
cusip8(:, end)=[];
cusip8=cellstr(cusip8);

fid=fopen('C:/Projects/Ivy/options_SPX_200401_201308.txt');
C=textscan(fid, repmat('%s', [1 21]), 1, 'Delimiter', '\t');
fclose(fid);

inputStr=[];

numIdx=1;
txtIdx=1;
isNum=[];

for i=1:length(C)
    switch char(C{1, i} )
        
        case 'secid'
            secidIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'cusip'
            cusipIdx=txtIdx;
            txtIdx=txtIdx+1;
            inputStr=[inputStr, '%s'];
            isNum=[isNum, false];
            
        case 'date'
            dateIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'exdate'
            exdateIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'cp_flag'
            cpflagIdx=txtIdx;
            txtIdx=txtIdx+1;
            inputStr=[inputStr, '%s'];
            isNum=[isNum, false];
            
        case 'strike_price'
            strikeIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'best_bid'
            bestbidIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'best_offer'
            bestaskIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'impl_volatility'
            impvolIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'delta'
            deltaIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'gamma'
            gammaIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'theta'
            thetaIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'vega'
            vegaIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'cfadj'
            cfadjIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'optionid'
            optionidIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];
            
        case 'symbol'
            inputStr=[inputStr, '%*s'];
            
        case 'volume'
            volumeIdx=numIdx;
            numIdx=numIdx+1;
            inputStr=[inputStr, '%f'];
            isNum=[isNum, true];

        case 'open_interest'
            inputStr=[inputStr, '%*f'];
            
        case 'ticker'
            inputStr=[inputStr, '%*s'];
            
        case 'index_flag'
            inputStr=[inputStr, '%*f'];
            
        case 'issuer'
            inputStr=[inputStr, '%*s'];
            
        case 'exercise_style'
            inputStr=[inputStr, '%*s'];
            
        otherwise
            assert(0);
    end
end

isNum=logical(isNum);
num=NaN(0, numIdx-1);
txt=cell(0, txtIdx-1);
len=1000000;

allIdx=1:(size(num, 2)+size(txt, 2));


for s=1:length(cusip8)
    firstPass=true;
    fprintf(1, '  stock %i cusip=%s ...\n', s, cusip8{s});
    fid=fopen('C:/Projects/Ivy/options_SPX_200401_201308.txt');
    C=textscan(fid, repmat('%s', [1 21]), 1, 'Delimiter', '\t');
    b=1;

    while (1)
        fprintf(1, 'Start block %i ...\n', b);
        b=b+1;

        C=textscan(fid, inputStr, len, 'Delimiter', '\t');
        num=cell2mat(C(:, allIdx(isNum)));
        
        if (isempty(num))
            break;
        elseif (num(end, dateIdx) < tday(1))
            continue;
        end
        
        txt= C(:, allIdx(~isNum));
        if (size(C{:, 1}, 1))==0
            break;
        end
        
        idxS=find(strcmp(cusip8(s), txt{:, cusipIdx})); 
        
        if (~isempty(idxS) )
            secid=unique(num(idxS, secidIdx));

            allOptionId=unique(num(idxS, optionidIdx)); 
            
            if (firstPass)
                optionid=allOptionId;

                strikes=NaN(size(cl_unadj, 1),  length(allOptionId));
                optExpireDate=NaN(size(cl_unadj, 1),  length(allOptionId));
                bid=NaN(size(cl_unadj, 1),  length(allOptionId));
                ask=NaN(size(cl_unadj, 1),  length(allOptionId));
                volume=NaN(size(cl_unadj, 1),  length(allOptionId));
                impVol=NaN(size(cl_unadj, 1),  length(allOptionId));
                delta=NaN(size(cl_unadj, 1),  length(allOptionId));
                gamma=NaN(size(cl_unadj, 1),  length(allOptionId));
                theta=NaN(size(cl_unadj, 1),  length(allOptionId));
                vega=NaN(size(cl_unadj, 1),  length(allOptionId));
                cfadj=NaN(size(cl_unadj, 1),  length(allOptionId));
                cpflag=cell(size(cl_unadj, 1),  length(allOptionId));
                
                firstPass=false;
            elseif ~isempty(setdiff(allOptionId, optionid))
                allOptionId=union(optionid, allOptionId);
                [foo, idxA, idxB]=intersect(optionid, allOptionId);
                optionid=allOptionId;

                strikes_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                optExpireDate_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                bid_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                ask_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                volume_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                impVol_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                delta_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                gamma_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                theta_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                vega_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                cfadj_new=NaN(size(cl_unadj, 1),  length(allOptionId));
                cpflag_new=cell(size(cl_unadj, 1),  length(allOptionId));

                strikes_new(:, idxB)=strikes(:, idxA);
                optExpireDate_new(:, idxB)=optExpireDate(:, idxA);
                bid_new(:, idxB)=bid(:, idxA);
                ask_new(:, idxB)=ask(:, idxA);
                volume_new(:, idxB)=volume(:, idxA);
                impVol_new(:, idxB)=impVol(:, idxA);
                delta_new(:, idxB)=delta(:, idxA);
                gamma_new(:, idxB)=gamma(:, idxA);
                theta_new(:, idxB)=theta(:, idxA);
                vega_new(:, idxB)=vega(:, idxA);
                cfadj_new(:, idxB)=cfadj(:, idxA);
                cpflag_new(:, idxB)=cpflag(:, idxA);
                
                strikes=strikes_new;
                optExpireDate=optExpireDate_new;
                bid=bid_new;
                ask=ask_new;
                volume=volume_new;
                impVol=impVol_new;
                delta=delta_new;
                gamma=gamma_new;
                theta=theta_new;
                vega=vega_new;
                cfadj=cfadj_new;
                cpflag=cpflag_new;
                
                clear *_new;
                
            end
            
            for t=1:length(tday)
                %                 numCalendarDaysToExpiration=datenum(cellstr(num2str(num(idxSP, dateIdx))), 'yyyymmdd')-datenum(num2str(tday(t)), 'yyyymmdd');
                idx=find(tday(t)==num(idxS, dateIdx));
                if (~isempty(idx))
                                        
                    myOptionid=num(idxS(idx), optionidIdx);
                    [foo, idxA, idxB]=intersect(optionid, myOptionid);
                    
                    strikes(t, idxA)=num(idxS(idx(idxB)), strikeIdx);
                    optExpireDate(t, idxA)=num(idxS(idx(idxB)), exdateIdx);
                    bid(t, idxA)=num(idxS(idx(idxB)), bestbidIdx);
                    ask(t, idxA)=num(idxS(idx(idxB)), bestaskIdx);
                    volume(t, idxA)=num(idxS(idx(idxB)), volumeIdx);
                    impVol(t, idxA)=num(idxS(idx(idxB)), impvolIdx);
                    delta(t, idxA)=num(idxS(idx(idxB)), deltaIdx);
                    gamma(t, idxA)=num(idxS(idx(idxB)), gammaIdx);
                    theta(t, idxA)=num(idxS(idx(idxB)), thetaIdx);
                    vega(t, idxA)= num(idxS(idx(idxB)), vegaIdx);
                    cfadj(t, idxA)=num(idxS(idx(idxB)), cfadjIdx);
                    
                    mycpflag=txt{:, cpflagIdx};
                    cpflag(t, idxA)=mycpflag(idxS(idx(idxB)));
                    
                end
                
            end
        end

    end
    
    if (1)
        save(['C:/Projects/Ivy/SPX_200701_201308/cusip_', cusip8{s}], 'secid', ...
            'strikes', 'optExpireDate', 'bid',  'ask', 'volume', 'impVol', 'delta',  'gamma',  'theta', 'vega', 'cfadj','cpflag', 'optionid');
    end
    fclose(fid);

end



