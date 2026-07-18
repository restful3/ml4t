% For SPX index option only
% capture ALL options of all
% strikes and expirations.

% Output one file for each day

%
% Time to expiration >= 7 days & <= 62 days
clear;


stk=load('C:/Projects/reversal_data/inputData_SPX_200401_201312', 'tday');

tday=stk.tday;

fid=fopen('C:/Projects/Ivy/options_SPX_index_200401_201308.txt');
C=textscan(fid, repmat('%s', [1 20]), 1, 'Delimiter', '\t');
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


fid=fopen('C:/Projects/Ivy/options_SPX_index_200401_201308.txt');
C=textscan(fid, repmat('%s', [1 20]), 1, 'Delimiter', '\t');
b=1;

lastDate=NaN;
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
    
    secid=unique(num(:, secidIdx));
    
    while (1)
    
        idx=find(num(1, dateIdx)==num(:, dateIdx)); % first date
        
        allOptionId=unique(num(idx, optionidIdx));
    
        if (num(1, dateIdx) ~= lastDate) % new day
            
            if (isfinite(lastDate))
                save(['C:/Projects/Ivy/SPX_index_200701_201308/', num2str(lastDate)], 'secid', ...
                    'strikes', 'optExpireDate', 'bid',  'ask', 'volume', 'impVol', 'delta',  'gamma',  'theta', 'vega', 'cfadj','cpflag', 'optionid');
            end
            
            optionid=allOptionId;
            
            strikes=NaN(1,  length(allOptionId));
            optExpireDate=NaN(1,  length(allOptionId));
            bid=NaN(1,  length(allOptionId));
            ask=NaN(1,  length(allOptionId));
            volume=NaN(1,  length(allOptionId));
            impVol=NaN(1,  length(allOptionId));
            delta=NaN(1,  length(allOptionId));
            gamma=NaN(1,  length(allOptionId));
            theta=NaN(1,  length(allOptionId));
            vega=NaN(1,  length(allOptionId));
            cfadj=NaN(1,  length(allOptionId));
            cpflag=cell(1,  length(allOptionId));
            
        else
            
            allOptionId=union(optionid, allOptionId);
            [foo, idxA, idxB]=intersect(optionid, allOptionId);
            optionid=allOptionId;
            
            strikes_new=NaN(1,  length(allOptionId));
            optExpireDate_new=NaN(1,  length(allOptionId));
            bid_new=NaN(1,  length(allOptionId));
            ask_new=NaN(1,  length(allOptionId));
            volume_new=NaN(1,  length(allOptionId));
            impVol_new=NaN(1,  length(allOptionId));
            delta_new=NaN(1,  length(allOptionId));
            gamma_new=NaN(1,  length(allOptionId));
            theta_new=NaN(1,  length(allOptionId));
            vega_new=NaN(1,  length(allOptionId));
            cfadj_new=NaN(1,  length(allOptionId));
            cpflag_new=cell(1,  length(allOptionId));
            
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
        
        myOptionid=num(idx, optionidIdx);
        [foo, idxA, idxB]=intersect(optionid, myOptionid);
        
        strikes(1, idxA)=num(idx(idxB), strikeIdx);
        optExpireDate(1, idxA)=num(idx(idxB), exdateIdx);
        bid(1, idxA)=num(idx(idxB), bestbidIdx);
        ask(1, idxA)=num(idx(idxB), bestaskIdx);
        volume(1, idxA)=num(idx(idxB), volumeIdx);
        impVol(1, idxA)=num(idx(idxB), impvolIdx);
        delta(1, idxA)=num(idx(idxB), deltaIdx);
        gamma(1, idxA)=num(idx(idxB), gammaIdx);
        theta(1, idxA)=num(idx(idxB), thetaIdx);
        vega(1, idxA)= num(idx(idxB), vegaIdx);
        cfadj(1, idxA)=num(idx(idxB), cfadjIdx);
        
        mycpflag=txt{:, cpflagIdx};
        cpflag(1, idxA)=mycpflag(idx(idxB));
        
        lastDate=num(idx(end), dateIdx);
        
        num(idx, :)=[];
        txt{:}(idx, :)=[];
        
        if (isempty(num))
            save(['C:/Projects/Ivy/SPX_index_200701_201308/', num2str(lastDate)], 'secid', ...
                'strikes', 'optExpireDate', 'bid',  'ask', 'volume', 'impVol', 'delta',  'gamma',  'theta', 'vega', 'cfadj','cpflag', 'optionid');
            break;
        end
      
    
    end
    
end



fclose(fid);





