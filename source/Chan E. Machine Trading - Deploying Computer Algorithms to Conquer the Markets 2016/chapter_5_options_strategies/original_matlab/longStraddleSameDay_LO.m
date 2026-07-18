% Buy at Wed 9:00, exit same day at 10:31.

clear;

entryDay=4; % Wed
exitDay=4; % Wed

entryTime=900;
exitTime=1031;

contracts={...
    'J12', ...
    'K12', ...
    'M12', ...
    'N12', ...
    'Q12', ...
    'U12', ...
    'V12', ...
    'X12', ...
    'Z12', ...
    'F13', ...
    'G13', ...
    'H13', ...
    'J13'};

dateRanges={...
    '20120301_20120331', ...
    '20120310_20120409', ...
    '20120410_20120509', ...
    '20120510_20120620', ...
    '20120610_20120720', ...
    '20120704_20120809', ...
    '20120804_20120909', ...
    '20120904_20121015', ...
    '20121004_20121115', ...
    '20121104_20121215', ...
    '20121204_20130115', ...
    '20130104_20130215', ...
    '20130204_20130227'};

firstDateTimes={...
    '20120301 08:30:00', ...
    '20120306 08:30:00', ...
    '20120406 08:30:00', ...
    '20120506 08:30:00', ...
    '20120606 08:30:00', ...
    '20120706 08:30:00', ...
    '20120806 08:30:00', ...
    '20120906 08:30:00', ...
    '20121006 08:30:00', ...
    '20121106 08:30:00', ...
    '20121206 08:30:00', ...
    '20130106 08:30:00', ...
    '20130206 08:30:00'};

lastDateTimes={...
    '20120305 10:30:00', ...
    '20120405 10:30:00', ...
    '20120505 10:30:00', ...
    '20120605 10:30:00', ...
    '20120705 10:30:00', ...
    '20120805 10:30:00', ...
    '20120905 10:30:00', ...
    '20121005 10:30:00', ...
    '20121105 10:30:00', ...
    '20121205 10:30:00', ...
    '20130105 10:30:00', ...
    '20130205 10:30:00', ...
    '20130305 10:30:00'};

assert(length(contracts)==length(dateRanges));
assert(length(contracts)==length(firstDateTimes));
assert(length(contracts)==length(lastDateTimes));

cumPL=0;
for c=1:length(contracts)
    contract=contracts{c};
    dateRange=dateRanges{c};
    firstDateTime=firstDateTimes{c};
    lastDateTime=lastDateTimes{c};
        
    % Get futures price to determine what strike price is ATM
    load(['//I3/Futures_data/inputData_CL', contract, '_BBO_', dateRange, '.mat'], 'dn', 'bid', 'ask');
    
    goodData=dn >= datenum(firstDateTime, 'yyyymmdd HH:MM:SS') & dn <= datenum(lastDateTime, 'yyyymmdd HH:MM:SS');
    
    dnFut=dn(goodData);
    bidFut=bid(goodData);
    askFut=ask(goodData);
    
    clear dn bid ask;
    
    midFut=(bidFut+askFut)/2;
    
    len=1000000;
        
    hhmm=str2double(cellstr(datestr(dnFut, 'HHMM')));
    
    isEntryFut=backshift(1, hhmm) < entryTime &  hhmm >= entryTime & weekday(dnFut')==entryDay;
    
    opn=midFut(isEntryFut);
    yyyymmddFut=yyyymmdd(datetime(dnFut, 'ConvertFrom', 'datenum'));
    yyyymmddEntry=yyyymmddFut(isEntryFut);
    
    isEntryFutIdx=find(isEntryFut);
    
    roundPrice=@(x) round(2*x)/2; % =1/minPriceIncr
    
    % Iterate through each event date
    for d=1:length(opn)
               
        stkPriceStr=num2str(roundPrice(opn(d)));
        if (isempty(regexp(stkPriceStr, '\.')))
            stkPriceStr=[stkPriceStr, '00'];
        elseif (isempty(regexp(stkPriceStr, '\.\d\d')))
            stkPriceStr=regexprep(stkPriceStr, '\.', '');
            stkPriceStr=[stkPriceStr, '0'];
        else
            stkPriceStr=regexprep(stkPriceStr, '\.', '');
        end
        
        %% Call
        fid=fopen(['//I3//Options_data/LO.', contract, '/pLO', contract, stkPriceStr, 'C_BBO_', dateRange, '.csv']);
        assert(fid~=-1);
        
        dn1=[];
        bid1=[];
        ask1=[];
        
        while (1)
            C=textscan(fid, '%u%s%f%f', len, 'Delimiter', ',');
            if (isempty(C))
                break;
            end
            if (length(C{1, 1})==0)
                break;
            end
            
            tday=num2str(C{1, 1});
            hhmmssfff=C{1, 2};
            
            bid1=[bid1; C{1, 3}];
            ask1=[ask1; C{1, 4}];
            
            dn1=[dn1; datenum(cellstr([tday, repmat(' ', size(hhmmssfff)), char(hhmmssfff)]), 'yyyymmdd HH:MM:SS.FFF')];
        end
        
        fclose(fid);
        
        %%%
        
        bid=NaN(size(bid1));
        ask=NaN(size(bid));
        dn=NaN(size(bid));
        
        t=1;
        t1=1;
        
        % Assume prices with same time stamp are in chronological order.
        while (t1 <= length(dn1) )
            if (bid1(t1) ~= 0 && ask1(t1) ~= 0)
                bid(t, 1)=bid1(t1);
                ask(t, 1)=ask1(t1);
                dn(t)=dn1(t1);
                t=t+1;
            end
            t1=t1+1;
        end
        
        lastGoodData=find(isfinite(dn));
        lastGoodData=lastGoodData(end);
        
        dn=dn(1:lastGoodData)';
        bid=bid(1:lastGoodData, :);
        ask=ask(1:lastGoodData, :);
        
        % Forward-fill quote prices when there are no new ticks.
        bid=fillMissingData(bid);
        ask=fillMissingData(ask);
        
        goodData=dn >= datenum(firstDateTime, 'yyyymmdd HH:MM:SS') & dn <= datenum(lastDateTime, 'yyyymmdd HH:MM:SS');
        
        dn=dn(goodData);
        bid=bid(goodData);
        ask=ask(goodData);
        
        hhmmCall=str2double(cellstr(datestr(dn, 'HHMM')));
        yyyymmddCall=yyyymmdd(datetime(dn, 'ConvertFrom', 'datenum'))';
        
        isEntry=hhmmCall < entryTime & yyyymmddCall==yyyymmddEntry(d);
        if (isempty(isEntry))
            fprintf(1, '    Missing call data on entry date %i: skipping...\n', yyyymmddEntry(d));
            continue;
        end

        isExit=hhmmCall < exitTime & yyyymmddCall == yyyymmddEntry(d);
        isEntryIdx=find(isEntry);
        isEntryIdx=isEntryIdx(end); % use latest entry 
        isExitIdx=find(isExit);
                     
        % Confirm futures dates are same as entry dates for options
        assert(yyyymmddEntry(d)==str2double(datestr(dn(isEntryIdx), 'yyyymmdd')));
        
        % Pick farthest exit date that is within a calendar week but more
        % than 3 calendar days
        ie=find(dn(isExitIdx)-dn(isEntryIdx) ==0);
        if (isempty(ie))
            fprintf(1, '    ***Cannot find call exit date: skipping entry on %i!\n', yyyymmddEntry(d));
            continue; % Do not enter on this event
        end
        ie=ie(end); 

        %         entryPriceC=bid(isEntryIdx); % At market
        %         entryPriceC=ask(isEntryIdx); % Limit
        entryPriceC=(bid(isEntryIdx)+ask(isEntryIdx))/2; % Midpoint
        exitPriceC=bid(isExitIdx(ie));
        
        hhmmssEntry=str2double(datestr(dn(isEntryIdx), 'HHMMSS'));
        
        yyyymmddExit=str2double(datestr(dn(isExitIdx(ie)), 'yyyymmdd'));
        hhmmssExit=str2double(datestr(dn(isExitIdx(ie)), 'HHMMSS'));

        %% Put
        fid=fopen(['//I3/Options_data/LO.', contract, '/pLO', contract, stkPriceStr, 'P_BBO_', dateRange, '.csv']);
        assert(fid~=-1);
                
        dn1=[];
        bid1=[];
        ask1=[];
        
        while (1)
            C=textscan(fid, '%u%s%f%f', len, 'Delimiter', ',');
            if (isempty(C))
                break;
            end
            if (length(C{1, 1})==0)
                break;
            end
            
            tday=num2str(C{1, 1});
            hhmmssfff=C{1, 2};
            
            bid1=[bid1; C{1, 3}];
            ask1=[ask1; C{1, 4}];
            
            dn1=[dn1; datenum(cellstr([tday, repmat(' ', size(hhmmssfff)), char(hhmmssfff)]), 'yyyymmdd HH:MM:SS.FFF')];
        end
        
        fclose(fid);
        
        %%%
        
        bid=NaN(size(bid1));
        ask=NaN(size(bid));
        dn=NaN(size(bid));
        
        t=1;
        t1=1;
        
        while (t1 <= length(dn1) )
            if (bid1(t1) ~= 0 && ask1(t1) ~= 0)
                bid(t, 1)=bid1(t1);
                ask(t, 1)=ask1(t1);
                dn(t)=dn1(t1);
                t=t+1;
            end
            t1=t1+1;
        end
        
        lastGoodData=find(isfinite(dn));
        lastGoodData=lastGoodData(end);
        
        dn=dn(1:lastGoodData)';
        bid=bid(1:lastGoodData, :);
        ask=ask(1:lastGoodData, :);
        
        bid=fillMissingData(bid);
        ask=fillMissingData(ask);
        
        goodData=dn >= datenum(firstDateTime, 'yyyymmdd HH:MM:SS') & dn <= datenum(lastDateTime, 'yyyymmdd HH:MM:SS');
        
        dn=dn(goodData);
        bid=bid(goodData);
        ask=ask(goodData);
                
        hhmmPut=str2double(cellstr(datestr(dn, 'HHMM')));
        yyyymmddPut=yyyymmdd(datetime(dn, 'ConvertFrom', 'datenum'))';

        isEntry=hhmmPut < entryTime & yyyymmddPut==yyyymmddEntry(d); 
        isExit=hhmmPut < exitTime & yyyymmddPut == yyyymmddExit;
        isEntryIdx=find(isEntry);
        isEntryIdx=isEntryIdx(end); % use latest entry
        isExitIdx=find(isExit);
        

        if (isempty(isExitIdx))
            fprintf(1, '    Missing put data on exit date %i: skipping...\n', yyyymmddExit);
            continue;
        end

        %         entryPriceP=bid(isEntryIdx); % At market
        %         entryPriceP=ask(isEntryIdx); % At limit
        entryPriceP=(bid(isEntryIdx)+ask(isEntryIdx))/2; % Midpoint
        exitPriceP=bid(isExitIdx(end)); % Select last tick to exit
        
        
        %%
        PL=(exitPriceC-entryPriceC)+(exitPriceP-entryPriceP);
        fprintf(1, '%s-%s: PL=%5.2f\n', datestr(dn(isEntryIdx), 'yyyymmdd HH:MM:SS'), datestr(dn(isExitIdx(end)), 'yyyymmdd HH:MM:SS'), PL);
        cumPL=cumPL+PL;
    end
    
end

fprintf(1, 'cumPL=%5.2f\n', cumPL);

% Entry at mid, exit at MKT
% 20120314 08:59:59-20120314 10:30:59: PL=-0.38
% 20120321 08:59:59-20120321 10:30:58: PL=-0.32
% 20120328 08:59:59-20120328 10:30:59: PL= 0.19
% 20120404 08:59:59-20120404 10:30:58: PL=-1.67
% 20120411 08:59:59-20120411 10:30:59: PL=-0.53
% 20120418 08:59:59-20120418 10:30:59: PL=-0.11
% 20120425 08:59:59-20120425 10:30:59: PL=-0.30
% 20120502 08:59:59-20120502 10:30:58: PL= 0.09
% 20120516 08:59:59-20120516 10:30:55: PL=-3.19
% 20120523 08:59:56-20120523 10:30:50: PL=-2.06
% 20120530 08:59:59-20120530 10:30:59: PL= 0.50
% 20120613 08:59:57-20120613 10:30:59: PL=-1.84
% 20120620 08:59:59-20120620 10:30:58: PL= 0.63
% 20120627 08:59:59-20120627 10:30:59: PL=-0.49
% 20120704 08:58:05-20120704 10:30:13: PL=-0.22
% 20120711 08:59:59-20120711 10:30:59: PL=-1.50
% 20120718 08:59:59-20120718 10:30:59: PL=-0.50
% 20120725 08:59:57-20120725 10:30:59: PL= 0.31
% 20120801 08:59:59-20120801 10:30:48: PL=-0.51
% 20120808 08:59:59-20120808 10:30:59: PL=-0.61
% 20120815 08:59:58-20120815 10:30:23: PL=-0.63
% 20120822 08:59:59-20120822 10:30:51: PL=-1.59
% 20120829 08:59:59-20120829 10:30:59: PL=-1.23
% 20120905 08:59:59-20120905 10:29:57: PL= 0.43
% 20120912 08:59:58-20120912 10:30:59: PL=-0.64
% 20120919 08:59:59-20120919 10:30:59: PL=-2.92
% 20120926 08:59:59-20120926 10:30:52: PL=-0.11
% 20121003 08:59:59-20121003 10:30:59: PL=-0.39
% 20121010 08:59:58-20121010 10:30:56: PL=-0.72
% 20121017 08:59:59-20121017 10:30:54: PL=-0.86
% 20121024 08:59:59-20121024 10:30:51: PL=-2.26
% 20121031 08:59:59-20121031 10:30:56: PL=-0.20
% 20121107 08:59:59-20121107 10:30:59: PL= 0.75
% 20121114 08:59:59-20121114 10:30:59: PL=-0.50
% 20121121 08:59:59-20121121 10:30:59: PL= 0.07
% 20121128 08:59:59-20121128 10:30:56: PL= 0.15
% 20121205 08:59:57-20121205 10:29:57: PL=-0.04
% 20121212 08:59:59-20121212 10:30:59: PL=-0.37
% 20121219 08:59:59-20121219 10:30:54: PL=-0.22
% 20121226 08:59:59-20121226 10:30:57: PL=-0.61
% 20130102 08:59:59-20130102 10:30:58: PL=-0.24
% 20130109 08:59:59-20130109 10:30:45: PL=-1.13
% 20130116 08:59:59-20130116 10:30:59: PL=-0.48
% 20130123 08:59:59-20130123 10:30:59: PL= 0.03
% 20130130 08:59:59-20130130 10:30:59: PL=-0.12
% 20130206 08:59:59-20130206 10:30:59: PL=-0.37
% 20130213 08:59:59-20130213 10:30:57: PL=-0.03
% 20130220 08:59:59-20130220 10:30:57: PL= 0.04
% 20130227 08:59:59-20130227 10:30:59: PL=-0.44
% cumPL=-27.11

