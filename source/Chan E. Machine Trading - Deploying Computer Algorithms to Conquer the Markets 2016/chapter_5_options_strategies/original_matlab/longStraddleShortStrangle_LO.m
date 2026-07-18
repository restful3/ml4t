% Short at Thursday 9:00, exit at next Wed 10:29.
clear;

entryDay=5; % Thurs
exitDay=4; % Wed
otm=10; % Buy $10 OTM put and call as hedge

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
    '20120310 08:30:00', ...
    '20120410 08:30:00', ...
    '20120510 08:30:00', ...
    '20120610 08:30:00', ...
    '20120710 08:30:00', ...
    '20120810 08:30:00', ...
    '20120910 08:30:00', ...
    '20121010 08:30:00', ...
    '20121110 08:30:00', ...
    '20121210 08:30:00', ...
    '20130110 08:30:00', ...
    '20130210 08:30:00'};

lastDateTimes={...
    '20120309 10:30:00', ...
    '20120409 10:30:00', ...
    '20120509 10:30:00', ...
    '20120609 10:30:00', ...
    '20120709 10:30:00', ...
    '20120809 10:30:00', ...
    '20120909 10:30:00', ...
    '20121009 10:30:00', ...
    '20121109 10:30:00', ...
    '20121209 10:30:00', ...
    '20130109 10:30:00', ...
    '20130209 10:30:00', ...
    '20130309 10:30:00'};

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
    
    entryTime=900;
    exitTime=1029;
    len=1000000;
        
    hhmm=str2double(cellstr(datestr(dnFut, 'HHMM')));
    
    isEntryFut=hhmm < entryTime & fwdshift(1, hhmm) >= entryTime & weekday(dnFut')==entryDay;
    
    opn=midFut(isEntryFut);
    yyyymmddFut=yyyymmdd(datetime(dnFut, 'ConvertFrom', 'datenum'));
    yyyymmddEntry=yyyymmddFut(isEntryFut);
    
    isEntryFutIdx=find(isEntryFut);
    
    roundPrice=@(x) round(2*x)/2; % =1/minPriceIncr
    
    % Iterate through each event date
    for d=1:length(opn)
               
        stkPriceStr=num2str(roundPrice(opn(d)));
        stkPriceOTMCallStr=num2str(roundPrice(opn(d)+otm));
        stkPriceOTMPutStr=num2str(roundPrice(opn(d)-otm));

        if (isempty(regexp(stkPriceStr, '\.')))
            stkPriceStr=[stkPriceStr, '00'];
        elseif (isempty(regexp(stkPriceStr, '\.\d\d')))
            stkPriceStr=regexprep(stkPriceStr, '\.', '');
            stkPriceStr=[stkPriceStr, '0'];
        else
            stkPriceStr=regexprep(stkPriceStr, '\.', '');
        end
        
        if (isempty(regexp(stkPriceOTMCallStr, '\.')))
            stkPriceOTMCallStr=[stkPriceOTMCallStr, '00'];
        elseif (isempty(regexp(stkPriceOTMCallStr, '\.\d\d')))
            stkPriceOTMCallStr=regexprep(stkPriceOTMCallStr, '\.', '');
            stkPriceOTMCallStr=[stkPriceOTMCallStr, '0'];
        else
            stkPriceOTMCallStr=regexprep(stkPriceOTMCallStr, '\.', '');
        end
        
        if (isempty(regexp(stkPriceOTMPutStr, '\.')))
            stkPriceOTMPutStr=[stkPriceOTMPutStr, '00'];
        elseif (isempty(regexp(stkPriceOTMPutStr, '\.\d\d')))
            stkPriceOTMPutStr=regexprep(stkPriceOTMPutStr, '\.', '');
            stkPriceOTMPutStr=[stkPriceOTMPutStr, '0'];
        else
            stkPriceOTMPutStr=regexprep(stkPriceOTMPutStr, '\.', '');
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

        isExit=hhmmCall < exitTime & yyyymmddCall > yyyymmddEntry(d);
        isEntryIdx=find(isEntry);
        isEntryIdx=isEntryIdx(end); % use latest entry 
        isExitIdx=find(isExit);
                     
        % Confirm futures dates are same as entry dates for options
        assert(yyyymmddEntry(d)==str2double(datestr(dn(isEntryIdx), 'yyyymmdd')));
        
        % Pick farthest exit date that is within a calendar week but more
        % than 3 calendar days
        ie=find(dn(isExitIdx)-dn(isEntryIdx) > 3 & dn(isExitIdx)-dn(isEntryIdx) < 6.5);
        if (isempty(ie))
            fprintf(1, '    ***Cannot find call exit date: skipping entry on %i!\n', yyyymmddEntry(d));
            continue; % Do not enter on this event
        end
        ie=ie(end); 

        entryPriceC=ask(isEntryIdx);
        exitPriceC=bid(isExitIdx(ie));
        
        hhmmssEntry=str2double(datestr(dn(isEntryIdx), 'HHMMSS'));
        
        yyyymmddExit=str2double(datestr(dn(isExitIdx(ie)), 'yyyymmdd'));
        hhmmssExit=str2double(datestr(dn(isExitIdx(ie)), 'HHMMSS'));

        %% OTM Call
        fid=fopen(['//I3/Options_data/LO.', contract, '/pLO', contract, stkPriceOTMCallStr, 'C_BBO_', dateRange, '.csv']);
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
                
        hhmmOMTCall=str2double(cellstr(datestr(dn, 'HHMM')));
        yyyymmddOMTCall=yyyymmdd(datetime(dn, 'ConvertFrom', 'datenum'))';

        isEntry=hhmmOMTCall < entryTime & yyyymmddOMTCall==yyyymmddEntry(d); 

        isExit=hhmmOMTCall < exitTime & yyyymmddOMTCall == yyyymmddExit;
        isEntryIdx=find(isEntry);
        isEntryIdx=isEntryIdx(end); % use latest entry
        isExitIdx=find(isExit);
        
        if (isempty(isExitIdx))
            fprintf(1, '    Missing OMT call data on exit date %i: skipping...\n', yyyymmddExit);
            continue;
        end

        entryPriceC2=bid(isEntryIdx); 
        exitPriceC2=ask(isExitIdx(end)); % Select last tick to exit

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

        entryPriceP=ask(isEntryIdx); 
        exitPriceP=bid(isExitIdx(end)); % Select last tick to exit
            
        %% OTM Put
        fid=fopen(['//I3/Options_data/LO.', contract, '/pLO', contract, stkPriceOTMPutStr, 'P_BBO_', dateRange, '.csv']);
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
                
        hhmmOMTPut=str2double(cellstr(datestr(dn, 'HHMM')));
        yyyymmddOMTPut=yyyymmdd(datetime(dn, 'ConvertFrom', 'datenum'))';

        isEntry=hhmmOMTPut < entryTime & yyyymmddOMTPut==yyyymmddEntry(d); 

        isExit=hhmmOMTPut < exitTime & yyyymmddOMTPut == yyyymmddExit;
        isEntryIdx=find(isEntry);
        isEntryIdx=isEntryIdx(end); % use latest entry
        isExitIdx=find(isExit);
        
        if (isempty(isExitIdx))
            fprintf(1, '    Missing OMT put data on exit date %i: skipping...\n', yyyymmddExit);
            continue;
        end

        entryPriceP2=bid(isEntryIdx); 
        exitPriceP2=ask(isExitIdx(end)); % Select last tick to exit
        
        %%
        PL=+(exitPriceC-entryPriceC)+(exitPriceP-entryPriceP) - (exitPriceC2-entryPriceC2)-(exitPriceP2-entryPriceP2);
        fprintf(1, '%s-%s: PL=%5.2f\n', datestr(dn(isEntryIdx), 'yyyymmdd HH:MM:SS'), datestr(dn(isExitIdx(end)), 'yyyymmdd HH:MM:SS'), PL);
        cumPL=cumPL+PL;
    end
    
end

fprintf(1, 'cumPL=%5.2f\n', cumPL);

% 
% 20120301 08:59:59-20120307 10:28:56: PL=-0.48
%     ***Cannot find call exit date: skipping entry on 20120308!
% 20120315 08:59:19-20120321 10:28:59: PL=-0.62
% 20120322 08:59:57-20120328 10:28:56: PL=-0.59
% 20120329 08:59:55-20120404 10:28:10: PL=-0.64
% 20120405 08:59:47-20120409 10:28:19: PL=-0.28
% 20120412 08:59:58-20120418 10:28:59: PL=-0.70
% 20120419 08:59:59-20120425 10:28:22: PL=-0.66
% 20120426 08:59:31-20120502 10:28:59: PL=-0.46
% 20120503 08:59:51-20120509 10:28:59: PL= 4.34
% 20120510 08:59:59-20120516 10:28:51: PL=-0.51
% 20120517 08:59:52-20120523 10:28:59: PL=-0.75
% 20120524 08:59:59-20120530 10:28:35: PL=-0.10
% 20120531 08:59:34-20120606 10:28:10: PL=-0.81
%     ***Cannot find call exit date: skipping entry on 20120607!
% 20120614 08:59:58-20120620 10:28:55: PL=-1.00
% 20120621 08:59:58-20120627 10:28:59: PL=-0.97
% 20120628 08:59:43-20120703 10:27:48: PL= 1.81
% 20120705 08:58:31-20120709 10:28:57: PL=-0.59
% 20120712 08:59:52-20120718 10:26:09: PL= 0.11
% 20120719 08:59:41-20120725 10:28:50: PL=-0.31
% 20120726 08:59:57-20120801 10:28:58: PL=-0.62
% 20120802 08:59:59-20120808 10:23:32: PL= 1.73
%     ***Cannot find call exit date: skipping entry on 20120809!
% 20120816 08:59:47-20120822 10:28:46: PL=-0.20
% 20120823 08:59:59-20120829 10:28:35: PL=-0.37
% 20120830 08:59:24-20120905 10:28:29: PL=-0.58
%     ***Cannot find call exit date: skipping entry on 20120906!
% 20120913 08:59:58-20120919 10:28:59: PL= 0.14
% 20120920 08:59:59-20120926 10:28:37: PL=-0.75
% 20120927 08:59:59-20121003 10:28:51: PL=-0.73
% 20121004 08:59:59-20121009 10:26:53: PL=-0.55
% 20121011 08:59:59-20121017 10:28:53: PL=-0.72
% 20121018 08:59:59-20121024 10:28:59: PL= 0.45
% 20121025 08:59:57-20121031 10:28:15: PL=-0.89
% 20121101 08:59:46-20121107 10:24:10: PL=-0.97
%     ***Cannot find call exit date: skipping entry on 20121108!
% 20121115 08:59:58-20121121 10:28:52: PL=-0.73
% 20121122 08:00:13-20121128 10:28:58: PL=-2.12
% 20121129 08:59:26-20121205 10:20:10: PL=-0.72
%     ***Cannot find call exit date: skipping entry on 20121206!
% 20121213 08:59:59-20121219 10:28:50: PL=-0.38
% 20121220 08:59:55-20121226 10:28:56: PL=-0.18
% 20121227 08:59:59-20130102 10:28:53: PL=-0.16
% 20130103 08:54:05-20130109 10:16:15: PL=-1.27
% 20130110 08:59:59-20130116 10:28:47: PL=-0.36
% 20130117 08:59:58-20130123 10:28:36: PL=-0.77
% 20130124 08:59:35-20130130 10:28:36: PL=-0.11
% 20130131 08:58:43-20130206 10:27:22: PL=-0.25
%     ***Cannot find call exit date: skipping entry on 20130207!
% 20130214 08:59:55-20130220 10:28:55: PL=-0.17
% 20130221 08:59:59-20130227 10:28:34: PL=-1.02
% cumPL=-15.51