% Short at Thursday 9:00, exit at Friday 14:25.
clear;

entryDay=5; % Thurs
exitDay=6; % Fri
entryTime=900;
exitTime=1425;
otm=0.05; % Buy 10% OTM put and call as hedge

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
               
        %         stkPriceStr=num2str(roundPrice(opn(d)));
        stkPriceOTMCallStr=num2str(roundPrice(opn(d)*(1+otm)));
        stkPriceOTMPutStr=num2str(roundPrice(opn(d)*(1-otm)));
        
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
        fid=fopen(['//I3//Options_data/LO.', contract, '/pLO', contract, stkPriceOTMCallStr, 'C_BBO_', dateRange, '.csv']);
        if (fid==-1)
            fprintf(1, '   Missing call strike price at %s on %i: skipping.\n', stkPriceOTMCallStr, yyyymmddEntry(d));
            continue;
        end
        
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
        ie=find(dn(isExitIdx)-dn(isEntryIdx) > 0 & dn(isExitIdx)-dn(isEntryIdx) < 1.5);
        if (isempty(ie))
            fprintf(1, '    ***Cannot find call exit date: skipping entry on %i!\n', yyyymmddEntry(d));
            continue; % Do not enter on this event
        end
        ie=ie(end); 

        entryPriceC=(bid(isEntryIdx)+ask(isEntryIdx))/2;
        exitPriceC=bid(isExitIdx(ie));
        
        hhmmssEntry=str2double(datestr(dn(isEntryIdx), 'HHMMSS'));
        
        yyyymmddExit=str2double(datestr(dn(isExitIdx(ie)), 'yyyymmdd'));
        hhmmssExit=str2double(datestr(dn(isExitIdx(ie)), 'HHMMSS'));

        %% Put
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

        entryPriceP=(bid(isEntryIdx)+ask(isEntryIdx))/2; 
        exitPriceP=bid(isExitIdx(end)); % Select last tick to exit
                  
        %%
        PL=(exitPriceC-entryPriceC)+(exitPriceP-entryPriceP);
        fprintf(1, '%s-%s: PL=%5.2f\n', datestr(dn(isEntryIdx), 'yyyymmdd HH:MM:SS'), datestr(dn(isExitIdx(end)), 'yyyymmdd HH:MM:SS'), PL);
        cumPL=cumPL+PL;
    end
    
end

fprintf(1, 'cumPL=%5.2f\n', cumPL);

% All PL are based on mid-quote entry, MKT exit.

% % otm=0.05
% 20120301 08:59:59-20120302 14:24:59: PL=-0.12
% 20120308 08:59:46-20120309 10:29:48: PL=-0.08
% 20120315 08:59:35-20120316 14:24:58: PL=-0.27
% 20120322 08:59:59-20120323 14:24:57: PL=-0.05
% 20120329 08:59:57-20120330 14:24:55: PL= 0.08
%     ***Cannot find call exit date: skipping entry on 20120405!
% 20120412 08:59:59-20120413 14:24:59: PL=-0.41
% 20120419 08:59:57-20120420 14:24:59: PL=-0.19
% 20120426 08:59:58-20120427 14:24:59: PL=-0.32
% 20120503 08:59:59-20120504 14:24:59: PL= 2.30
% 20120510 08:59:59-20120511 14:24:59: PL=-0.19
% 20120517 08:59:59-20120518 14:24:58: PL= 0.23
% 20120524 08:59:59-20120525 14:24:58: PL=-0.17
% 20120531 08:59:59-20120601 14:24:57: PL= 1.58
% 20120607 08:59:59-20120608 14:24:56: PL= 0.19
% 20120614 08:59:59-20120615 14:24:59: PL=-0.18
% 20120621 08:59:59-20120622 14:24:58: PL=-0.24
% 20120628 08:59:59-20120629 14:24:57: PL= 0.77
% 20120705 08:59:59-20120706 14:24:59: PL=-0.14
% 20120712 08:59:57-20120713 14:24:58: PL=-0.19
% 20120719 08:59:56-20120720 14:24:59: PL=-0.02
% 20120726 08:59:59-20120727 14:24:59: PL=-0.27
% 20120802 08:59:59-20120803 14:24:44: PL= 0.47
%     ***Cannot find call exit date: skipping entry on 20120809!
% 20120816 08:59:58-20120817 14:24:59: PL=-0.10
% 20120823 08:59:59-20120824 14:24:59: PL=-0.07
% 20120830 08:59:59-20120831 14:24:53: PL=-0.18
% 20120906 08:59:59-20120907 14:24:59: PL=-0.25
% 20120913 08:59:40-20120914 14:24:59: PL= 0.27
% 20120920 08:59:59-20120921 14:24:59: PL=-0.67
% 20120927 08:59:59-20120928 14:24:57: PL=-0.46
% 20121004 08:59:59-20121005 14:24:59: PL=-0.22
% 20121011 08:59:59-20121012 14:24:59: PL=-0.31
% 20121018 08:59:59-20121019 14:24:56: PL=-0.15
% 20121025 08:59:59-20121026 14:24:58: PL=-0.28
% 20121101 08:59:16-20121102 14:24:59: PL=-0.01
% 20121108 08:59:57-20121109 10:29:35: PL=-0.15
% 20121115 08:59:59-20121116 14:24:59: PL= 0.01
% 20121122 08:58:41-20121123 13:44:42: PL=-0.70
% 20121129 08:59:59-20121130 14:24:59: PL=-0.02
% 20121206 08:59:58-20121207 14:24:59: PL=-0.15
% 20121213 08:59:59-20121214 14:24:58: PL=-0.05
% 20121220 08:59:59-20121221 14:24:59: PL=-0.02
% 20121227 08:59:59-20121228 14:24:58: PL=-0.04
% 20130103 08:59:59-20130104 14:24:58: PL=-0.28
% 20130110 08:59:59-20130111 14:24:59: PL=-0.05
% 20130117 08:59:59-20130118 14:24:58: PL=-0.27
% 20130124 08:59:52-20130125 14:24:59: PL= 0.00
% 20130131 08:59:57-20130201 14:24:56: PL=-0.11
% 20130207 08:59:48-20130208 14:24:58: PL=-0.07
% 20130214 08:59:59-20130215 14:24:55: PL= 0.29
% 20130221 08:59:59-20130222 14:24:57: PL=-0.32
% cumPL=-1.59