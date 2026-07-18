% Short at Thursday 9:00, exit at next Wed 10:29.

clear;

entryDay=6; % Fri
exitDay=6; % Fri

entryTime=930;
exitTime=1530;

contracts={...
    'H12', ...
    'M12', ...
    'U12', ...
    'U12', ...
    'Z12', ...
    'Z12', ...
    'H13'};

FdateRanges={...
    '20120301_20120319', ...
    '20120320_20120519', ...
    '20120601_20120930', ...
    '20120601_20120930', ...
    '20120901_20121231', ...
    '20120901_20121231', ...
    '20121201_20130227'};
    
dateRanges={...
    '20120301_20120319', ...
    '20120310_20120619', ...
    '20120601_20120617', ...
    '', ...
    '20120901_20120917', ...
    '20120913_20121213', ...
    '20121213_20130227'};

firstDateTimes={...
    '20120301 9:00:00', ...
    '20120310 9:00:00', ...
    '20120610 9:00:00', ...
    '20120618 9:00:00', ...
    '20120910 9:00:00', ...
    '20120918 9:00:00', ...
    '20121210 9:00:00'};

lastDateTimes={...
    '20120309 16:15:00', ...
    '20120609 16:15:00', ...
    '20120617 16:15:00', ...
    '20120909 16:15:00', ...
    '20120917 16:15:00', ...
    '20121209 16:15:00', ...
    '20130227 16:15:00'};

assert(length(contracts)==length(dateRanges));
assert(length(contracts)==length(firstDateTimes));
assert(length(contracts)==length(lastDateTimes));

cumPL=0;
for c=1:length(contracts)
    contract=contracts{c};
    FdateRange=FdateRanges{c};
    dateRange=dateRanges{c};
    firstDateTime=firstDateTimes{c};
    lastDateTime=lastDateTimes{c};
        
    % Get futures price to determine what strike price is ATM
    load(['//I3/Futures_data/inputData_ES', contract, '_BBO_', FdateRange, '.mat'], 'dn', 'bid', 'ask');
    
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
    
    roundPrice=@(x) round(0.2*x)*5; % round(x/minIncrement)*minIncrement
    
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
        if (~isempty(dateRange))
            fid=fopen(['//I3//Options_data/ES.', contract, '/pES', contract, stkPriceStr, 'C_BBO_', dateRange, '.csv']);
        else
            fid=fopen(['//I3//Options_data/ES.', contract, '/pES', contract, stkPriceStr, 'C_BBO.csv']);
        end
        
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
        ie=find(dn(isExitIdx)-dn(isEntryIdx) > 0 & dn(isExitIdx)-dn(isEntryIdx) < 1);
        if (isempty(ie))
            fprintf(1, '    ***Cannot find call exit date: skipping entry on %i!\n', yyyymmddEntry(d));
            continue; % Do not enter on this event
        end
        ie=ie(end); 

        %         entryPriceC=bid(isEntryIdx); % At market
        %         entryPriceC=ask(isEntryIdx); % Limit
        entryPriceC=(bid(isEntryIdx)+ask(isEntryIdx))/2; % Midpoint
        exitPriceC=ask(isExitIdx(ie));
        
        hhmmssEntry=str2double(datestr(dn(isEntryIdx), 'HHMMSS'));
        
        yyyymmddExit=str2double(datestr(dn(isExitIdx(ie)), 'yyyymmdd'));
        hhmmssExit=str2double(datestr(dn(isExitIdx(ie)), 'HHMMSS'));

        %% Put
        if (~isempty(dateRange))
            fid=fopen(['//I3/Options_data/ES.', contract, '/pES', contract, stkPriceStr, 'P_BBO_', dateRange, '.csv']);
        else
            fid=fopen(['//I3//Options_data/ES.', contract, '/pES', contract, stkPriceStr, 'P_BBO.csv']);
        end

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
        exitPriceP=ask(isExitIdx(end)); % Select last tick to exit
        
        
        %%
        PL=-(exitPriceC-entryPriceC)-(exitPriceP-entryPriceP);
        fprintf(1, '%s %s-%s: PL=%5.2f\n',  contract, datestr(dn(isEntryIdx), 'yyyymmdd HH:MM:SS'), datestr(dn(isExitIdx(end)), 'yyyymmdd HH:MM:SS'), PL);
        cumPL=cumPL+PL;
    end
    
end

fprintf(1, 'cumPL=%5.2f\n', cumPL);



% Use LMT at ask for entry, MKT for exit at 16:00
% H12 20120302 09:29:44-20120302 15:59:59: PL= 1.38
% H12 20120309 09:29:59-20120309 15:59:59: PL= 0.00
% M12 20120323 09:29:58-20120323 15:59:59: PL= 1.88
% M12 20120330 09:29:59-20120330 15:59:59: PL=-1.75
% M12 20120413 09:29:59-20120413 15:59:59: PL=-3.88
% M12 20120420 09:29:59-20120420 15:59:59: PL= 0.75
% M12 20120427 09:29:59-20120427 15:59:59: PL=-1.13
% M12 20120504 09:29:59-20120504 15:59:59: PL=-1.88
% M12 20120511 09:29:59-20120511 15:59:59: PL=-2.00
% M12 20120518 09:29:59-20120518 15:59:59: PL=-3.00
% U12 20120615 09:29:59-20120615 15:59:59: PL= 1.50
% U12 20120622 09:29:57-20120622 15:59:56: PL= 1.63
% U12 20120629 09:29:59-20120629 15:59:59: PL=-2.63
% U12 20120706 09:29:59-20120706 15:59:59: PL= 2.50
% U12 20120713 09:29:57-20120713 15:59:59: PL=-1.38
% U12 20120720 09:29:55-20120720 15:59:59: PL= 0.13
% U12 20120727 09:29:59-20120727 15:59:59: PL=-4.88
% U12 20120803 09:29:59-20120803 15:59:59: PL=-1.13
% U12 20120810 09:29:55-20120810 15:59:59: PL= 0.13
% U12 20120817 09:29:59-20120817 15:59:59: PL= 2.38
% U12 20120824 09:29:59-20120824 15:59:56: PL=-2.00
% U12 20120831 09:29:59-20120831 15:59:59: PL=-0.25
% U12 20120907 09:29:59-20120907 15:59:59: PL= 1.75
% Z12 20120914 09:29:44-20120914 15:59:59: PL=-5.25
% Z12 20120921 09:29:59-20120921 15:59:59: PL= 0.38
% Z12 20120928 09:29:56-20120928 15:59:59: PL=-2.88
% H13 20121214 09:29:59-20121214 15:59:59: PL= 0.25
% H13 20121221 09:29:59-20121221 15:59:59: PL= 2.75
% H13 20121228 09:29:56-20121228 15:59:59: PL=-2.88
% H13 20130104 09:29:59-20130104 15:59:59: PL=-1.88
% H13 20130111 09:29:55-20130111 15:59:59: PL= 0.38
% H13 20130118 09:29:59-20130118 15:59:59: PL= 2.50
% H13 20130125 09:29:59-20130125 15:59:59: PL=-1.00
% H13 20130201 09:29:59-20130201 15:59:59: PL=-0.75
% H13 20130208 09:29:59-20130208 15:59:59: PL=-1.00
% H13 20130215 09:29:59-20130215 15:59:59: PL= 0.75
% H13 20130222 09:29:59-20130222 15:59:59: PL=-0.25
% cumPL=-20.75

% Use LMT at ask for entry, MKT for exit at 15:30

% H12 20120302 09:29:44-20120302 15:29:59: PL= 0.63
% H12 20120309 09:29:59-20120309 15:29:59: PL= 0.00
% M12 20120323 09:29:58-20120323 15:29:56: PL= 2.13
% M12 20120330 09:29:59-20120330 15:29:59: PL=-1.50
% M12 20120413 09:29:59-20120413 15:29:40: PL=-2.63
% M12 20120420 09:29:59-20120420 15:29:55: PL= 0.75
% M12 20120427 09:29:59-20120427 15:29:58: PL=-0.88
% M12 20120504 09:29:59-20120504 15:29:36: PL=-1.88
% M12 20120511 09:29:59-20120511 15:29:56: PL=-1.25
% M12 20120518 09:29:59-20120518 15:29:59: PL=-2.00
% U12 20120615 09:29:59-20120615 15:29:59: PL= 0.75
% U12 20120622 09:29:57-20120622 15:29:58: PL= 1.88
% U12 20120629 09:29:59-20120629 15:29:40: PL=-2.88
% U12 20120706 09:29:59-20120706 15:29:59: PL= 1.75
% U12 20120713 09:29:57-20120713 15:29:49: PL=-1.63
% U12 20120720 09:29:55-20120720 15:29:45: PL= 0.38
% U12 20120727 09:29:59-20120727 15:29:59: PL=-5.13
% U12 20120803 09:29:59-20120803 15:29:58: PL=-2.63
% U12 20120810 09:29:55-20120810 15:29:57: PL= 0.38
% U12 20120817 09:29:59-20120817 15:29:48: PL= 2.88
% U12 20120824 09:29:59-20120824 15:29:58: PL=-2.00
% U12 20120831 09:29:59-20120831 15:29:59: PL= 1.00
% U12 20120907 09:29:59-20120907 15:29:55: PL= 2.50
% Z12 20120914 09:29:44-20120914 15:29:59: PL=-4.25
% Z12 20120921 09:29:59-20120921 15:29:50: PL= 0.63
% Z12 20120928 09:29:56-20120928 15:29:59: PL=-1.38
% H13 20121214 09:29:59-20121214 15:29:57: PL=-0.25
% H13 20121221 09:29:59-20121221 15:29:54: PL= 2.25
% H13 20121228 09:29:56-20121228 15:29:52: PL=-1.13
% H13 20130104 09:29:59-20130104 15:29:43: PL=-1.38
% H13 20130111 09:29:55-20130111 15:29:55: PL= 0.63
% H13 20130118 09:29:59-20130118 15:29:54: PL= 3.00
% H13 20130125 09:29:59-20130125 15:29:56: PL=-0.75
% H13 20130201 09:29:59-20130201 15:29:59: PL=-1.00
% H13 20130208 09:29:59-20130208 15:29:39: PL=-0.75
% H13 20130215 09:29:59-20130215 15:29:37: PL= 0.25
% H13 20130222 09:29:59-20130222 15:29:59: PL=-0.50
% cumPL=-14.00
